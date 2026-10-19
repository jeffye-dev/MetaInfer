#!/usr/bin/env python3
"""MetaInfer claude agent 环境检查。

检查项根据 MetaInfer 调用 claude agent 的方式推导：

* 二进制
  - orchestrator / WebUI QA：``$METAINFER_CLAUDE_BIN``，默认 ``ccb``
    （metainfer/tasks/*/orchestrator/cli.py、metainfer/server/qa.py）
  - multi-node worker：``$METAINFER_CCB`` / ``$CLAUDE_BIN``，否则 PATH 上的
    ``ccb`` / ``claude``（metainfer/worker/jobs.py）
* 命令行（SubAgentManager._build_claude_command）::

    <bin> -p --output-format stream-json --input-format text --verbose \\
          --permission-mode <$METAINFER_PERMISSION_MODE，默认 bypassPermissions> \\
          --add-dir <workdir> [--model M] [--effort <$METAINFER_EFFORT，默认 max>] \\
          [--resume ID | --session-id ID]          # prompt 从 stdin 读入

* 环境（SubAgentManager._build_env）：继承父进程环境 + ``DISABLE_INTERACTIVITY=1``；
  bypassPermissions 时 orchestrator / QA 额外设置 ``IS_SANDBOX=1``（root 下必需），
  worker 不设置。
* 鉴权：MetaInfer 从不传 API key，完全依赖 claude 自身配置——
  ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN``（+ ``ANTHROPIC_BASE_URL``）/
  ``CLAUDE_CODE_OAUTH_TOKEN`` 环境变量、``$CLAUDE_CONFIG_DIR/settings.json`` 的
  ``env`` / ``apiKeyHelper``、``/login`` 写入的凭据，或 Bedrock / Vertex / Foundry。

运行时环境（agent 生成的 kernel / 推理代码依赖）：

* CUDA：``nvidia-smi``（驱动版本、驱动支持的最高 CUDA、GPU 列表）和 ``nvcc``
  （PATH → ``$CUDA_HOME`` / ``$CUDA_PATH`` → /usr/local/cuda；toolkit 与 cudart 版本）
* torch / vLLM / transformers：在目标解释器（``--python``，默认
  ``$METAINFER_PYTHON``，否则当前解释器——与 MetaInfer launcher 启动 orchestrator
  的解释器一致）的子进程里 import，崩溃或卡住都不会拖垮本脚本；并交叉检查
  驱动 / toolkit / torch 的 CUDA 版本、vLLM 编译扩展与 torch 的 ABI、vLLM 依赖约束。

用法::

    python tools/env_check.py                  # 静态检查，不发起 API 请求
    python tools/env_check.py --live           # 再用 MetaInfer 同款命令真实调用一次 agent
    python tools/env_check.py --live --model claude-sonnet-5
    python tools/env_check.py --python /opt/venv/bin/python   # 检查指定解释器里的 torch / vLLM
    python tools/env_check.py --skip-runtime   # 只检查 claude agent

只依赖标准库。退出码：有 FAIL 为 1，否则为 0。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

# Defaults mirrored from metainfer/tasks/*/orchestrator/cli.py and
# metainfer/orchestrator/subagent_manager.py.
DEFAULT_CLAUDE_BIN = "ccb"
DEFAULT_PERMISSION_MODE = "bypassPermissions"
DEFAULT_EFFORT = "max"
VALID_PERMISSION_MODES = ("default", "acceptEdits", "plan", "bypassPermissions", "auto")
VALID_EFFORTS = ("low", "medium", "high", "max")
BACKEND_ALIASES = {
    "claude": "claude", "claude-code": "claude", "ccb": "claude",
    "codex": "codex", "openai-codex": "codex",
    "pi": "pi", "pi-coding-agent": "pi", "earendil": "pi",
}
# Every flag SubAgentManager._build_claude_command may pass. A CLI lacking
# one fails every sub-agent launch with "unknown option".
REQUIRED_FLAGS = (
    "--print", "--output-format", "--input-format", "--verbose",
    "--permission-mode", "--add-dir", "--model", "--effort",
    "--resume", "--session-id",
)
AUTH_ENV_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")
PROVIDER_ENV_VARS = ("CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY")
MODEL_ENV_VARS = (
    "ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL",
)
MIN_NODE_MAJOR = 18
LIVE_MARKER = "METAINFER_ENV_OK"
LIVE_PROMPT = f"Reply with exactly: {LIVE_MARKER}\n"
PROBE_MARKER = "__METAINFER_ENV_PROBE__"
DEFAULT_PROBE_TIMEOUT = 120
TORCH_INDEX_HINT = (
    "pip install torch --index-url https://download.pytorch.org/whl/cu<主版本><次版本>"
    "（例如 cu130 / cu129 / cu126；CUDA 主版本要 <= 驱动支持的最高 CUDA，并与 nvcc 一致）"
)

OK, INFO, WARN, FAIL = "OK", "INFO", "WARN", "FAIL"

INSTALL_HINT = (
    "安装 ccb: npm i -g claude-code-best\n"
    "或改用官方 Claude Code: npm i -g @anthropic-ai/claude-code，"
    "然后 export METAINFER_CLAUDE_BIN=claude"
)


@dataclass
class Check:
    level: str
    title: str
    detail: str = ""
    hint: str = ""


@dataclass
class Ctx:
    """Everything the checks read, injectable for tests."""
    env: Dict[str, str]
    home: Path
    cwd: Path
    euid: int
    platform: str

    @classmethod
    def current(cls) -> "Ctx":
        return cls(
            env=dict(os.environ),
            home=Path.home(),
            cwd=Path.cwd(),
            euid=os.geteuid() if hasattr(os, "geteuid") else -1,
            platform=sys.platform,
        )

    # Resolution below copies MetaInfer's exact lookups (``.get(k, default)``
    # vs ``or``), so an empty-string env var is judged the way MetaInfer
    # would treat it.
    @property
    def claude_bin(self) -> str:
        return self.env.get("METAINFER_CLAUDE_BIN", DEFAULT_CLAUDE_BIN)

    @property
    def permission_mode(self) -> str:
        return self.env.get("METAINFER_PERMISSION_MODE", DEFAULT_PERMISSION_MODE)

    @property
    def effort(self) -> str:
        return self.env.get("METAINFER_EFFORT", DEFAULT_EFFORT)

    @property
    def config_dir(self) -> Path:
        v = self.env.get("CLAUDE_CONFIG_DIR")
        return Path(v).expanduser() if v else self.home / ".claude"

    @property
    def metainfer_root(self) -> Path:
        v = self.env.get("METAINFER_ROOT")
        return Path(v).expanduser() if v else self.cwd

    def which(self, name: str) -> Optional[str]:
        return shutil.which(name, path=self.env.get("PATH")) if name else None

    def worker_bin(self) -> str:
        """metainfer.worker.jobs._find_ccb()."""
        explicit = self.env.get("METAINFER_CCB") or self.env.get("CLAUDE_BIN")
        if explicit:
            return explicit
        for name in ("ccb", "claude"):
            found = self.which(name)
            if found:
                return found
        return "ccb"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _truthy(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on")


def _tail(text: str, n: int = 12) -> str:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines[-n:])


def _run(cmd: List[str], ctx: Ctx, timeout: float = 30) -> Tuple[Optional[int], str]:
    """Run a short command; returns (returncode or None, stdout+stderr)."""
    env = dict(ctx.env)
    env.setdefault("DISABLE_INTERACTIVITY", "1")
    try:
        p = subprocess.run(cmd, env=env, stdin=subprocess.DEVNULL,
                           capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, f"超时（{timeout:.0f}s）"
    except OSError as e:
        return None, str(e)
    return p.returncode, (p.stdout + p.stderr).decode("utf-8", "replace")


def _writable(d: Path) -> bool:
    """Probe-write ``d`` (or its nearest existing parent, without creating
    anything) so read-only mounts / root_squash are caught too."""
    p = d
    while not p.exists():
        if p.parent == p:
            return False
        p = p.parent
    if not p.is_dir():
        return False
    try:
        fd, tmp = tempfile.mkstemp(prefix=".metainfer-env-check-", dir=str(p))
    except OSError:
        return False
    os.close(fd)
    os.unlink(tmp)
    return True


def _shebang_program(path: str) -> Optional[str]:
    try:
        with open(os.path.realpath(path), "rb") as f:
            first = f.readline(256).decode("utf-8", "replace").strip()
    except OSError:
        return None
    if not first.startswith("#!"):
        return None
    parts = first[2:].split()
    if parts and os.path.basename(parts[0]) == "env":
        parts = [p for p in parts[1:] if not p.startswith("-")]
    return parts[0] if parts else None


# --------------------------------------------------------------------------- #
# Static checks
# --------------------------------------------------------------------------- #


def check_knobs(ctx: Ctx) -> List[Check]:
    checks: List[Check] = []
    raw_backend = ctx.env.get("METAINFER_AGENT_BACKEND") or "claude"
    backend = BACKEND_ALIASES.get(raw_backend.strip().lower())
    mode, effort = ctx.permission_mode, ctx.effort

    if backend is None:
        checks.append(Check(FAIL, f"METAINFER_AGENT_BACKEND={raw_backend!r} 无效",
                            hint="取值 claude / codex / pi（默认 claude）"))
    elif backend != "claude":
        checks.append(Check(INFO, f"agent backend: {backend}",
                            "主任务 sub-agent 不走 claude；但 WebUI QA 分析和 multi-node worker "
                            "的 agent job 固定调用 claude，以下检查仍然适用"))

    if mode not in VALID_PERMISSION_MODES:
        checks.append(Check(FAIL, f"METAINFER_PERMISSION_MODE={mode!r} 无效（orchestrator 启动即退出）",
                            hint=f"取值 {' / '.join(VALID_PERMISSION_MODES)}；"
                                 "推荐 unset，使用默认 bypassPermissions"))
    elif mode in ("default", "plan"):
        checks.append(Check(WARN, f"METAINFER_PERMISSION_MODE={mode}",
                            "sub-agent 以 -p 非交互运行：default 模式会卡在每次 Edit/Write 的权限确认上，"
                            "plan 模式不会修改文件",
                            "unset METAINFER_PERMISSION_MODE（默认 bypassPermissions），或设为 acceptEdits / auto"))

    if effort not in VALID_EFFORTS:
        checks.append(Check(FAIL, f"METAINFER_EFFORT={effort!r} 无效（orchestrator 启动即退出）",
                            hint=f"取值 {' / '.join(VALID_EFFORTS)}；默认 max"))

    if not any(c.level in (WARN, FAIL) for c in checks):
        checks.append(Check(OK, f"MetaInfer agent 参数: backend={backend} "
                                f"permission_mode={mode} effort={effort}"))
    return checks


def check_interpreter(path: str, ctx: Ctx) -> List[Check]:
    """npm-installed CLIs are scripts (``#!/usr/bin/env node``): the runtime
    must be on the PATH the orchestrator hands to the agent."""
    prog = _shebang_program(path)
    name = os.path.basename(prog) if prog else ""
    if name not in ("node", "bun"):
        return []
    found = ctx.which(prog)
    if not found:
        hint = (f"安装 Node.js >= {MIN_NODE_MAJOR}（https://nodejs.org 或 apt / brew），并确保在 PATH 上"
                if name == "node" else "npm i -g bun，并确保在 PATH 上")
        return [Check(FAIL, f"{os.path.basename(path)} 依赖 {name}，但 PATH 上找不到",
                      f"shebang 来自 {os.path.realpath(path)}", hint)]
    if name != "node":
        return []
    rc, out = _run([found, "--version"], ctx)
    m = re.match(r"v?(\d+)\.", out.strip())
    if rc != 0 or not m:
        return [Check(WARN, f"无法确认 Node.js 版本（{found}）", _tail(out))]
    if int(m.group(1)) < MIN_NODE_MAJOR:
        return [Check(FAIL, f"Node.js 版本过低: {out.strip()}", found,
                      f"升级到 Node.js >= {MIN_NODE_MAJOR}")]
    return [Check(OK, f"Node.js {out.strip()}", found)]


def check_claude_bin(ctx: Ctx) -> List[Check]:
    name = ctx.claude_bin
    if not name:
        return [Check(FAIL, "METAINFER_CLAUDE_BIN 被设置为空字符串",
                      hint="unset METAINFER_CLAUDE_BIN（默认 ccb），或设为 claude / 绝对路径")]
    path = ctx.which(name)
    if not path:
        return [Check(FAIL, f"找不到 claude 二进制 {name!r}（$METAINFER_CLAUDE_BIN，默认 ccb）",
                      f"PATH={ctx.env.get('PATH', '')}", INSTALL_HINT)]

    checks = check_interpreter(path, ctx)
    rc, out = _run([path, "--version"], ctx)
    if rc != 0:
        checks.append(Check(FAIL, f"{name} --version 执行失败", _tail(out),
                            "确认安装完整，或重新安装:\n" + INSTALL_HINT))
        return checks
    version = out.strip().splitlines()[0] if out.strip() else ""
    checks.append(Check(OK, f"claude 二进制: {path}", version))

    _, help_text = _run([path, "--help"], ctx)
    missing = [f for f in REQUIRED_FLAGS
               if not re.search(re.escape(f) + r"(?![\w-])", help_text)]
    if missing:
        checks.append(Check(FAIL, f"{name} 不支持 MetaInfer 需要的参数: {' '.join(missing)}",
                            "版本过旧，或不是 Claude Code 兼容的 CLI",
                            "升级: npm i -g claude-code-best@latest（或 @anthropic-ai/claude-code@latest）"))
    return checks


def _load_settings(ctx: Ctx) -> Tuple[dict, Optional[Check]]:
    p = ctx.config_dir / "settings.json"
    if not p.is_file():
        return {}, None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {}, Check(FAIL, f"settings.json 无法解析: {p}", str(e),
                         "修正 JSON 语法，或先移走该文件")
    if not isinstance(data, dict):
        return {}, Check(FAIL, f"settings.json 顶层不是 JSON object: {p}", hint="改成 {...} 形式")
    return data, None


def check_auth(ctx: Ctx, model: Optional[str] = None) -> List[Check]:
    """Credential *presence* only; values are never printed. Whether they
    actually work is what ``--live`` answers."""
    checks: List[Check] = []
    settings, err = _load_settings(ctx)
    if err:
        checks.append(err)
    settings_env = settings.get("env") if isinstance(settings.get("env"), dict) else {}

    def lookup(key: str) -> Tuple[str, str]:
        if ctx.env.get(key):
            return ctx.env[key], f"环境变量 {key}"
        v = settings_env.get(key)
        if isinstance(v, str) and v:
            return v, f"settings.json env.{key}"
        return "", ""

    sources: List[str] = []
    for key in AUTH_ENV_VARS:
        value, src = lookup(key)
        if value:
            sources.append(src)
    for key in PROVIDER_ENV_VARS:
        value, src = lookup(key)
        if _truthy(value):
            sources.append(src)
    if settings.get("apiKeyHelper"):
        sources.append("settings.json apiKeyHelper")
    cred = ctx.config_dir / ".credentials.json"
    if cred.is_file() and cred.stat().st_size > 0:
        sources.append(f"/login 凭据 {cred}")

    if sources:
        checks.append(Check(OK, "鉴权: 已找到凭据", "来源: " + "; ".join(sources)))
    else:
        hint = "\n".join([
            "任选其一:",
            "  Anthropic API key:   export ANTHROPIC_API_KEY=<key>",
            "  API 网关 / 代理:     export ANTHROPIC_BASE_URL=<url> ANTHROPIC_AUTH_TOKEN=<token>",
            "  订阅账号长期 token:  claude setup-token，然后 export CLAUDE_CODE_OAUTH_TOKEN=<token>",
            f"  交互登录:            运行 {ctx.claude_bin or DEFAULT_CLAUDE_BIN}，输入 /login"
            f"（凭据写入 {ctx.config_dir}）",
            "  上述变量也可以写进 settings.json 的 \"env\" 字段，对所有 claude 进程生效",
            "  Docker: 用 -e 传入上述变量，或把已登录的 ~/.claude 挂载到容器的 $CLAUDE_CONFIG_DIR",
            "          （macOS 的 /login 凭据存在钥匙串里，挂载 ~/.claude 带不进 Linux 容器）",
        ])
        detail = "MetaInfer 不传 API key，鉴权完全依赖 claude 自身的配置"
        if ctx.platform == "darwin":
            checks.append(Check(WARN, "鉴权: 环境变量和配置文件中都没有凭据",
                                detail + "；macOS 上 /login 凭据存在钥匙串中，无法静态检查，"
                                         "用 --live 实际验证", hint))
        else:
            checks.append(Check(FAIL, "鉴权: 未找到任何凭据", detail, hint))

    base_url, src = lookup("ANTHROPIC_BASE_URL")
    if base_url:
        u = urlparse(base_url)
        if u.scheme not in ("http", "https") or not u.netloc:
            checks.append(Check(FAIL, f"ANTHROPIC_BASE_URL 格式错误（{src}）", base_url,
                                "形如 https://gateway.example.com"))
        else:
            # Drop any user:pass@ before echoing the URL.
            shown = f"{u.scheme}://{u.netloc.rpartition('@')[2]}{u.path}"
            if u.path.rstrip("/").endswith("/v1"):
                checks.append(Check(WARN, f"ANTHROPIC_BASE_URL 以 /v1 结尾: {shown}",
                                    "claude 会自己拼接 /v1/messages，请求变成 /v1/v1/messages 而 404",
                                    "去掉末尾的 /v1"))
            else:
                checks.append(Check(OK, f"ANTHROPIC_BASE_URL: {shown}", f"来源: {src}"))
            if not model and not any(lookup(k)[0] for k in MODEL_ENV_VARS):
                checks.append(Check(INFO, "使用自定义 API 端点，但没有设置模型环境变量",
                                    "端点不认识 claude 的默认模型 ID 时会报 model not found / 404",
                                    "export ANTHROPIC_MODEL=<端点支持的模型 ID>（或在任务表单里指定模型），"
                                    "可用 --live --model <id> 验证"))
    return checks


def check_dirs(ctx: Ctx) -> List[Check]:
    checks: List[Check] = []
    cfg = ctx.config_dir
    if _writable(cfg):
        checks.append(Check(OK, f"claude 配置目录可写: {cfg}"))
    else:
        checks.append(Check(WARN, f"claude 配置目录不可写: {cfg}",
                            "claude 在这里保存会话（MetaInfer 用 --resume 续接多轮 agent）、设置和登录状态",
                            f"sudo chown -R $(id -u):$(id -g) {cfg}，或 export CLAUDE_CONFIG_DIR=<可写目录>"))
    root = ctx.metainfer_root
    if _writable(root):
        checks.append(Check(OK, f"METAINFER_ROOT 可写: {root}"))
    else:
        checks.append(Check(FAIL, f"METAINFER_ROOT 不可写: {root}",
                            "agent 的 workdir 和任务状态都在这个目录下",
                            "把该目录 chown 给当前用户，或 export METAINFER_ROOT=<可写目录>"))
    return checks


def check_root_sandbox(ctx: Ctx) -> List[Check]:
    if ctx.euid != 0:
        return []
    if ctx.env.get("IS_SANDBOX") == "1" or _truthy(ctx.env.get("CLAUDE_CODE_BUBBLEWRAP", "")):
        return [Check(OK, "以 root 运行，IS_SANDBOX=1 已设置")]
    return [Check(WARN, "以 root 运行且未设置 IS_SANDBOX=1",
                  "orchestrator / WebUI QA 会自动设置；multi-node worker 的 agent job 固定用 "
                  "bypassPermissions 却不设置，claude 会以 \"cannot be used with root/sudo "
                  "privileges\" 拒绝启动",
                  "启动 worker 前 export IS_SANDBOX=1（docker/Dockerfile 已默认设置）")]


def check_worker_bin(ctx: Ctx) -> List[Check]:
    worker = ctx.worker_bin()
    worker_path = ctx.which(worker)
    if not worker_path:
        return [Check(INFO, f"multi-node worker 找不到 claude 二进制 {worker!r}",
                      "只有运行 python -m metainfer.worker 时需要",
                      "export METAINFER_CCB=<claude 二进制路径>")]
    orch_path = ctx.which(ctx.claude_bin)
    if orch_path and os.path.realpath(orch_path) != os.path.realpath(worker_path):
        return [Check(WARN, "worker 与 orchestrator 使用的 claude 二进制不同",
                      f"orchestrator: {orch_path}\nworker:       {worker_path}",
                      "worker 只读 METAINFER_CCB / CLAUDE_BIN（不读 METAINFER_CLAUDE_BIN），"
                      f"需要一致时: export METAINFER_CCB={orch_path}")]
    return []


def run_checks(ctx: Ctx, model: Optional[str] = None) -> List[Check]:
    return [
        *check_knobs(ctx),
        *check_claude_bin(ctx),
        *check_auth(ctx, model),
        *check_dirs(ctx),
        *check_root_sandbox(ctx),
        *check_worker_bin(ctx),
    ]


# --------------------------------------------------------------------------- #
# Runtime checks: CUDA / torch / vLLM / transformers
# --------------------------------------------------------------------------- #

# Runs inside the *target* interpreter via ``python -c``. Python 3.8+ only,
# stdlib + optional ``packaging``. Prints PROBE_MARKER + one JSON line so
# import-time log noise on stdout can't corrupt the result.
_PROBE_SCRIPT = r'''
import json, os, sys, sysconfig, traceback, warnings
name, marker = sys.argv[1], sys.argv[2]
out = {"name": name}
try:
    from importlib import metadata
    try:
        out["pkg"] = metadata.version(name)
    except metadata.PackageNotFoundError:
        out["pkg"] = None
except Exception:
    out["pkg"] = None
out["scripts_dir"] = sysconfig.get_paths().get("scripts")
try:
    mod = __import__(name)
except ModuleNotFoundError as e:
    if e.name != name:
        out["import_error"] = "".join(traceback.format_exception_only(type(e), e)).strip()
except BaseException as e:
    out["import_error"] = "".join(traceback.format_exception_only(type(e), e)).strip()
else:
    out["version"] = str(getattr(mod, "__version__", "") or "")
    out["file"] = getattr(mod, "__file__", None)
    if name == "torch":
        v = getattr(mod, "version", None)
        out["cuda_build"] = getattr(v, "cuda", None)
        out["hip_build"] = getattr(v, "hip", None)
        try:
            out["cudnn"] = mod.backends.cudnn.version()
        except Exception:
            out["cudnn"] = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                out["cuda_available"] = bool(mod.cuda.is_available())
            except Exception as e:
                out["cuda_available"] = False
                out["cuda_error"] = str(e)
        out["cuda_warnings"] = [str(w.message) for w in caught]
        devices = []
        if out.get("cuda_available"):
            try:
                for i in range(mod.cuda.device_count()):
                    p = mod.cuda.get_device_properties(i)
                    devices.append({"name": p.name, "cc": "%d.%d" % mod.cuda.get_device_capability(i),
                                    "mem_gb": round(p.total_memory / 2 ** 30, 1)})
            except Exception as e:
                out["cuda_error"] = str(e)
        out["devices"] = devices
        try:
            v = mod.cuda.nccl.version()  # tuple on new torch, int on old
            out["nccl"] = ".".join(str(x) for x in v) if isinstance(v, tuple) else str(v)
        except Exception:
            out["nccl"] = None
    if name == "vllm":
        try:
            import vllm._C
            out["ext"] = "ok"
        except ModuleNotFoundError as e:
            out["ext"] = "missing" if e.name == "vllm._C" else "error: " + str(e)
        except BaseException as e:
            out["ext"] = "error: " + "".join(traceback.format_exception_only(type(e), e)).strip()
        reqs = []
        try:
            from importlib import metadata
            from packaging.requirements import Requirement
            for spec in metadata.requires("vllm") or []:
                r = Requirement(spec)
                if r.name.lower() not in ("torch", "transformers"):
                    continue
                if r.marker is not None and not r.marker.evaluate({"extra": ""}):
                    continue
                try:
                    have = metadata.version(r.name)
                except metadata.PackageNotFoundError:
                    have = None
                ok = have is not None and r.specifier.contains(have, prereleases=True)
                reqs.append({"name": r.name.lower(), "spec": str(r.specifier), "have": have, "ok": ok})
        except Exception:
            pass
        out["reqs"] = reqs
print("\n" + marker + json.dumps(out))
'''


def _vtuple(v: Optional[str]) -> Optional[Tuple[int, int]]:
    m = re.match(r"(\d+)\.(\d+)", str(v or ""))
    return (int(m.group(1)), int(m.group(2))) if m else None


def _find_nvcc(ctx: Ctx) -> Tuple[Optional[str], bool]:
    """Returns (nvcc path, found via PATH)."""
    on_path = ctx.which("nvcc")
    if on_path:
        return on_path, True
    for root in (ctx.env.get("CUDA_HOME"), ctx.env.get("CUDA_PATH"), "/usr/local/cuda"):
        if root:
            p = os.path.join(root, "bin", "nvcc")
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return p, False
    return None, False


def _cudart_version(root: str) -> Optional[str]:
    """CUDA runtime version from the toolkit's version.json, else from the
    libcudart.so.<ver> file name."""
    try:
        with open(os.path.join(root, "version.json"), encoding="utf-8") as f:
            v = (json.load(f).get("cuda_cudart") or {}).get("version")
        if v:
            return str(v)
    except (OSError, ValueError, AttributeError):
        pass
    found = []
    for sub in ("lib64", "lib", os.path.join("targets", "*", "lib")):
        for p in glob.glob(os.path.join(root, sub, "libcudart.so.*")):
            m = re.search(r"libcudart\.so\.(\d+(?:\.\d+)+)$", p)
            if m:
                found.append(m.group(1))
    return max(found, key=lambda s: tuple(int(x) for x in s.split("."))) if found else None


def _driver_hint(text: str) -> str:
    low = text.lower()
    if "driver/library version mismatch" in low:
        return "升级驱动后内核模块还是旧版本: 重启机器，或重新加载 nvidia 内核模块"
    if "couldn't communicate with the nvidia driver" in low or "no devices were found" in low:
        return ("驱动没有加载或 GPU 不可见: 检查 lsmod | grep nvidia、dmesg；"
                "容器里用 docker run --gpus all（需要 nvidia-container-toolkit）")
    return "按上面的输出排查驱动安装"


def _probe_cuda(ctx: Ctx) -> Tuple[List[Check], dict]:
    """Driver + toolkit checks. Returns (checks, facts) where facts carries
    driver_cuda / nvcc_cuda / gpus for the torch cross-checks."""
    checks: List[Check] = []
    facts: dict = {"driver_cuda": None, "nvcc_cuda": None, "gpus": 0}

    smi = ctx.which("nvidia-smi")
    if not smi:
        if ctx.platform == "darwin":
            checks.append(Check(INFO, "CUDA: macOS 不支持 CUDA，跳过驱动 / GPU 检查"))
        else:
            checks.append(Check(WARN, "NVIDIA 驱动: PATH 上找不到 nvidia-smi",
                                "GPU 任务（kernel 优化、推理 oracle）需要 NVIDIA GPU；只跑 WebUI / orchestrator 的节点可以忽略",
                                "物理机安装 NVIDIA 驱动；容器里用 docker run --gpus all"
                                "（驱动和 nvidia-smi 由 nvidia-container-toolkit 注入）"))
    else:
        rc, out = _run([smi, "--query-gpu=driver_version,name,memory.total",
                        "--format=csv,noheader"], ctx)
        if rc != 0:
            checks.append(Check(FAIL, "nvidia-smi 执行失败", _tail(out), _driver_hint(out)))
        else:
            # A query row has exactly 3 fields; drop any stderr warning
            # lines _run() merged in so they aren't counted as GPUs.
            rows = [r for r in ([c.strip() for c in ln.split(",")] for ln in out.splitlines())
                    if len(r) == 3]
            _, header = _run([smi], ctx)
            m = re.search(r"CUDA Version:\s*([\d.]+)", header)
            facts["driver_cuda"] = _vtuple(m.group(1)) if m else None
            facts["gpus"] = len(rows)
            driver = rows[0][0] if rows and rows[0] else "?"
            detail = [f"nvidia-smi: {smi}"]
            detail += [f"GPU {i}: {r[1] if len(r) > 1 else '?'}"
                       + (f", {r[2]}" if len(r) > 2 else "") for i, r in enumerate(rows)]
            checks.append(Check(OK, f"NVIDIA 驱动 {driver}（支持 CUDA <= {m.group(1) if m else '?'}），"
                                    f"{len(rows)} 块 GPU", "\n".join(detail)))

    nvcc, on_path = _find_nvcc(ctx)
    if not nvcc:
        if ctx.platform != "darwin":
            checks.append(Check(WARN, "CUDA toolkit: 找不到 nvcc（PATH / $CUDA_HOME / /usr/local/cuda）",
                                "agent 编译 CUDA kernel、torch cpp_extension / vLLM 源码编译都需要 nvcc",
                                "安装 CUDA toolkit（或用 nvidia/cuda:*-devel 镜像），"
                                "export PATH=<cuda>/bin:$PATH CUDA_HOME=<cuda>"))
        return checks, facts

    rc, out = _run([nvcc, "--version"], ctx)
    m = re.search(r"release\s+([\d.]+)", out)
    if rc != 0 or not m:
        checks.append(Check(FAIL, f"nvcc --version 执行失败: {nvcc}", _tail(out),
                            "CUDA toolkit 安装不完整，重新安装"))
        return checks, facts
    facts["nvcc_cuda"] = _vtuple(m.group(1))
    root = os.path.dirname(os.path.dirname(os.path.realpath(nvcc)))
    cudart = _cudart_version(root)
    detail = [f"nvcc: {nvcc}" + (f" -> {os.path.realpath(nvcc)}" if os.path.realpath(nvcc) != nvcc else ""),
              f"toolkit 目录: {root}",
              f"CUDA_HOME={ctx.env.get('CUDA_HOME') or '(未设置)'}",
              f"CUDA runtime (cudart): {cudart or '未知'}"]
    checks.append(Check(OK, f"CUDA toolkit {m.group(1)}", "\n".join(detail)))
    if not on_path:
        checks.append(Check(WARN, "nvcc 不在 PATH 上",
                            "agent / 构建脚本直接调用 nvcc 会找不到",
                            f"export PATH={os.path.dirname(nvcc)}:$PATH CUDA_HOME={root}"))

    drv, tk = facts["driver_cuda"], facts["nvcc_cuda"]
    if drv and tk and tk[0] > drv[0]:
        checks.append(Check(FAIL, f"CUDA toolkit {tk[0]}.{tk[1]} 比驱动支持的 CUDA {drv[0]}.{drv[1]} 新（主版本不同）",
                            "编译出的程序在这个驱动上无法运行（CUDA driver version is insufficient）",
                            "升级 NVIDIA 驱动，或换用主版本 <= 驱动的 CUDA toolkit / 镜像（--build-arg CUDA_VERSION=12.x）"))
    elif drv and tk and tk > drv:
        checks.append(Check(WARN, f"CUDA toolkit {tk[0]}.{tk[1]} 比驱动支持的 CUDA {drv[0]}.{drv[1]} 新",
                            "同主版本靠 minor version compatibility 一般能运行，但 PTX JIT 等特性可能失败",
                            "升级驱动，或使用与驱动一致的 toolkit"))
    return checks, facts


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, AttributeError):
        proc.kill()


def probe_package(python: str, name: str, ctx: Ctx,
                  timeout: float = DEFAULT_PROBE_TIMEOUT) -> dict:
    """Import ``name`` in a child ``python`` (own process group, cwd=tmp so a
    source checkout in cwd can't shadow the installed package)."""
    with tempfile.TemporaryDirectory(prefix="metainfer-env-probe-") as cwd:
        try:
            proc = subprocess.Popen([python, "-c", _PROBE_SCRIPT, name, PROBE_MARKER],
                                    cwd=cwd, env=dict(ctx.env),
                                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, start_new_session=True)
        except OSError as e:
            return {"probe_error": "crash", "detail": str(e)}
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            proc.communicate()
            return {"probe_error": "timeout", "detail": f"{timeout:.0f}s 内 import 未完成"}
    stdout = out.decode("utf-8", "replace")
    for line in reversed(stdout.splitlines()):
        if line.startswith(PROBE_MARKER):
            try:
                return json.loads(line[len(PROBE_MARKER):])
            except ValueError:
                break
    return {"probe_error": "crash",
            "detail": f"exit={proc.returncode}\n" + _tail(err.decode("utf-8", "replace") + "\n" + stdout)}


def _target_python(ctx: Ctx, override: Optional[str] = None) -> str:
    """Same interpreter MetaInfer's launcher uses for orchestrators."""
    return override or ctx.env.get("METAINFER_PYTHON") or sys.executable


def _check_interpreter(python: str, ctx: Ctx) -> Tuple[Optional[str], Check]:
    path = ctx.which(python)
    if not path:
        return None, Check(FAIL, f"找不到 Python 解释器 {python!r}",
                           hint="--python 或 $METAINFER_PYTHON 指向可执行的 python")
    rc, out = _run([path, "-c", "import platform, sys; "
                                "print(platform.python_version(), sys.executable)"], ctx)
    if rc != 0:
        return None, Check(FAIL, f"Python 解释器无法运行: {path}", _tail(out))
    version, _, exe = out.strip().splitlines()[-1].partition(" ")
    return path, Check(OK, f"Python {version}（torch / vLLM / transformers 在这个解释器里检查）",
                       f"解释器: {path}" + (f" -> {exe}" if exe and exe != path else ""))


def _import_problem(label: str, module: str, info: dict, missing: Check, fix: str) -> Optional[Check]:
    """Shared failure triage for a probed package; None means it imported."""
    if info.get("probe_error") == "timeout":
        return Check(WARN, f"import {module} 超时", info["detail"],
                     f"首次 import 可能在加载 CUDA 库；用 --probe-timeout 加大，或手动 python -c 'import {module}' 排查")
    if "probe_error" in info:
        return Check(FAIL, f"import {module} 时进程崩溃", info["detail"],
                     f"手动运行 python -c 'import {module}' 排查（常见原因: CUDA 库 / 驱动不匹配、ABI 冲突）")
    if "import_error" in info:
        return Check(FAIL, f"import {module} 失败", info["import_error"], fix)
    if "version" not in info:
        if info.get("pkg"):
            return Check(FAIL, f"{label} {info['pkg']} 的包元数据存在，但找不到 {module} 模块",
                         "安装损坏（模块文件被删除或装到了别的 site-packages）", "重新安装: " + fix)
        return missing
    return None


def _env_cli(info: dict, name: str) -> Optional[str]:
    """``name`` console script in the probed interpreter's scripts dir."""
    scripts = info.get("scripts_dir") or ""
    p = os.path.join(scripts, name) if scripts else ""
    return p if p and os.access(p, os.X_OK) else None


def check_torch(info: dict, facts: dict) -> List[Check]:
    problem = _import_problem(
        "torch", "torch", info,
        Check(WARN, "torch 未安装", "evolve-kernel、opt_GEMM_kernel、推理框架 oracle 等任务需要 torch",
              TORCH_INDEX_HINT),
        "重新安装与 CUDA 匹配的 torch: " + TORCH_INDEX_HINT)
    if problem:
        return [problem]

    checks: List[Check] = []
    cuda_b = info.get("cuda_build")
    detail = [f"模块: {info.get('file')}",
              f"CLI: torchrun {_env_cli(info, 'torchrun') or '(未找到)'}",
              f"构建: " + (f"CUDA {cuda_b}" if cuda_b else f"ROCm/HIP {info['hip_build']}"
                           if info.get("hip_build") else "CPU-only"),
              f"cuDNN: {info.get('cudnn') or '-'}    NCCL: {info.get('nccl') or '-'}"]
    for i, d in enumerate(info.get("devices") or []):
        detail.append(f"cuda:{i} {d['name']}  compute capability {d['cc']}  {d['mem_gb']} GiB")
    checks.append(Check(OK, f"torch {info.get('version') or info.get('pkg')}"
                            f"（cuda.is_available={info.get('cuda_available')}）", "\n".join(detail)))

    gpus = facts.get("gpus", 0)
    if not cuda_b and not info.get("hip_build") and gpus:
        checks.append(Check(FAIL, "torch 是 CPU-only 版本，但机器上有 GPU",
                            "PyPI 默认的 torch 可能是 CPU 版或 CUDA 版本不对",
                            "重新安装 CUDA 版 torch: " + TORCH_INDEX_HINT))
    elif cuda_b and not info.get("cuda_available") and gpus:
        msgs = [*info.get("cuda_warnings", []), info.get("cuda_error") or ""]
        checks.append(Check(FAIL, "torch.cuda.is_available() 为 False，但 nvidia-smi 能看到 GPU",
                            _tail("\n".join(m for m in msgs if m)) or "(torch 没有给出原因)",
                            "常见原因: torch 的 CUDA 版本比驱动新、CUDA_VISIBLE_DEVICES 为空 / 写错、"
                            "容器没加 --gpus all"))
    elif cuda_b and not info.get("cuda_available"):
        checks.append(Check(INFO, "torch 是 CUDA 版本，但当前没有可用 GPU（GPU 任务只能在 GPU 节点上运行）"))

    tb, drv, tk = _vtuple(cuda_b), facts.get("driver_cuda"), facts.get("nvcc_cuda")
    if tb and drv and tb[0] > drv[0]:
        checks.append(Check(FAIL, f"torch 的 CUDA {cuda_b} 比驱动支持的 CUDA {drv[0]}.{drv[1]} 新（主版本不同）",
                            hint="升级 NVIDIA 驱动，或安装 CUDA 主版本 <= 驱动的 torch: " + TORCH_INDEX_HINT))
    if tb and tk and tb[0] != tk[0]:
        checks.append(Check(WARN, f"nvcc 的 CUDA {tk[0]}.{tk[1]} 与 torch 的 CUDA {cuda_b} 主版本不一致",
                            "torch.utils.cpp_extension / vLLM 源码编译会因 CUDA 版本不匹配而失败",
                            "让 CUDA_HOME 指向与 torch 同主版本的 toolkit，或换装匹配的 torch"))
    return checks


def check_vllm(info: dict, torch_info: dict) -> List[Check]:
    problem = _import_problem(
        "vLLM", "vllm", info,
        Check(WARN, "vLLM 未安装",
              hint="pip install vllm（vLLM 锁定 torch 版本，会替换已装的 torch；"
                   "CUDA 版本选择见 https://docs.vllm.ai）"),
        "通常是与 torch / transformers 版本冲突: 重新 pip install vllm，让它装回自己锁定的 torch")
    if problem:
        return [problem]

    checks: List[Check] = []
    scripts = info.get("scripts_dir") or ""
    env_cli = _env_cli(info, "vllm")
    path_cli = info.get("cli")
    detail = [f"模块: {info.get('file')}",
              f"CLI: {path_cli or env_cli or '找不到 vllm 命令'}",
              "编译扩展 vllm._C: " + {"ok": "已加载", "missing": "不存在（CPU / 非 CUDA 构建？）"}.get(
                  info.get("ext", ""), info.get("ext", "?"))]
    checks.append(Check(OK, f"vLLM {info.get('version') or info.get('pkg')}", "\n".join(detail)))

    ext = info.get("ext", "")
    if ext.startswith("error"):
        checks.append(Check(FAIL, "vllm._C 加载失败（vLLM 的编译扩展与当前 torch 不匹配）", ext,
                            f"vLLM 是针对固定 torch 版本编译的；当前 torch={torch_info.get('version') or '?'}。"
                            "重新 pip install vllm 装回它锁定的 torch，或换装与 torch 匹配的 vLLM"))
    if path_cli and env_cli and os.path.realpath(path_cli) != os.path.realpath(env_cli):
        checks.append(Check(WARN, "PATH 上的 vllm 命令不属于这个解释器",
                            f"PATH:   {path_cli}\n解释器: {env_cli}",
                            f"export PATH={scripts}:$PATH（或激活对应的 venv）"))
    elif not path_cli and env_cli:
        checks.append(Check(WARN, "vllm 命令不在 PATH 上", f"在 {env_cli}",
                            f"export PATH={scripts}:$PATH（或激活对应的 venv）"))

    for r in info.get("reqs", []):
        if r["name"] == "torch" and not r["ok"]:
            checks.append(Check(FAIL, f"vLLM 要求 torch{r['spec']}，已安装 {r['have'] or '无'}",
                                "torch 版本不符时 vllm._C 常报 undefined symbol",
                                f"pip install 'torch{r['spec']}'（用对应 CUDA 的 index），或重新 pip install vllm"))
    return checks


def check_transformers(info: dict, vllm_info: dict) -> List[Check]:
    problem = _import_problem(
        "transformers", "transformers", info,
        Check(WARN, "transformers 未安装", "gen-infer-framework 的 perf oracle 用它加载 tokenizer",
              "pip install transformers"),
        "pip install -U transformers，或按报错补齐依赖")
    if problem:
        return [problem]
    checks = [Check(OK, f"transformers {info.get('version') or info.get('pkg')}",
                    f"模块: {info.get('file')}")]
    for r in vllm_info.get("reqs", []):
        if r["name"] == "transformers" and not r["ok"]:
            checks.append(Check(WARN, f"vLLM 要求 transformers{r['spec']}，已安装 {r['have'] or '无'}",
                                hint=f"pip install 'transformers{r['spec']}'"))
    return checks


def run_runtime_checks(ctx: Ctx, python: Optional[str] = None,
                       timeout: float = DEFAULT_PROBE_TIMEOUT) -> List[Check]:
    checks, facts = _probe_cuda(ctx)
    path, interp = _check_interpreter(_target_python(ctx, python), ctx)
    checks.append(interp)
    if not path:
        return checks

    torch_info = probe_package(path, "torch", ctx, timeout)
    checks += check_torch(torch_info, facts)
    vllm_info = probe_package(path, "vllm", ctx, timeout)
    if "probe_error" not in vllm_info:
        vllm_info["cli"] = ctx.which("vllm")
    checks += check_vllm(vllm_info, torch_info if "probe_error" not in torch_info else {})
    checks += check_transformers(probe_package(path, "transformers", ctx, timeout),
                                 vllm_info if "probe_error" not in vllm_info else {})
    return checks


# --------------------------------------------------------------------------- #
# Live probe
# --------------------------------------------------------------------------- #

# (regex on lower-cased output, hint) — first match wins.
_DIAGNOSES = (
    (r"cannot be used with root/sudo",
     "root 下 bypassPermissions 被拒绝: export IS_SANDBOX=1"),
    (r"not logged in|/login|invalid api key|invalid x-api-key|authentication_error|oauth token|\b401\b",
     "鉴权失败: 检查 ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / CLAUDE_CODE_OAUTH_TOKEN 是否有效，"
     "或重新 /login（见上面的“鉴权”项）"),
    (r"not_found_error|invalid model|model.{0,40}(not found|not exist|invalid)|\b404\b",
     "模型不可用: 用 --model 或 ANTHROPIC_MODEL 指定账号 / 网关支持的模型 ID；"
     "ANTHROPIC_BASE_URL 不要以 /v1 结尾"),
    (r"permission_error|forbidden|\b403\b",
     "403: key 无权访问该模型或区域，或被网关拒绝"),
    (r"rate_limit|overloaded|\b429\b|\b529\b",
     "限流 / 过载: 稍后重试"),
    (r"econnrefused|enotfound|etimedout|econnreset|fetch failed|connection error|"
     r"unable to connect|getaddrinfo|certificate|self.signed",
     "网络不通: 检查 ANTHROPIC_BASE_URL 和 HTTPS_PROXY / HTTP_PROXY / NO_PROXY，"
     "容器内要能访问 API 端点；自签证书设置 NODE_EXTRA_CA_CERTS"),
)


def _diagnose(text: str) -> str:
    low = text.lower()
    for pattern, hint in _DIAGNOSES:
        if re.search(pattern, low):
            return hint
    return "未识别的错误，按上面的输出排查"


def _live_command(path: str, mode: str, workdir: str,
                  model: Optional[str], effort: str) -> List[str]:
    """Same argv as SubAgentManager._build_claude_command for a fresh turn."""
    cmd = [path, "-p", "--output-format", "stream-json", "--input-format", "text",
           "--verbose", "--permission-mode", mode, "--add-dir", workdir]
    if model:
        cmd += ["--model", model]
    if effort:
        cmd += ["--effort", effort]
    return cmd


def live_probe(ctx: Ctx, model: Optional[str] = None, timeout: float = 180) -> Check:
    path = ctx.which(ctx.claude_bin)
    if not path:
        return Check(FAIL, "live 探测未执行: 找不到 claude 二进制", hint=INSTALL_HINT)
    mode = ctx.permission_mode if ctx.permission_mode in VALID_PERMISSION_MODES else DEFAULT_PERMISSION_MODE
    effort = ctx.effort if ctx.effort in VALID_EFFORTS else DEFAULT_EFFORT
    env = dict(ctx.env)
    env.setdefault("DISABLE_INTERACTIVITY", "1")
    sandbox_prefix = ""
    if mode == "bypassPermissions":
        env["IS_SANDBOX"] = "1"
        sandbox_prefix = "IS_SANDBOX=1 "
    repro = ('cd "$(mktemp -d)" && echo ' + shlex.quote(LIVE_PROMPT.strip()) + " | "
             + sandbox_prefix + shlex.join(_live_command(path, mode, ".", model, effort)))

    with tempfile.TemporaryDirectory(prefix="metainfer-env-check-") as workdir:
        started = time.time()
        try:
            proc = subprocess.Popen(
                _live_command(path, mode, workdir, model, effort),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=workdir, env=env, start_new_session=True,
            )
        except OSError as e:
            return Check(FAIL, "live 探测启动失败", str(e), INSTALL_HINT)
        try:
            out, err = proc.communicate(LIVE_PROMPT.encode(), timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            proc.communicate()
            return Check(FAIL, f"live 探测超时（{timeout:.0f}s 内未完成）",
                         hint="检查网络 / 代理是否可达 API 端点；effort=max 时首个响应较慢，"
                              "可加 --timeout 300 重试\n手动复现: " + repro)
    elapsed = time.time() - started

    stdout = out.decode("utf-8", "replace")
    stderr = err.decode("utf-8", "replace")
    events: List[dict] = []
    plain: List[str] = []
    for line in stdout.splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            plain.append(line)
            continue
        if isinstance(ev, dict):
            events.append(ev)
    init = next((e for e in events if e.get("type") == "system" and e.get("subtype") == "init"), {})
    result = next((e for e in reversed(events) if e.get("type") == "result"), None)

    if proc.returncode == 0 and result is not None and not result.get("is_error"):
        detail = f"model={init.get('model', '?')}，耗时 {elapsed:.1f}s"
        cost = result.get("total_cost_usd")
        if isinstance(cost, (int, float)):
            detail += f"，费用 ${cost:.4f}"
        text = str(result.get("result", ""))
        if LIVE_MARKER not in text:
            detail += f"\n回复: {text[:200]}"
        return Check(OK, "live 探测: agent 正常完成一轮对话", detail)

    message = str(result.get("result", "")) if result else ""
    detail = _tail("\n".join([stderr, *plain, message])) or "(无输出)"
    return Check(FAIL, f"live 探测失败 (exit={proc.returncode})", detail,
                 _diagnose(stdout + "\n" + stderr) + "\n手动复现: " + repro)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

_COLORS = {OK: "32", INFO: "36", WARN: "33", FAIL: "31"}


def render(checks: List[Check], color: bool = False) -> str:
    lines: List[str] = []
    for c in checks:
        tag = f"[{c.level:^4}]"
        if color:
            tag = f"\033[{_COLORS[c.level]}m{tag}\033[0m"
        lines.append(f"{tag} {c.title}")
        lines += [f"       {ln}" for ln in c.detail.splitlines()]
        hint = c.hint.splitlines()
        if hint:
            lines.append(f"       → {hint[0]}")
            lines += [f"         {ln}" for ln in hint[1:]]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None, ctx: Optional[Ctx] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="env_check.py",
        description="检查 MetaInfer 调用 claude agent 的环境是否就绪，不正确时给出修复提示。",
    )
    parser.add_argument("--live", action="store_true",
                        help="用 MetaInfer 同款命令真实调用一次 agent（发起一次 API 请求，消耗少量 token）")
    parser.add_argument("--model", default=None,
                        help="--live 时传给 agent 的 --model（默认不传，与任务未指定模型时一致）")
    parser.add_argument("--timeout", type=float, default=180,
                        help="--live 超时秒数（默认 180）")
    parser.add_argument("--python", default=None,
                        help="检查 torch / vLLM / transformers 所用的解释器（默认 $METAINFER_PYTHON，否则当前解释器）")
    parser.add_argument("--skip-runtime", action="store_true",
                        help="跳过 CUDA / torch / vLLM / transformers 检查，只检查 claude agent")
    parser.add_argument("--probe-timeout", type=float, default=DEFAULT_PROBE_TIMEOUT,
                        help=f"每个 Python 包 import 的超时秒数（默认 {DEFAULT_PROBE_TIMEOUT}）")
    args = parser.parse_args(argv)

    ctx = ctx or Ctx.current()
    agent_checks = run_checks(ctx, args.model)
    if args.live:
        agent_checks.append(live_probe(ctx, args.model, args.timeout))
    runtime_checks = [] if args.skip_runtime else run_runtime_checks(ctx, args.python, args.probe_timeout)

    color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    print("MetaInfer claude agent 环境检查")
    print(f"  bin={ctx.claude_bin or '(空)'}  permission_mode={ctx.permission_mode}  "
          f"effort={ctx.effort}  uid={ctx.euid}")
    print(f"  CLAUDE_CONFIG_DIR={ctx.config_dir}  METAINFER_ROOT={ctx.metainfer_root}")
    print()
    print(render(agent_checks, color=color))
    if runtime_checks:
        print()
        print("运行时环境（CUDA / torch / vLLM / transformers）")
        print()
        print(render(runtime_checks, color=color))
    print()
    checks = agent_checks + runtime_checks
    n_fail = sum(c.level == FAIL for c in checks)
    n_warn = sum(c.level == WARN for c in checks)
    print(f"结果: {n_fail} FAIL, {n_warn} WARN")
    if not args.live and not any(c.level == FAIL for c in agent_checks):
        print("提示: 加 --live 用 MetaInfer 同款命令真实调用一次 agent，确认鉴权和模型可用（消耗少量 token）")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
