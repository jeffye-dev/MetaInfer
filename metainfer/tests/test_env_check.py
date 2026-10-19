"""tools/env_check.py: static checks + --live probe of the claude agent env.

The claude binary is replaced by a /bin/sh stub, so no real agent runs and
no API request is made.
"""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "env_check.py"
_spec = importlib.util.spec_from_file_location("metainfer_tools_env_check", _SCRIPT)
env_check = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = env_check  # dataclasses resolve annotations via sys.modules
_spec.loader.exec_module(env_check)

OK, INFO, WARN, FAIL = env_check.OK, env_check.INFO, env_check.WARN, env_check.FAIL
FULL_HELP = "-p, --print " + " ".join(env_check.REQUIRED_FLAGS)

FAKE_CLAUDE = r"""#!/bin/sh
case "$1" in
    --version) echo "9.9.9 (Claude Code)"; exit 0 ;;
    --help) printf '%s\n' "$FAKE_HELP"; exit 0 ;;
esac
cat >/dev/null
if [ -n "$FAKE_ARGS" ]; then
    printf '%s\n' "$@" > "$FAKE_ARGS"
    echo "IS_SANDBOX=$IS_SANDBOX" >> "$FAKE_ARGS"
fi
case "$FAKE_MODE" in
    ok)
        echo '{"type":"system","subtype":"init","model":"fake-model"}'
        echo '{"type":"result","subtype":"success","is_error":false,"result":"METAINFER_ENV_OK","total_cost_usd":0.0012}'
        ;;
    login)
        echo '{"type":"result","subtype":"success","is_error":true,"result":"Not logged in - Please run /login"}'
        exit 1
        ;;
    root)
        echo "--dangerously-skip-permissions cannot be used with root/sudo privileges for security reasons" >&2
        exit 1
        ;;
    hang) exec sleep 30 ;;
esac
"""

needs_non_root = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root ignores directory permissions",
)


@pytest.fixture()
def ctx(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    env = {
        "PATH": os.pathsep.join([str(bin_dir), "/usr/bin", "/bin"]),
        "HOME": str(home),
        "METAINFER_ROOT": str(tmp_path / "root"),
        "FAKE_HELP": FULL_HELP,
    }
    return env_check.Ctx(env=env, home=home, cwd=tmp_path, euid=1000, platform="linux")


def _install(ctx, name: str = "ccb", body: str = FAKE_CLAUDE) -> Path:
    p = Path(ctx.env["PATH"].split(os.pathsep)[0]) / name
    p.write_text(body, encoding="utf-8")
    p.chmod(0o755)
    return p


def _find(checks, needle: str):
    hits = [c for c in checks if needle in c.title]
    assert hits, f"no check titled ~{needle!r}: {[c.title for c in checks]}"
    return hits[0]


def _settings(ctx, data) -> Path:
    cfg = ctx.home / ".claude"
    cfg.mkdir(exist_ok=True)
    p = cfg / "settings.json"
    p.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# Binary
# --------------------------------------------------------------------------- #


def test_missing_binary_fails_with_install_hint(ctx) -> None:
    c = _find(env_check.run_checks(ctx), "找不到 claude 二进制")
    assert c.level == FAIL
    assert "npm i -g claude-code-best" in c.hint


def test_installed_binary_passes_version_and_flag_checks(ctx) -> None:
    _install(ctx)
    checks = env_check.run_checks(ctx)
    c = _find(checks, "claude 二进制:")
    assert c.level == OK and "9.9.9" in c.detail
    assert not [x for x in checks if "不支持" in x.title]


def test_old_binary_missing_flags_fails(ctx) -> None:
    _install(ctx)
    ctx.env["FAKE_HELP"] = "-p, --print --output-format --verbose"
    c = _find(env_check.run_checks(ctx), "不支持 MetaInfer 需要的参数")
    assert c.level == FAIL
    assert "--effort" in c.title and "--print" not in c.title


def test_node_shebang_without_node_fails(ctx) -> None:
    _install(ctx, body="#!/usr/bin/env node\n")
    ctx.env["PATH"] = ctx.env["PATH"].split(os.pathsep)[0]
    c = _find(env_check.run_checks(ctx), "依赖 node")
    assert c.level == FAIL and "Node.js" in c.hint


def test_worker_binary_mismatch_warns(ctx) -> None:
    _install(ctx, "ccb")
    other = _install(ctx, "claude-alt")
    ctx.env["METAINFER_CLAUDE_BIN"] = str(other)
    c = _find(env_check.run_checks(ctx), "worker 与 orchestrator")
    assert c.level == WARN and f"METAINFER_CCB={other}" in c.hint


# --------------------------------------------------------------------------- #
# MetaInfer knobs / root
# --------------------------------------------------------------------------- #


def test_invalid_knobs_fail(ctx) -> None:
    ctx.env.update(METAINFER_AGENT_BACKEND="gpt", METAINFER_PERMISSION_MODE="yolo",
                   METAINFER_EFFORT="ultra")
    checks = env_check.run_checks(ctx)
    for needle in ("METAINFER_AGENT_BACKEND", "METAINFER_PERMISSION_MODE", "METAINFER_EFFORT"):
        assert _find(checks, needle).level == FAIL


def test_default_permission_mode_warns(ctx) -> None:
    ctx.env["METAINFER_PERMISSION_MODE"] = "default"
    assert _find(env_check.run_checks(ctx), "METAINFER_PERMISSION_MODE=default").level == WARN


def test_root_needs_is_sandbox_for_worker(ctx) -> None:
    ctx.euid = 0
    assert _find(env_check.run_checks(ctx), "IS_SANDBOX").level == WARN
    ctx.env["IS_SANDBOX"] = "1"
    assert _find(env_check.run_checks(ctx), "IS_SANDBOX").level == OK


@needs_non_root
def test_unwritable_metainfer_root_fails(ctx, tmp_path) -> None:
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        ctx.env["METAINFER_ROOT"] = str(ro)
        assert _find(env_check.run_checks(ctx), "METAINFER_ROOT 不可写").level == FAIL
    finally:
        ro.chmod(0o755)


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("platform,level", [("linux", FAIL), ("darwin", WARN)])
def test_no_credentials(ctx, platform, level) -> None:
    ctx.platform = platform
    c = _find(env_check.run_checks(ctx), "鉴权")
    assert c.level == level
    assert "ANTHROPIC_API_KEY" in c.hint


def test_credentials_from_settings_env_found_but_never_printed(ctx, capsys) -> None:
    _install(ctx)
    _settings(ctx, {"env": {
        "ANTHROPIC_AUTH_TOKEN": "sk-super-secret-123",
        "ANTHROPIC_BASE_URL": "https://user:pw@gw.example.com/v1",
    }})
    env_check.main(["--skip-runtime"], ctx)
    out = capsys.readouterr().out
    assert "sk-super-secret-123" not in out and "user:pw" not in out

    checks = env_check.run_checks(ctx)
    auth = _find(checks, "鉴权")
    assert auth.level == OK
    assert "settings.json env.ANTHROPIC_AUTH_TOKEN" in auth.detail
    assert _find(checks, "以 /v1 结尾").level == WARN
    assert _find(checks, "没有设置模型").level == INFO


def test_env_var_and_login_file_are_credential_sources(ctx) -> None:
    ctx.env["ANTHROPIC_API_KEY"] = "k"
    cfg = ctx.home / ".claude"
    cfg.mkdir()
    (cfg / ".credentials.json").write_text('{"claudeAiOauth": {}}')
    auth = _find(env_check.run_checks(ctx), "鉴权")
    assert auth.level == OK
    assert "环境变量 ANTHROPIC_API_KEY" in auth.detail and ".credentials.json" in auth.detail


def test_invalid_settings_json_fails(ctx) -> None:
    _settings(ctx, "{not json")
    assert _find(env_check.run_checks(ctx), "settings.json 无法解析").level == FAIL


# --------------------------------------------------------------------------- #
# Live probe
# --------------------------------------------------------------------------- #


def test_live_probe_success_uses_metainfer_command(ctx, tmp_path) -> None:
    _install(ctx)
    args_file = tmp_path / "args.txt"
    ctx.env.update(FAKE_MODE="ok", FAKE_ARGS=str(args_file))
    c = env_check.live_probe(ctx, model="m-1", timeout=20)
    assert c.level == OK, c
    assert "fake-model" in c.detail and "$0.0012" in c.detail
    argv = args_file.read_text().splitlines()
    assert argv[:8] == ["-p", "--output-format", "stream-json", "--input-format", "text",
                        "--verbose", "--permission-mode", "bypassPermissions"]
    assert argv[argv.index("--model") + 1] == "m-1"
    assert argv[argv.index("--effort") + 1] == "max"
    assert "IS_SANDBOX=1" in argv


@pytest.mark.parametrize("mode,needle", [("login", "鉴权失败"), ("root", "IS_SANDBOX=1")])
def test_live_probe_failure_hints(ctx, mode, needle) -> None:
    _install(ctx)
    ctx.env["FAKE_MODE"] = mode
    c = env_check.live_probe(ctx, timeout=20)
    assert c.level == FAIL
    assert needle in c.hint and "手动复现" in c.hint


def test_live_probe_timeout(ctx) -> None:
    _install(ctx)
    ctx.env["FAKE_MODE"] = "hang"
    c = env_check.live_probe(ctx, timeout=1)
    assert c.level == FAIL and "超时" in c.title


def test_main_exit_code(ctx, capsys) -> None:
    assert env_check.main(["--skip-runtime"], ctx) == 1  # no binary, no credentials
    _install(ctx)
    ctx.env["ANTHROPIC_API_KEY"] = "k"
    assert env_check.main(["--skip-runtime"], ctx) == 0
    assert "运行时环境" not in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Runtime: CUDA / torch / vLLM / transformers
#
# nvidia-smi / nvcc are shell stubs; torch / vllm / transformers are fake
# packages (with dist-info, so importlib.metadata sees them) on PYTHONPATH,
# imported by the real probe subprocess.
# --------------------------------------------------------------------------- #

FAKE_SMI = r"""#!/bin/sh
if [ "$FAKE_SMI_MODE" = mismatch ]; then
    echo "Failed to initialize NVML: Driver/library version mismatch"; exit 18
fi
case "$1" in
    --query-gpu=*)
        echo "WARNING: infoROM is corrupted at gpu 0000:4E:00.0" >&2
        i=0
        while [ "$i" -lt "${FAKE_GPUS:-2}" ]; do
            echo "${FAKE_DRIVER:-580.65.06}, NVIDIA H100 80GB HBM3, 81559 MiB"; i=$((i + 1))
        done ;;
    *) echo "| NVIDIA-SMI ${FAKE_DRIVER:-580.65.06}  Driver Version: ${FAKE_DRIVER:-580.65.06}  CUDA Version: ${FAKE_DRIVER_CUDA:-13.0} |" ;;
esac
"""

FAKE_NVCC = r"""#!/bin/sh
echo "nvcc: NVIDIA (R) Cuda compiler driver"
echo "Cuda compilation tools, release ${FAKE_NVCC:-13.0}, V${FAKE_NVCC:-13.0}.88"
"""

FAKE_TORCH_INIT = r"""
import os, warnings
from . import version, backends, cuda
__version__ = os.environ.get("FAKE_TORCH_VERSION", "2.13.0+cu130")
"""

FAKE_TORCH_CUDA = r"""
import os, warnings
from . import nccl

def is_available():
    if os.environ.get("FAKE_TORCH_AVAILABLE", "1") != "1":
        warnings.warn("CUDA initialization: The NVIDIA driver on your system is too old (found version 12080).")
        return False
    return True

def device_count():
    return int(os.environ.get("FAKE_GPUS", "2"))

class _Props:
    name = "NVIDIA H100 80GB HBM3"
    total_memory = 80 * 2 ** 30

def get_device_properties(i):
    return _Props()

def get_device_capability(i):
    return (9, 0)
"""

FAKE_VLLM_INIT = r"""
import os
if os.environ.get("FAKE_VLLM_IMPORT") == "crash":
    os._exit(139)
if os.environ.get("FAKE_VLLM_IMPORT") == "hang":
    import time; time.sleep(60)
__version__ = "0.13.0"
"""

FAKE_VLLM_C = r"""
import os
if os.environ.get("FAKE_VLLM_C") == "abi":
    raise ImportError("vllm/_C.abi3.so: undefined symbol: _ZN3c104cuda9SetDeviceEi")
"""


def _write_pkg(site: Path, name: str, version: str, files: dict, requires=()) -> None:
    for rel, body in files.items():
        p = site / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    dist = site / f"{name}-{version}.dist-info"
    dist.mkdir()
    meta = [f"Metadata-Version: 2.1", f"Name: {name}", f"Version: {version}",
            *[f"Requires-Dist: {r}" for r in requires]]
    (dist / "METADATA").write_text("\n".join(meta) + "\n", encoding="utf-8")


@pytest.fixture()
def rt(ctx, tmp_path: Path):
    bin_dir = Path(ctx.env["PATH"].split(os.pathsep)[0])
    _install(ctx, "nvidia-smi", FAKE_SMI)
    cuda = tmp_path / "cuda"
    (cuda / "bin").mkdir(parents=True)
    (cuda / "lib64").mkdir()
    (cuda / "lib64" / "libcudart.so.13.0.96").write_text("")
    nvcc = cuda / "bin" / "nvcc"
    nvcc.write_text(FAKE_NVCC)
    nvcc.chmod(0o755)
    _install(ctx, "vllm", "#!/bin/sh\n")  # vllm CLI on PATH

    site = tmp_path / "site"
    _write_pkg(site, "torch", "2.13.0+cu130", {
        "torch/__init__.py": FAKE_TORCH_INIT,
        "torch/version.py": "import os\ncuda = os.environ.get('FAKE_TORCH_CUDA', '13.0') or None\nhip = None\n",
        "torch/backends/__init__.py": "from . import cudnn\n",
        "torch/backends/cudnn.py": "def version():\n    return 91300\n",
        "torch/cuda/__init__.py": FAKE_TORCH_CUDA,
        "torch/cuda/nccl.py": "def version():\n    return (2, 27, 7)\n",
    })
    _write_pkg(site, "vllm", "0.13.0", {
        "vllm/__init__.py": FAKE_VLLM_INIT,
        "vllm/_C.py": FAKE_VLLM_C,
    }, requires=["torch==2.13.0", "transformers>=4.56.0", "pytest; extra == 'test'"])
    _write_pkg(site, "transformers", "4.57.1", {
        "transformers/__init__.py": "__version__ = '4.57.1'\n",
    })

    ctx.env["PATH"] = os.pathsep.join([str(bin_dir), str(cuda / "bin"), "/usr/bin", "/bin"])
    ctx.env["PYTHONPATH"] = str(site)
    ctx.env["CUDA_HOME"] = str(cuda)
    ctx.site = site
    ctx.cuda = cuda
    return ctx


def _runtime(ctx, **kw):
    return env_check.run_runtime_checks(ctx, sys.executable, **kw)


def test_runtime_happy_path_reports_versions_and_paths(rt) -> None:
    checks = _runtime(rt)
    assert not [c for c in checks if c.level in (FAIL, WARN)], env_check.render(checks)

    drv = _find(checks, "NVIDIA 驱动 580.65.06")
    assert "CUDA <= 13.0" in drv.title and "2 块 GPU" in drv.title and "H100" in drv.detail
    tk = _find(checks, "CUDA toolkit 13.0")
    assert f"nvcc: {rt.cuda / 'bin' / 'nvcc'}" in tk.detail
    assert "CUDA runtime (cudart): 13.0.96" in tk.detail
    assert _find(checks, "Python ").detail.startswith(f"解释器: {sys.executable}")

    torch = _find(checks, "torch 2.13.0+cu130")
    assert "cuda.is_available=True" in torch.title
    assert str(rt.site / "torch" / "__init__.py") in torch.detail
    assert "构建: CUDA 13.0" in torch.detail and "NCCL: 2.27.7" in torch.detail
    assert "cuda:1 NVIDIA H100 80GB HBM3  compute capability 9.0  80.0 GiB" in torch.detail

    vllm = _find(checks, "vLLM 0.13.0")
    assert str(rt.site / "vllm" / "__init__.py") in vllm.detail
    assert "vllm._C: 已加载" in vllm.detail and "CLI: " in vllm.detail
    assert str(rt.site / "transformers") in _find(checks, "transformers 4.57.1").detail


def test_cudart_version_prefers_version_json(rt) -> None:
    (rt.cuda / "version.json").write_text('{"cuda_cudart": {"version": "13.0.88"}}')
    assert "CUDA runtime (cudart): 13.0.88" in _find(_runtime(rt), "CUDA toolkit").detail


def test_driver_older_than_torch_and_toolkit_fails(rt) -> None:
    rt.env.update(FAKE_DRIVER="570.124.06", FAKE_DRIVER_CUDA="12.8", FAKE_TORCH_AVAILABLE="0")
    checks = _runtime(rt)
    assert _find(checks, "CUDA toolkit 13.0 比驱动支持的 CUDA 12.8 新").level == FAIL
    assert _find(checks, "torch 的 CUDA 13.0 比驱动支持的 CUDA 12.8 新").level == FAIL
    unavailable = _find(checks, "is_available() 为 False")
    assert unavailable.level == FAIL and "driver on your system is too old" in unavailable.detail


def test_cpu_only_torch_on_gpu_host_fails(rt) -> None:
    rt.env["FAKE_TORCH_CUDA"] = ""
    c = _find(_runtime(rt), "CPU-only 版本")
    assert c.level == FAIL and "download.pytorch.org/whl/cu" in c.hint


def test_nvcc_major_differs_from_torch_warns(rt) -> None:
    rt.env.update(FAKE_NVCC="12.9", FAKE_DRIVER_CUDA="13.0")
    assert _find(_runtime(rt), "nvcc 的 CUDA 12.9 与 torch 的 CUDA 13.0").level == WARN


def test_nvcc_only_under_cuda_home_warns(rt) -> None:
    rt.env["PATH"] = os.pathsep.join(p for p in rt.env["PATH"].split(os.pathsep)
                                     if p != str(rt.cuda / "bin"))
    c = _find(_runtime(rt), "nvcc 不在 PATH 上")
    assert c.level == WARN and f"export PATH={rt.cuda / 'bin'}:$PATH" in c.hint


def test_nvidia_smi_failure_points_at_reboot(rt) -> None:
    rt.env["FAKE_SMI_MODE"] = "mismatch"
    c = _find(_runtime(rt), "nvidia-smi 执行失败")
    assert c.level == FAIL and "重启" in c.hint


def test_vllm_extension_abi_mismatch_fails(rt) -> None:
    rt.env["FAKE_VLLM_C"] = "abi"
    c = _find(_runtime(rt), "vllm._C 加载失败")
    assert c.level == FAIL and "undefined symbol" in c.detail and "2.13.0+cu130" in c.hint


def test_vllm_pinned_torch_and_transformers_mismatch(rt) -> None:
    (rt.site / "torch-2.13.0+cu130.dist-info" / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: torch\nVersion: 2.12.1\n")
    (rt.site / "transformers-4.57.1.dist-info" / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: transformers\nVersion: 4.50.0\n")
    checks = _runtime(rt)
    assert _find(checks, "vLLM 要求 torch==2.13.0，已安装 2.12.1").level == FAIL
    assert _find(checks, "vLLM 要求 transformers>=4.56.0，已安装 4.50.0").level == WARN
    # The `extra == 'test'` requirement must not be evaluated as a hard dep.
    assert not [c for c in checks if "pytest" in c.title]


@pytest.mark.parametrize("mode,needle,level", [
    ("crash", "import vllm 时进程崩溃", FAIL),
    ("hang", "import vllm 超时", WARN),
])
def test_vllm_import_crash_or_hang_is_contained(rt, mode, needle, level) -> None:
    rt.env["FAKE_VLLM_IMPORT"] = mode
    checks = _runtime(rt, timeout=5)
    assert _find(checks, needle).level == level
    # torch / transformers are still reported from their own probes.
    assert _find(checks, "torch 2.13.0").level == OK
    assert _find(checks, "transformers 4.57.1").level == OK


def test_metadata_without_module_is_broken_install(rt) -> None:
    import shutil as _shutil
    _shutil.rmtree(rt.site / "transformers")
    c = _find(_runtime(rt), "包元数据存在，但找不到 transformers 模块")
    assert c.level == FAIL


@pytest.mark.skipif(any(importlib.util.find_spec(m) for m in ("torch", "vllm", "transformers")),
                    reason="real torch / vllm / transformers installed in the test interpreter")
def test_packages_missing_warn(rt) -> None:
    rt.env.pop("PYTHONPATH")
    checks = _runtime(rt)
    for needle in ("torch 未安装", "vLLM 未安装", "transformers 未安装"):
        assert _find(checks, needle).level == WARN


@pytest.mark.skipif(os.path.exists("/usr/local/cuda/bin/nvcc"), reason="host has /usr/local/cuda")
def test_darwin_skips_cuda(rt) -> None:
    rt.platform = "darwin"
    rt.env["PATH"] = "/usr/bin:/bin"
    rt.env["CUDA_HOME"] = str(rt.cuda / "missing")
    checks = _runtime(rt)
    assert _find(checks, "macOS 不支持 CUDA").level == INFO
    assert not [c for c in checks if "nvcc" in c.title]


def test_python_override_and_main_prints_runtime_section(rt, tmp_path, capsys) -> None:
    wrapper = tmp_path / "bin" / "py-wrapper"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    wrapper.chmod(0o755)
    _install(rt)
    rt.env["ANTHROPIC_API_KEY"] = "k"
    assert env_check.main(["--python", str(wrapper)], rt) == 0
    out = capsys.readouterr().out
    assert "运行时环境（CUDA / torch / vLLM / transformers）" in out
    assert f"解释器: {wrapper} -> " in out
    assert "torch 2.13.0+cu130" in out and "vLLM 0.13.0" in out


def test_bad_python_override_fails(rt) -> None:
    c = _find(env_check.run_runtime_checks(rt, "/nonexistent/python"), "找不到 Python 解释器")
    assert c.level == FAIL
