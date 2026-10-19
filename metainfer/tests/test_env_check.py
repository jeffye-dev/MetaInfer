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
    env_check.main([], ctx)
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
    assert env_check.main([], ctx) == 1  # no binary, no credentials
    _install(ctx)
    ctx.env["ANTHROPIC_API_KEY"] = "k"
    assert env_check.main([], ctx) == 0
