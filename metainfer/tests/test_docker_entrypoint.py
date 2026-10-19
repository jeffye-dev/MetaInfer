"""docker/entrypoint.sh preflight: METAINFER_ROOT must be writable.

A fake ``python`` on PATH stands in for the real server so the tests only
exercise the shell dispatch + preflight, never start MetaInfer itself.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ENTRYPOINT = Path(__file__).resolve().parents[2] / "docker" / "entrypoint.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash not installed",
)

# Root bypasses mode bits, so read-only dirs are still writable for it.
needs_non_root = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root ignores directory permissions",
)


@pytest.fixture()
def fake_bin(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "python"
    fake.write_text('#!/bin/sh\necho "FAKE-PYTHON $*"\n')
    fake.chmod(0o755)
    return bin_dir


def _run(args: list[str], env_extra: dict[str, str], fake_bin: Path,
         cwd: Path) -> subprocess.CompletedProcess:
    env = {
        "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
        "HOME": str(cwd),
        "XDG_CACHE_HOME": str(cwd / "cache"),
        **env_extra,
    }
    return subprocess.run(
        ["bash", str(ENTRYPOINT), *args], env=env, cwd=cwd,
        capture_output=True, text=True, timeout=30,
    )


def _read_only_dir(path: Path) -> Path:
    path.mkdir()
    path.chmod(stat.S_IRUSR | stat.S_IXUSR)
    return path


@pytest.fixture()
def restore_perms():
    dirs: list[Path] = []
    yield dirs
    for d in dirs:
        d.chmod(0o755)


def test_serve_creates_missing_root_and_execs(tmp_path, fake_bin) -> None:
    root = tmp_path / "data" / "metainfer"
    r = _run(["serve", "--reload"], {"METAINFER_ROOT": str(root)}, fake_bin, tmp_path)
    assert r.returncode == 0, r.stderr
    assert root.is_dir()
    assert "FAKE-PYTHON /opt/MetaInfer/serve.py" in r.stdout
    assert "--reload" in r.stdout
    # Probe file must not be left behind in the user's data dir.
    assert list(root.iterdir()) == []


@needs_non_root
def test_serve_unwritable_root_fails_with_hint(tmp_path, fake_bin, restore_perms) -> None:
    root = _read_only_dir(tmp_path / "data")
    restore_perms.append(root)
    r = _run(["serve"], {"METAINFER_ROOT": str(root)}, fake_bin, tmp_path)
    assert r.returncode == 1
    assert "FAKE-PYTHON" not in r.stdout
    assert f"METAINFER_ROOT={root} 不可写" in r.stderr
    assert f"chown -R {os.getuid()}:{os.getgid()}" in r.stderr
    assert "root_squash" not in r.stderr


@needs_non_root
def test_root_unwritable_root_points_at_ro_or_root_squash(
        tmp_path, fake_bin, restore_perms) -> None:
    # Fake `id` reports root; the dir stays genuinely unwritable for the
    # real (non-root) test user, mimicking a :ro mount / NFS root_squash.
    fake_id = fake_bin / "id"
    fake_id.write_text(
        '#!/bin/sh\ncase "$1" in -un) echo root ;; *) echo 0 ;; esac\n')
    fake_id.chmod(0o755)
    root = _read_only_dir(tmp_path / "data")
    restore_perms.append(root)
    r = _run(["serve"], {"METAINFER_ROOT": str(root)}, fake_bin, tmp_path)
    assert r.returncode == 1
    assert "容器用户 root (uid=0 gid=0)" in r.stderr
    assert "root_squash" in r.stderr
    assert "chown -R 0:0" not in r.stderr


@needs_non_root
def test_worker_honors_metainfer_root_flag(tmp_path, fake_bin, restore_perms) -> None:
    bad = _read_only_dir(tmp_path / "bad")
    restore_perms.append(bad)
    good = tmp_path / "good"
    # Env points at an unwritable dir, but the flag wins (both flag forms).
    for flag in (["--metainfer-root", str(good)], [f"--metainfer-root={good}"]):
        r = _run(["worker", *flag], {"METAINFER_ROOT": str(bad)}, fake_bin, tmp_path)
        assert r.returncode == 0, r.stderr
        assert "FAKE-PYTHON -m metainfer.worker" in r.stdout
    r = _run(["worker"], {"METAINFER_ROOT": str(bad)}, fake_bin, tmp_path)
    assert r.returncode == 1
    assert "不可写" in r.stderr


@needs_non_root
def test_unwritable_agent_config_only_warns(tmp_path, fake_bin, restore_perms) -> None:
    claude_dir = _read_only_dir(tmp_path / "claude")
    restore_perms.append(claude_dir)
    r = _run(["serve"], {
        "METAINFER_ROOT": str(tmp_path / "root"),
        "CLAUDE_CONFIG_DIR": str(claude_dir),
    }, fake_bin, tmp_path)
    assert r.returncode == 0, r.stderr
    assert f"警告: CLAUDE_CONFIG_DIR={claude_dir} 不可写" in r.stderr
    assert "FAKE-PYTHON" in r.stdout


def test_other_commands_skip_preflight(tmp_path, fake_bin, restore_perms) -> None:
    root = _read_only_dir(tmp_path / "data")
    restore_perms.append(root)
    r = _run(["echo", "hello"], {"METAINFER_ROOT": str(root)}, fake_bin, tmp_path)
    assert r.returncode == 0
    assert r.stdout.strip() == "hello"
    assert r.stderr == ""
