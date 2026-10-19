#!/usr/bin/env bash
# MetaInfer container entrypoint.
#   serve  [args...]  -> WebUI (python /opt/MetaInfer/serve.py)
#   worker [args...]  -> multi-node worker daemon (python -m metainfer.worker)
#   anything else     -> exec as-is (e.g. bash, codex login, ccb)
#
# serve/worker first preflight that METAINFER_ROOT is writable by the
# container user, so a bind mount with the wrong owner fails with a fix
# hint instead of a PermissionError traceback deep inside the server.
set -euo pipefail

log() { printf '[metainfer] %s\n' "$*" >&2; }

# "uid:gid" of a path; GNU stat in the image, BSD stat as a fallback.
owner_of() {
    stat -c '%u:%g' "$1" 2>/dev/null || stat -f '%u:%g' "$1" 2>/dev/null || echo '?'
}

# Create the dir if missing, then prove writability with a real probe file
# (catches read-only mounts and NFS root-squash that a mode check misses).
is_writable_dir() {
    local dir=$1 probe
    mkdir -p "$dir" 2>/dev/null || return 1
    probe=$(mktemp "$dir/.metainfer-write-probe.XXXXXX" 2>/dev/null) || return 1
    rm -f "$probe"
}

require_writable() {
    local dir=$1
    is_writable_dir "$dir" && return 0
    local uid gid
    uid=$(id -u)
    gid=$(id -g)
    log "错误: METAINFER_ROOT=$dir 不可写"
    log "  目录属主 $(owner_of "$dir")，容器用户 $(id -un 2>/dev/null || echo '?') (uid=$uid gid=$gid)"
    if [[ "$uid" == 0 ]]; then
        log "  root 仍不可写，通常是只读挂载（-v ...:ro）或 NFS root_squash（root 被映射成 nobody）"
        log "  修复: 去掉 :ro；NFS 服务端改用 no_root_squash 导出，或把该目录属主改为 65534:65534"
    else
        log "  通常是 -v 挂载的宿主机目录属主与容器用户不一致（不存在的目录会被 dockerd 以 root 创建）"
        log "  修复: 宿主机上执行 mkdir -p <宿主机目录> && sudo chown -R $uid:$gid <宿主机目录>，或去掉 --user 以默认 root 运行"
    fi
    exit 1
}

# Agent CLI config dirs: agents can still run read-only (e.g. API key via
# env), so only warn.
warn_if_unwritable() {
    local name=$1 dir=$2
    [[ -n "$dir" ]] || return 0
    is_writable_dir "$dir" && return 0
    log "警告: $name=$dir 不可写（属主 $(owner_of "$dir")，容器 uid=$(id -u)），agent 可能无法保存登录态/会话"
}

preflight() {
    require_writable "$1"
    warn_if_unwritable CLAUDE_CONFIG_DIR "${CLAUDE_CONFIG_DIR:-}"
    warn_if_unwritable CODEX_HOME "${CODEX_HOME:-}"
}

# The worker accepts --metainfer-root, which overrides $METAINFER_ROOT.
worker_root() {
    local root=${METAINFER_ROOT:-} prev=
    for arg in "$@"; do
        case "$arg" in
            --metainfer-root=*) root=${arg#*=} ;;
            *) [[ "$prev" == "--metainfer-root" ]] && root=$arg ;;
        esac
        prev=$arg
    done
    printf '%s' "$root"
}

mkdir -p "${XDG_CACHE_HOME:-/tmp/cache}" 2>/dev/null || true

cmd="${1:-serve}"
[[ $# -gt 0 ]] && shift

case "$cmd" in
    serve)
        # Matches metainfer.server.paths.root_dir(): unset -> cwd.
        preflight "${METAINFER_ROOT:-$PWD}"
        exec python /opt/MetaInfer/serve.py \
            --host "${METAINFER_HOST:-0.0.0.0}" \
            --port "${METAINFER_PORT:-8765}" "$@"
        ;;
    worker)
        root=$(worker_root "$@")
        preflight "${root:-$PWD}"
        exec python -m metainfer.worker "$@"
        ;;
    *)
        exec "$cmd" "$@"
        ;;
esac
