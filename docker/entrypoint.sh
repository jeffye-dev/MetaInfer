#!/usr/bin/env bash
# MetaInfer container entrypoint.
#   serve  [args...]  -> WebUI (python /opt/MetaInfer/serve.py)
#   worker [args...]  -> multi-node worker daemon (python -m metainfer.worker)
#   anything else     -> exec as-is (e.g. bash, codex login, ccb)
set -euo pipefail

mkdir -p "${METAINFER_ROOT:-/workspace/metainfer}" "${XDG_CACHE_HOME:-/tmp/cache}"

cmd="${1:-serve}"
[[ $# -gt 0 ]] && shift

case "$cmd" in
    serve)
        exec python /opt/MetaInfer/serve.py \
            --host "${METAINFER_HOST:-0.0.0.0}" \
            --port "${METAINFER_PORT:-8765}" "$@"
        ;;
    worker)
        exec python -m metainfer.worker "$@"
        ;;
    *)
        exec "$cmd" "$@"
        ;;
esac
