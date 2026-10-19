# MetaInfer runtime image: CUDA toolkit + PyTorch + MetaInfer, plus the coding
# agent CLIs it drives (Claude Code, claude-code-best/ccb, Codex).
#
# Build (context = repo root):
#   docker build -f docker/Dockerfile -t metainfer:latest .
#
# CUDA 13 needs NVIDIA driver >= 580. On older drivers pick a CUDA 12.x that
# PyTorch ships wheels for (torch 2.13.0: cu126 / cu129), e.g.:
#   docker build -f docker/Dockerfile --build-arg CUDA_VERSION=12.9.1 -t metainfer:cu129 .
#
# Run:
#   docker run --rm -it --gpus all -p 8765:8765 \
#     -v $PWD/data:/workspace/metainfer \
#     -v ~/.claude:/root/.claude \
#     -v ~/.codex:/root/.codex \
#     -e METAINFER_AGENT_BACKEND=claude \
#     metainfer:latest

ARG CUDA_VERSION=13.0.3
# devel (not base/runtime): agents compile CUDA kernels and extensions with
# nvcc. Parameterized so a private registry mirror can be substituted.
ARG BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG TARGETARCH
ARG CUDA_VERSION

# Pinned tool versions; override with --build-arg when upgrading.
ARG PYTHON_VERSION=3.12
# The torch wheel comes from ${PYTORCH_CUDA_INDEX_BASE_URL}/cu<major><minor>
# derived from CUDA_VERSION, so that pair must exist on the PyTorch index.
ARG TORCH_VERSION=2.13.0
ARG PYTORCH_CUDA_INDEX_BASE_URL=https://download.pytorch.org/whl
ARG UV_VERSION=0.12.23
ARG NODE_VERSION=24.22.0
ARG BUN_VERSION=1.4.3
ARG CLAUDE_CODE_VERSION=2.1.301
ARG CLAUDE_CODE_BEST_VERSION=2.8.4
ARG CODEX_VERSION=0.163.0

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        jq \
        less \
        vim \
        ninja-build \
        openssh-client \
        procps \
        ripgrep \
        tini \
        unzip \
        vim-tiny \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

# PyPI mirror for every uv install below except torch (which pins its own
# --index-url). Build-time only, never baked into the image env:
#   --build-arg UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG UV_INDEX_URL

# Python ${PYTHON_VERSION} via uv (Ubuntu 22.04 only ships 3.10). uv, its
# managed CPython and the venv all live under /opt, independent of $HOME.
ENV UV_INSTALL_DIR=/opt/uv/bin \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_CACHE_DIR=/opt/uv/cache \
    UV_HTTP_TIMEOUT=500 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/opt/uv/bin:/root/.local/bin:${PATH}
RUN curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | env UV_NO_MODIFY_PATH=1 sh \
    && uv venv --seed --python "${PYTHON_VERSION}" /opt/venv \
    && python --version \
    && python -m pip --version
# Only after `uv venv` (which hardlinks back to the managed CPython): copy
# mode avoids hardlink failures across BuildKit cache mounts.
ENV UV_LINK_MODE=copy

# PyTorch built for this image's CUDA (13.0.3 -> cu130). Only the PyTorch
# index is used so PyPI's default-CUDA torch can never win resolution, and
# the build fails if torch's CUDA doesn't match the image. Installed before
# MetaInfer is copied so code changes keep this ~3GB layer cached.
RUN --mount=type=cache,target=/opt/uv/cache \
    cuda_mm=$(echo "${CUDA_VERSION}" | cut -d. -f1,2) \
    && uv pip install --python /opt/venv/bin/python \
        --index-url "${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo "${cuda_mm}" | tr -d .)" \
        "torch==${TORCH_VERSION}" numpy \
    && python -c "import sys, torch; v = torch.version.cuda; print('torch', torch.__version__, 'CUDA', v); sys.exit(0 if v == sys.argv[1] else f'torch CUDA {v} != image CUDA {sys.argv[1]}')" "${cuda_mm}"

# Node.js from the official release tarball (ccb / Claude Code / Codex are
# distributed via npm).
RUN case "${TARGETARCH:-$(dpkg --print-architecture)}" in \
        amd64) node_arch=x64 ;; \
        arm64) node_arch=arm64 ;; \
        *) echo "unsupported arch: ${TARGETARCH}" >&2; exit 1 ;; \
    esac \
    && curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${node_arch}.tar.xz" \
        | tar -xJ -C /usr/local --strip-components=1 --no-same-owner \
        --exclude='*/CHANGELOG.md' --exclude='*/README.md' --exclude='*/LICENSE' \
    && node --version \
    && npm --version

# Coding agent CLIs. ccb declares bun>=1.3 as its engine, so bun is
# installed alongside node.
RUN npm install --global --no-fund --no-audit \
        "bun@${BUN_VERSION}" \
        "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
        "claude-code-best@${CLAUDE_CODE_BEST_VERSION}" \
        "@openai/codex@${CODEX_VERSION}" \
    && npm cache clean --force \
    && bun --version \
    && claude --version \
    && ccb --version \
    && codex --version

# MetaInfer itself, editable in /opt/venv so the console scripts
# (metainfer-web, metainfer-orchestrator) are on PATH. The import check runs
# against a throwaway METAINFER_ROOT so it leaves no nodes/ dir in the image.
COPY . /opt/MetaInfer
RUN --mount=type=cache,target=/opt/uv/cache \
    uv pip install --python /opt/venv/bin/python -e /opt/MetaInfer \
    && tmp_root=$(mktemp -d) \
    && METAINFER_ROOT="${tmp_root}" python -c "import metainfer.server.app" \
    && rm -rf "${tmp_root}"

RUN mkdir -p \
        /root/.claude \
        /root/.codex \
        /root/.config \
        /root/.local/share \
        /root/.local/state \
        /workspace/metainfer

COPY docker/entrypoint.sh /usr/local/bin/metainfer-entrypoint
RUN chmod 0755 /usr/local/bin/metainfer-entrypoint

# Runs as root. Claude Code / ccb refuse bypassPermissions (the MetaInfer
# default) as root unless IS_SANDBOX=1, which marks the container as the
# sandbox boundary.
ENV HOME=/root \
    IS_SANDBOX=1 \
    CODEX_HOME=/root/.codex \
    CLAUDE_CONFIG_DIR=/root/.claude \
    XDG_CONFIG_HOME=/root/.config \
    XDG_DATA_HOME=/root/.local/share \
    XDG_STATE_HOME=/root/.local/state \
    XDG_CACHE_HOME=/tmp/cache \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    METAINFER_ROOT=/workspace/metainfer \
    METAINFER_NODE_ID=local \
    METAINFER_HOST=0.0.0.0 \
    METAINFER_PORT=8765 \
    METAINFER_AGENT_BACKEND=claude \
    METAINFER_CLAUDE_BIN=ccb \
    METAINFER_CODEX_BIN=codex

USER root
WORKDIR /workspace/metainfer
VOLUME ["/workspace/metainfer"]
EXPOSE 8765

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/metainfer-entrypoint"]
CMD ["serve"]
