# MetaInfer runtime image: MetaInfer WebUI/orchestrator + coding agent CLIs
# (Claude Code, claude-code-best/ccb, Codex).
#
# Build (context = repo root):
#   docker build -f docker/Dockerfile -t metainfer:latest .
#
# Run:
#   docker run --rm -it -p 8765:8765 \
#     -v $PWD/data:/workspace/metainfer \
#     -v ~/.claude:/root/.claude \
#     -v ~/.codex:/root/.codex \
#     -e METAINFER_AGENT_BACKEND=claude \
#     metainfer:latest

FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TARGETARCH

# Pinned tool versions; override with --build-arg when upgrading.
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
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        ripgrep \
        tini \
        unzip \
        vim-tiny \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

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

# MetaInfer itself, installed into a dedicated venv so the console scripts
# (metainfer-web, metainfer-orchestrator) are on PATH.
COPY . /opt/MetaInfer
RUN python3 -m venv /opt/metainfer-venv \
    && /opt/metainfer-venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/metainfer-venv/bin/pip install --no-cache-dir -e /opt/MetaInfer \
    && /opt/metainfer-venv/bin/python -c "import metainfer.server.app"

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
    PATH=/opt/metainfer-venv/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
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
