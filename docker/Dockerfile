# MetaInfer runtime image: CUDA toolkit + PyTorch + vLLM + MetaInfer, plus the
# coding agent CLIs it drives (Claude Code, claude-code-best/ccb, Codex).
#
# Build (context = repo root):
#   docker build -f docker/Dockerfile -t metainfer:latest .
#
# Also build UCX + NIXL from a local NIXL checkout (see the NIXL section
# below; empty NIXL_SRC_PATH, the default, skips it). The checkout is passed
# as a named build context, so it can live anywhere on the build host:
#   NIXL_SRC_PATH=$HOME/linke/nixl-int
#   docker build -f docker/Dockerfile -t metainfer:latest \
#     --build-arg NIXL_SRC_PATH="$NIXL_SRC_PATH" \
#     --build-context nixl_src="$NIXL_SRC_PATH" .
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

# NIXL source for the optional UCX + NIXL build (NIXL section below). Empty
# placeholder; `--build-context nixl_src=<NIXL checkout>` replaces it.
FROM scratch AS nixl_src

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
# Must pin the same torch as TORCH_VERSION (vllm 0.32.0 requires torch==2.13.0).
ARG VLLM_VERSION=0.32.0
ARG UV_VERSION=0.12.23
# Rust toolchain via rustup (rustup-init sha256-verified per arch).
ARG RUST_VERSION=1.99.0
ARG RUSTUP_VERSION=1.29.1
ARG RUSTUP_SHA256_X86_64=dda7234360b7f578ca8b0ddcb80145646fa61a67c1720a5abc7051b35c9fcb71
ARG RUSTUP_SHA256_AARCH64=15f6e4ce9f583b929c996c91562bad6d4454f3281de858b02cdfdef615fac433
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

# PyPI mirror for every uv install below (vllm, MetaInfer) except torch
# (which pins its own --index-url). Build-time only, never baked into the
# image env:
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

# vLLM, matching the rtx-4090 linke_metainfer container (vllm 0.32.0 on
# torch 2.13.0+cu130, torchvision 0.28.0, torchaudio 2.11.0, flashinfer
# 0.7.0.post1, transformers 5.19.0). vllm 0.32.0 pins torch==2.13.0, so the
# CUDA torch above already satisfies it and is kept; the PyPI vllm wheel is
# built for CUDA 13 (links libcudart.so.13). The build fails if vllm swapped
# torch, or if its compiled extensions have unresolved shared libraries.
# They can't be imported here: they link libcuda.so.1, which only exists once
# the NVIDIA runtime injects the driver (docker run --gpus), so ldd checks
# everything except that.
RUN --mount=type=cache,target=/opt/uv/cache \
    torch_before=$(python -c "import torch; print(torch.__version__)") \
    && uv pip install --python /opt/venv/bin/python "vllm==${VLLM_VERSION}" \
    && python -c "import sys, importlib.metadata as m; t = m.version('torch'); print('vllm', m.version('vllm'), 'torch', t); sys.exit(0 if t == sys.argv[1] else f'vllm replaced torch {sys.argv[1]} with {t}')" "${torch_before}" \
    && sp=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") \
    && missing=$(for so in "${sp}"/vllm/_C*.so; do LD_LIBRARY_PATH="${sp}/torch/lib" ldd "${so}"; done \
        | grep 'not found' | grep -v 'libcuda\.so\.1' || true) \
    && if [ -n "${missing}" ]; then echo "vllm extensions have unresolved libs:"; echo "${missing}"; exit 1; fi

# Rust toolchain (rustc, cargo, rustfmt, clippy) for agents building Rust
# code, e.g. NIXL's Rust bindings (nixl-sys). rustup and the toolchain live
# under /opt, independent of $HOME, like uv. Minimal profile keeps docs out of
# the image. pkg-config, libssl-dev and libclang-dev cover the common native
# deps: openssl-sys, and bindgen (used by nixl-sys) needs libclang.
ENV RUSTUP_HOME=/opt/rust/rustup \
    CARGO_HOME=/opt/rust/cargo \
    PATH=/opt/rust/cargo/bin:${PATH}
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        pkg-config \
        libssl-dev \
        libclang-dev \
    && rm -rf /var/lib/apt/lists/* \
    && case "${TARGETARCH:-$(dpkg --print-architecture)}" in \
        amd64) rust_arch=x86_64; rustup_sha="${RUSTUP_SHA256_X86_64}" ;; \
        arm64) rust_arch=aarch64; rustup_sha="${RUSTUP_SHA256_AARCH64}" ;; \
        *) echo "unsupported arch: ${TARGETARCH}" >&2; exit 1 ;; \
    esac \
    && curl -fsSL -o /tmp/rustup-init \
        "https://static.rust-lang.org/rustup/archive/${RUSTUP_VERSION}/${rust_arch}-unknown-linux-gnu/rustup-init" \
    && echo "${rustup_sha}  /tmp/rustup-init" | sha256sum -c - \
    && chmod +x /tmp/rustup-init \
    && /tmp/rustup-init -y --no-modify-path --profile minimal \
        --default-toolchain "${RUST_VERSION}" --component rustfmt,clippy \
    && rm -f /tmp/rustup-init \
    && rm -rf "${CARGO_HOME}/registry" "${CARGO_HOME}/git" \
    && rustc --version \
    && cargo --version

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

# --------------------------------------------------------------------------- #
# Optional: UCX + NIXL built from source (same flow as the validated build in
# nixl-source-analysis/nixl-build). Enabled by --build-arg NIXL_SRC_PATH=<dir>
# together with --build-context nixl_src=<dir> (the checkout itself; it may be
# outside the MetaInfer context and is not copied into /opt/MetaInfer). Empty
# NIXL_SRC_PATH (default) skips both layers.
#   docker build -f docker/Dockerfile --build-arg NIXL_SRC_PATH=$HOME/nixl-int \
#     --build-context nixl_src=$HOME/nixl-int .
# The checkout is bind-mounted read-write so meson can fetch its subprojects;
# those writes (and the build dir) are discarded, the checkout is untouched.
# A pristine copy of it (including .git) is kept in the image at /opt/nixl,
# taken before the build, so it has no build dir, downloaded subprojects or
# renamed pyproject.toml. Not created when NIXL_SRC_PATH is empty.
# Declared here, after the heavy torch/vllm/CLI layers, so changing them only
# re-runs the layers below.
# --------------------------------------------------------------------------- #
ARG NIXL_SRC_PATH=
# debug like NIXL CI: meson only builds test/ (gtest, ucx_backend_test, ...)
# when buildtype != release. Use release / debugoptimized for a lean runtime.
ARG NIXL_BUILDTYPE=debug
# UCX v1.23.x (the version NIXL CI tests against), pinned to the commit that
# was validated: 8094f69 "UCT/CUDA_IPC/DEVICE: fix atomic after put ordering".
ARG UCX_GIT_REF=8094f699daa0d53ea84d11b6cefd7dffa9c9eb46
# Parallel jobs for make/ninja; empty = nproc.
ARG BUILD_JOBS=

# UCX with CUDA, configure flags mirrored from NIXL CI (.gitlab/build.sh).
# Builds against the toolkit's stub libcuda, so no GPU is needed at build time.
RUN --mount=type=cache,target=/opt/uv/cache \
    set -eu; \
    if [ -z "${NIXL_SRC_PATH:-}" ]; then echo "NIXL_SRC_PATH empty: skipping UCX/NIXL"; exit 0; fi; \
    apt-get update; \
    apt-get install --yes --no-install-recommends \
        autoconf automake autotools-dev libtool pkg-config \
        libnuma-dev numactl libz-dev uuid-dev libhwloc-dev hwloc \
        libgtest-dev libgmock-dev libaio-dev libgflags-dev libgoogle-glog-dev \
        libibverbs-dev librdmacm-dev rdma-core ibverbs-providers \
        libssl-dev libcurl4-openssl-dev pybind11-dev patchelf parallel \
        iproute2 net-tools pciutils libpci-dev; \
    rm -rf /var/lib/apt/lists/*; \
    uv pip install --python /opt/venv/bin/python \
        meson meson-python pybind11 tomlkit patchelf pyyaml types-PyYAML build \
        click tabulate 'setuptools>=80.9.0' pytest pytest-timeout pyzmq; \
    jobs=${BUILD_JOBS:-$(nproc)}; \
    src=$(mktemp -d); \
    git -C "${src}" init -q; \
    git -C "${src}" remote add origin https://github.com/openucx/ucx.git; \
    git -C "${src}" fetch -q --depth 1 origin "${UCX_GIT_REF}"; \
    git -C "${src}" checkout -q FETCH_HEAD; \
    git -C "${src}" log --oneline -1; \
    cd "${src}"; \
    ./autogen.sh; \
    ./contrib/configure-release-mt \
        --prefix=/usr/local/ucx \
        --enable-shared --disable-static --disable-doxygen-doc \
        --enable-optimizations --without-avx --enable-cma \
        --enable-devel-headers --with-verbs --with-dm --without-gdrcopy \
        --with-cuda=/usr/local/cuda; \
    make -j"${jobs}"; \
    make -j install-strip; \
    echo /usr/local/ucx/lib > /etc/ld.so.conf.d/ucx.conf; \
    ldconfig; \
    cd /; rm -rf "${src}"; \
    /usr/local/ucx/bin/ucx_info -v | head -2

# NIXL: C++ core + plugins + tests into /usr/local/nixl (meson), then the
# Python packages: nixl-cu<major> (compiled wheel; pyproject's nixl-cu12 is
# renamed per CUDA major like CI) and the "nixl" meta wheel that dispatches to
# it. Plugins resolve relative to each libnixl.so, so NIXL_PLUGIN_DIR is left
# unset (the Python wheel uses its own bundled plugins).
RUN --mount=type=bind,from=nixl_src,target=/nixl-src,rw \
    --mount=type=cache,target=/opt/uv/cache \
    set -eu; \
    if [ -z "${NIXL_SRC_PATH:-}" ]; then echo "NIXL_SRC_PATH empty: skipping NIXL"; exit 0; fi; \
    if [ ! -f /nixl-src/meson.build ]; then \
        echo "NIXL_SRC_PATH=${NIXL_SRC_PATH} but no NIXL source mounted:" \
             "add --build-context nixl_src=${NIXL_SRC_PATH}" >&2; \
        exit 1; \
    fi; \
    # meson clones wrap-git subprojects into the bind-mounted checkout, which
    # may be owned by a host uid; trust it for this step only (not persisted).
    export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*'; \
    jobs=${BUILD_JOBS:-$(nproc)}; \
    # Keep the source in the image before anything writes to it; root-owned
    # so git works in it at runtime.
    mkdir -p /opt/nixl; \
    cp -a --no-preserve=ownership /nixl-src/. /opt/nixl/; \
    echo "NIXL source kept at /opt/nixl: $(git -C /opt/nixl log --oneline -1 2>/dev/null || echo 'no git metadata')"; \
    build=$(mktemp -d); \
    cd /nixl-src; \
    meson setup "${build}" \
        --prefix=/usr/local/nixl \
        --buildtype="${NIXL_BUILDTYPE}" \
        -Ducx_path=/usr/local/ucx \
        -Dbuild_docs=false \
        -Drust=false; \
    ninja -C "${build}" -j"${jobs}"; \
    ninja -C "${build}" install; \
    libdir=/usr/local/nixl/lib/$(uname -m)-linux-gnu; \
    printf '%s\n%s\n' "${libdir}" "${libdir}/plugins" > /etc/ld.so.conf.d/nixl.conf; \
    ldconfig; \
    echo "NIXL plugins: $(ls "${libdir}/plugins" | tr '\n' ' ')"; \
    cuda_major=$(nvcc --version | grep -oP 'release \K[0-9]+'); \
    python ./contrib/tomlutil.py --wheel-name "nixl-cu${cuda_major}" pyproject.toml; \
    uv pip install --python /opt/venv/bin/python --no-build-isolation \
        -C setup-args="-Ducx_path=/usr/local/ucx" \
        -C compile-args="-j${jobs}" .; \
    uv pip install --python /opt/venv/bin/python \
        "${build}"/src/bindings/python/nixl-meta/nixl-*-py3-none-any.whl; \
    rm -rf "${build}"; \
    cd /; \
    python -c "import nixl, nixl._bindings; print('nixl ->', nixl._pkg.__name__)"

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
    METAINFER_CODEX_BIN=codex \
    PATH=/usr/local/nixl/bin:/usr/local/ucx/bin:${PATH}

USER root
WORKDIR /workspace/metainfer
VOLUME ["/workspace/metainfer"]
EXPOSE 8765

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/metainfer-entrypoint"]
CMD ["serve"]
