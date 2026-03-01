#!/usr/bin/env bash
# install.sh — Build llama.cpp from source with CUDA + RPC for 2× DGX Spark cluster
#
# Run as root on BOTH nodes:  sudo bash scripts/install.sh
# Tested on: NVIDIA DGX Spark (GB10, sm_121), DGX OS (Ubuntu 24.04), CUDA 13
set -euo pipefail

LLAMA_REPO="https://github.com/ggml-org/llama.cpp"
CUDA_ARCH="121"   # sm_121 (GB10) — CUDA 13 supports this natively

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[install]${NC} $*"; }
warn()    { echo -e "${YELLOW}[install]${NC} $*"; }
die()     { echo -e "${RED}[install] ERROR:${NC} $*" >&2; exit 1; }

# ── 0. Root check ─────────────────────────────────────────────────────────────
[[ $EUID -eq 0 ]] || die "Run this script as root (sudo bash $0)"

# Resolve the real user (not root)
TARGET_USER="${SUDO_USER:-$(logname 2>/dev/null || echo nobody)}"
TARGET_HOME=$(eval echo "~${TARGET_USER}")
INSTALL_DIR="${TARGET_HOME}/llama.cpp"

# ── 1. Dependencies ───────────────────────────────────────────────────────────
info "Installing build dependencies..."
apt-get update -qq
apt-get install -y --no-install-recommends \
    git cmake build-essential patchelf python3-venv curl

# ── 2. Clone or update llama.cpp ──────────────────────────────────────────────
if [[ -d "${INSTALL_DIR}/.git" ]]; then
    info "llama.cpp already cloned at ${INSTALL_DIR}, pulling latest..."
    su - "${TARGET_USER}" -c "git -C '${INSTALL_DIR}' pull --ff-only"
else
    info "Cloning llama.cpp into ${INSTALL_DIR}..."
    su - "${TARGET_USER}" -c "git clone '${LLAMA_REPO}' '${INSTALL_DIR}'"
fi

# ── 3. Build with CUDA + RPC ─────────────────────────────────────────────────
info "Configuring CMake (CUDA arch ${CUDA_ARCH}, RPC enabled)..."
su - "${TARGET_USER}" -c "
    export PATH=/usr/local/cuda/bin:\$PATH
    cmake -S '${INSTALL_DIR}' -B '${INSTALL_DIR}/build' \\
        -DGGML_CUDA=ON \\
        -DGGML_RPC=ON \\
        -DCMAKE_BUILD_TYPE=Release \\
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}
"

info "Building (this takes a few minutes on ARM)..."
su - "${TARGET_USER}" -c "
    export PATH=/usr/local/cuda/bin:\$PATH
    cmake --build '${INSTALL_DIR}/build' --config Release -j \$(nproc)
"

# ── 4. Create Python venv for huggingface-cli ─────────────────────────────────
VENV_DIR="${INSTALL_DIR}/.venv"
if [[ ! -d "${VENV_DIR}" ]]; then
    info "Creating Python venv for huggingface-cli..."
    su - "${TARGET_USER}" -c "
        python3 -m venv '${VENV_DIR}'
        '${VENV_DIR}/bin/pip' install huggingface_hub hf_transfer
    "
else
    info "Python venv already exists at ${VENV_DIR}"
fi

# ── 5. Verify ─────────────────────────────────────────────────────────────────
info "Verifying build..."
for bin in llama-server rpc-server; do
    if [[ -f "${INSTALL_DIR}/build/bin/${bin}" ]]; then
        info "  ✓ ${bin}"
    else
        die "  ✗ ${bin} not found!"
    fi
done

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN} llama.cpp installed successfully!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Binaries:  ${INSTALL_DIR}/build/bin/llama-server"
echo "             ${INSTALL_DIR}/build/bin/rpc-server"
echo "  Venv:      ${VENV_DIR}/bin/hf"
echo ""
echo "Next steps:"
echo "  1. Run this script on BOTH Spark nodes"
echo "  2. Download a model:  bash scripts/download-model.sh <repo> <pattern>"
echo "  3. Set up services:   sudo bash scripts/setup-openclaw.sh"
echo ""
