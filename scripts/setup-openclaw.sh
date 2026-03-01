#!/usr/bin/env bash
# setup-openclaw.sh — Install llama-proxy + systemd services on the MASTER node
#
# Run as root on the MASTER node:  sudo bash scripts/setup-openclaw.sh
# Requires:  llama.cpp already built (run install.sh on BOTH nodes first)
#
# Environment variables (override defaults):
#   MODEL_DIR    — path to model directory (default: ~/llama-models)
#   MODEL_GLOB   — glob pattern for the .gguf shard(s) (auto-detected)
#   WORKER_RPC   — worker IP:port for RPC (default: 192.168.0.122:50052)
#   CTX_SIZE     — context window size (default: 65536)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"
SYSTEMD_DIR="/etc/systemd/system"

# Detect the user who invoked sudo
SERVICE_USER="${SUDO_USER:-$(logname 2>/dev/null || echo nobody)}"
SERVICE_HOME=$(eval echo "~${SERVICE_USER}")
INSTALL_DIR="${SERVICE_HOME}/llama.cpp"
PROXY_SRC="${REPO_DIR}/proxy/llama-proxy.py"
PYTHON_BIN="$(su - "${SERVICE_USER}" -c 'which python3' 2>/dev/null || which python3)"

# Configurable defaults
MODEL_BASE="${MODEL_DIR:-${SERVICE_HOME}/llama-models}"
WORKER_RPC="${WORKER_RPC:-192.168.0.122:50052}"
CTX_SIZE="${CTX_SIZE:-65536}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info() { echo -e "${GREEN}[setup]${NC} $*"; }
warn() { echo -e "${YELLOW}[setup]${NC} $*"; }
die()  { echo -e "${RED}[setup] ERROR:${NC} $*" >&2; exit 1; }

# ── 0. Checks ─────────────────────────────────────────────────────────────────
[[ $EUID -eq 0 ]] || die "Run this script as root (sudo bash $0)"
[[ -f "${PROXY_SRC}" ]] || die "Proxy script not found at ${PROXY_SRC}"
[[ -f "${INSTALL_DIR}/build/bin/llama-server" ]] || die "llama-server not found — run install.sh first"

# ── 1. Auto-detect model ──────────────────────────────────────────────────────
if [[ -n "${MODEL_GLOB:-}" ]]; then
    MODEL_FILE="${MODEL_GLOB}"
else
    # Find the first .gguf file in any subdirectory of the model base
    MODEL_FILE=$(find "${MODEL_BASE}" -name '*.gguf' -print -quit 2>/dev/null || true)
    if [[ -z "${MODEL_FILE}" ]]; then
        die "No .gguf model found under ${MODEL_BASE}. Run download-model.sh first."
    fi
    # For sharded models, use the -00001-of- shard (llama.cpp auto-finds the rest)
    FIRST_SHARD=$(find "${MODEL_BASE}" -name '*-00001-of-*.gguf' -print -quit 2>/dev/null || true)
    if [[ -n "${FIRST_SHARD}" ]]; then
        MODEL_FILE="${FIRST_SHARD}"
    fi
fi
info "Using model: ${MODEL_FILE}"

# ── 2. Copy proxy script ──────────────────────────────────────────────────────
info "Installing proxy to ${INSTALL_DIR}/llama-proxy.py..."
cp "${PROXY_SRC}" "${INSTALL_DIR}/llama-proxy.py"
chmod 755 "${INSTALL_DIR}/llama-proxy.py"

# ── 3. Write systemd units ────────────────────────────────────────────────────
info "Writing systemd unit: llama-server.service (port 8001)..."
cat > "${SYSTEMD_DIR}/llama-server.service" << EOF
[Unit]
Description=llama.cpp server (distributed, 2× DGX Spark)
After=network-online.target
Wants=network-online.target
Before=llama-proxy.service

[Service]
Type=simple
User=${SERVICE_USER}
Environment=PATH=/usr/local/cuda/bin:/usr/bin:/bin
ExecStart=${INSTALL_DIR}/build/bin/llama-server \\
    --model ${MODEL_FILE} \\
    --ctx-size ${CTX_SIZE} \\
    --parallel 1 \\
    --host 0.0.0.0 \\
    --port 8001 \\
    -ngl 99 \\
    -fa on \\
    --rpc ${WORKER_RPC}
Restart=on-failure
RestartSec=10
StandardOutput=append:/var/log/llama-server.log
StandardError=append:/var/log/llama-server.log
TimeoutStartSec=600

[Install]
WantedBy=multi-user.target
EOF

info "Writing systemd unit: llama-proxy.service (port 8000)..."
cat > "${SYSTEMD_DIR}/llama-proxy.service" << EOF
[Unit]
Description=llama-proxy (role rewrite + thinking control, port 8000→8001)
After=network.target llama-server.service
Requires=llama-server.service

[Service]
Type=simple
User=${SERVICE_USER}
ExecStart=${PYTHON_BIN} ${INSTALL_DIR}/llama-proxy.py
Restart=on-failure
RestartSec=5
StandardOutput=append:/var/log/llama-proxy.log
StandardError=append:/var/log/llama-proxy.log

[Install]
WantedBy=multi-user.target
EOF

# ── 4. Enable + start ─────────────────────────────────────────────────────────
info "Enabling and starting services..."
systemctl daemon-reload
systemctl enable llama-server llama-proxy

# Start llama-server and wait for it to be ready
systemctl start llama-server
info "Waiting for llama-server to load model (up to 5 min)..."
timeout 300 bash -c \
    'until curl -sf http://127.0.0.1:8001/health 2>/dev/null | grep -q ok; do sleep 5; printf "."; done' \
    && echo ""

systemctl start llama-proxy
sleep 2

# ── 5. Verify ─────────────────────────────────────────────────────────────────
info "Verifying proxy health check..."
HEALTH=$(curl -sf http://127.0.0.1:8000/health 2>/dev/null || echo "FAILED")
if echo "${HEALTH}" | grep -q ok; then
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN} Setup complete! (2× DGX Spark cluster)${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "  llama-server  →  http://0.0.0.0:8001   (master, with --rpc to worker)"
    echo "  llama-proxy   →  http://127.0.0.1:8000  (openclaw connects here)"
    echo "  rpc-server    →  ${WORKER_RPC}         (worker node)"
    echo ""
    echo "  Model: ${MODEL_FILE}"
    echo ""
    echo "Next step: add the llamacpp provider to ~/.openclaw/openclaw.json"
    echo "  See: openclaw/provider-snippet.json"
    echo ""
else
    warn "Proxy health check failed."
    echo "  Check logs:  journalctl -u llama-proxy -u llama-server"
    echo "  Make sure the RPC worker is running on ${WORKER_RPC}"
    exit 1
fi
