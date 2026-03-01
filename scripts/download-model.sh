#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# download-model.sh — Download a GGUF model and sync to the second DGX Spark
#
# Usage:
#   ./download-model.sh <hf_repo> <include_pattern> [model_dir]
#
# Examples:
#   # Qwen3.5-122B-A10B at Q4_K_M (single file, ~70 GB)
#   ./download-model.sh unsloth/Qwen3.5-122B-A10B-GGUF "Qwen3.5-122B-A10B-Q4_K_M.gguf"
#
#   # Qwen3.5-397B-A17B at Q3_K_M (sharded, ~189 GB)
#   ./download-model.sh unsloth/Qwen3.5-397B-A17B-GGUF "Q3_K_M/*"
#
#   # Qwen3-235B-A22B at UD-Q4_K_XL (single file, ~140 GB)
#   ./download-model.sh unsloth/Qwen3-235B-A22B-GGUF "Qwen3-235B-A22B-UD-Q4_K_XL.gguf"
#
# Environment variables (override defaults):
#   MODEL_BASE_DIR   — where models are stored (default: ~/llama-models)
#   REMOTE_HOST      — high-speed IP of worker Spark (default: 192.168.0.122)
#   REMOTE_USER      — SSH user on worker (default: $USER)
###############################################################################

HF_REPO="${1:?Usage: $0 <hf_repo> <include_pattern> [model_dir]}"
INCLUDE="${2:?Usage: $0 <hf_repo> <include_pattern> [model_dir]}"

# Derive a clean model directory name from the repo
DEFAULT_DIR=$(echo "$HF_REPO" | sed 's|/|--|g')
MODEL_DIR="${3:-$DEFAULT_DIR}"

MODEL_BASE_DIR="${MODEL_BASE_DIR:-$HOME/llama-models}"
REMOTE_HOST="${REMOTE_HOST:-192.168.0.122}"
REMOTE_USER="${REMOTE_USER:-$USER}"

LOCAL_PATH="$MODEL_BASE_DIR/$MODEL_DIR"

echo "============================================="
echo "  DGX Spark Model Downloader"
echo "============================================="
echo "  Repo:        $HF_REPO"
echo "  Pattern:     $INCLUDE"
echo "  Local path:  $LOCAL_PATH"
echo "  Remote:      $REMOTE_USER@$REMOTE_HOST:$LOCAL_PATH"
echo "============================================="

# --- Step 0: Ensure huggingface-cli is available via venv ---
VENV_DIR="$HOME/llama.cpp/.venv"
if [ ! -f "$VENV_DIR/bin/huggingface-cli" ]; then
    echo "[*] Creating venv at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
    echo "[*] Installing huggingface_hub + hf_transfer..."
    "$VENV_DIR/bin/pip" install huggingface_hub hf_transfer
fi

# Use venv binaries directly (more reliable than source activate over SSH)
VENV_PYTHON="$VENV_DIR/bin/python3"
# huggingface_hub >= 1.x uses 'hf', older versions use 'huggingface-cli'
if [ -f "$VENV_DIR/bin/hf" ]; then
    HF_CLI="$VENV_DIR/bin/hf"
elif [ -f "$VENV_DIR/bin/huggingface-cli" ]; then
    HF_CLI="$VENV_DIR/bin/huggingface-cli"
else
    echo "ERROR: Cannot find hf or huggingface-cli in $VENV_DIR/bin/"
    exit 1
fi

# Enable fast Rust-based transfers if available
if "$VENV_PYTHON" -c "import hf_transfer" &>/dev/null; then
    export HF_HUB_ENABLE_HF_TRANSFER=1
    echo "[*] hf_transfer enabled (fast downloads)"
fi

# --- Step 1: Download ---
mkdir -p "$LOCAL_PATH"
echo ""
echo "[1/3] Downloading from $HF_REPO ..."
echo "      Pattern: $INCLUDE"
echo ""

DOWNLOAD_TMP="${LOCAL_PATH}/.dl_tmp"
mkdir -p "$DOWNLOAD_TMP"

"$HF_CLI" download \
    "$HF_REPO" \
    --include "$INCLUDE" \
    --local-dir "$DOWNLOAD_TMP"

# Flatten: move all .gguf files to LOCAL_PATH regardless of subdirectory
find "$DOWNLOAD_TMP" -name '*.gguf' -exec mv {} "$LOCAL_PATH/" \;
rm -rf "$DOWNLOAD_TMP"

echo ""
echo "[*] Download complete. Contents:"
ls -lhS "$LOCAL_PATH"/*.gguf 2>/dev/null || ls -lhR "$LOCAL_PATH" | head -20
TOTAL=$(du -sh "$LOCAL_PATH" | cut -f1)
echo "    Total: $TOTAL"

# --- Step 2: Sync to worker Spark ---
echo ""
echo "[2/3] Syncing to $REMOTE_USER@$REMOTE_HOST ..."
echo "      (over ConnectX-7 high-speed link)"
echo ""

ssh "$REMOTE_USER@$REMOTE_HOST" "mkdir -p '$LOCAL_PATH'"

rsync -avh --progress \
    -e "ssh" \
    "$LOCAL_PATH/" \
    "$REMOTE_USER@$REMOTE_HOST:$LOCAL_PATH/"

# --- Step 3: Verify ---
echo ""
echo "[3/3] Verifying remote copy..."
LOCAL_SIZE=$(du -sb "$LOCAL_PATH" | cut -f1)
REMOTE_SIZE=$(ssh "$REMOTE_USER@$REMOTE_HOST" "du -sb '$LOCAL_PATH' | cut -f1")

if [ "$LOCAL_SIZE" = "$REMOTE_SIZE" ]; then
    echo "  ✓ Sizes match: $TOTAL"
    echo ""
    echo "============================================="
    echo "  Done! Model ready on both Sparks at:"
    echo "  $LOCAL_PATH"
    echo "============================================="
else
    echo "  ✗ Size mismatch! Local=$LOCAL_SIZE Remote=$REMOTE_SIZE"
    echo "    Re-run the script to retry rsync."
    exit 1
fi
