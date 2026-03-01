# llama.cpp + openclaw on 2× NVIDIA DGX Spark (GB10)

> **Fork of [ZengboJamesWang/Qwen3.5-35B-A3B-openclaw-dgx-spark](https://github.com/ZengboJamesWang/Qwen3.5-35B-A3B-openclaw-dgx-spark)** — adapted for **distributed inference across two DGX Sparks**, enabling much larger models.

Run **Qwen3.5-122B-A10B** (or larger) locally on a 2× DGX Spark cluster and use it inside **openclaw** as a fully functional AI agent — including tool calls and on-demand reasoning mode.

## What changed from the original

| Change | Why |
|--------|-----|
| Build with `-DGGML_RPC=ON -DCMAKE_CUDA_ARCHITECTURES=121` | RPC enables distributed inference; CUDA 13 supports sm_121 natively |
| Added `rpc-server` on worker node | Exposes the second Spark's GPU + memory to the master |
| `llama-server --rpc <worker_ip>:50052` | Splits model weights and KV cache across both nodes |
| New `scripts/download-model.sh` | Downloads GGUF once, syncs to both Sparks over ConnectX-7 |
| New `systemd/llama-rpc-worker.service` | Manages the RPC worker as a systemd service |
| Updated model & context sizes | Larger models need adjusted context budgets |

---

## Hardware

Two **NVIDIA DGX Spark** units (GB10 Superchip, sm_121) connected via ConnectX-7.

| Metric | Single Spark | 2× Cluster |
|--------|-------------|------------|
| Unified memory | ~128 GB | **~256 GB** |
| Memory bandwidth | 273 GB/s | 273 GB/s per node |
| Interconnect | — | 200 Gb/s RDMA (ConnectX-7) |
| Max model (Q4) | ~70 GB | **~240 GB** |

### Which models fit

| Model | Quant | Size | Fits on | Notes |
|-------|-------|------|---------|-------|
| Qwen3.5-35B-A3B | UD-Q4_K_XL | ~21 GB | 1× Spark | Use [original guide](https://github.com/ZengboJamesWang/Qwen3.5-35B-A3B-openclaw-dgx-spark) |
| Qwen3.5-122B-A10B | Q4_K_M | ~70 GB | 1× or 2× Spark | Splits load across both GPUs on 2× |
| Qwen3-235B-A22B | UD-Q4_K_XL | ~140 GB | 2× Spark | Qwen3 flagship MoE |
| **Qwen3.5-397B-A17B** | **Q3_K_M** | **~189 GB** | **2× Spark** | **Qwen3.5 flagship** |
| Qwen3.5-397B-A17B | Q4_K_M | ~241 GB | 2× Spark | Tight — ~15 GB for KV cache |

---

## What this repo provides

| File | Purpose |
|------|---------|
| `scripts/install.sh` | Builds llama.cpp with CUDA + **RPC** on both nodes |
| `scripts/download-model.sh` | Downloads GGUF model, syncs to both Sparks over ConnectX-7 |
| `scripts/setup-openclaw.sh` | Installs proxy + systemd units on the master node |
| `proxy/llama-proxy.py` | Proxy that makes llama-server compatible with openclaw |
| `systemd/llama-rpc-worker.service` | systemd unit for rpc-server on the **worker** node (port 50052) |
| `systemd/llama-server.service` | systemd unit for llama-server on the **master** node (port 8001) |
| `systemd/llama-proxy.service` | systemd unit for the proxy on the **master** node (port 8000) |
| `openclaw/provider-snippet.json` | Drop-in config snippet for `~/.openclaw/openclaw.json` |

---

## Quick start

```bash
# === On BOTH nodes ===
# 1. Build llama.cpp with CUDA + RPC
sudo bash scripts/install.sh

# === On the MASTER node only ===
# 2. Download model and sync to worker
bash scripts/download-model.sh unsloth/Qwen3.5-122B-A10B-GGUF "Q4_K_M/*"

# === On the WORKER node ===
# 3. Start the RPC worker
sudo cp systemd/llama-rpc-worker.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now llama-rpc-worker

# === On the MASTER node ===
# 4. Install proxy + systemd services
sudo bash scripts/setup-openclaw.sh

# 5. Add the provider to openclaw (see Section 4 below)
```

---

## Section 0 — Network setup

Connect the two Sparks via QSFP cable using the ConnectX-7 ports.

Your DGX OS setup script likely already configured this. Verify:

```bash
ip link show | grep -A1 'enp1s0f1np1\|enP2p1s0f1np1'
ip -4 addr show enp1s0f1np1
```

Note the IPs. Throughout this guide:
- **Master** (runs llama-server + proxy): `192.168.0.124`
- **Worker** (runs rpc-server): `192.168.0.122`

Replace these with your actual IPs.

---

## Section 1 — Build llama.cpp (both nodes)

See [`scripts/install.sh`] for the full automated script, or follow manually:

```bash
sudo apt-get install -y git cmake build-essential patchelf
export PATH=/usr/local/cuda/bin:$PATH

git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cd ~/llama.cpp

cmake -B build \
  -DGGML_CUDA=ON \
  -DGGML_RPC=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=121

cmake --build build --config Release -j $(nproc)

# Create venv for huggingface-cli
python3 -m venv ~/llama.cpp/.venv
~/llama.cpp/.venv/bin/pip install huggingface_hub hf_transfer
```

> **Key differences from single-node guide:**
> - `-DGGML_RPC=ON` — enables the RPC backend for distributed inference
> - `-DCMAKE_CUDA_ARCHITECTURES=121` — CUDA 13 supports sm_121 natively (the original guide used `120`)

---

## Section 2 — Download the model

```bash
# Qwen3.5-122B-A10B at Q4_K_M (~70 GB, 3 shards)
bash scripts/download-model.sh unsloth/Qwen3.5-122B-A10B-GGUF "Q4_K_M/*"

# Or: Qwen3.5-397B-A17B at Q3_K_M (~189 GB)
bash scripts/download-model.sh unsloth/Qwen3.5-397B-A17B-GGUF "Q3_K_M/*"

# Override worker IP:
REMOTE_HOST=192.168.0.122 bash scripts/download-model.sh ...
```

---

## Section 3 — Start the cluster

```
openclaw  →  port 8000 (llama-proxy)  →  port 8001 (llama-server, MASTER)
                                               ↕ RPC (TCP)
                                         port 50052 (rpc-server, WORKER)
```

### Worker: start rpc-server

```bash
~/llama.cpp/build/bin/rpc-server --host 192.168.0.122 --port 50052 -c
```

### Master: start llama-server

```bash
~/llama.cpp/build/bin/llama-server \
  --model ~/llama-models/unsloth--Qwen3.5-122B-A10B-GGUF/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf \
  --ctx-size 65536 \
  --parallel 1 \
  --host 0.0.0.0 \
  --port 8001 \
  -ngl 99 \
  -fa on \
  --rpc 192.168.0.122:50052
```

| Flag | Effect |
|------|--------|
| `--rpc <worker_ip>:50052` | Offload to the worker node |
| `--ctx-size 65536` | Adjusted for larger models (reduce if OOM) |
| `-ngl 99` | Offload all layers to GPU (both local and remote) |
| `-fa on` | Flash attention |

llama.cpp automatically splits model weights across both nodes proportionally.

---

## Section 4 — Configure openclaw

Add the `llamacpp` provider to `~/.openclaw/openclaw.json` (see `openclaw/provider-snippet.json`):

```json
"llamacpp": {
  "baseUrl": "http://127.0.0.1:8000/v1",
  "apiKey": "llamacpp-local",
  "api": "openai-completions",
  "models": [
    {
      "id": "Qwen3.5-122B-A10B",
      "name": "Qwen3.5-122B-A10B (local, 2× Spark)",
      "reasoning": true,
      "input": ["text"],
      "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
      "contextWindow": 65536,
      "maxTokens": 16384
    }
  ]
}
```

Select with `/model qwen` after adding the alias.

---

## Performance

Tested with Qwen3.5-122B-A10B Q4_K_M on 2× DGX Spark:

| Metric | Value |
|--------|-------|
| Prefill speed | ~70 tok/s |
| Generation speed | ~20 tok/s |
| Context window | 65536 tokens |

---

## Troubleshooting

### `CUDA error: no kernel image is available for execution`
Rebuild **both nodes** with `-DCMAKE_CUDA_ARCHITECTURES=121`.

### `Remote RPC server crashed or returned malformed response`
Check rpc-server log: `sudo journalctl -u llama-rpc-worker -n 50`
Most common cause: CUDA arch mismatch.

### Worker unreachable
```bash
sudo ufw allow 50052/tcp  # on worker
iperf3 -s                 # on worker
iperf3 -c <worker_ip>     # on master
```

### `HTTP 500: Unexpected message role`
The proxy is not running: `systemctl status llama-proxy`

### `huggingface-cli: command not found`
Use `~/llama.cpp/.venv/bin/hf download ...` — the CLI is named `hf` in huggingface_hub >= 1.x.
