<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/icon-rounded-dark.svg" width="140">
    <source media="(prefers-color-scheme: light)" srcset="docs/images/icon-rounded-light.svg" width="140">
    <img alt="cMLX" src="docs/images/icon-rounded-light.svg" width="140">
  </picture>
</p>

<h1 align="center">cMLX (Optimized Fork)</h1>
<p align="center"><b>Next-Generation LLM Inference for Apple Silicon</b><br>Native C++ Extensions · vLLM Metal PagedAttention · Predictive Memory Safety · Dynamic KV Quantization</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-0.5.3-blue" alt="Version">
  <img src="https://img.shields.io/badge/optimized-M4%20Pro-orange" alt="M4 Pro Optimized">
  <img src="https://img.shields.io/badge/runtime-C++%20Native-red" alt="C++ Native">
  <img src="https://img.shields.io/badge/vllm--metal-PagedAttention-green" alt="vLLM Metal">
</p>

---

## 📖 Overview

This repository is a high-performance fork of [jundot/omlx](https://github.com/jundot/omlx), specifically engineered to maximize throughput and stability on high-spec Apple Silicon (MacBook M4 Pro). 

While the original `cmlx` brought continuous batching and tiered caching to MLX, this fork introduces a **C++ Native Runtime** and **Predictive Memory Management** to eliminate the Metal kernel panics and Python interpreter bottlenecks associated with high-concurrency LLM/VLM inference.

---

## 🚀 Key Enhancements (Latest Native Core)

### 1. ⚡ Full C++ Native Inference Core (`NativeEngine`)
We have migrated the entire performance-critical inference path to C++, eliminating Python GIL bottlenecks:
- **Continuous Batching**: Native C++ state machine managing `WAITING`, `PREFILLING`, and `GENERATING` states.
- **Metal Partitioning**: Integrated "Partitioned Reduction" from `vllm-metal`. Sequences >4096 tokens are split into parallel chunks, eliminating the sequential attention bottleneck.
- **On-GPU Sampling**: Temperature and Greedy sampling happen entirely on the GPU via `mx::random::categorical`, saving PCIe bandwidth.
- **Dynamic Chunked Prefill**: Automatically scales prefill chunks (2048 -> 128) based on real-time memory pressure to prevent system stalls.

### 2. 🤖 Standalone Agent Runner (`bin/agent_runner`)
A zero-Python C++ executable designed for maximum stability with **Claude Code**:
- **JSON-RPC Interface**: Communicates via `stdin/stdout` for seamless agentic integration.
- **Ultra-Low Latency**: Bypasses the entire Python runtime and FastAPI overhead.
- **Stability**: Immune to Python-level crashes or async event loop blockages during long reasoning tasks.

### 🛡️ Hardware-Aware Memory Safety
Specifically tuned for **M4 Pro** and high-spec Apple Silicon:
- **Auto-Calculated Limits**: Automatically detects system RAM and sets safe boundaries (e.g., 37.44GB on 48GB machines).
- **Emergency Abort Gates**: Proactively cancels or defers requests before the GPU hits hard memory limits.
- **Low-Latency SSE**: 3-second keep-alive heartbeats ensure Claude Code never times out during long prefills.
- **Persistent Logging**: All activity is automatically captured in `~/.cmlx/logs/server.log` with daily rotation, in addition to console output.

---

## ⚙️ Installation & Build

For the best performance on **M4 Pro** and other Apple Silicon Macs, use the unified build script. This creates a virtual environment and compiles the high-performance C++ core.

```bash
# 1. Clone & Build
git clone https://github.com/shantibsharma/cmlx.git
cd cmlx
./build.sh
```

---

## 🏃 Running cMLX

### Option 1: Standard API Server (OpenAI Compatible)
Starts the server with hardware-aware auto-scaling. Defaults to models in `~/.cmlx/models`.
```bash
./run.sh
```

**Custom Directory/Port:**
```bash
./run.sh /path/to/your/models 8080
```

### Option 2: Standalone Agent Runner (For Claude Code)
Ideal for high-stability agentic reasoning loops.
```bash
./bin/agent_runner ~/.cmlx/models/your-model-name
```

### Option 3: Performance Verification
Verify the raw speed of the C++ core:
```bash
python3 scratch/perf_test_native.py
```

---

## 🏎️ Hardware Optimization Guide

### 💎 MacBook M4 Pro (Optimized for 48GB+)
The flagship configuration. Moves everything to C++ and uses large SSD prefix caches.
```bash
# Server will auto-calculate optimal limits
python3 -m cmlx.server --model-dir ~/.cmlx/models
```

### 🛠️ Tuning Tips
1. **SSD Prefix Cache**: Automatically enabled for models like Gemma 4. Check for `paged SSD cache enabled` in logs for near-zero TTFT on repeats.
2. **Data Migration**: All local settings and models have moved from `~/.omlx` to `~/.cmlx`.
3. **Clean Logs**: Web interface polling noise (`/admin/api/stats`) is automatically filtered from your console.
4. **Claude Code**: Use the `--port 8000` flag to connect Claude Code via the OpenAI-compatible bridge.

### ⚡ MacBook M1/M2/M3 (Base / Pro - 16GB to 32GB)
Focuses on aggressive memory gates and SSD offloading to keep the system responsive.
```bash
cmlx serve \
  --model-dir ~/.cmlx/models \
  --max-process-memory 80% \
  --paged-ssd-cache-dir ~/.cmlx/cache \
  --paged-ssd-cache-quantize \
  --max-concurrent-requests 4
```

> [!IMPORTANT]
> Apple Silicon uses Unified Memory. Setting `--max-process-memory` higher than the **Metal Hard Limit** (usually 75-80% of RAM) will cause the OS to swap heavily or trigger the "Spin Lock" kernel panic. This fork automatically detects your **Metal Wired Limit** and clamps `auto` safely.

### Verify Native Runtime
Check the server logs for:
`✅ cmlx_fast_io extension loaded successfully. Native C++ Runtime is ACTIVE.`

---

## 🏗️ Architecture

| Component | Ported to C++? | Rationale |
|-----------|----------------|-----------|
| **Request Parser** | No | Python/FastAPI is efficient enough for I/O. |
| **KV Management** | **YES** | LRU and Hash lookups are too slow in Python loops. |
| **Memory Monitor** | **YES** | Python's `psutil` or `os` calls were too high-latency. |
| **Scheduler Queue**| **YES** | Atomic multi-producer priority queueing. |
| **KV Quantization**| **YES** (partially) | Core math is MLX-native, orchestration is Python. |
| **PagedAttention** | **YES** (vllm-metal) | Varlen Metal kernel compiled via nanobind + Rust. |

---

## 📊 Advanced CLI Flags

| Flag | Description |
|------|-------------|
| `--initial-cache-blocks` | Pre-allocate blocks at startup to reduce dynamic latency. |
| `--paged-ssd-cache-quantize` | Enable dynamic **INT8** KV quantization for SSD storage. |
| `--hot-cache-max-size` | Allocate a dedicated RAM buffer to "pin" KV blocks in memory. |
| `--fp8-kv-cache` | Use FP8 (E4M3) instead of INT8 for KV cache quantization. |
| `cmlx convert-fp8 <model>` | Convert FP16/BF16 model weights to FP8 for ~50% memory savings. |
| `--hf-endpoint` / `--ms-endpoint` | Custom HuggingFace or ModelScope (HF-mirror/ModelScope) endpoints. |
| `--log-level trace` | Enable full raw message tracing (useful for debugging agent loops). |
| `cmlx launch [tool]` | Launch integrated tools like `codex`, `opencode`, or `openclaw`. |

---

## 🗺️ Developer Roadmap
- [x] Phase 1: Native LRU & Memory Monitoring
- [x] Phase 2: Predictive Abort Gates
- [x] Phase 3: int8 KV Quantization
- [x] Phase 4: vLLM Metal Varlen PagedAttention (integrated via [vllm-metal](https://github.com/vllm-project/vllm-metal))
- [x] Phase 5: FP8 Weight Support (native loading, conversion, KV cache quantization)

---

## 📄 License
[Apache 2.0](LICENSE)
