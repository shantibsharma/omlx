<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/icon-rounded-dark.svg" width="140">
    <source media="(prefers-color-scheme: light)" srcset="docs/images/icon-rounded-light.svg" width="140">
    <img alt="oMLX" src="docs/images/icon-rounded-light.svg" width="140">
  </picture>
</p>

<h1 align="center">oMLX (Optimized Fork)</h1>
<p align="center"><b>Next-Generation LLM Inference for Apple Silicon</b><br>Native C++ Extensions · Predictive Memory Safety · Dynamic KV Quantization</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-0.5.3-blue" alt="Version">
  <img src="https://img.shields.io/badge/optimized-M4%20Pro-orange" alt="M4 Pro Optimized">
  <img src="https://img.shields.io/badge/runtime-C++%20Native-red" alt="C++ Native">
</p>

---

## 📖 Overview

This repository is a high-performance fork of [jundot/omlx](https://github.com/jundot/omlx), specifically engineered to maximize throughput and stability on high-spec Apple Silicon (MacBook M4 Pro). 

While the original `omlx` brought continuous batching and tiered caching to MLX, this fork introduces a **C++ Native Runtime** and **Predictive Memory Management** to eliminate the Metal kernel panics and Python interpreter bottlenecks associated with high-concurrency LLM/VLM inference.

---

## 🚀 Key Enhancements (The "Added" Features)

### 1. ⚡ Native C++ Accelerator (`omlx_fast_io.so`)
We migrated the most frequency-intensive logic from Python to a high-performance C++ core:
- **O(1) LRU Management**: Near-zero overhead for managing thousands of KV cache blocks.
- **Native Memory Monitoring**: Sub-millisecond tracking of Metal active/cache memory and system pressure.
- **Parallel Model Warmup**: Multi-threaded model loading directly into Metal Unified Memory.
- **Fast I/O Bridge**: Zero-copy `mmap` transfers for SSD-to-RAM cache restoration.

### 2. 🛡️ Predictive Memory Safety & Gates
To solve the "Metal Memory Panic" problem on M4 Pro:
- **Emergency Abort Gates**: Proactively cancels or defers requests before the GPU hits hard memory limits.
- **Hard/Soft Limits**: Configurable thresholds (e.g., Soft at 80%, Hard at 90%) to gracefully shed load.
- **Atomic Memory Tracking**: Precise accounting of model activation memory + KV cache overhead.

### 3. 🧠 oQ: Universal Dynamic Quantization
- **Data-Driven Mixed-Precision**: Uses real-world calibration to allocate bits where they matter most, achieving 80%+ accuracy in 3-bit modes (surpassing standard rounding).
- **oQ+ (GPTQ Enhanced)**: Optimized Hessian-based error compensation for rounding decisions, including a **15x faster batched MoE GPTQ** algorithm.
- **Universal Format**: Produces standard `mlx-lm` compatible models that work in any MLX-compatible environment.
- **Streaming Path**: Processes massive models (70B+) via `mmap` without requiring full RAM instantiation.

### 4. 🧩 Advanced Cache & Architecture
- **Rotating KV Cache**: Full support for sliding-window models (e.g., Qwen, Mistral).
- **BatchRotatingKVCache**: Optimized multi-request handling for rotating caches.
- **CacheList & ArraysCache**: Native support for **DeepSeek-V3 (MLA)**, Mamba, and hybrid SSM architectures.
- **Fix for mlx-lm**: `SizedArraysCache` wrapper fixes critical size-reporting bugs in the upstream framework.

### 5. 🛠️ Tool & Agent Integrations
- **One-Click Launch**: Integrated support for agents like **Codex**, **OpenCode**, and **OpenClaw** via `omlx launch`.
- **Claude Code Optimized**: Explicit handling of raw request tracing and SSE keep-alives to prevent connection drops with [Claude Code](https://claude.ai/code).

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- macOS 15.0+ (Sequoia)
- Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools (`xcode-select --install`)

### 2. Automated Installation (Recommended)
This script creates a virtual environment, installs dependencies, and builds the native C++ extensions.

```bash
# 1. Clone & Setup
git clone https://github.com/jundot/omlx.git
cd omlx
./scripts/setup_omlx.sh

# 2. Activate Environment
source .omlxvnv/bin/activate

# 3. Launch Server
omlx serve --model-dir ~/.omlx/models --max-process-memory 80%
```

After starting, visit the **Admin Dashboard** at [http://localhost:8000/admin](http://localhost:8000/admin).

---

## ⚔️ Comparison vs. Original Fork (`jundot/omlx`)

While the base `jundot/omlx` provides an excellent foundation for multi-model serving, this **Optimized Fork** introduces critical features to handle high-concurrency and large-scale (70B+) models reliably:

| Feature | Original oMLX | **Optimized Fork** | Impact |
|---------|---------------|-------------------|--------|
| **C++ Native Core** | No | **YES** (`omlx_fast_io`) | O(1) LRU, sub-ms memory tracking. |
| **Emergency Abort Gates** | No | **YES** | Thwart Metal kernel panics under load. |
| **KV Quantization** | No | **YES** (INT8) | 50% less SSD bandwidth / storage. |
| **oQ Quantization** | No | **YES** (oQ/oQ+) | High-precision 3-bit / 4-bit weights. |
| **Hot Cache Tier** | No | **YES** (`--hot-cache-max-size`) | Drastically reduces RAM-to-SSD swap. |
| **MLA / DeepSeek-V3** | Limited | **Full** | Native support for advanced architectures. |

---

## 🏎️ Hardware Optimization Guide

### 💎 MacBook M4 Pro (Optimized for 48GB+)
The flagship configuration. Moves critical scheduling to C++ and uses large Hot Cache tiers.
```bash
omlx serve \
  --model-dir ~/.omlx/models \
  --max-process-memory 42GB \
  --hot-cache-max-size 12GB \
  --initial-cache-blocks 1024 \
  --paged-ssd-cache-quantize
```

### ⚡ MacBook M1/M2/M3 (Base / Pro - 16GB to 32GB)
Focuses on aggressive memory gates and SSD offloading to keep the system responsive.
```bash
omlx serve \
  --model-dir ~/.omlx/models \
  --max-process-memory 80% \
  --paged-ssd-cache-dir ~/.omlx/cache \
  --paged-ssd-cache-quantize \
  --max-concurrent-requests 4
```

> [!IMPORTANT]
> Apple Silicon uses Unified Memory. Setting `--max-process-memory` higher than the **Metal Hard Limit** (usually 75-80% of RAM) will cause the OS to swap heavily or trigger the "Spin Lock" kernel panic. This fork automatically detects your **Metal Wired Limit** and clamps `auto` safely.

### Verify Native Runtime
Check the server logs for:
`✅ omlx_fast_io extension loaded successfully. Native C++ Runtime is ACTIVE.`

---

## 🏗️ Architecture

| Component | Ported to C++? | Rationale |
|-----------|----------------|-----------|
| **Request Parser** | No | Python/FastAPI is efficient enough for I/O. |
| **KV Management** | **YES** | LRU and Hash lookups are too slow in Python loops. |
| **Memory Monitor** | **YES** | Python's `psutil` or `os` calls were too high-latency. |
| **Scheduler Queue**| **YES** | Atomic multi-producer priority queueing. |
| **KV Quantization**| **YES** (partially) | Core math is MLX-native, orchestration is Python. |

---

## 📊 Advanced CLI Flags

| Flag | Description |
|------|-------------|
| `--initial-cache-blocks` | Pre-allocate blocks at startup to reduce dynamic latency. |
| `--paged-ssd-cache-quantize` | Enable dynamic **INT8** KV quantization for SSD storage. |
| `--hot-cache-max-size` | Allocate a dedicated RAM buffer to "pin" KV blocks in memory. |
| `--hf-endpoint` / `--ms-endpoint` | Custom HuggingFace or ModelScope (HF-mirror/ModelScope) endpoints. |
| `--log-level trace` | Enable full raw message tracing (useful for debugging agent loops). |
| `omlx launch [tool]` | Launch integrated tools like `codex`, `opencode`, or `openclaw`. |

---

## 🗺️ Developer Roadmap
- [x] Phase 1: Native LRU & Memory Monitoring
- [x] Phase 2: Predictive Abort Gates
- [x] Phase 3: int8 KV Quantization
- [ ] Phase 4: Custom Metal Kernels for Paged Attention
- [ ] Phase 5: FP8 Weight Support

---

## 📄 License
[Apache 2.0](LICENSE)
