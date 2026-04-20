# cMLX Architecture & Functionality Documentation

## 1. Overview
cMLX is a high-performance LLM inference server specifically optimized for Apple Silicon (M1/M2/M3/M4 series). It is an optimized fork of the original `jundot/omlx` project, engineered to handle high-concurrency workloads, large-scale models (70B+ parameters), and multi-modal (VLM) tasks with improved stability and throughput.

The primary goal of cMLX is to maximize the utility of Apple Silicon's Unified Memory architecture by introducing native C++ extensions for performance-critical tasks and advanced memory management strategies to prevent common "Metal Memory Panic" issues.

---

## 2. Core Architecture

The cMLX architecture is built around a coordinated set of components that manage the lifecycle of a request from API reception to token generation.

### 2.1 Component Diagram (High-Level)
`Client Request` $\rightarrow$ `FastAPI Server` $\rightarrow$ `EnginePool` $\rightarrow$ `EngineCore` $\rightarrow$ `Scheduler` $\rightarrow$ `MLX Model`

### 2.2 Key Components

#### **A. FastAPI Server (`cmlx/server.py`)**
The entry point for all external communications. It provides:
- **OpenAI/Anthropic Compatibility**: API endpoints that mirror standard LLM providers.
- **Admin Dashboard**: A web UI at `/admin` for real-time monitoring of model usage, engine status, and cache observability.
- **Streaming Support**: Robust implementation of Server-Sent Events (SSE) with specialized handling for agentic loops (e.g., Claude Code) to prevent connection drops.

#### **B. EnginePool (`cmlx/model_registry.py` & `cmlx/engine/`)**
Manages the lifecycle of multiple model engines.
- **Multi-Model Support**: Can simultaneously host LLMs, VLMs (Vision Language Models), Embedding models, and Rerankers.
- **LRU Eviction**: Automatically manages model residency in memory, evicting least-recently-used models when memory pressure is high.
- **Lifecycle Management**: Handles loading, warming up, and unloading models from Metal Unified Memory.

#### **C. EngineCore (`cmlx/engine_core.py`)**
The heart of the inference process, implementing the continuous batching loop.
- **Continuous Batching**: Unlike traditional batching, it allows new requests to join the current batch as soon as others finish, significantly increasing throughput.
- **Single-Threaded MLX Execution**: Uses a dedicated `ThreadPoolExecutor` to serialize all MLX GPU operations. This is a critical design choice to prevent Metal command buffer races that cause system-wide segfaults.
- **Async/Sync Interfaces**: Provides high-performance `generate_batch_sync` for throughput-oriented tasks and `stream_outputs` for low-latency interactive tasks.

#### **D. Scheduler (`cmlx/scheduler.py`)**
Coordinates the execution of batched requests.
- **Request Management**: Tracks active requests, their status, and their priority.
- **Batching Logic**: Interacts with `mlx-lm`'s `BatchGenerator` to perform the actual step-by-step token generation.
- **Speculative Prefill**: Supports advanced prefill optimizations to reduce time-to-first-token (TTFT).

#### **E. Tiered KV Cache System (`cmlx/cache/`)**
The most distinctive feature of cMLX, designed to handle massive context windows and many concurrent users.
- **Hot Tier (RAM)**: A block-based, prefix-sharing cache stored in fast Unified Memory.
- **Cold Tier (SSD)**: A paged SSD cache that offloads KV blocks to disk in `safetensors` format. This allows for "persistence across restarts" and massive context offloading.
- **Hybrid/Tiered Management**:
    - **Memory Monitoring**: Continuously tracks Metal memory pressure.
    - **Predictive Eviction**: When memory usage hits high thresholds (e.g., 90%), the `TieredCacheManager` automatically moves Least-Recently-Used (LRU) blocks from RAM to SSD (Cold Tier) to prevent a system panic.
    - **Emergency Abort Gates**: If memory pressure is critical, the system proactively cancels or defers new requests to save the process.

---

## 3. Key Functionalities

### 3.1 High-Performance Inference
- **Native C++ Acceleration (`cmlx_fast_io`)**: Offloads intensive operations like LRU management, memory monitoring, and fast I/O to a compiled C++ layer, bypassing Python overhead.
- **Universal Dynamic Quantization (oQ)**: Implements high-precision quantization (3-bit/4-bit) that maintains accuracy while significantly reducing the memory and bandwidth footprint of large models.
- **Multi-Modal Support**: Specialized engines for VLMs that handle image/video tokenization and vision-text embedding integration.

### 3.2 Advanced Memory Safety
- **Predictive Memory Gates**: Uses configured hard and soft limits (e.g., 80% and 90% of total RAM) to gracefully shed load or evict cache rather than triggering a macOS kernel panic.
- **Atomic Memory Tracking**: Precise accounting of model activations plus KV cache overhead to ensure the system stays within the Metal Wired Limit.

### 3.3 Agentic & Developer Tooling
- **Claude Code Optimization**: Specialized handling of request tracing and keep-alives to support advanced AI coding agents.
- **Integrated Launchers**: One-click support for specialized tools via `cmlx launch`.

---

## 4. Summary Table: Architectural Decisions

| Feature | Implementation | Rationale |
| :--- | :--- | :--- |
| **Concurrency** | Continuous Batching | Maximize throughput by minimizing idle GPU time. |
| **Stability** | Single-worker MLX Executor | Prevent Metal command buffer races and segfaults. |
| **Scale** | Tiered KV Cache (RAM + SSD) | Support 70B+ models and massive contexts on consumer Mac hardware. |
| **Performance** | C++ Native Extensions | Eliminate Python interpreter bottlenecks for O(1) cache management. |
| **Reliability** | Predictive Memory Gates | Prevent system-wide "Metal Memory Panics" during high load. |
