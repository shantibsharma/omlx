# cMLX Architecture and Optimization Plan

## 1. System Architecture Overview

`cmlx` is a high-performance, multi-model inference server optimized for Apple Silicon. It leverages the MLX framework and implements several advanced techniques to maximize throughput and minimize latency, such as continuous batching, a tiered KV cache system, and agentic model swapping.

### Core Components

- **API Layer (FastAPI)**: Handles incoming HTTP requests (OpenAI and Anthropic compatible), performs validation, and converts external request formats into a unified internal format.
- **Management Layer (EnginePool & ServerState)**: Manages the lifecycle of multiple model engines, handles memory limits via LRU eviction, and coordinates model loading/unloading.
- **Scheduling & Execution Layer (EngineCore & Scheduler)**: Implements continuous batching. The `Scheduler` manages request queues and coordinates with the `BatchGenerator` (from `mlx-lm`) to execute generation steps.
- **Inference Engine Layer (Engines)**: Specialized implementations for different model types (LLM, VLM, Embedding, Reranker, Audio).
- **Memory & Cache Layer (Tiered KV Cache)**: Provides efficient KV cache management, including prefix sharing and tiered storage (GPU memory $\leftrightarrow$ SSD) to support large context windows and multi-model serving.

### Component Details

- **FastAPI Server (`cmlx/server.py`)**: The entry point for all external interactions. It uses adapters to translate external JSON schemas into `InternalRequest` objects.
- **EnginePool (`cmlx/engine_pool.py`)**: The central authority for model management. It maintains a registry of all discovered models and manages an LRU (Least Recently Used) eviction policy to ensure memory usage stays within limits. It also implements **Eager Swapping** to overlap model unloading with SSD pre-warming.
- **Scheduler (`cmlx/scheduler.py`)**: The heart of continuous batching. It manages `waiting` and `running` request queues and interfaces with the `BatchGenerator` to perform generation steps. It coordinates the `TieredCacheManager` and `BlockAwarePrefixCache`.
- **Tiered KV Cache (`cmlx/cache/`)**:
    - **Hot Cache (`PagedCacheManager`)**: Stores KV cache blocks in GPU memory for immediate access.
    - **Cold Cache (`PagedSSDCacheManager`)**: Offloads cache blocks to the SSD when GPU memory is under pressure.
    - **`TieredCacheManager`**: Coordinates the movement of blocks between hot (GPU) and cold (SSD) tiers.
    - **`BlockAwarePrefixCache`**: Enables prefix sharing to avoid redundant computation.

### Data Flow: API Request to Inference

1. **Ingress**: A client sends a POST request to `/v1/chat/completions`.
2. **Adaptation**: The `FastAPI` server receives the request and converts it into an `InternalRequest`.
3. **Engine Acquisition**: The server calls `EnginePool.get_engine(model_id)`. If necessary, the pool performs LRU eviction or eager swapping.
4. **Request Submission**: The server calls `BatchedEngine.chat()`, submitting it to the `EngineCore`.
5. **Scheduling**: The `EngineCore` passes the request to the `Scheduler`, which places it in the `waiting` queue.
6. **Continuous Batching**: The `EngineCore` loop calls `Scheduler.step()`, which moves requests from `waiting` to `running` and bundles them into a batch for the `BatchGenerator`.
7. **Output Collection**: As tokens are generated, `EngineCore` pushes outputs to a `RequestOutputCollector`.
8. **Egress**: The server consumes the output from the collector and sends it back to the client.

---

## 2. Optimization Plan

Based on research into the core MLX ecosystem (`mlx`, `mlx-lm`, and `mlx-c`), the following optimizations are proposed for `cmlx`.

### Phase 1: Memory & Stability (High Priority)
*Objective: Prevent Metal OOM panics and optimize memory footprint.*

- [x] **Enhanced Memory Gates**: Implement more granular C++-level memory accounting in `src/cache_core.cpp` (inspired by `mlx-c` tracking) to proactively reject or defer requests before Metal/system-level OOM occurs.
- [x] **Predictive Limit Management**: Utilize the pattern of setting "hard" and "soft" limits within the C++ scheduler to proactively manage memory pressure.
- [x] **Explicit GPU Synchronization**: Ensure rigorous `mx::synchronize()` calls are used during critical model swaps and cache clearing to prevent IOKit memory callback underflows.

### Phase 2: KV Cache Efficiency (High Priority)
*Objective: Reduce latency in the generation loop and manage large-scale context efficiently.*

- [x] **Native O(1) LRU Management**: Implement the LRU management for cache blocks in C++ using doubly-linked lists to achieve near-instantaneous eviction candidate selection.
- [ ] **Dynamic KV Cache Quantization**: Implement a transition mechanism where KV cache starts in FP16 and automatically migrates to 4-bit or 8-bit quantized formats after a certain token threshold (similar to `mlx-lm` patterns).
- [ ] **Custom Metal Kernels for Paged Attention**: Implement specialized Metal kernels for `gather` and `scatter` operations related to KV cache block management in the `cmlx` C++ core to bypass the overhead of generic tensor operations.
- [x] **Cryptographically Secure Prefix Hashing**: Upgrade prefix sharing to use 256-bit SHA256 hashing to prevent collisions in high-concurrency, multi-tenant environments.

### Phase 3: Throughput & Latency (Medium Priority)
*Objective: Maximize request concurrency and minimize per-token latency.*

- [x] **Low-Overhead Scheduler State**: Use the "opaque pointer" pattern from `mlx-c` to manage complex scheduling metadata and request states in `src/scheduler_core.cpp`, minimizing Python $\leftrightarrow$ C++ boundary latency.
- [x] **Lock-Free Command Submission**: Implement a multi-producer, single-consumer (MPSC) command queue in C++ to allow high-concurrency request submission to the engine with minimal synchronization overhead.
- [ ] **Speculative Decoding Optimization**: Optimize the "Rewind" mechanism by implementing native C++ logic for cache trimming and state rollback during speculative rejection.
- [ ] **Native Samplers & Logit Processors**: Port common samplers (Min-P, Top-P, Top-K) and logits processors to the native C++ runtime to eliminate Python-to-C++ boundary overhead in the generation loop.
- [x] **Chunked Prefilling**: Implement chunked prefilling to process large prompts in smaller steps, reducing massive memory spikes during the initial context processing stage.

### Phase 4: Advanced Features (Future Research)
- [ ] **Parallel Prefill Prefetching**: Use high-speed I/O to stream prefix blocks into the cache in parallel with the initial computation.
- [ ] **Advanced Multi-Stream Isolation**: Utilize `mx.new_stream()` more extensively to isolate generation tasks from background management tasks for smoother performance.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
