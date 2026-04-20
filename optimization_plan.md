# Optimization Plan: Native C++ Performance Enhancements

## Context
Following the `ARCHITECTURE_AND_OPTIMIZATION.md` document, we are implementing a series of performance optimizations to `cmlx` by moving critical management logic from Python into the existing C++ runtime. The primary goals are to prevent Metal OOM panics, reduce KV cache management latency, and minimize the Python-C++ boundary overhead during the generation loop.

The codebase already has a `src/` directory containing C++ core implementations (`cache_core.cpp`, `scheduler_core.cpp`, `cmlx_fast_io.cpp`) which are integrated into Python via `ctypes` in `cmlx/c_bindings.py`.

## Implementation Strategy

### Phase 1: Enhanced Memory Gates (High Priority)
*Objective: Prevent Metal OOM panics by implementing more granular memory accounting.*

- **Task 1.1: Granular Memory Accounting in `src/cache_core.cpp`**
  - **Why:** Current Python-side checks may be too slow or coarse-grained to catch rapid memory spikes.
  - **How to apply:** Implement a more precise memory usage tracker in C++ that accounts for both the allocated KV cache blocks and the overhead of the model weights. This will provide an atomic, low-latency flag for the Python scheduler.
  - **Files:** `src/cache_core.cpp`, `cmlx/c_bindings.py`, `cmlx/cache/paged_cache.py`.

- **Task 1.2: Predictive Limit Management**
  - **Why:** To proactively reject or defer requests before hitting hard OS/Metal limits.
  - **How to apply:** Introduce "soft" and "hard" memory thresholds in the C++ core. When the soft threshold is crossed, the C++ layer can signal the Python scheduler to stop admitting new requests.
  - **Files:** `src/scheduler_core.cpp`, `cmlx/scheduler.py`.

### Phase 2: KV Cache Efficiency (High Priority)
*Objective: Reduce latency in the generation loop and manage large-scale context efficiently.*

- **Task 2.1: Native O(1) LRU Management Optimization**
  - **Why:** Python-based LRU management for cache blocks introduces significant overhead in high-concurrency scenarios.
  - **How to apply:** Fully migrate the LRU logic from Python's `FreeKVCacheBlockQueue` to the `PagedCacheCore` in C++. Ensure all metadata updates (ref counting, eviction) happen natively.
  - **Files:** `src/cache_core.cpp`, `cmlx/cache/paged_cache.py`, `cmlx/c_bindings.py`.

- **Task 2.2: Dynamic KV Cache Quantization (Research/Implementation)**
  - **Why:** To save memory by migrating older tokens to lower precision.
  - **How to apply:** Implement a mechanism where cache blocks can be re-quantized (e.g., FP16 -> INT8/INT4). This will require new Metal kernels or integration with `mlx` quantization utilities.
  - **Files:** `src/cache_core.cpp`, `cmlx/cache/paged_cache.py`.

### Phase 3: Throughput & Latency (Medium Priority)
*Objective: Maximize request concurrency and minimize per-token latency.*

- **Task 3.1: Low-Overhead Scheduler State**
  - **Why:** Reducing the frequency and weight of Python $\leftrightarrow$ C++ boundary crossings.
  - **How to apply:** Move more of the request state (status, token counts, priority) into the `SchedulerCore` in C++. The Python `Scheduler` will then primarily act as an orchestrator, querying the C++ core for the next batch of requests.
  - **Files:** `src/scheduler_core.cpp`, `cmlx/scheduler.py`.

- **Task 3.2: Native Samplers & Logit Processors**
  - **Why:** Python-side sampling is a bottleneck in the generation loop.
  - **How to apply:** Port common samplers (Min-P, Top-P, Top-K) to C++.
  - **Files:** `src/` (new or existing), `cmlx/engine/`.

## Verification Plan

### 1. Unit Testing
- Run existing tests: `pytest -m "not slow"`
- Add new C++ unit tests for `cache_core` and `scheduler_core` using a native testing framework (e.g., GTest) or via `ctypes` bindings in `pytest`.

### 2. Integration Testing
- Run integration tests to ensure Python-C++ bindings are correct: `pytest -m integration`
- Verify that memory gates correctly trigger under simulated memory pressure.

### 3. End-to-End Performance Benchmarking
- Use `cmlx serve` to start a server.
- Run a script to benchmark throughput (tokens/sec) and latency (time to first token) with and without the new optimizations.
- Monitor GPU memory usage during high-concurrency requests to ensure no OOMs occur.

## Critical Files to Modify
- `src/cache_core.cpp`
- `src/scheduler_core.cpp`
- `cmlx/c_bindings.py`
- `cmlx/cache/paged_cache.py`
- `cmlx/scheduler.py`
