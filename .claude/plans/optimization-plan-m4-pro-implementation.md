---
name: optimization-plan-m4-pro-implementation
description: Detailed implementation plan for optimizing cmlx for MacBook M4 Pro (48GB)
type: project
---

# Implementation Plan: cmlx Optimization for M4 Pro (48GB)

## Context
The goal is to maximize performance and reliability of cmlx on high-spec MacBook M4 Pro hardware. This requires preventing Out-Of-Memory (OOM) errors during heavy inference loads and increasing throughput by optimizing critical paths. The existing Python implementation of KV cache management and scheduling contains high-frequency operations that are prime candidates for C/C++ optimization.

## Implementation Strategy

### Phase 1: Benchmarking & Profiling (Baseline) [DONE]
Before making changes, we must establish a performance baseline on the target hardware.
- **Task 1.1 [DONE]**: Create a benchmarking script to measure:
    - Tokens Per Second (TPS) for varying batch sizes.
    - Time-To-First-Token (TTFT).
    - Inter-token latency.
- **Task 1.2 [DONE]**: Profile the current Python implementation using `cProfile` or `py-spy` to confirm that `paged_cache.py` and `scheduler.py` are indeed the primary bottlenecks.

### Phase 2: Robust Memory Management (Python)
Refine the existing Python logic to be more proactive before moving to C++.
- **Task 2.1: Dynamic Memory Monitoring**:
    - Replace or augment the current `MemoryMonitor` to use more precise macOS-specific memory metrics.
    - Implement a "safety buffer" in `cmlx/scheduler.py` that accounts for model activation memory, not just the KV cache.
- **Task 2.2: Improved Eviction Logic**:
    - Refine the `TieredCacheManager` eviction policy to be more proactive when approaching the 48GB limit.

### Phase 3: Critical Path Optimization (C/C++ via PyO3)
Convert the most intensive Python logic to C++ to reduce interpreter overhead.
- **Task 3.1: C++ Paged Cache Core**:
    - Rewrite the `PagedCacheManager` logic in C++.
    - Focus on: `CacheBlock` metadata management, `FreeKVCacheBlockQueue` (doubly linked list), and `BlockTable` operations.
    - Use `PyO3` to expose these to Python.
- **Task 3.2: C++ Scheduler Loop**:
    - Move high-frequency scheduling decisions in `cmlx/scheduler.py` to a C++ extension.
- **Task 3.3: Fast I/O for SSD Cache**:
    - Optimize the transition between RAM and SSD in `cmlx/cache/paged_ssd_cache.py` using C++ and `mmap` for faster block transfers.

### Phase 4: Integration & Verification
- **Task 4.1**: Integrate the new C++ extensions into the `cmlx` package.
- **Task 4.2**: Run the stress tests developed in Phase 1 to ensure no OOMs occur under maximum load.
- **Task 4.3**: Compare TPS and latency against the Phase 1 baseline to quantify improvements.

## Critical Files to Modify
- `cmlx/scheduler.py` (Scheduling logic and memory guards)
- `cmlx/cache/paged_cache.py` (Core block management)
- `cmlx/cache/tiered_manager.py` (Cache orchestration)
- `cmlx/cache/paged_ssd_cache.py` (SSD offloading)
- `cmlx/cache/paged_ssd_cache.py` (I/O logic)
- *New files*: `cmlx/src/cache_core.cpp`, `cmlx/src/scheduler_core.cpp` (C++ implementations)

## Verification Plan
- **Memory Stability**: Run continuous inference for 1 hour on a 48GB machine with high batch sizes; check for memory growth or OOM.
- **Performance Delta**: Verify at least a X% improvement in TPS and Y% reduction in TTFT compared to baseline.
- **SSD Swap Latency**: Measure the latency overhead of a block being moved to/from SSD.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
