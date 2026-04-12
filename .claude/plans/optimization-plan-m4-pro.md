---
name: optimization-plan-m4-pro
description: Plan to optimize omlx for MacBook M4 Pro (48GB) focusing on memory efficiency and speed.
type: project
---

# Optimization Plan for omlx on MacBook M4 Pro (48GB)

## Context
The goal is to maximize the performance and reliability of omlx on a high-spec MacBook M4 Pro with 48GB of unified memory. This involves preventing Out-Of-Memory (OOM) errors while maintaining high throughput and low latency. Current Python-based implementations of critical paths (KV cache management, scheduling, and potentially block allocation) are targets for optimization, including potential conversion to C/C++.

## Proposed Approach

### 1. Memory Management & OOM Prevention
- **Dynamic Memory Monitoring**: Implement a more robust system memory monitor that uses `psutil` or macOS-specific APIs to get real-time available RAM, rather than relying on hardcoded limits in the scheduler.
- **Tiered Cache Optimization**:
    - Optimize the "Cold Tier" (SSD) transition. Replace Python-based serialization with a faster binary format or direct memory mapping (`mmap`) if possible.
    - Refine the eviction policy (LRU/LFU) to be more proactive, preventing the "Hot Tier" from hitting hard limits.
- **Predictive Allocation**: Implement a "safety buffer" in the scheduler that accounts for the memory overhead of new requests and model activations, not just the KV cache.

### 2. Performance Optimization (C/C++ Integration)
- **Critical Path Conversion**: Identify and rewrite the following in C/C++ (via PyO3 or CFFI):
    - **Paged Cache Logic**: The core logic for block allocation, tracking, and bitmask management in `omlx/paged_cache.py`.
    - **Scheduler Loop**: The high-frequency scheduling logic in `omlx/scheduler.py`.
- **Fast IO for SSD Cache**: Use C++ for the SSD offloading/loading logic to minimize Python overhead during cache swaps.

### 3. Execution Strategy
- **Phase 1: Benchmarking & Profiling**: Establish a baseline using current Python implementation on the M4 Pro.
- **Phase 2: Python Refinement**: Implement the dynamic memory monitor and improved eviction logic in Python first to validate the logic.
- **Phase 3: C++ Extension Development**: Implement the C++ versions of the identified critical paths.
- **Phase 4: Integration & Verification**: Integrate extensions and run performance/stability tests.

## Critical Files
- `omlx/scheduler.py` (Scheduling & Memory Limits)
- `omlx/paged_cache.py` (Cache Logic)
- `omlx/cache/cache_manager.py` (Cache Orchestration)
- `omlx/engine_core.py` (Inference Loop)

## Verification Plan
- **Memory Stress Tests**: Run long-duration inference tasks with varying batch sizes to ensure no OOM occurs.
- **Throughput Benchmarking**: Compare tokens per second (TPS) before and after C++ integration.
- **Latency Analysis**: Measure time-to-first-token (TTFT) and inter-token latency.
- **Cache Swap Test**: Measure the overhead of transitioning blocks between RAM and SSD.
