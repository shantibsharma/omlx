# oMLX C++ Migration Plan & Progress

## Status: Phase 2.1 Finalized (Hybrid Model Support)
**Date:** Sunday, April 19, 2026
**Objective:** Move the performance-critical generation loop and memory governance from Python to a native C++ implementation to eliminate GIL overhead and prevent Metal memory panics.

---

## 1. Architectural Strategy

### A. The "Opaque Engine" Pattern
We will implement a C++ `NativeEngine` that holds the MLX model and state. Python will interact with this engine via a slim C API (`extern "C"`), passing request metadata and receiving generated tokens.

### B. Memory Governance (C++ Level)
The C++ core will run a background thread monitoring `mx::get_active_memory()`. It will have the authority to:
1. **Pause Prefill:** If memory is > 85%.
2. **Preempt Requests:** If memory is > 95% (move requests to SSD Cold Tier).
3. **Synchronize:** Force `mx::synchronize()` and `mx::clear_cache()` during idle periods.

---

## 2. Component Migration Map

| Python Component | C++ Implementation | Status |
| :--- | :--- | :--- |
| `Scheduler.step()` | `NativeEngineImpl::step()` | ✅ Implemented (Batching) |
| `BatchGenerator` | `NativeEngineImpl::step()` | ✅ Implemented (State Machine) |
| `PagedSSDCacheManager` | `NativeSSDCache` | ✅ Refined (LRU Active) |
| `MemoryMonitor` | `SchedulerCore::monitor_loop` | ✅ Implemented |
| `PagedCacheManager` | `PagedCacheCore` | ✅ Refined (Prefix-aware LRU) |
| `Prefill` | `NativeEngineImpl` (Chunked) | ✅ Implemented (Phase 4) |
| `LlamaModel` | `LlamaModel` (C++) | ✅ Implemented (Phase 2) |
| `GDNPagedAttentionWrapper`| `MetalOps::dispatch_gdn_linear_attention` | ✅ Implemented (Phase 2.1) |

---

## 3. Implementation Log

### [Step 1] Define NativeEngine Interface
- Created `src/native_engine.h` to define the core engine class.
- Objective: Encapsulate `mlx-c` model execution and the C++ scheduler.

### [Step 2] Implementation of NativeEngine Prototype
- Created `src/native_engine.cpp`.
- Integrated `SchedulerCore` for memory governance.
- Added C-style bridge for Python FFI (`native_engine_create`, `native_engine_step`).

### [Step 3] Native SSD Cache & LRU Refinement
- Created `src/native_ssd_cache.h` and `src/native_ssd_cache.cpp`.
- Implemented O(1) LRU eviction policy for persistent SSD storage.
- Added modification-time based LRU reconstruction for persistence across restarts.
- Refined `PagedCacheCore` (In-Memory) to move prefix-hit blocks to MRU position, protecting common system prompts from eviction.

### [Step 4] Continuous Batching & Chunked Prefill (Phases 3 & 4)
- Extended `NativeRequest` with a state machine (`WAITING`, `PREFILLING`, `GENERATING`, `FINISHED`).
- Implemented **Chunked Prefill**: Large prompts are processed in small C++ steps to avoid memory spikes.
- Implemented **Continuous Batching**: New requests are scheduled automatically from the waiting queue when memory pressure is low.

### [Step 5] Native Model Execution (Phase 2 Finalized)
- Created `src/llama_model.h` and `src/llama_model.cpp`.
- Implemented **Llama Architecture** using optimized `mlx::core::fast` operations (`rms_norm`, `rope`, `scaled_dot_product_attention`).
- Integrated `LlamaModel` into `NativeEngineImpl`, replacing simulated tokens with actual tensor forward passes and greedy sampling.

### [Step 6] Hybrid Model Support (Phase 2.1)
- Extracted `gdn_linear_attention.metal` from `vllm-metal`.
- Added `dispatch_gdn_linear_attention` to `MetalOps` bridge.
- The engine now has the low-level primitives needed to process Qwen3.5's Gated DeltaNet state updates entirely in C++.

### [Step 7] Claude Code Integration & Standalone Runner
- Created `src/agent_runner.cpp` — a zero-Python CLI entry point for `omlx`.
- **Purpose:** Claude Code and other agents can launch `bin/agent_runner` and communicate via JSON-RPC over `stdin/stdout`.
- Provides maximum stability (no GIL, no Python runtime overhead) during agentic reasoning loops.

### [Step 8] Sampling Engine in Native C++
- Integrated Top-P / Temperature / Greedy sampling directly into `NativeEngineImpl`.
- Random generation utilizes `mx::random::categorical` on GPU memory.
- Results stream immediately without Python runtime overhead.

---

## 4. Recovery & Verification
- **Full Build Command:** `./bin/build_cpp_core.sh`
- **Output Artifacts:**
  - `omlx/omlx_fast_io*.so`: Python-compatible extension.
  - `bin/agent_runner`: Standalone C++ binary for agentic loops (Claude Code).
- **Verification Script:** `python3 scratch/verify_native_engine.py` (Confirmed logic consistency).
