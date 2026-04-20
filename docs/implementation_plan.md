# Optimization & Execution Plan: cMLX on M4 Pro 48GB

This document outlines the strategy for running Opus-equivalent (large/complex) and Haiku-equivalent (small/fast) models via `cmlx` to serve Claude Code on a MacBook Pro M4 Pro with 48 GB of Unified Memory.

## 1. Goal & Hardware Constraints
**Hardware:** M4 Pro with 48 GB Unified Memory.
**Available Metal RAM allowance:** macOS typically reserves ~4-6 GB of RAM strictly for the OS and kernel. The absolute safe ceiling for `mlx` inference is around **~40-42 GB** of active memory.

**Model Targets:**
- **Opus-Equivalent (Heavy Tasks):** Needs a ~70B parameter model.
  - *Recommendation:* `mlx-community/Meta-Llama-3.1-70B-Instruct-4bit` (~39.5 GB) or `mlx-community/Qwen2.5-72B-Instruct-4bit` (~41 GB).
- **Haiku-Equivalent (Fast/Simple Tasks):** Needs an ~8B parameter model.
  - *Recommendation:* `mlx-community/Meta-Llama-3.1-8B-Instruct-8bit` (~8.5 GB) or `mlx-community/Qwen2.5-7B-Instruct-8bit` (~7.7 GB) or 4-bit (~4.3 GB).

**The Clash:** 
Loading a 70B (40 GB) and an 8B (4-8 GB) simultaneously plus the KV cache will exceed 48 GB, triggering severe macOS swap thrashing. Because Claude Code switches rapidly between the "heavy" and "fast" agent prompts, dynamically juggling these in 48 GB is our primary problem.

---

## 2. Configuration & Model Selection Strategy

### Option A: The "Intelligent Swap" Strategy (70B + 8B)
Use the 70B (4-bit) as Opus and 8B (4-bit) as Haiku.
We must tightly enforce memory boundaries so `cmlx` unloads the idle model seamlessly when Claude Code switches agents, rather than letting the OS swap.
- **`max-process-memory`:** Set strictly to `42GB`. This triggers `ProcessMemoryEnforcer` in `engine_pool.py`. When the 70B model is invoked, the `engine_pool` will gracefully kick the 8B out, and vice-versa, saving you from complete system hangs.
- **Model Choice:** Use 4-bit for the 8B model as well (takes ~4.3 GB), meaning when the 70B model returns its memory, the 8B load overhead is blazing fast (<2 seconds off NVMe SSD).

### Option B: The "Simultaneous Hosting" Strategy (32B + 8B) 
If the <2 second reload penalty of Option A during Claude Code's agent-switching is unacceptable, we must shrink the "Opus" model so both persist in memory simultaneously.
- **Opus:** `mlx-community/Qwen2.5-32B-Instruct-4bit` (~18.5 GB)
- **Haiku:** `mlx-community/Llama-3.1-8B-Instruct-8bit` (~8.5 GB)
- **Calculation:** 18.5 + 8.5 = 27 GB. This leaves massive headroom (~15 GB) for deep KV caches, meaning context windows can go up to 32k safely without eviction.

---

## 3. Proposed Code Modifications

To better optimize `cmlx` specifically for the M4 Pro and these tighter constraints, I propose the following targeted code tweaks:

### [MODIFY] `cmlx/process_memory_enforcer.py`
Currently, the hard limit for inline prefill check is strictly:
`max(get_system_memory() - 4 * 1024**3, self._max_bytes)`
For a 48GB machine pushing heavy boundaries, we need to make KV cache eviction more aggressive *before* hitting swap. We will adjust the prefill threshold multiplier or add dynamic context-length limiting depending on available buffers.

### [MODIFY] `cmlx/cli.py`
Add explicit handling for the new M4 architecture constraints.
While issue #300 handling `mx.set_cache_limit(total_mem)` is present, we need to enforce optimal `--initial-cache-blocks` sizing for 48GB. If the user invokes `cmlx serve` we should conditionally define `auto` logic for `max_model_memory` uniquely for 48GB models, clamping it effectively at 85% instead of raw system polling limits. 

### [MODIFY] `cmlx/engine_pool.py`
Tune `kv_headroom`. Right now it defaults to `25%` of estimated model size (`int(entry.estimated_size * 0.25)`). On a 70B model (40GB), 25% is 10GB, pushing the check to 50GB, triggering eviction loop paradoxes. I will scale `kv_headroom` dynamically based on total system RAM instead of flat model multipliers.

## 4. Open Questions

1. **Which Strategy Do You Prefer?** 
   Do you want to run the massive 70B model (Option A) and let `cmlx` automatically swap models in and out of GPU RAM when Claude Code changes its focus? Or do you prefer the 32B model (Option B) for instantaneous 0-latency swapping?
2. **Persistence:** Do you want specific endpoints pinned via `--pin` that block eviction?

## 5. Verification Plan
- Make config default edits.
- Run `cmlx serve` locally passing synthetic boundaries.
- Inspect `EnginePool` metrics via `/status` endpoint to guarantee LRU unloading thresholds function perfectly mathematically.
