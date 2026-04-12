# Phase 1: Benchmarking & Profiling Results (Baseline)

## Hardware Environment
- **Device**: Apple M4 Pro
- **RAM**: 48GB
- **MLX Device Info**: `{'resource_limit': 499000, 'max_buffer_length': 30150672384, 'architecture': 'applegpu_g16s', 'memory_size': 51539607552, 'max_recommended_working_set_size': 40200896512, 'device_name': 'Apple M4 Pro'}`

## Benchmark Results (Qwen3-Coder-30B-4bit)
| Metric | Prompt 1024 | Prompt 4096 |
| :--- | :--- | :--- |
| **TTFT (ms)** | 1299.4 | 5080.1 |
| **Gen TPS** | 79.6 | 66.1 |
| **PP TPS** | 788.1 | 806.3 |

### Batch Inference (Prompt 1024)
| Batch Size | Avg TTFT (ms) | TG TPS (Aggregate) | PP TPS (Aggregate) |
| :--- | :--- | :--- | :--- |
| 1 | 1299.4 | 79.6 | 788.1 |
| 2 | 2468.8 | 109.4 | 807.4 |
| 4 | 4868.8 | 132.6 | 801.8 |

## Profiling Analysis
The `cProfile` results for a 4096 token prefill + 64 token generation:

1. **Prefill Bottleneck**: `_do_external_prefill` dominates the start of execution (4.9s out of 6s). This includes both MLX computation and the overhead of managing the paged cache blocks in Python.
2. **Step Overhead**: `scheduler.py:step` is called for every block/step. While individual `tottime` is low, the cumulative orchestration involves significant Python activity.
3. **Cache Management**: `_schedule_waiting` and block table updates in Python occur at high frequency during multi-request scenarios.

## Files
- `benchmark_results_m4_pro.json`: Raw benchmark metrics.
- `profile_results_m4_pro.txt`: Full `cProfile` cumulative stats.
