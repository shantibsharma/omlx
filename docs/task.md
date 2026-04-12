# oMLX 48GB M4 Pro Optimization Tasks

- `[x]` Update `kv_headroom` calculation in `engine_pool.py`
  - Ensure 25% scaler is capped to a flat bound (e.g. 4-6GB) to prevent large models from triggering paradox eviction.
- `[x]` Adjust `process_memory_enforcer.py` prefill boundaries
  - Limit the prefill safety bounds for 48GB to maximize inference capability.
- `[x]` Tune `cli.py` defaults
  - Apply custom parsing for M4 environments to ensure `initial-cache-blocks` and `max_process_memory` play well with tighter constraints.
- `[x]` Verify changes locally using dry-run tests and memory math checks.
