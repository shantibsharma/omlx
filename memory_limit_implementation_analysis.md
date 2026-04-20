# Memory Limit Configuration Implementation Plan

## Goal
Implement a feature in the Admin Dashboard UI to allow users to specify their available memory and the amount of memory they want to allocate for cMLX. Update the engine and scheduler to respect these limits to prevent OOM crashes.

## Current State
- The system currently manages memory through `process_memory_enforcer.py` and `scheduler.py`.
- There is an admin dashboard at `/admin`.
- Settings are stored in `settings.json`.
- `cmlx/settings.py` has `MemorySettings` with `max_process_memory`.

## Findings from Research
- **Admin UI**: Located in `cmlx/admin/`. Templates in `cmlx/admin/templates/`, routes in `cmlx/admin/routes.py`.
- **Memory Management**: `cmlx/process_memory_enforcer.py` and `cmlx/scheduler.py` are key. `cmlx/settings.py` uses `MemorySettings`.
- **Configuration**: Settings are persisted in a JSON file. `GET /api/global-settings` and `POST /api/global-settings` exist.

## Implementation Plan

### Stage 1: Backend & Configuration (Current Focus)
- Verify how `max_process_memory` is used in `process_memory_enforcer.py` and `scheduler.py`.
- Ensure `GlobalSettings` and the `POST /api/global-settings` endpoint correctly handle user-provided memory limits.
- If we want to allow users to specify "available memory", we might need to add a field for it, or just let them set the limit they want to use. The user's request is: "accept how much memory User have and want to use". This implies two things:
    1. Total available memory (for context/calculation).
    2. Amount they want to use (the actual limit).
- I will add `available_system_memory` to `MemorySettings` to store what the user *thinks* they have, and use `max_process_memory` as the actual enforcement limit.

### Stage 2: Frontend Implementation
- Modify `cmlx/admin/templates/dashboard.html` (or relevant part) to add input fields for:
    - System Available Memory (e.g., in GB).
    - Max Process Memory (the limit to enforce).
- Update the JavaScript in the Admin UI to send these values via `POST /api/global-settings`.

### Stage 3: Verification
- Run tests in `tests/test_process_memory_enforcer.py`.
- Verify that changing settings via the UI updates the settings file and is respected by the enforcer.

## Notes
- The user wants to "accept how much memory User have and want to use".
- I'll interpret "User have" as a reference value and "want to use" as the enforcement limit.
