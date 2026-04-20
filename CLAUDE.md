# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

IMPORTANT - Monitor the conversation length. When the context usage exceeds roughly 65,000 tokens (approx. 33% of the 200k limit), pro-actively run the /compact command to summarize the previous conversation, ensuring important architectural decisions, file paths, and current goals are preserved. Do not wait for the automatic 95% compaction.


### Development Setup
- Install development dependencies: `pip install -e ".[dev]"`
- Install with MCP support: `pip install -e ".[mcp]"`

### Testing
- Run fast tests: `pytest -m "not slow"`
- Run a specific test file: `pytest tests/test_<module>.py -v`
- Run slow tests (requires models): `pytest -m slow`
- Test markers:
    - `@pytest.mark.slow`: Requires loading models.
    - `@pytest.mark.integration`: Requires a running server.

### macOS App Packaging
- Build full app bundle (requires `venvstacks`): `cd packaging && python build.py`
- Build without `venvstacks` (code changes only): `cd packaging && python build.py --skip-venv`
- Build DMG only: `cd packaging && python build.py --dmg-only`

### Running the Server
- Start server: `cmlx serve --model-dir <path_to_models>`

## Architecture

cMLX is an LLM inference server optimized for Apple Silicon, featuring continuous batching and a tiered KV cache system.

### Core Components
- **FastAPI Server**: Provides OpenAI and Anthropic compatible API endpoints.
- **EnginePool**: Manages multiple model engines (LLM, VLM, Embedding, Reranker) with LRU eviction and TTL.
- **Scheduler**: Manages request concurrency using `mlx-lm`'s `BatchGenerator`.
- **Tiered KV Cache**:
    - **Hot Tier (RAM)**: Block-based, prefix-sharing, and Copy-on-Write (CoW) cache.
    - **Cold Tier (SSD)**: Offloads cache blocks to SSD in `safetensors` format for persistence across restarts.
- **Admin Dashboard**: A web UI at `/admin` for model management, monitoring, and chat.

### Project Structure
- `cmlx/api/`: API models and adapters.
- `cmlx/cache/`: KV cache management (paged, prefix, SSD).
- `cmlx/engine/`: Inference engine implementations.
- `cmlx/mcp/`: Model Context Protocol integration.
- `cmlx/models/`: Model wrappers.
- `cmlx/server.py`: FastAPI server entry point.
- `cmlx/scheduler.py`: Request scheduling logic.
- `cmlx/engine_core.py`: Core async inference engine.
- `cmlx/paged_cache.py`: Block-based KV cache logic.
- `packaging/`: macOS menubar app implementation.
- `tests/`: Test suite.

## Related Working Directories
- `~/work/code/mlx`
- `~/work/code/mlx-lm`
- `~/work/code/mlx-framework.org`
- `~/work/code/mlx-examples`
- `~/work/code/mlx-data`
