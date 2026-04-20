# SPDX-License-Identifier: Apache-2.0
"""
cmlx: LLM inference server, optimized for your Mac

This package provides native Apple Silicon GPU acceleration using
Apple's MLX framework and mlx-lm for LLMs.

Features:
- Continuous batching via vLLM-style scheduler
- OpenAI-compatible API server
- Paged KV cache with prefix sharing
- Tiered cache (GPU + paged SSD offloading)
"""

from cmlx._version import __version__

# Continuous batching engine (core functionality, no torch required)
from cmlx.request import Request, RequestOutput, RequestStatus, SamplingParams
from cmlx.scheduler import Scheduler, SchedulerConfig, SchedulerOutput
from cmlx.engine_core import EngineCore, AsyncEngineCore, EngineConfig
from cmlx.cache.prefix_cache import BlockAwarePrefixCache
from cmlx.cache.paged_cache import PagedCacheManager, CacheBlock, BlockTable
from cmlx.cache.stats import PrefixCacheStats, PagedCacheStats
from cmlx.model_registry import get_registry, ModelOwnershipError

# Backward compatibility alias
CacheStats = PagedCacheStats

__all__ = [
    # Request management
    "Request",
    "RequestOutput",
    "RequestStatus",
    "SamplingParams",
    # Scheduler
    "Scheduler",
    "SchedulerConfig",
    "SchedulerOutput",
    # Engine
    "EngineCore",
    "AsyncEngineCore",
    "EngineConfig",
    # Model registry
    "get_registry",
    "ModelOwnershipError",
    # Prefix cache (paged SSD-only)
    "BlockAwarePrefixCache",
    # Paged cache (memory efficiency)
    "PagedCacheManager",
    "CacheBlock",
    "BlockTable",
    "PagedCacheStats",
    "CacheStats",  # Backward compatibility alias
    # Version
    "__version__",
]
