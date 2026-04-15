# SPDX-License-Identifier: Apache-2.0
"""
Centralized memory management and synchronization for oMLX.

Provides safe wrappers for MLX memory operations to prevent race conditions
between Metal buffer reclamation and IOKit asynchronous callbacks,
avoiding 'completeMemory() prepare count underflow' kernel panics.
"""

import logging
import mlx.core as mx
from mlx_lm.generate import generation_stream

logger = logging.getLogger(__name__)

def sync_and_clear_cache():
    """Synchronize ALL in-flight GPU work before clearing the Metal buffer cache.

    Without sufficient synchronization and delay, mx.clear_cache() can release 
    Metal buffers that are still referenced by in-flight command buffers 
    or whose 'completeMemory' callbacks are still pending in IOKit. 
    This is especially critical on M4 Pro hardware.

    This helper synchronizes both the generation stream (used by mlx-lm)
    and the default stream, then calls clear_cache.
    """
    try:
        # 1. Synchronize the specific generation stream
        mx.synchronize(generation_stream)
        
        # 2. Synchronize the default stream (used for vision encoding, fast I/O, etc.)
        mx.synchronize()
        
        # 3. Clear the cache
        mx.clear_cache()
        
    except Exception as e:
        logger.warning(f"Failed to synchronize and clear cache: {e}")
