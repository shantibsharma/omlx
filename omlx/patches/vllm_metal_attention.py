# SPDX-License-Identifier: Apache-2.0
"""Patch scaled_dot_product_attention to support varlen PagedAttention via vllm-metal.

When a PagedKVCache is detected, routes continuous batch attention to:
  - vllm_metal.metal.metal_unified_attention()
"""

import logging
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

_PATCHED = False

def apply_metal_attention_patch() -> bool:
    """Monkey-patch mlx-lm's scaled_dot_product_attention for vllm-metal PagedAttention."""
    global _PATCHED
    if _PATCHED:
        return False

    try:
        from mlx_lm.models import base as mlx_base
        from vllm_metal.metal import metal_unified_attention
    except ImportError as e:
        logger.warning(f"Could not load vllm-metal or mlx_lm: {e}")
        return False

    original_sdpa = mlx_base.scaled_dot_product_attention

    def patched_varlen_sdpa(
        queries,
        keys,
        values,
        cache,
        scale: float,
        mask: Optional[mx.array],
        sinks: Optional[mx.array] = None,
    ) -> mx.array:
        """Intercept the SDPA loop to execute the C++ vllm-metal varlen kernel."""
        # Unpack the cache if it matches our PagedCacheManager
        from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
        
        real_cache = cache
        if hasattr(cache, "_cache"):
            real_cache = cache._cache
            
        if isinstance(real_cache, PagedSSDCacheManager):
            # To interface with vllm_metal.metal_unified_attention
            # We must map the PagedSSDCacheManager block tracking layout into:
            # q, k_cache, v_cache, out, cu_seqlens_q, seqused_k, max_seqlen_q...
            
            # CHECK: Does the cache have the experimental vllm-metal pre-allocated tracking pools?
            if not hasattr(real_cache, 'k_pool') or not hasattr(real_cache, 'current_block_tables'):
                return original_sdpa(queries, keys, values, cache, scale, mask, sinks)
            
            # 1. Prepare an empty output array identically shaped to queries
            out = mx.zeros_like(queries)
            
            # 2. Extract block_tables and sequence boundaries from the scheduler via real_cache
            # (Note: Requires real_cache to track cu_seqlens_q during Step generation)
            block_tables = real_cache.current_block_tables
            cu_seqlens_q = real_cache.current_cu_seqlens
            seqused_k = real_cache.current_seqused
            max_seqlen_k = int(mx.max(seqused_k).item())
            max_seqlen_q = int(queries.shape[1])
            
            # 3. Fire the custom Metal Kernel directly 
            metal_unified_attention(
                q=queries,
                k=real_cache.k_pool,
                v=real_cache.v_pool,
                out=out,
                cu_seqlens_q=cu_seqlens_q,
                seqused_k=seqused_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=scale,
                causal=True,
                window_size=(-1, -1),
                block_table=block_tables,
                softcap=0.0
            )
            return out
        
        # Fallback to standard MLX attention for non-paged models
        return original_sdpa(queries, keys, values, cache, scale, mask, sinks)

    # Apply Patch directly mapping to MLX-LM
    mlx_base.scaled_dot_product_attention = patched_varlen_sdpa
    
    _PATCHED = True
    logger.info("Varlen Metal Attention patch applied dynamically")
    return True
