# SPDX-License-Identifier: Apache-2.0
"""
Dynamic KV Cache Quantization for oMLX.

Enables memory-efficient and I/O-efficient storage of KV cache blocks
by quantizing fp16/bf16 tensors to int8 before SSD persistence.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None

logger = logging.getLogger(__name__)

def quantize_kv_block(
    cache_data: List[Tuple[Any, Any]],
    bits: int = 8,
) -> List[Tuple[Any, Any]]:
    """
    Quantize fp16/bf16 KV cache tensors to int8 for storage.

    Args:
        cache_data: List of (keys, values) pairs for each layer.
        bits: Quantization bits (default 8 for safe compression).

    Returns:
        New cache_data list with quantized tensors and scales.
        Standard KVCache (k, v) becomes (k_quant, v_quant, k_scales, v_scales, k_biases, v_biases).
    """
    if not HAS_MLX:
        return cache_data

    quantized_data = []
    for layer in cache_data:
        if not isinstance(layer, tuple) or len(layer) != 2:
            # Skip non-standard layers (CacheList, TurboQuant)
            quantized_data.append(layer)
            continue

        keys, values = layer
        if not hasattr(keys, "dtype") or keys.dtype not in (mx.float16, mx.bfloat16):
            quantized_data.append(layer)
            continue

        # Quantize keys
        qk, sk, bk = mx.quantize(keys, group_size=64, bits=bits, mode="affine")
        # Quantize values
        qv, sv, bv = mx.quantize(values, group_size=64, bits=bits, mode="affine")

        # Package as extended tuple
        quantized_data.append(('__omlx_quant_v1__', (qk, qv, sk, sv, bk, bv)))

    return quantized_data

def dequantize_kv_block(
    layer_data: Any,
) -> Tuple[Any, Any]:
    """
    Dequantize int8 KV cache tensors back to original type.

    Args:
        layer_data: Quantized layer data as packaged by quantize_kv_block.

    Returns:
        (keys, values) tuple in dequantized fp16/bf16.
    """
    if not HAS_MLX or not isinstance(layer_data, tuple) or layer_data[0] != '__omlx_quant_v1__':
        return layer_data

    _, payload = layer_data
    qk, qv, sk, sv, bk, bv = payload

    k = mx.dequantize(qk, sk, bk, group_size=64, bits=8, mode="affine")
    v = mx.dequantize(qv, sv, bv, group_size=64, bits=8, mode="affine")

    return k, v
