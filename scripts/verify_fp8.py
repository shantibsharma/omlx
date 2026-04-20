import mlx.core as mx
import numpy as np
import os
import shutil
from pathlib import Path
from omlx.fp8 import quantize_weight_to_fp8, dequantize_fp8_weight
from omlx.cache.quantization import quantize_kv_block, dequantize_kv_block
from omlx.fp8 import quantize_kv_block_fp8

def test_weight_roundtrip():
    print("Testing FP8 Weight Roundtrip...")
    original = mx.random.normal((512, 512)).astype(mx.float16)
    fp8, scale = quantize_weight_to_fp8(original)
    
    assert fp8.dtype == mx.uint8
    assert scale.dtype == mx.float32
    print(f"  Scale: {scale.item():.6f}")
    
    restored = dequantize_fp8_weight(fp8, scale, mx.float16)
    
    mse = mx.mean((original - restored)**2).item()
    print(f"  MSE: {mse:.8f}")
    assert mse < 1e-3

def test_kv_cache_integration():
    print("\nTesting KV Cache FP8 Integration...")
    # Mock KV cache block (2 layers)
    k1 = mx.random.normal((1, 8, 64, 128)).astype(mx.float16)
    v1 = mx.random.normal((1, 8, 64, 128)).astype(mx.float16)
    k2 = mx.random.normal((1, 8, 64, 128)).astype(mx.float16)
    v2 = mx.random.normal((1, 8, 64, 128)).astype(mx.float16)
    
    cache_data = [(k1, v1), (k2, v2)]
    
    # 1. Test INT8 path (default)
    print("  Testing INT8 path...")
    q_int8 = quantize_kv_block(cache_data)
    assert q_int8[0][0] == '__omlx_quant_v1__'
    
    # 2. Test FP8 path
    print("  Testing FP8 path...")
    q_fp8 = quantize_kv_block_fp8(cache_data)
    assert q_fp8[0][0] == '__omlx_fp8_v1__'
    
    # 3. Test Auto-dequantization
    print("  Testing Auto-dequantization...")
    r_int8_k, r_int8_v = dequantize_kv_block(q_int8[0])
    r_fp8_k, r_fp8_v = dequantize_kv_block(q_fp8[0])
    
    mse_int8 = mx.mean((k1 - r_int8_k)**2).item()
    mse_fp8 = mx.mean((k1 - r_fp8_k)**2).item()
    
    print(f"    INT8 MSE: {mse_int8:.8f}")
    print(f"    FP8 MSE:  {mse_fp8:.8f}")
    
    assert mse_int8 < 1e-3
    assert mse_fp8 < 1e-3

def test_native_bindings():
    print("\nTesting Native C++ Bindings...")
    from omlx.c_bindings import HAS_NATIVE, native_fp8_encode, native_fp8_decode
    
    if not HAS_NATIVE:
        print("  SKIPPED: Native extension not found")
        return
        
    print("  Native extension found. Verifying symbols...")
    # Test simple encoding
    data = np.random.randn(1024).astype(np.float32)
    out_fp8 = np.zeros(1024, dtype=np.uint8)
    
    scale, success = native_fp8_encode(data.ctypes.data, 1024, out_fp8.ctypes.data)
    print(f"  Encoding Success: {success}, Scale: {scale:.6f}")
    assert success
    
    # Test decoding
    out_float = np.zeros(1024, dtype=np.float32)
    success = native_fp8_decode(out_fp8.ctypes.data, 1024, scale, out_float.ctypes.data)
    print(f"  Decoding Success: {success}")
    assert success
    
    mse = np.mean((data - out_float)**2)
    print(f"  Native MSE: {mse:.8f}")
    assert mse < 1e-3

if __name__ == "__main__":
    test_weight_roundtrip()
    test_kv_cache_integration()
    test_native_bindings()
    print("\n✅ All FP8 integration tests passed!")
