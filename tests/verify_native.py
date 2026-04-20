import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from cmlx.c_bindings import get_memory_stats, estimate_model_size, parallel_warmup_dir

def test_native_runtime():
    print("=== cMLX Native Runtime Verification ===")
    
    # 1. Test Memory Stats
    print("\n1. Testing get_memory_stats()...")
    stats = get_memory_stats()
    if stats:
        print(f"   Total System RAM: {stats.total_system_memory / 1024**3:.2f} GB")
        print(f"   Available RAM:    {stats.available_memory / 1024**3:.2f} GB")
        print(f"   Metal Active:     {stats.metal_active_memory / 1024**2:.2f} MB")
        print(f"   Metal Cache:      {stats.metal_cache_memory / 1024**2:.2f} MB")
        print(f"   Metal Peak:       {stats.metal_peak_memory / 1024**2:.2f} MB")
    else:
        print("   FAILED to get memory stats")

    # 2. Test Size Estimation
    models_dir = Path.home() / ".cmlx" / "models"
    first_model = None
    if models_dir.exists():
        for d in models_dir.iterdir():
            if d.is_dir() and (d / "config.json").exists():
                first_model = d
                break
    
    if first_model:
        print(f"\n2. Testing estimate_model_size('{first_model.name}')...")
        size = estimate_model_size(str(first_model))
        if size > 0:
            print(f"   Estimated Size: {size / 1024**3:.2f} GB")
        else:
            print("   FAILED to estimate size")
            
        # 3. Test Parallel Warmup
        print(f"\n3. Testing parallel_warmup_dir('{first_model.name}')...")
        shards = parallel_warmup_dir(str(first_model))
        if shards >= 0:
            print(f"   Successfully warmed up {shards} shards")
        else:
            print("   FAILED parallel warmup")
    else:
        print("\n2/3. SKIPPING size/warmup (no models found in ~/.cmlx/models)")

if __name__ == "__main__":
    test_native_runtime()
