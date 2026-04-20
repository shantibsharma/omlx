import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from cmlx.c_bindings import (
        HAS_NATIVE, 
        scheduler_core_init, 
        scheduler_core_is_critical,
        scheduler_core_get_memory_gb,
        cache_core_init,
        cache_core_get_free_count
    )
    
    print(f"Native Runtime Detected: {HAS_NATIVE}")
    if not HAS_NATIVE:
        print("ERROR: Native runtime enabled is False!")
        sys.exit(1)
        
    # Test initialization
    print("Testing Scheduler Core init (40GB limit)...")
    scheduler_core_init(40.0)
    print(f"Current Memory: {scheduler_core_get_memory_gb():.2f} GB")
    print(f"Is Critical: {scheduler_core_is_critical()}")
    
    # Test Cache Core init
    print("Testing Cache Core init (1000 blocks)...")
    cache_core_init(1000)
    print(f"Free Block Count: {cache_core_get_free_count()}")
    
    print("\n✅ Native cMLX runtime verified successfully!")
    
except Exception as e:
    print(f"\n❌ Binding Verification Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
