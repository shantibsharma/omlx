import ctypes
import os
import sys

# Find the compiled library
lib_path = "cmlx/cmlx_fast_io.cpython-314-darwin.so"
if not os.path.exists(lib_path):
    # Try finding any .so in cmlx/
    import glob
    sos = glob.glob("cmlx/cmlx_fast_io*.so")
    if sos:
        lib_path = sos[0]
    else:
        print("❌ Could not find compiled .so library")
        sys.exit(1)

print(f"✅ Found library: {lib_path}")
lib = ctypes.CDLL(os.path.abspath(lib_path))

# Define NativeEngineResult structure to match C++
class NativeEngineResult(ctypes.Structure):
    _fields_ = [
        ("request_id", ctypes.c_char * 64),
        ("token", ctypes.c_int),
        ("state", ctypes.c_int), # 0=WAITING, 1=PREFILLING, 2=GENERATING, 3=FINISHED, 4=ABORTED
        ("finish_reason", ctypes.c_char * 32),
    ]

# Setup argument and return types
lib.native_engine_create.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_char_p]
lib.native_engine_create.restype = ctypes.c_void_p

lib.native_engine_destroy.argtypes = [ctypes.c_void_p]

lib.native_engine_step.argtypes = [
    ctypes.c_void_p, 
    ctypes.POINTER(NativeEngineResult), 
    ctypes.c_int
]
lib.native_engine_step.restype = ctypes.c_int

# Bridge for add_request
lib.native_engine_add_request_simple.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]

print("🚀 Initializing NativeEngine...")
cache_dir = b"./.native_cache"
engine = lib.native_engine_create(8.0, 12.0, cache_dir)

if engine:
    print("✅ Engine created successfully")
    
    # Add a request with 300 tokens (will take 3 prefill steps of 128 tokens)
    print("📝 Adding test request...")
    lib.native_engine_add_request_simple(engine, b"test-req-1", 300)
    
    # Run steps to see transition
    for i in range(10):
        results = (NativeEngineResult * 10)()
        count = lib.native_engine_step(engine, results, 10)
        
        state_str = "IDLE"
        if count > 0:
            s = results[0].state
            if s == 0: state_str = "WAITING"
            elif s == 1: state_str = "PREFILLING"
            elif s == 2: state_str = "GENERATING"
            elif s == 3: state_str = "FINISHED"
            elif s == 4: state_str = "ABORTED"
            print(f"Step {i}: Request {results[0].request_id.decode()} state={state_str}, token={results[0].token}")
        else:
            print(f"Step {i}: {state_str}")
    
    # Clean up
    lib.native_engine_destroy(engine)
    print("✅ Engine destroyed")
else:
    print("❌ Failed to create engine")
