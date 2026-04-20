import ctypes
import os
import time
import numpy as np

# Find the compiled library
lib_path = "cmlx/cmlx_fast_io.cpython-314-darwin.so"
if not os.path.exists(lib_path):
    import glob
    sos = glob.glob("cmlx/cmlx_fast_io*.so")
    if sos:
        lib_path = sos[0]

lib = ctypes.CDLL(os.path.abspath(lib_path))

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
lib.native_engine_step.argtypes = [ctypes.c_void_p, ctypes.POINTER(NativeEngineResult), ctypes.c_int]
lib.native_engine_step.restype = ctypes.c_int
lib.native_engine_add_request_simple.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]

def run_perf_test(num_requests=5, prompt_len=1000, max_tokens=100):
    print(f"🚀 Starting C++ Core Performance Test")
    print(f"Requests: {num_requests}, Prompt Len: {prompt_len}, Max Gen: {max_tokens}")
    
    cache_dir = b"./.native_cache"
    engine = lib.native_engine_create(8.0, 12.0, cache_dir)
    
    # Add requests
    for i in range(num_requests):
        req_id = f"test-batch-{i}".encode()
        lib.native_engine_add_request_simple(engine, req_id, prompt_len)
    
    results = (NativeEngineResult * 32)()
    
    start_time = time.perf_counter()
    total_tokens = 0
    finished_count = 0
    
    while finished_count < num_requests:
        count = lib.native_engine_step(engine, results, 32)
        for i in range(count):
            if results[i].state == 2: # GENERATING
                total_tokens += 1
            elif results[i].state == 3: # FINISHED
                finished_count += 1
        
        # Limit test duration
        if time.perf_counter() - start_time > 10:
            print("Timeout reached")
            break
            
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    print(f"\n✅ Performance Test Results:")
    print(f"Total Duration: {duration:.2f}s")
    print(f"Total Tokens Generated: {total_tokens}")
    print(f"Throughput: {total_tokens / duration:.2f} tokens/s")
    
    lib.native_engine_destroy(engine)

if __name__ == "__main__":
    run_perf_test()
