import ctypes
import os
import logging
from typing import NamedTuple

logger = logging.getLogger(__name__)

# Locate the compiled C library - for performance.
try:
    # When installed as a package, the extension is a module within cmlx
    from . import cmlx_fast_io as _lib_module
    _lib = ctypes.CDLL(_lib_module.__file__)
    HAS_NATIVE = True
except (ImportError, AttributeError, OSError):
    # Fallback for local development/non-installed mode
    import os
    try:
        _lib_path = os.path.join(os.path.dirname(__file__), "..", "src", "cmlx_fast_io.so")
        _lib = ctypes.CDLL(_lib_path)
        HAS_NATIVE = True
    except Exception:
        _lib = None
        HAS_NATIVE = False
        logger.warning("cmlx_fast_io extension not found. C++ Native Runtime is partially disabled.")

class NativeMemoryStats(NamedTuple):
    total_system_memory: int
    available_memory: int
    metal_active_memory: int
    metal_cache_memory: int
    metal_peak_memory: int

class MemoryStats(ctypes.Structure):
    _fields_ = [
        ("total_system_memory", ctypes.c_longlong),
        ("available_memory", ctypes.c_longlong),
        ("metal_active_memory", ctypes.c_longlong),
        ("metal_cache_memory", ctypes.c_longlong),
        ("metal_peak_memory", ctypes.c_longlong),
    ]

try:
    if not _lib:
        raise OSError("Library not loaded")
    
    # 1. fast_cache_warmup(const char*)
    _lib.fast_cache_warmup.argtypes = [ctypes.c_char_p]
    _lib.fast_cache_warmup.restype = ctypes.c_int

    # 2. parallel_warmup_dir(const char*, int)
    _lib.parallel_warmup_dir.argtypes = [ctypes.c_char_p, ctypes.c_int]
    _lib.parallel_warmup_dir.restype = ctypes.c_int

    # 3. estimate_model_size(const char*)
    _lib.estimate_model_size.argtypes = [ctypes.c_char_p]
    _lib.estimate_model_size.restype = ctypes.c_longlong

    # 4. get_memory_stats(MemoryStats*)
    _lib.get_memory_stats.argtypes = [ctypes.POINTER(MemoryStats)]
    _lib.get_memory_stats.restype = ctypes.c_int

    # 5. fast_tensor_count(const char*)
    _lib.fast_tensor_count.argtypes = [ctypes.c_char_p]
    _lib.fast_tensor_count.restype = ctypes.c_int

    def fast_cache_warmup(file_path: str) -> bool:
        """Stream a single safetensors file directly into Metal Unified Memory."""
        return _lib.fast_cache_warmup(file_path.encode('utf-8')) == 1

    def parallel_warmup_dir(dir_path: str, num_threads: int = 0) -> int:
        """Load all safetensors in a directory using native C++ threads."""
        return _lib.parallel_warmup_dir(dir_path.encode('utf-8'), num_threads)

    def estimate_model_size(dir_path: str) -> int:
        """Estimate model size (bytes) using native filesystem scan."""
        return _lib.estimate_model_size(dir_path.encode('utf-8'))

    def get_memory_stats() -> NativeMemoryStats | None:
        """Get real-time system and Metal memory stats via native calls."""
        stats = MemoryStats()
        if _lib.get_memory_stats(ctypes.byref(stats)) == 1:
            return NativeMemoryStats(
                total_system_memory=stats.total_system_memory,
                available_memory=stats.available_memory,
                metal_active_memory=stats.metal_active_memory,
                metal_cache_memory=stats.metal_cache_memory,
                metal_peak_memory=stats.metal_peak_memory,
            )
        return None

    def fast_tensor_count(file_path: str) -> int:
        """Count tensors in a safetensors file without full loading."""
        return _lib.fast_tensor_count(file_path.encode('utf-8'))

    # --- Cache Core (Task 3.1) ---
    # 6. cache_core_init(int32_t, int32_t, size_t, int32_t)
    _lib.cache_core_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_size_t, ctypes.c_int32]
    _lib.cache_core_init.restype = None

    _lib.cache_core_resolve_prefix.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.c_char_p
    ]
    _lib.cache_core_resolve_prefix.restype = ctypes.c_int32

    _lib.cache_core_register_block.argtypes = [ctypes.c_int32]
    _lib.cache_core_register_block.restype = None

    # 7. cache_core_allocate()
    _lib.cache_core_allocate.argtypes = []
    _lib.cache_core_allocate.restype = ctypes.c_int32

    # 8. cache_core_free(int32_t)
    _lib.cache_core_free.argtypes = [ctypes.c_int32]
    _lib.cache_core_free.restype = None

    # 9. cache_core_find_hash(const uint64_t*)
    _lib.cache_core_find_hash.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
    _lib.cache_core_find_hash.restype = ctypes.c_int32

    # 10. cache_core_set_hash(int32_t, const uint64_t*)
    _lib.cache_core_set_hash.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_uint64)]
    _lib.cache_core_set_hash.restype = None

    # 11. cache_core_get_free_count()
    _lib.cache_core_get_free_count.argtypes = []
    _lib.cache_core_get_free_count.restype = ctypes.c_int32

    # 12. cache_core_touch(int32_t)
    _lib.cache_core_touch.argtypes = [ctypes.c_int32]
    _lib.cache_core_touch.restype = None

    # 13. cache_core_allocate_specific(int32_t)
    _lib.cache_core_allocate_specific.argtypes = [ctypes.c_int32]
    _lib.cache_core_allocate_specific.restype = ctypes.c_int32

    # 14. cache_core_get_eviction_candidates(int32_t*, int)
    _lib.cache_core_get_eviction_candidates.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
    _lib.cache_core_get_eviction_candidates.restype = ctypes.c_int32

    # 20. cache_core_set_model_weight_bytes(size_t)
    _lib.cache_core_set_model_weight_bytes.argtypes = [ctypes.c_size_t]
    _lib.cache_core_set_model_weight_bytes.restype = None

    # 21. cache_core_get_total_usage()
    _lib.cache_core_get_total_usage.argtypes = []
    _lib.cache_core_get_total_usage.restype = ctypes.c_longlong

    def _to_hash_ptr(block_hash: bytes | None):
        if block_hash is None:
            return None
        if len(block_hash) != 32:
            return None
        h_array = (ctypes.c_uint64 * 4).from_buffer_copy(block_hash)
        return ctypes.cast(h_array, ctypes.POINTER(ctypes.c_uint64))

    def cache_core_init(max_blocks: int, initial_blocks: int, block_size_bytes: int = 0, tokens_per_block: int = 0) -> None:
        if HAS_NATIVE:
            try:
                _lib.cache_core_init(max_blocks, initial_blocks, block_size_bytes, tokens_per_block)
            except Exception:
                pass

    def cache_core_resolve_prefix(token_ids: list[int], max_blocks: int, model_name: str | None = None) -> list[int]:
        """Resolve an entire prefix chain in a single atomic native call (C++)."""
        if not HAS_NATIVE:
            return []
        num_tokens = len(token_ids)
        c_tokens = (ctypes.c_int32 * num_tokens)(*token_ids)
        c_out = (ctypes.c_int32 * max_blocks)()
        m_name = model_name.encode('utf-8') if model_name else None
        
        count = _lib.cache_core_resolve_prefix(c_tokens, num_tokens, c_out, max_blocks, m_name)
        return [int(c_out[i]) for i in range(count)]

    def cache_core_register_block(block_id: int) -> None:
        if HAS_NATIVE:
            try:
                _lib.cache_core_register_block(block_id)
            except Exception:
                pass

    def cache_core_allocate() -> int:
        """Allocate a block from the native C++ cache core."""
        return int(_lib.cache_core_allocate())

    def cache_core_free(block_id: int):
        """Free a block in the native C++ cache core."""
        _lib.cache_core_free(block_id)

    def cache_core_find_hash(block_hash: bytes) -> int:
        """Find a block by 256-bit SHA256 hash in the native C++ cache core."""
        ptr = _to_hash_ptr(block_hash)
        if ptr is None:
            return -1
        return int(_lib.cache_core_find_hash(ptr))

    def cache_core_set_hash(block_id: int, block_hash: bytes | None):
        """Set a block's 256-bit SHA256 hash in the native C++ cache core."""
        ptr = _to_hash_ptr(block_hash)
        _lib.cache_core_set_hash(block_id, ptr)

    def cache_core_get_free_count() -> int:
        """Get the number of free blocks in the native C++ cache core."""
        return int(_lib.cache_core_get_free_count())

    def cache_core_touch(block_id: int):
        """Update last access time and move to MRU in the native C++ cache core."""
        _lib.cache_core_touch(block_id)

    def cache_core_allocate_specific(block_id: int) -> bool:
        """Allocate a specific block (e.g. for prefix cache hit)."""
        return _lib.cache_core_allocate_specific(block_id) == 1

    def cache_core_get_eviction_candidates(count: int) -> list[int]:
        """Get list of block IDs for eviction (LRU order)."""
        out_ids = (ctypes.c_int32 * count)()
        found = _lib.cache_core_get_eviction_candidates(out_ids, count)
        return [int(out_ids[i]) for i in range(found)]

    def cache_core_set_model_weight_bytes(bytes_count: int):
        """Set the model weight memory overhead in bytes."""
        _lib.cache_core_set_model_weight_bytes(bytes_count)

    def cache_core_get_total_usage() -> int:
        """Get total tracked memory usage (cache + weights) in bytes."""
        return int(_lib.cache_core_get_total_usage())

    # --- FP8 Acceleration (Phase 5) ---
    class FP8EncodeResult(ctypes.Structure):
        _fields_ = [
            ("scale", ctypes.c_float),
            ("success", ctypes.c_int),
        ]

    _lib.fp8_encode_tensor.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    _lib.fp8_encode_tensor.restype = FP8EncodeResult

    _lib.fp8_decode_tensor.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_void_p]
    _lib.fp8_decode_tensor.restype = ctypes.c_int

    def native_fp8_encode(data_ptr: int, n_elements: int, out_ptr: int, dtype: str = "float32") -> tuple[float, bool]:
        """Encode a float32/fp16/bf16 buffer to FP8 E4M3 via native C++/Metal path."""
        dtype_map = {"float32": 0, "float16": 1, "bfloat16": 2}
        dtype_code = dtype_map.get(dtype, 0)
        res = _lib.fp8_encode_tensor(data_ptr, n_elements, dtype_code, out_ptr)
        return float(res.scale), res.success == 1

    def native_fp8_decode(fp8_ptr: int, n_elements: int, scale: float, out_ptr: int, dtype: str = "float32") -> bool:
        """Decode a FP8 E4M3 buffer back to float32/fp16/bf16 via native C++/Metal path."""
        dtype_map = {"float32": 0, "float16": 1, "bfloat16": 2}
        dtype_code = dtype_map.get(dtype, 0)
        return _lib.fp8_decode_tensor(fp8_ptr, n_elements, scale, dtype_code, out_ptr) == 1

    # --- Native I/O ---
    _lib.native_save_safetensors.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t)
    ]
    _lib.native_save_safetensors.restype = ctypes.c_int

    def native_save_safetensors(
        path: str,
        header_json: bytes,
        data_ptrs: list[int],
        data_sizes: list[int]
    ) -> bool:
        """Write a safetensors file natively via C++ to bypass GIL during SSD writes."""
        num_tensors = len(data_ptrs)
        # Create ctypes arrays for data pointers and sizes
        c_ptrs = (ctypes.c_void_p * num_tensors)(*data_ptrs)
        c_sizes = (ctypes.c_size_t * num_tensors)(*data_sizes)

        res = _lib.native_save_safetensors(
            path.encode('utf-8'),
            header_json,
            len(header_json),
            num_tensors,
            c_ptrs,
            c_sizes
        )
        return res == 1

    # --- Hashing & Metadata ---
    _lib.native_compute_block_hash.argtypes = [
        ctypes.c_void_p,      # parent_hash (32 bytes or NULL)
        ctypes.POINTER(ctypes.c_int32), # token_ids
        ctypes.c_int,         # num_tokens
        ctypes.c_char_p,      # model_name
        ctypes.c_void_p,      # extra_keys_data
        ctypes.c_size_t,      # extra_keys_size
        ctypes.c_void_p       # out_hash (32 bytes)
    ]
    _lib.native_compute_block_hash.restype = ctypes.c_int

    _lib.native_read_safetensors_metadata.argtypes = [
        ctypes.c_char_p,      # path
        ctypes.c_char_p,      # out_json_buffer
        ctypes.c_size_t,      # buffer_size
        ctypes.POINTER(ctypes.c_size_t) # out_actual_len
    ]
    _lib.native_read_safetensors_metadata.restype = ctypes.c_int

    def native_compute_block_hash(
        parent_hash: bytes | None,
        token_ids: list[int],
        model_name: str | None = None,
        extra_keys: bytes | None = None
    ) -> bytes:
        """Compute SHA256 block hash natively for high performance."""
        num_tokens = len(token_ids)
        c_tokens = (ctypes.c_int32 * num_tokens)(*token_ids)
        out_hash = ctypes.create_string_buffer(32)
        
        # parent_hash pointer
        p_ptr = ctypes.cast(ctypes.c_char_p(parent_hash), ctypes.c_void_p) if parent_hash else None
        # model_name
        m_name = model_name.encode('utf-8') if model_name else None
        # extra_keys
        extra_ptr = ctypes.cast(ctypes.c_char_p(extra_keys), ctypes.c_void_p) if extra_keys else None
        extra_size = len(extra_keys) if extra_keys else 0

        res = _lib.native_compute_block_hash(
            p_ptr,
            c_tokens,
            num_tokens,
            m_name,
            extra_ptr,
            extra_size,
            out_hash
        )
        return out_hash.raw if res == 1 else b""

    def native_read_safetensors_metadata(path: str, max_size: int = 1024*1024) -> str | None:
        """Fast native extraction of Safetensors JSON metadata."""
        buf = ctypes.create_string_buffer(max_size)
        actual_len = ctypes.c_size_t(0)
        res = _lib.native_read_safetensors_metadata(
            path.encode('utf-8'),
            buf,
            max_size,
            ctypes.byref(actual_len)
        )
        if res == 1:
            return buf.value.decode('utf-8')
        return None

    # --- Scheduler Core ---
    # 15. scheduler_core_init(float, float)
    _lib.scheduler_core_init.argtypes = [ctypes.c_float, ctypes.c_float]
    _lib.scheduler_core_init.restype = None

    # 16. scheduler_core_is_soft_critical()
    _lib.scheduler_core_is_soft_critical.argtypes = []
    _lib.scheduler_core_is_soft_critical.restype = ctypes.c_int32

    # 22. scheduler_core_is_hard_critical()
    _lib.scheduler_core_is_hard_critical.argtypes = []
    _lib.scheduler_core_is_hard_critical.restype = ctypes.c_int32

    # 17. scheduler_core_get_memory_gb()
    _lib.scheduler_core_get_memory_gb.argtypes = []
    _lib.scheduler_core_get_memory_gb.restype = ctypes.c_float

    # 18. scheduler_core_gpu_sync()
    _lib.scheduler_core_gpu_sync.argtypes = []
    _lib.scheduler_core_gpu_sync.restype = None

    # 19. scheduler_core_set_limits(float, float)
    _lib.scheduler_core_set_limits.argtypes = [ctypes.c_float, ctypes.c_float]
    _lib.scheduler_core_set_limits.restype = None

    # Native Queue Tasks
    _lib.scheduler_core_waiting_append.argtypes = [ctypes.c_char_p, ctypes.c_int]
    _lib.scheduler_core_waiting_append.restype = None

    _lib.scheduler_core_waiting_appendleft.argtypes = [ctypes.c_char_p, ctypes.c_int]
    _lib.scheduler_core_waiting_appendleft.restype = None

    _lib.scheduler_core_waiting_popleft.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    _lib.scheduler_core_waiting_popleft.restype = ctypes.c_int32

    _lib.scheduler_core_waiting_remove.argtypes = [ctypes.c_char_p]
    _lib.scheduler_core_waiting_remove.restype = None

    _lib.scheduler_core_waiting_size.argtypes = []
    _lib.scheduler_core_waiting_size.restype = ctypes.c_int32

    _lib.scheduler_core_waiting_clear.argtypes = []
    _lib.scheduler_core_waiting_clear.restype = None

    # Native Abort Queue
    _lib.scheduler_core_abort_enqueue.argtypes = [ctypes.c_char_p]
    _lib.scheduler_core_abort_enqueue.restype = None

    _lib.scheduler_core_abort_dequeue.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    _lib.scheduler_core_abort_dequeue.restype = ctypes.c_int32

    _lib.scheduler_core_abort_has_pending.argtypes = []
    _lib.scheduler_core_abort_has_pending.restype = ctypes.c_int32

    _lib.scheduler_core_abort_contains.argtypes = [ctypes.c_char_p]
    _lib.scheduler_core_abort_contains.restype = ctypes.c_int32

    _lib.scheduler_core_abort_clear.argtypes = []
    _lib.scheduler_core_abort_clear.restype = None

    if hasattr(_lib, 'scheduler_core_shutdown'):
        _lib.scheduler_core_shutdown.argtypes = []
        _lib.scheduler_core_shutdown.restype = None

    def scheduler_core_init(soft_limit_gb: float, hard_limit_gb: float):
        """Initialize the native C++ scheduler core with predictive thresholds."""
        _lib.scheduler_core_init(soft_limit_gb, hard_limit_gb)

    def scheduler_core_is_soft_critical() -> bool:
        """Check if soft memory pressure threshold is crossed."""
        return _lib.scheduler_core_is_soft_critical() == 1

    def scheduler_core_is_hard_critical() -> bool:
        """Check if hard memory pressure threshold is crossed."""
        return _lib.scheduler_core_is_hard_critical() == 1

    def scheduler_core_get_memory_gb() -> float:
        """Get current active memory usage in GB via native core."""
        return float(_lib.scheduler_core_get_memory_gb())

    def scheduler_core_gpu_sync():
        """Force a GPU device synchronization via native core."""
        _lib.scheduler_core_gpu_sync()

    def scheduler_core_set_limits(soft_gb: float, hard_gb: float):
        """Update both soft and hard memory limits in the native core."""
        _lib.scheduler_core_set_limits(soft_gb, hard_gb)

    def scheduler_core_waiting_append(request_id: str, priority: int = 0):
        _lib.scheduler_core_waiting_append(request_id.encode('utf-8'), priority)

    def scheduler_core_waiting_appendleft(request_id: str, priority: int = 0):
        _lib.scheduler_core_waiting_appendleft(request_id.encode('utf-8'), priority)

    def scheduler_core_waiting_popleft() -> str | None:
        buf = ctypes.create_string_buffer(256)
        res = _lib.scheduler_core_waiting_popleft(buf, 256)
        if res == 1:
            return buf.value.decode('utf-8')
        return None

    def scheduler_core_waiting_remove(request_id: str):
        _lib.scheduler_core_waiting_remove(request_id.encode('utf-8'))

    def scheduler_core_waiting_size() -> int:
        return _lib.scheduler_core_waiting_size()

    def scheduler_core_waiting_clear():
        _lib.scheduler_core_waiting_clear()

    def scheduler_core_abort_enqueue(request_id: str):
        _lib.scheduler_core_abort_enqueue(request_id.encode('utf-8'))

    def scheduler_core_abort_dequeue() -> str | None:
        buf = ctypes.create_string_buffer(256)
        res = _lib.scheduler_core_abort_dequeue(buf, 256)
        if res == 1:
            return buf.value.decode('utf-8')
        return None

    def scheduler_core_abort_has_pending() -> bool:
        return _lib.scheduler_core_abort_has_pending() == 1

    def scheduler_core_abort_contains(request_id: str) -> bool:
        return _lib.scheduler_core_abort_contains(request_id.encode('utf-8')) == 1

    def scheduler_core_abort_clear():
        _lib.scheduler_core_abort_clear()

    def scheduler_core_shutdown():
        """Shut down and join the native background monitor thread."""
        if HAS_NATIVE and hasattr(_lib, 'scheduler_core_shutdown'):
            _lib.scheduler_core_shutdown()

    HAS_NATIVE = True

except (OSError, FileNotFoundError):
    logger.warning("cmlx_fast_io.so not found. C++ Native Runtime is partially disabled.")
    
    def fast_cache_warmup(file_path: str) -> bool: return False
    def parallel_warmup_dir(dir_path: str, num_threads: int = 0) -> int: return -1
    def estimate_model_size(dir_path: str) -> int: return -1
    def get_memory_stats() -> NativeMemoryStats | None: return None
    def fast_tensor_count(file_path: str) -> int: return -1
    
    def cache_core_init(max_blocks: int, block_size_bytes: int = 0): pass
    def cache_core_allocate() -> int: return -2
    def cache_core_free(block_id: int): pass
    def cache_core_find_hash(block_hash: bytes) -> int: return -1
    def cache_core_set_hash(block_id: int, block_hash: bytes | None): pass
    def cache_core_get_free_count() -> int: return 0
    def cache_core_touch(block_id: int): pass
    def cache_core_allocate_specific(block_id: int) -> bool: return False
    def cache_core_get_eviction_candidates(count: int) -> list[int]: return []
    def cache_core_set_model_weight_bytes(bytes_count: int): pass
    def cache_core_get_total_usage() -> int: return 0

    def scheduler_core_init(soft_limit_gb: float, hard_limit_gb: float): pass
    def scheduler_core_is_soft_critical() -> bool: return False
    def scheduler_core_is_hard_critical() -> bool: return False
    def scheduler_core_get_memory_gb() -> float: return 0.0
    def scheduler_core_gpu_sync(): pass
    def scheduler_core_set_limits(soft_gb: float, hard_gb: float): pass
    def scheduler_core_waiting_append(request_id: str, priority: int = 0): pass
    def scheduler_core_waiting_appendleft(request_id: str, priority: int = 0): pass
    def scheduler_core_waiting_popleft() -> str | None: return None
    def scheduler_core_waiting_remove(request_id: str): pass
    def scheduler_core_waiting_size() -> int: return 0
    def scheduler_core_waiting_clear(): pass
    def scheduler_core_abort_enqueue(request_id: str): pass
    def scheduler_core_abort_dequeue() -> str | None: return None
    def scheduler_core_abort_has_pending() -> bool: return False
    def scheduler_core_abort_contains(request_id: str) -> bool: return False
    def scheduler_core_abort_clear(): pass

    def native_fp8_encode(data_ptr: int, n_elements: int, out_ptr: int) -> tuple[float, bool]: return 1.0, False
    def native_fp8_decode(fp8_ptr: int, n_elements: int, scale: float, out_ptr: int) -> bool: return False
    def native_save_safetensors(path: str, header_json: bytes, data_ptrs: list[int], data_sizes: list[int]) -> bool: return False
    def native_compute_block_hash(parent_hash: bytes | None, token_ids: list[int], model_name: str | None = None, extra_keys: bytes | None = None) -> bytes: return b""
    def native_read_safetensors_metadata(path: str, max_size: int = 1024*1024) -> str | None: return None

    HAS_NATIVE = False

