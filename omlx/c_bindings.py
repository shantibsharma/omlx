import ctypes
import os
import logging
from typing import NamedTuple

logger = logging.getLogger(__name__)

# Locate the compiled C library - for performance.
try:
    # When installed as a package, the extension is a module within omlx
    from . import omlx_fast_io as _lib_module
    _lib = ctypes.CDLL(_lib_module.__file__)
    HAS_NATIVE = True
except (ImportError, AttributeError, OSError):
    # Fallback for local development/non-installed mode
    import os
    try:
        _lib_path = os.path.join(os.path.dirname(__file__), "..", "src", "omlx_fast_io.so")
        _lib = ctypes.CDLL(_lib_path)
        HAS_NATIVE = True
    except Exception:
        _lib = None
        HAS_NATIVE = False
        logger.warning("omlx_fast_io extension not found. C++ Native Runtime is partially disabled.")

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
    # 6. cache_core_init(int32_t, int32_t, size_t)
    _lib.cache_core_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_size_t]
    _lib.cache_core_init.restype = None

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

    def cache_core_init(max_blocks: int, initial_blocks: int, block_size_bytes: int = 0) -> None:
        if HAS_NATIVE:
            try:
                _lib.cache_core_init(max_blocks, initial_blocks, block_size_bytes)
            except Exception:
                pass

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

    HAS_NATIVE = True

except (OSError, FileNotFoundError):
    logger.warning("omlx_fast_io.so not found. C++ Native Runtime is partially disabled.")
    
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

    HAS_NATIVE = False

