# SPDX-License-Identifier: Apache-2.0
"""
Memory Monitor for oMLX paged SSD-based KV cache.

This module provides memory utilities for paged SSD-based KV cache management
on Apple Silicon unified memory.

Key features:
- GPU memory utilization tracking via MLX Metal API
- Block memory estimation for cache management
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from omlx.cache.paged_cache import PagedCacheManager

from omlx.utils.hardware import get_max_working_set_bytes, format_bytes

logger = logging.getLogger(__name__)

# Check if MLX Metal is available
try:
    import mlx.core as mx

    HAS_MLX_METAL = mx.metal.is_available()
except ImportError:
    HAS_MLX_METAL = False
    mx = None


@dataclass
class MemoryInfo:
    """
    Current GPU memory state.

    Attributes:
        total_bytes: Total available GPU memory
        used_bytes: Currently used memory (estimated)
        available_bytes: Available memory
        utilization: Memory utilization ratio (0.0 to 1.0)
    """

    total_bytes: int
    used_bytes: int
    available_bytes: int
    utilization: float


class MemoryMonitor:
    """
    Memory monitor for paged SSD-based KV cache.

    In paged SSD-only mode, KV cache data is stored on paged SSD, not GPU memory.
    This class provides memory estimation utilities for block management
    but does not trigger GPU memory-based eviction.

    Example:
        >>> monitor = MemoryMonitor(max_kv_cache_memory=2 * 1024**3)
        >>> block_mem = monitor.estimate_block_memory(64)  # 64 tokens
    """

    def __init__(
        self,
        max_kv_cache_memory: int,
        check_interval: float = 1.0,
    ):
        """
        Initialize the memory monitor.

        Args:
            max_kv_cache_memory: Maximum memory for KV cache in bytes (required).
                This is the absolute limit for KV cache memory usage.
            check_interval: Minimum seconds between memory checks (for throttling).
        """
        if max_kv_cache_memory <= 0:
            raise ValueError(
                f"max_kv_cache_memory must be positive, got {max_kv_cache_memory}"
            )

        self._max_kv_cache_memory = max_kv_cache_memory
        self._check_interval = check_interval
        self._max_memory = self._get_max_memory()

        self._last_check_time = 0.0
        self._last_memory_info: Optional[MemoryInfo] = None
        self._lock = threading.Lock()

        # Model info for memory estimation (set by scheduler)
        self._num_layers: Optional[int] = None
        self._num_kv_heads: Optional[int] = None
        self._head_dim: Optional[int] = None
        self._dtype_size: int = 2  # Default float16
        self._num_attention_heads: Optional[int] = None
        self._num_kv_cache_layers: Optional[int] = None

        # PagedCacheManager for KV cache memory measurement
        self._paged_cache_manager: Optional["PagedCacheManager"] = None
        self._block_size: int = 256  # Default block size

        # Baseline memory (model weights) - set after model load
        self._baseline_memory: int = 0

        # Request stats (set by scheduler for logging)
        self._running_requests: int = 0
        self._waiting_requests: int = 0

        # Safety buffer for model activation memory (Task 2.1)
        self._safety_buffer_bytes: int = 0

        logger.info(
            f"MemoryMonitor initialized: max_kv_cache={format_bytes(max_kv_cache_memory)}"
        )

        # Cooldown for pressure-triggered eviction.
        # Eviction from the index does not free Metal memory, so we must
        # prevent re-triggering every generation step when the GPU reading
        # remains at the same high-water-mark between checks.
        self._last_eviction_time: float = 0.0
        self._eviction_cooldown: float = 2.0  # seconds between eviction cycles

    def _get_max_memory(self) -> int:
        """
        Get max_recommended_working_set_size from MLX Metal.

        Falls back to system memory heuristic if MLX Metal unavailable.

        Returns:
            Maximum memory in bytes that can be used.
        """
        return get_max_working_set_bytes()

    def set_paged_cache_manager(
        self, manager: "PagedCacheManager", block_size: int = 64
    ) -> None:
        """
        Set PagedCacheManager for memory monitoring.

        Args:
            manager: PagedCacheManager instance
            block_size: Number of tokens per block
        """
        self._paged_cache_manager = manager
        self._block_size = block_size
        logger.info(
            f"PagedCacheManager connected for memory monitoring "
            f"(block_size={block_size})"
        )

    def set_baseline_memory(self) -> None:
        """
        Set baseline memory after model load.

        Call this after loading the model to capture the baseline memory usage
        (model weights, etc.). The KV cache memory is calculated as:
        active_memory - baseline_memory

        This allows accurate detection of memory pressure from KV cache growth
        while ignoring static model memory.
        """
        if HAS_MLX_METAL:
            try:
                self._baseline_memory = mx.get_active_memory()
                logger.info(
                    f"Baseline memory set: {format_bytes(self._baseline_memory)}"
                )
            except Exception as e:
                logger.warning(f"Failed to set baseline memory: {e}")
                self._baseline_memory = 0
        else:
            self._baseline_memory = 0
            logger.warning("MLX Metal not available, baseline memory set to 0")

    def set_request_stats(self, running: int, waiting: int) -> None:
        """
        Update request stats for logging.

        Args:
            running: Number of currently running requests
            waiting: Number of waiting requests
        """
        self._running_requests = running
        self._waiting_requests = waiting

    def set_safety_buffer(self, bytes: int) -> None:
        """
        Set safety buffer in bytes.
        
        This buffer is reserved for peak activation memory during prefill.
        If current_memory + safety_buffer > hard_limit * threshold,
        pressure is detected.
        """
        self._safety_buffer_bytes = bytes
        logger.info(f"MemoryMonitor safety buffer set to {format_bytes(bytes)}")

    def _get_current_memory_usage(self) -> int:
        """
        Get current GPU memory usage.

        Returns:
            Current Metal active memory in bytes.
        """
        if HAS_MLX_METAL:
            return mx.get_active_memory()
        return self._get_process_rss()

    def _get_process_rss(self) -> int:
        """
        Get process RSS memory (fallback method).

        Returns:
            Process resident set size in bytes.
        """
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            return 0

    def get_memory_info(self) -> MemoryInfo:
        """
        Get current memory state.

        Returns:
            MemoryInfo with current memory statistics.
        """
        with self._lock:
            current_time = time.time()

            # Throttle checks to avoid overhead
            if (
                self._last_memory_info is not None
                and current_time - self._last_check_time < self._check_interval
            ):
                return self._last_memory_info

            used = self._get_current_memory_usage()
            available = max(0, self._max_memory - used)
            utilization = used / self._max_memory if self._max_memory > 0 else 0.0

            self._last_memory_info = MemoryInfo(
                total_bytes=self._max_memory,
                used_bytes=used,
                available_bytes=available,
                utilization=utilization,
            )
            self._last_check_time = current_time

            return self._last_memory_info

    def is_under_pressure(self, threshold: float = 0.9) -> bool:
        """
        Check if memory pressure exists based on a utilization threshold.

        Includes a cooldown to prevent repeated eviction storms when in
        paged-SSD-only mode where index eviction doesn't free Metal memory.

        Accounts for the safety buffer if set (Task 2.1).

        Args:
            threshold: Utilization ratio (0.0 to 1.0) above which pressure is detected.

        Returns:
            True if (utilization + safety_buffer_ratio) exceeds threshold AND
            the eviction cooldown has elapsed.
        """
        info = self.get_memory_info()

        # Effective utilization including safety buffer
        effective_used = info.used_bytes + self._safety_buffer_bytes
        effective_utilization = (
            effective_used / self._max_memory if self._max_memory > 0 else 0.0
        )

        if effective_utilization < threshold:
            return False

        # Enforce cooldown: even if still under pressure, don't evict
        # more often than once per _eviction_cooldown seconds.
        now = time.time()
        if now - self._last_eviction_time < self._eviction_cooldown:
            return False

        return True

    def is_critically_over_limit(self, hard_limit_bytes: int, threshold: float = 0.92) -> bool:
        """
        Check if memory is critically close to the hard limit.

        Unlike is_under_pressure(), this NEVER applies cooldown — it is a
        real-time safety gate that must fire every time to prevent SIGABRT.

        Args:
            hard_limit_bytes: Absolute hard memory limit (system_ram - 4GB).
            threshold: Fraction of hard_limit above which we are critical.

        Returns:
            True if active Metal memory exceeds threshold * hard_limit.
        """
        if hard_limit_bytes <= 0 or not HAS_MLX_METAL:
            return False
        current = mx.get_active_memory()
        return current > int(hard_limit_bytes * threshold)

    def should_skip_cache_store(self, hard_limit_bytes: int) -> bool:
        """
        Check whether store_cache should be skipped to prevent OOM.

        When active memory is above 85% of the hard limit, the temporary
        memory spike from cloning KV tensors into paged blocks will likely
        push past the Metal limit and crash the process.

        Args:
            hard_limit_bytes: Absolute hard memory limit.

        Returns:
            True if cache storage should be skipped.
        """
        if hard_limit_bytes <= 0 or not HAS_MLX_METAL:
            return False
        current = mx.get_active_memory()
        # Skip cache store when we're above 85% of hard limit — the
        # store_cache tensor cloning would spike us past the limit.
        return current > int(hard_limit_bytes * 0.85)

    def record_eviction(self) -> None:
        """Record that an eviction cycle just completed (resets cooldown)."""
        self._last_eviction_time = time.time()

    def bytes_to_free(self, target_utilization: float = 0.8) -> int:
        """
        Calculate bytes needed to free to reach target utilization.

        Args:
            target_utilization: Goal utilization ratio.

        Returns:
            Bytes to free, or 0 if already under target.
        """
        info = self.get_memory_info()
        target_bytes = int(self._max_memory * target_utilization)
        
        if info.used_bytes > target_bytes:
            return info.used_bytes - target_bytes
        return 0

    def set_model_info(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype_size: int = 2,
        num_attention_heads: Optional[int] = None,
        num_kv_cache_layers: Optional[int] = None,
    ) -> None:
        """
        Set model information for memory estimation.

        Args:
            num_layers: Number of transformer layers
            num_kv_heads: Number of KV attention heads
            head_dim: Dimension per attention head
            dtype_size: Bytes per element (2 for float16, 4 for float32)
            num_attention_heads: Number of query attention heads (for SDPA
                peak estimation). Defaults to num_kv_heads if not set.
            num_kv_cache_layers: Number of layers that use KVCache
                (full attention). For hybrid models this may be less than
                num_layers. Defaults to num_layers.
        """
        self._num_layers = num_layers
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._dtype_size = dtype_size
        self._num_attention_heads = num_attention_heads or num_kv_heads
        self._num_kv_cache_layers = num_kv_cache_layers or num_layers

        # Log estimated memory per block
        if num_layers and num_kv_heads and head_dim:
            sample_block_mem = self.estimate_block_memory(64)  # 64 tokens
            logger.info(
                f"Model info set: {num_layers} layers "
                f"({self._num_kv_cache_layers} KVCache), "
                f"{num_kv_heads} KV heads, "
                f"{self._num_attention_heads} Q heads, "
                f"{head_dim} head_dim. Estimated memory per 64-token block: "
                f"{format_bytes(sample_block_mem)}"
            )

    def estimate_block_memory(
        self,
        block_size: int,
        num_layers: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        dtype_size: Optional[int] = None,
    ) -> int:
        """
        Estimate memory usage for a KV cache block.

        Args:
            block_size: Number of tokens in the block
            num_layers: Override stored num_layers
            num_kv_heads: Override stored num_kv_heads
            head_dim: Override stored head_dim
            dtype_size: Override stored dtype_size

        Returns:
            Estimated memory in bytes for one block.
        """
        layers = num_layers or self._num_layers or 32  # Default for ~7B model
        kv_heads = num_kv_heads or self._num_kv_heads or 8
        dim = head_dim or self._head_dim or 128
        dtype = dtype_size or self._dtype_size

        # Memory per layer: keys + values
        # Shape: (batch=1, kv_heads, block_size, head_dim)
        per_layer = block_size * kv_heads * dim * dtype * 2  # *2 for keys+values
        total = per_layer * layers

        return total

    def estimate_prompt_kv_bytes(self, num_tokens: int) -> int:
        """
        Estimate KV cache memory for a prompt of given length.

        Uses per-layer cache type info if available (hybrid models),
        otherwise falls back to uniform num_layers estimate.

        Args:
            num_tokens: Number of prompt tokens.

        Returns:
            Estimated KV cache memory in bytes.
        """
        layers = self._num_kv_cache_layers or self._num_layers or 0
        kv_heads = self._num_kv_heads or 0
        dim = self._head_dim or 0
        dtype = self._dtype_size

        if not (layers and kv_heads and dim):
            return 0

        # KVCache layers: memory grows with num_tokens
        per_token = layers * kv_heads * dim * dtype * 2  # keys + values
        return num_tokens * per_token

    def estimate_prefill_peak_bytes(
        self, total_prompt_tokens: int, chunk_size: int
    ) -> int:
        """
        Estimate worst-case peak memory during prefill (last chunk).

        MLX SDPA internals (C++ fallback path, head_dim > 128):
          1. scores = scale*Q @ K^T → [B, n_q, chunk, kv_len] float32
          2. softmax(scores) → in-place (already float32)
          3. out = scores @ V → [B, n_q, chunk, head_dim] float32
          GQA: K/V broadcast, no extra allocation.

        MLX SDPA fused kernel (head_dim <= 128):
          Tiled computation, O(n) memory. Only output buffer allocated.

        Args:
            total_prompt_tokens: Total tokens in the prompt.
            chunk_size: Prefill step size (default 2048).

        Returns:
            Estimated peak memory in bytes (KV cache + SDPA activation).
            Returns 0 if model info is not available.
        """
        hd = self._head_dim or 0
        n_q = self._num_attention_heads or 0

        if n_q == 0 or hd == 0:
            return 0  # can't estimate

        if hd > 128:
            # Fallback: full attention matrix materialized in float32
            # scores [B, n_q, chunk, total_tokens] + output [B, n_q, chunk, hd]
            attn = n_q * chunk_size * total_prompt_tokens * 4
            attn += n_q * chunk_size * hd * 4  # output buffer (small)
        else:
            # Fused kernel: tiled, only output buffer
            attn = n_q * chunk_size * hd * 4

        kv = self.estimate_prompt_kv_bytes(total_prompt_tokens)
        return attn + kv

    def estimate_blocks_to_free(self, bytes_to_free: int, block_size: int) -> int:
        """
        Estimate number of blocks to evict to free the given bytes.

        Args:
            bytes_to_free: Target bytes to free
            block_size: Tokens per block

        Returns:
            Number of blocks to evict.
        """
        block_mem = self.estimate_block_memory(block_size)
        if block_mem <= 0:
            return 0

        # Round up to ensure we free enough
        num_blocks = (bytes_to_free + block_mem - 1) // block_mem
        return max(1, num_blocks)

    @property
    def max_memory(self) -> int:
        """Get maximum system memory limit."""
        return self._max_memory

    @property
    def max_kv_cache_memory(self) -> int:
        """Get maximum KV cache memory limit."""
        return self._max_kv_cache_memory

    def get_stats(self) -> dict:
        """
        Get memory statistics as a dictionary.

        Returns:
            Dictionary with memory statistics.
        """
        info = self.get_memory_info()
        return {
            "total_bytes": info.total_bytes,
            "used_bytes": info.used_bytes,
            "available_bytes": info.available_bytes,
            "utilization": info.utilization,
            "max_kv_cache_memory": self._max_kv_cache_memory,
            "total_formatted": format_bytes(info.total_bytes),
            "used_formatted": format_bytes(info.used_bytes),
            "available_formatted": format_bytes(info.available_bytes),
        }

    def __repr__(self) -> str:
        info = self.get_memory_info()
        return (
            f"MemoryMonitor(max_kv_cache={format_bytes(self._max_kv_cache_memory)}, "
            f"used={format_bytes(info.used_bytes)})"
        )
