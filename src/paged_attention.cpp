/**
 * paged_attention.cpp — cMLX integration shim for vllm-metal PagedAttention.
 *
 * The actual Metal kernel compilation and dispatch is handled entirely by
 * the vllm_metal Python package (installed separately via pip).
 *
 * This file provides a thin C++ stub so the symbol is available in
 * cmlx_fast_io.so for future direct-dispatch optimizations without
 * going through vllm_metal's Python layer.
 */

#include <cstdio>

namespace cmlx {
namespace paged_attention {

static bool _vllm_metal_available = false;

void set_available(bool available) {
    _vllm_metal_available = available;
}

bool is_available() {
    return _vllm_metal_available;
}

} // namespace paged_attention
} // namespace cmlx
