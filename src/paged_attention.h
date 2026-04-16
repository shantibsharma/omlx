#pragma once

/**
 * paged_attention.h — oMLX integration shim for vllm-metal PagedAttention.
 *
 * The real compute kernel lives inside the vllm_metal Python package.
 * This header exposes a simple availability flag so that omlx_fast_io
 * callers can check at runtime whether vllm-metal was loaded.
 */

namespace omlx {
namespace paged_attention {

void set_available(bool available);
bool is_available();

} // namespace paged_attention
} // namespace omlx
