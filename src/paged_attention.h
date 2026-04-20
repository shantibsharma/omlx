#pragma once

/**
 * paged_attention.h — cMLX integration shim for vllm-metal PagedAttention.
 *
 * The real compute kernel lives inside the vllm_metal Python package.
 * This header exposes a simple availability flag so that cmlx_fast_io
 * callers can check at runtime whether vllm-metal was loaded.
 */

namespace cmlx {
namespace paged_attention {

void set_available(bool available);
bool is_available();

} // namespace paged_attention
} // namespace cmlx
