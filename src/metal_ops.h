#pragma once

#include <string>
#include <vector>
#include <memory>
#include "mlx/mlx.h"
#include "mlx/backend/metal/device.h"

namespace mx = mlx::core;

namespace omlx {

/**
 * MetalOps
 * Low-level Metal kernel manager for oMLX.
 * Ported from vllm-metal for high-performance paged attention.
 */
class MetalOps {
public:
    static MetalOps& instance();

    // Initialize Metal libraries with provided shader source
    void init_libraries(const std::string& reshape_src, const std::string& paged_attn_src);
    void init_v2_library(const std::string& v2_src);
    void init_gdn_library(const std::string& gdn_src);

    // High-performance partitioned attention dispatch
    void dispatch_paged_attention_v2_partitioned(
        mx::array& out,
        const mx::array& query,
        const mx::array& key_cache,
        const mx::array& value_cache,
        int num_kv_heads,
        float scale,
        float softcap,
        const mx::array& block_tables,
        const mx::array& seq_lens,
        const mx::array& cu_seqlens_q,
        int block_size,
        int max_seq_len,
        int sliding_window,
        mx::Stream s,
        mx::array& exp_sums,
        mx::array& max_logits,
        mx::array& tmp_out
    );

    // GDN Linear Attention (Qwen3.5 Support)
    void dispatch_gdn_linear_attention(
        const mx::array& q,
        const mx::array& k,
        const mx::array& v,
        const mx::array& g,
        const mx::array& beta,
        mx::array& state_pool,
        const mx::array& cu_seqlens,
        const mx::array& slot_mapping,
        mx::array& y,
        int Hk, int Hv, int Dk, int Dv,
        mx::Stream s
    );

private:
    MetalOps() = default;
    
    std::string reshape_cache_source_;
    std::string paged_attention_source_;
    std::string v2_paged_attention_source_;
    std::string gdn_source_;
};

} // namespace omlx
