#include "metal_ops.h"
#include "mlx/backend/metal/device.h"
#include <iostream>

namespace omlx {

MetalOps& MetalOps::instance() {
    static MetalOps instance;
    return instance;
}

void MetalOps::init_libraries(const std::string& reshape_src, const std::string& paged_attn_src) {
    reshape_cache_source_ = reshape_src;
    paged_attention_source_ = paged_attn_src;

    auto& d = mx::metal::device(mx::Device::gpu);
    d.get_library("paged_reshape_cache", [&]() { return reshape_cache_source_; });
    d.get_library("paged_attention_kern", [&]() { return paged_attention_source_; });
}

void MetalOps::init_v2_library(const std::string& v2_src) {
    v2_paged_attention_source_ = v2_src;
    auto& d = mx::metal::device(mx::Device::gpu);
    d.get_library("paged_attention_v2_kern", [&]() { return v2_paged_attention_source_; });
}

void MetalOps::init_gdn_library(const std::string& gdn_src) {
    gdn_source_ = gdn_src;
    auto& d = mx::metal::device(mx::Device::gpu);
    d.get_library("gdn_kern", [&]() { return gdn_source_; });
}

static std::string dtype_to_metal(mx::Dtype dt) {
    if (dt == mx::float16) return "half";
    if (dt == mx::bfloat16) return "bfloat16_t";
    if (dt == mx::float32) return "float";
    return "half";
}

void MetalOps::dispatch_paged_attention_v2_partitioned(
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
) {
    auto& d = mx::metal::device(s.device);

    int total_q_tokens = static_cast<int>(query.shape(0));
    int num_heads  = static_cast<int>(query.shape(1));
    int head_size  = static_cast<int>(query.shape(2));
    int max_blocks = static_cast<int>(block_tables.shape(1));
    int num_seqs   = static_cast<int>(cu_seqlens_q.shape(0)) - 1;
    
    // PARTITION_SIZE from constants.py / paged_ops.cpp
    const int kPartitionSize = 512; 
    int max_num_partitions = std::max(1, (max_seq_len + kPartitionSize - 1) / kPartitionSize);

    auto dt = dtype_to_metal(query.dtype());
    std::string kname = "paged_attention_" + dt + "_cache_" + dt +
                        "_hs" + std::to_string(head_size) +
                        "_bs" + std::to_string(block_size) +
                        "_nt256_nsl32_ps" + std::to_string(kPartitionSize);

    bool use_partitioning = true;
    bool use_alibi = false;
    bool use_fp8 = false;
    bool use_sinks = false;

    auto* lib = d.get_library("paged_attention_v2_kern");
    auto* kernel = d.get_kernel(kname, lib, kname + "_v2",
        {{&use_partitioning, MTL::DataType::DataTypeBool, NS::UInteger(10)},
         {&use_alibi,        MTL::DataType::DataTypeBool, NS::UInteger(20)},
         {&use_fp8,          MTL::DataType::DataTypeBool, NS::UInteger(30)},
         {&use_sinks,        MTL::DataType::DataTypeBool, NS::UInteger(40)}});

    // Shmem sizing
    constexpr int NUM_THREADS = 256;
    constexpr int NUM_SIMD_LANES = 32;
    constexpr int NUM_WARPS = NUM_THREADS / NUM_SIMD_LANES;
    int warp_scores_bytes = NUM_WARPS * block_size * sizeof(float);
    int merge_bytes = (2 * NUM_WARPS + NUM_WARPS * head_size) * sizeof(float);
    size_t shmem = static_cast<size_t>(std::max(warp_scores_bytes, merge_bytes));

    auto& enc = d.get_command_encoder(s.index);
    enc.set_compute_pipeline_state(kernel);
    enc.set_threadgroup_memory_length(shmem, 0);

    enc.set_output_array(exp_sums, 0);
    enc.set_output_array(max_logits, 1);
    enc.set_output_array(tmp_out, 2);
    enc.set_input_array(query, 3);
    enc.set_input_array(key_cache, 4);
    enc.set_input_array(value_cache, 5);

    int32_t nkv = static_cast<int32_t>(num_kv_heads);
    enc.set_bytes(nkv, 8);
    enc.set_bytes(scale, 9);
    enc.set_bytes(softcap, 10);

    enc.set_input_array(block_tables, 11);
    enc.set_input_array(seq_lens, 12);
    int32_t max_blocks_i = static_cast<int32_t>(max_blocks);
    enc.set_bytes(max_blocks_i, 13);

    int32_t q_stride = static_cast<int32_t>(num_heads * head_size);
    int32_t kv_block_stride = static_cast<int32_t>(key_cache.strides()[0]);
    int32_t kv_head_stride = static_cast<int32_t>(key_cache.strides()[2]);
    enc.set_bytes(q_stride, 15);
    enc.set_bytes(kv_block_stride, 16);
    enc.set_bytes(kv_head_stride, 17);

    enc.set_input_array(cu_seqlens_q, 19);
    int32_t num_seqs_i = static_cast<int32_t>(num_seqs);
    enc.set_bytes(num_seqs_i, 20);
    int32_t sliding_window_i = -1; // Default disabled
    enc.set_bytes(sliding_window_i, 21);

    enc.dispatch_threadgroups(
        MTL::Size::Make(num_heads, total_q_tokens, max_num_partitions),
        MTL::Size::Make(NUM_THREADS, 1, 1));

    // Reduce Kernel
    std::string reduce_kname = "paged_attention_v2_reduce_" + dt +
                               "_hs" + std::to_string(head_size) +
                               "_nt256_nsl32_ps" + std::to_string(kPartitionSize);
    auto* reduce_kernel = d.get_kernel(reduce_kname, lib, reduce_kname + "_v2_reduce",
        {{&use_sinks, MTL::DataType::DataTypeBool, NS::UInteger(40)}});

    size_t reduce_shmem = static_cast<size_t>(2 * max_num_partitions * sizeof(float));
    enc.set_compute_pipeline_state(reduce_kernel);
    enc.set_threadgroup_memory_length(reduce_shmem, 0);

    enc.set_output_array(out, 0);
    enc.set_input_array(exp_sums, 1);
    enc.set_input_array(max_logits, 2);
    enc.set_input_array(tmp_out, 3);
    enc.set_input_array(seq_lens, 4);
    int32_t max_num_partitions_i = static_cast<int32_t>(max_num_partitions);
    enc.set_bytes(max_num_partitions_i, 5);
    enc.set_input_array(cu_seqlens_q, 7);
    enc.set_bytes(num_seqs_i, 8);

    enc.dispatch_threadgroups(
        MTL::Size::Make(num_heads, total_q_tokens, 1),
        MTL::Size::Make(NUM_THREADS, 1, 1));

    // arrays are managed by MLX graph, no need for add_temporary in this C++ integrated path
}

void MetalOps::dispatch_gdn_linear_attention(
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
) {
    auto& d = mx::metal::device(s.device);
    int num_requests = static_cast<int>(cu_seqlens.shape(0)) - 1;

    if (Dk > 256) {
        throw std::runtime_error(
            "GDN kernel supports Dk <= 256 (state[8] * 32 threads). "
            "Got Dk=" + std::to_string(Dk));
    }

    auto dt = dtype_to_metal(q.dtype());
    std::string kname = "gdn_linear_attention_" + dt;
    auto* lib = d.get_library("gdn_kern");
    auto* kernel = d.get_kernel(kname, lib, kname, {});

    auto& enc = d.get_command_encoder(s.index);
    enc.set_compute_pipeline_state(kernel);

    enc.set_input_array(q, 0);
    enc.set_input_array(k, 1);
    enc.set_input_array(v, 2);
    enc.set_input_array(g, 3);
    enc.set_input_array(beta, 4);
    enc.set_output_array(state_pool, 5);
    enc.set_input_array(cu_seqlens, 6);
    enc.set_input_array(slot_mapping, 7);
    enc.set_output_array(y, 8);

    int32_t num_req_i = static_cast<int32_t>(num_requests);
    int32_t Hk_i = static_cast<int32_t>(Hk);
    int32_t Hv_i = static_cast<int32_t>(Hv);
    int32_t Dk_i = static_cast<int32_t>(Dk);
    int32_t Dv_i = static_cast<int32_t>(Dv);
    enc.set_bytes(num_req_i, 9);
    enc.set_bytes(Hk_i, 10);
    enc.set_bytes(Hv_i, 11);
    enc.set_bytes(Dk_i, 12);
    enc.set_bytes(Dv_i, 13);

    // Grid: (Dv, 1, num_requests * Hv)  Threadgroup: (32, 1, 1)
    enc.dispatch_threadgroups(
        MTL::Size::Make(Dv, 1, num_requests * Hv),
        MTL::Size::Make(32, 1, 1));
}

} // namespace omlx
