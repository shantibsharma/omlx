#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include "mlx/mlx.h"

namespace mx = mlx::core;

namespace omlx {

struct LlamaArgs {
    int hidden_size;
    int num_hidden_layers;
    int intermediate_size;
    int num_attention_heads;
    int num_key_value_heads;
    int vocab_size;
    float rms_norm_eps;
    int head_dim;
    float rope_theta = 10000.0f;
    bool rope_traditional = false;
};

class LlamaModel {
public:
    LlamaModel(const LlamaArgs& args);
    
    // Load weights from a map (e.g., from safetensors)
    void load_weights(const std::unordered_map<std::string, mx::array>& weights);

    // Forward pass
    // x: input tokens [batch_size, sequence_length]
    // caches: optional KV caches (one per layer)
    // returns: logits [batch_size, sequence_length, vocab_size]
    mx::array forward(
        const mx::array& x,
        std::vector<std::optional<std::pair<mx::array, mx::array>>>& caches
    );

private:
    LlamaArgs args;
    std::unordered_map<std::string, mx::array> weights;

    mx::array attention(
        const mx::array& x,
        int layer_idx,
        std::optional<std::pair<mx::array, mx::array>>& cache
    );

    mx::array feed_forward(const mx::array& x, int layer_idx);
};

} // namespace omlx
