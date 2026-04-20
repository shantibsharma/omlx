#include "native_engine.h"
#include "scheduler_core.h"
#include "native_ssd_cache.h"
#include "llama_model.h"
#include "metal_ops.h"
#include "mlx/mlx.h"
#include <iostream>
#include <algorithm>
#include <mutex>
#include <map>
#include <unordered_map>
#include <fstream>
#include <sstream>

// Metal Partitioning constants for long-context scaling (from vllm-metal)
constexpr int PARTITION_SIZE = 512;
constexpr int PARTITION_THRESHOLD = 4096;

namespace omlx {

static std::string read_file_to_string(const std::string& path) {
    std::ifstream t(path);
    if (!t.is_open()) return "";
    std::stringstream buffer;
    buffer << t.rdbuf();
    std::string s = buffer.str();
    
    // Strip local #include "..." lines to avoid JIT errors during concatenation
    std::string result;
    std::stringstream ss(s);
    std::string line;
    while (std::getline(ss, line)) {
        if (line.find("#include \"") == std::string::npos) {
            result += line + "\n";
        }
    }
    return result;
}

class NativeEngineImpl : public NativeEngine {
private:
    std::unique_ptr<SchedulerCore> scheduler;
    std::unique_ptr<NativeSSDCache> ssd_cache;
    std::unique_ptr<LlamaModel> model;
    std::map<std::string, std::shared_ptr<NativeRequest>> active_requests;

    // Model state
    std::vector<std::optional<std::pair<mx::array, mx::array>>> global_caches;

    // We serialize MLX calls to prevent Metal command buffer races
    std::mutex mlx_mtx;

public:
    NativeEngineImpl(float soft_limit, float hard_limit, const std::string& cache_dir) {
        scheduler = std::make_unique<SchedulerCore>(soft_limit, hard_limit);

        NativeSSDCache::Config cache_cfg;
        cache_cfg.cache_dir = cache_dir;
        ssd_cache = std::make_unique<NativeSSDCache>(cache_cfg);
        
        // Initialize Metal Kernels
        std::string metal_dir = "src/metal/";
        std::string utils_src = read_file_to_string(metal_dir + "utils.metal");
        std::string float8_src = read_file_to_string(metal_dir + "float8.metal");
        std::string paged_attn_src = read_file_to_string(metal_dir + "pagedattention.metal");
        std::string gdn_src = read_file_to_string(metal_dir + "gdn_linear_attention.metal");
        
        // Simplified source concatenation for JIT
        std::string full_src = "#define VLLM_METAL_PARTITION_SIZE 512\n" + float8_src + "\n" + utils_src + "\n" + paged_attn_src;
        std::string full_gdn = utils_src + "\n" + gdn_src;
        
        MetalOps::instance().init_libraries("", full_src);
        MetalOps::instance().init_v2_library(full_src);
        MetalOps::instance().init_gdn_library(full_gdn);

        // Default args...
        LlamaArgs args;
        args.hidden_size = 4096;
        args.num_hidden_layers = 32;
        args.intermediate_size = 11008;
        args.num_attention_heads = 32;
        args.num_key_value_heads = 32;
        args.vocab_size = 32000;
        args.rms_norm_eps = 1e-6f;
        args.head_dim = 128;

        model = std::make_unique<LlamaModel>(args);
        global_caches.resize(args.num_hidden_layers);

        std::cout << "[NativeEngine] Initialized with cache dir: " << cache_dir << std::endl;
    }

    bool load_model(const std::string& model_path) override {
        std::lock_guard<std::mutex> lock(mlx_mtx);
        try {
            std::cout << "[NativeEngine] Loading model weights from: " << model_path << std::endl;

            // Try to find safetensors in the directory
            std::string st_path = model_path;
            if (st_path.back() != '/') st_path += "/";
            st_path += "model.safetensors";

            auto [weights, metadata] = mx::load_safetensors(st_path);
            model->load_weights(weights);

            // Trigger eval to warm up
            std::vector<mx::array> all_weights;
            for (auto& [k, v] : weights) all_weights.push_back(v);
            mx::eval(all_weights);
            mx::synchronize();

            return true;
        } catch (const std::exception& e) {
            std::cerr << "[NativeEngine] Load failed: " << e.what() << std::endl;
            return false;
        }
    }

    void unload_model() override {
        std::lock_guard<std::mutex> lock(mlx_mtx);
        model->load_weights({});
        global_caches.clear();
        mx::clear_cache();
    }

    void add_request(std::shared_ptr<NativeRequest> request) override {
        std::lock_guard<std::mutex> lock(mlx_mtx);
        active_requests[request->request_id] = request;
        scheduler->waiting_append(request->request_id.c_str(), 0);
    }

    void abort_request(const std::string& request_id) override {
        std::lock_guard<std::mutex> lock(mlx_mtx);
        active_requests.erase(request_id);
        scheduler->abort_enqueue(request_id.c_str());
    }

    std::vector<NativeEngineResult> step() override {
        std::lock_guard<std::mutex> lock(mlx_mtx);
        std::vector<NativeEngineResult> results;

        // 1. Process Pending Aborts
        char abort_id[64];
        while (scheduler->abort_dequeue(abort_id, 64)) {
            std::string id(abort_id);
            if (active_requests.count(id)) {
                active_requests[id]->state = RequestState::ABORTED;
                active_requests[id]->finish_reason = "aborted";
                std::cout << "[NativeEngine] Request aborted: " << id << std::endl;
            }
        }

        // 2. Schedule Waiting Requests
        while (!scheduler->is_pressure_soft() && scheduler->waiting_size() > 0) {
            char next_id[64];
            if (scheduler->waiting_popleft(next_id, 64)) {
                std::string id(next_id);
                if (active_requests.count(id)) {
                    active_requests[id]->state = RequestState::PREFILLING;
                }
            }
        }

        // 3. Inference Step
        // For simplicity in Phase 2, we process one request at a time if they are in PREFILLING or GENERATING
        // Real continuous batching would bundle them.
        for (auto it = active_requests.begin(); it != active_requests.end(); ) {
            auto& req = it->second;

            if (req->state == RequestState::ABORTED || req->state == RequestState::FINISHED) {
                NativeEngineResult res;
                std::strncpy(res.request_id, req->request_id.c_str(), 63);
                res.request_id[63] = '\0';
                res.token = -1;
                res.state = static_cast<int>(req->state);
                std::strncpy(res.finish_reason, req->finish_reason.c_str(), 31);
                res.finish_reason[31] = '\0';
                results.push_back(res);
                it = active_requests.erase(it);
                continue;
            }

            NativeEngineResult res;
            std::strncpy(res.request_id, req->request_id.c_str(), 63);
            res.request_id[63] = '\0';

            if (req->state == RequestState::PREFILLING) {
                int remaining = req->prompt_tokens.size() - req->tokens_processed;
                int chunk = std::min(remaining, req->prefill_chunk_size);
                
                std::vector<int> chunk_tokens(
                    req->prompt_tokens.begin() + req->tokens_processed,
                    req->prompt_tokens.begin() + req->tokens_processed + chunk
                );
                
                mx::array input(chunk_tokens.data(), {1, (int)chunk_tokens.size()}, mx::int32);
                
                // --- Phase 4 Optimization: Partitioned Reduction for Long Context ---
                int current_seq_len = req->tokens_processed + chunk;
                if (current_seq_len > PARTITION_THRESHOLD) {
                    // Logic for partitioned prefill (future implementation)
                    // For now, we still use the model->forward but we'll eventually
                    // switch to direct Metal kernel dispatch here.
                }

                auto logits = model->forward(input, global_caches);
                mx::eval(logits);
                
                req->tokens_processed += chunk;
                if (req->tokens_processed >= (int)req->prompt_tokens.size()) {
                    req->state = RequestState::GENERATING;
                }
                
                res.token = -1;
                res.state = static_cast<int>(RequestState::PREFILLING);
            } 
            else if (req->state == RequestState::GENERATING) {
                int last_token = req->generated_tokens.empty() ? 
                                 req->prompt_tokens.back() : 
                                 req->generated_tokens.back();
                
                mx::array input({last_token}, {1, 1}, mx::int32);
                
                // --- Phase 3 Optimization: Partitioned Reduction for Long Decode ---
                int total_len = req->prompt_tokens.size() + req->generated_tokens.size();
                if (total_len > PARTITION_THRESHOLD) {
                    // In a full implementation, we would bypass model->forward() here
                    // and call our custom partitioned Metal kernel directly to merge 
                    // partial attention results from GPU threadgroups.
                    // This prevents the bottleneck of a single threadgroup reducing 
                    // the entire KV cache.
                }

                auto logits = model->forward(input, global_caches);
                
                // Greedy sampling vs Top-P / Temp
                int next_token = 0;
                
                // Fallback to standard greedy
                if (req->temperature < 1e-5) {
                    auto next_token_arr = mx::argmax(logits, -1);
                    mx::eval(next_token_arr);
                    next_token = next_token_arr.item<int>();
                } else {
                    // Fast Temperature Sampling
                    auto scaled_logits = logits / req->temperature;
                    auto next_token_arr = mx::random::categorical(scaled_logits);
                    mx::eval(next_token_arr);
                    next_token = next_token_arr.item<int>();
                }
                
                req->generated_tokens.push_back(next_token);
                
                if (req->generated_tokens.size() >= (size_t)req->max_tokens) {
                    req->state = RequestState::FINISHED;
                    req->finish_reason = "length";
                }
                
                res.token = next_token;
                res.state = static_cast<int>(RequestState::GENERATING);
            }

            results.push_back(res);
            ++it;
        }

        // 4. Emergency Memory Shedding
        if (scheduler->is_pressure_hard() && !active_requests.empty()) {
            auto newest_it = std::prev(active_requests.end());
            std::string rid = newest_it->first;
            newest_it->second->state = RequestState::WAITING;
            scheduler->waiting_appendleft(rid.c_str(), 0);
            active_requests.erase(newest_it);

            // Clear caches on preemption to free memory
            for (auto& c : global_caches) c = std::nullopt;
            mx::clear_cache();
        }

        mx::synchronize();
        return results;
    }



    int active_requests_count() const override {
        return active_requests.size();
    }

    float get_memory_usage_gb() const override {
        return scheduler->get_current_memory_gb();
    }

    // SSD Cache Access
    NativeSSDCache* get_ssd_cache() {
        return ssd_cache.get();
    }
};

} // namespace omlx

// C-compatible bridge for Python/FFI
extern "C" {
    void* native_engine_create(float soft_limit, float hard_limit, const char* cache_dir) {
        return new omlx::NativeEngineImpl(soft_limit, hard_limit, std::string(cache_dir));
    }

    void native_engine_destroy(void* engine) {
        delete static_cast<omlx::NativeEngineImpl*>(engine);
    }

    int native_engine_step(void* engine, omlx::NativeEngineResult* results, int max_results) {
        auto e = static_cast<omlx::NativeEngineImpl*>(engine);
        auto step_results = e->step();
        
        int count = std::min((int)step_results.size(), max_results);
        for (int i = 0; i < count; ++i) {
            results[i] = step_results[i];
        }
        return count;
    }

    // SSD Cache C-API
    int native_engine_cache_save_block(void* engine, const char* block_id, const mx::array* keys, const mx::array* values) {
        auto e = static_cast<omlx::NativeEngineImpl*>(engine);
        return e->get_ssd_cache()->save_block(std::string(block_id), *keys, *values) ? 1 : 0;
    }

    // Simple add_request for FFI testing
    void native_engine_add_request_simple(void* engine, const char* request_id, int prompt_len) {
        auto e = static_cast<omlx::NativeEngineImpl*>(engine);
        std::vector<int> tokens(prompt_len, 1); // Dummy tokens
        auto req = std::make_shared<omlx::NativeRequest>(std::string(request_id), tokens);
        req->prefill_chunk_size = 128; // Small chunk for testing
        e->add_request(req);
    }
}
