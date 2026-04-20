#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include "mlx/mlx.h"

namespace mx = mlx::core;

namespace omlx {

enum class RequestState {
    WAITING,
    PREFILLING,
    GENERATING,
    FINISHED,
    ABORTED
};

/**
 * NativeRequest
 * Represents a single inference request within the C++ core.
 */
struct NativeRequest {
    std::string request_id;
    std::vector<int> prompt_tokens;
    std::vector<int> generated_tokens;
    
    // Chunked prefill tracking
    int tokens_processed = 0;
    int prefill_chunk_size = 512;
    
    // Sampling parameters
    float temperature = 0.7f;
    float top_p = 1.0f;
    int max_tokens = 512;
    
    // Internal state
    RequestState state = RequestState::WAITING;
    std::string finish_reason = "";
    
    // Cache management
    std::vector<int32_t> block_table;
    
    NativeRequest(const std::string& id, const std::vector<int>& tokens)
        : request_id(id), prompt_tokens(tokens) {}
};

/**
 * NativeEngineResult
 * Captured output from a single engine step.
 * Using fixed-size char arrays for C-compatibility in FFI.
 */
struct NativeEngineResult {
    char request_id[64];
    int token;
    int state; // 0=WAITING, 1=PREFILLING, 2=GENERATING, 3=FINISHED, 4=ABORTED
    char finish_reason[32];
};

/**
 * NativeEngine
 * The C++ core engine that manages the MLX model and request lifecycle.
 */
class NativeEngine {
public:
    virtual ~NativeEngine() = default;

    // Model Lifecycle
    virtual bool load_model(const std::string& model_path) = 0;
    virtual void unload_model() = 0;

    // Request Management
    virtual void add_request(std::shared_ptr<NativeRequest> request) = 0;
    virtual void abort_request(const std::string& request_id) = 0;

    // Core Loop Step
    // Executes one generation step for all active requests in the batch.
    virtual std::vector<NativeEngineResult> step() = 0;

    // Stats & Monitoring
    virtual int active_requests_count() const = 0;
    virtual float get_memory_usage_gb() const = 0;
};

} // namespace omlx

// C-API Export
extern "C" {
    void* native_engine_create(float soft_limit, float hard_limit, const char* cache_dir);
    void native_engine_destroy(void* engine);
    int native_engine_step(void* engine, omlx::NativeEngineResult* results, int max_results);
    void native_engine_add_request_simple(void* engine, const char* request_id, int prompt_len);
}
