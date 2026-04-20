/*
 * oMLX Native Runtime — C++ Extension Module
 * ============================================
 * High-performance native code for the oMLX inference server.
 * Optimized for M4 Pro Unified Memory with C++ multi-threading.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <string>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <thread>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <atomic>
#include <CommonCrypto/CommonDigest.h>

#include "mlx/mlx.h"

namespace mx = mlx::core;

extern "C" {

/**
 * Stream a single .safetensors file into Metal Unified Memory.
 *
 * NOTE: Load::eval_gpu is not implemented in MLX because loading
 * happens via CPU mmap/read. We eval on the CPU stream to fault
 * the pages into the Unified Memory pool.
 */
int fast_cache_warmup(const char* file_path) {
    try {
        // Load the graph. We use the CPU stream for the load ops.
        auto [arrays, metadata] = mx::load_safetensors(file_path);

        std::vector<mx::array> to_eval;
        for (auto& [name, arr] : arrays) {
            to_eval.push_back(arr);
        }

        // Evaluate on CPU stream to trigger Page Faults and warm up the cache.
        // Because M4 uses Unified Memory, these warm pages are immediately
        // visible to the GPU without copying.
        mx::eval(to_eval);
        mx::synchronize();
        
        return 1;
    } catch (const std::exception& e) {
        fprintf(stderr, "[omlx_native] warmup failed for %s: %s\n",
                file_path, e.what());
        return 0;
    }
}

/**
 * Load ALL .safetensors files in a directory using C++ threads.
 */
int parallel_warmup_dir(const char* dir_path, int num_threads) {
    try {
        std::vector<std::string> shard_paths;
        DIR* dir = opendir(dir_path);
        if (!dir) return -1;

        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string name(entry->d_name);
            if (name.size() > 12 &&
                name.substr(name.size() - 12) == ".safetensors") {
                shard_paths.push_back(std::string(dir_path) + "/" + name);
            }
        }
        closedir(dir);

        if (shard_paths.empty()) return 0;

        unsigned int hw_threads = std::thread::hardware_concurrency();
        unsigned int pool_size = (num_threads > 0)
            ? static_cast<unsigned int>(num_threads)
            : std::min(hw_threads, static_cast<unsigned int>(shard_paths.size()));

        std::mutex arrays_mutex;
        std::vector<mx::array> all_arrays;
        std::atomic<int> success_count{0};

        auto worker = [&](size_t start, size_t end) {
            for (size_t i = start; i < end; ++i) {
                try {
                    auto [arrays, metadata] = mx::load_safetensors(shard_paths[i]);
                    std::vector<mx::array> local;
                    for (auto& [name, arr] : arrays) {
                        local.push_back(arr);
                    }
                    {
                        std::lock_guard<std::mutex> lock(arrays_mutex);
                        all_arrays.insert(all_arrays.end(), local.begin(), local.end());
                    }
                    success_count.fetch_add(1);
                } catch (...) {
                    // skip failed shards
                }
            }
        };

        std::vector<std::thread> threads;
        size_t shards_per_thread = (shard_paths.size() + pool_size - 1) / pool_size;
        for (unsigned int t = 0; t < pool_size; ++t) {
            size_t start = t * shards_per_thread;
            size_t end = std::min(start + shards_per_thread, shard_paths.size());
            if (start < end) threads.emplace_back(worker, start, end);
        }
        for (auto& t : threads) t.join();

        if (!all_arrays.empty()) {
            // Bulk eval on the default stream to map all pages in parallel.
            mx::eval(all_arrays);
            // Synchronize with the device to ensure all 'completeMemory' callbacks 
            // from IOKit are processed before we drop the arrays and release buffers.
            // This prevents kernel panics on M4 hardware during parallel swaps.
            mx::synchronize();
        }

        return success_count.load();
    } catch (const std::exception& e) {
        fprintf(stderr, "[omlx_native] parallel_warmup_dir failed: %s\n", e.what());
        return -1;
    }
}

long long estimate_model_size(const char* dir_path) {
    DIR* dir = opendir(dir_path);
    if (!dir) return -1;
    long long total = 0;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name(entry->d_name);
        if (name.size() > 12 && name.substr(name.size() - 12) == ".safetensors") {
            struct stat st;
            std::string full_path = std::string(dir_path) + "/" + name;
            if (stat(full_path.c_str(), &st) == 0) total += st.st_size;
        }
    }
    closedir(dir);
    if (total == 0) return -1;
    return static_cast<long long>(total * 1.05); // 5% overhead
}

typedef struct {
    long long total_system_memory;
    long long available_memory;
    long long metal_active_memory;
    long long metal_cache_memory;
    long long metal_peak_memory;
} MemoryStats;

int get_memory_stats(MemoryStats* out) {
    if (!out) return 0;
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    int64_t memsize = 0;
    size_t len = sizeof(memsize);
    if (sysctl(mib, 2, &memsize, &len, nullptr, 0) == 0) out->total_system_memory = memsize;
    try {
        out->metal_active_memory = static_cast<long long>(mx::get_active_memory());
        out->metal_cache_memory = static_cast<long long>(mx::get_cache_memory());
        out->metal_peak_memory = static_cast<long long>(mx::get_peak_memory());
    } catch (...) {}
    out->available_memory = out->total_system_memory - out->metal_active_memory - out->metal_cache_memory;
    if (out->available_memory < 0) out->available_memory = 0;
    return 1;
}

int fast_tensor_count(const char* file_path) {
    try {
        auto [arrays, metadata] = mx::load_safetensors(file_path);
        return static_cast<int>(arrays.size());
    } catch (...) { return -1; }
}

// ---------------------------------------------------------------------------
// FP8 Native Acceleration
// ---------------------------------------------------------------------------

/**
 * Encode a float16/bfloat16 tensor to FP8 E4M3 in-place via MLX Metal.
 *
 * Returns the per-tensor absmax scale as a float. The encoded uint8
 * data is written into `out_fp8_ptr` (caller must allocate n_elements bytes).
 *
 * This bypasses Python iteration entirely — a single C++ call encodes
 * an entire weight matrix or KV cache block.
 */
typedef struct {
    float scale;
    int success;
} FP8EncodeResult;

FP8EncodeResult fp8_encode_tensor(const void* data, int n_elements, int dtype_code, unsigned char* out_fp8) {
    FP8EncodeResult result = {1.0f, 0};
    try {
        mx::Dtype dtype = mx::float32;
        if (dtype_code == 1) dtype = mx::float16;
        else if (dtype_code == 2) dtype = mx::bfloat16;

        // Wrap raw data into an MLX array (zero-copy view)
        // Wrap raw data into an MLX array (zero-copy view)
        mx::array input = (dtype == mx::float16 || dtype == mx::bfloat16) ?
            mx::array(static_cast<const uint16_t*>(data), {n_elements}, dtype) :
            mx::array(static_cast<const float*>(data), {n_elements}, dtype);

        // Compute per-tensor scale: absmax / 448.0 (FP8 E4M3 max)
        auto absmax = mx::max(mx::abs(input));
        mx::eval(absmax);
        float max_val = absmax.item<float>();
        float scale = max_val / 448.0f;
        if (scale < 1e-12f) scale = 1e-12f;

        // Scale and encode to FP8
        auto scaled = input / mx::array(scale);
        auto fp8 = mx::to_fp8(mx::astype(scaled, mx::float16));
        mx::eval(fp8);

        // Copy encoded bytes out
        const uint8_t* fp8_data = fp8.data<uint8_t>();
        std::memcpy(out_fp8, fp8_data, n_elements);

        result.scale = scale;
        result.success = 1;
    } catch (const std::exception& e) {
        fprintf(stderr, "[omlx_native] fp8_encode_tensor failed: %s\n", e.what());
    }
    return result;
}

/**
 * Decode FP8 E4M3 data back to float32 using MLX Metal kernels.
 */
int fp8_decode_tensor(const unsigned char* fp8_data, int n_elements,
                      float scale, int dtype_code, void* out_data) {
    try {
        mx::Dtype dtype = mx::float32;
        if (dtype_code == 1) dtype = mx::float16;
        else if (dtype_code == 2) dtype = mx::bfloat16;

        auto fp8 = mx::array(fp8_data, {n_elements}, mx::uint8);
        auto decoded = mx::from_fp8(fp8, dtype);
        auto result = decoded * mx::array(scale);
        mx::eval(result);

        size_t bytes = n_elements * mx::size_of(dtype);
        if (dtype == mx::float16 || dtype == mx::bfloat16) {
            std::memcpy(out_data, result.data<uint16_t>(), bytes);
        } else {
            std::memcpy(out_data, result.data<float>(), bytes);
        }
        return 1;
    } catch (const std::exception& e) {
        fprintf(stderr, "[omlx_native] fp8_decode_tensor failed: %s\n", e.what());
        return 0;
    }
}

/**
 * Write a safetensors file natively to bypass Python GIL during SSD cache persistence.
 *
 * This function performs pure binary I/O, writing the safetensors header and 
 * concatenated tensor data directly to disk. Because it releases the GIL 
 * (when called via ctypes), it allows the background writer thread to persist 
 * cache blocks without stalling model inference.
 */
int native_save_safetensors(
    const char* path,
    const char* header_json,
    size_t header_len,
    int num_tensors,
    const void** data_ptrs,
    const size_t* data_sizes
) {
    FILE* f = std::fopen(path, "wb");
    if (!f) return -1;

    try {
        // Safetensors format:
        // [8 bytes: header_len as little-endian uint64]
        // [header_json bytes]
        // [tensor data bytes]

        uint64_t h_len = static_cast<uint64_t>(header_len);
        std::fwrite(&h_len, 8, 1, f);
        std::fwrite(header_json, 1, header_len, f);

        for (int i = 0; i < num_tensors; ++i) {
            if (data_ptrs[i] && data_sizes[i] > 0) {
                std::fwrite(data_ptrs[i], 1, data_sizes[i], f);
            }
        }

        std::fclose(f);
        return 1;
    } catch (const std::exception& e) {
        std::fclose(f);
        std::fprintf(stderr, "[omlx_native] native_save_safetensors failed for %s: %s\n",
                     path, e.what());
        return 0;
    } catch (...) {
        std::fclose(f);
        return 0;
    }
}

/**
 * Compute SHA256 block hash natively for better performance than Python str(tuple()).
 * 
 * Uses Apple's CommonCrypto CommonDigest for optimized binary-level hashing.
 * Bypasses the overhead of converting token lists to strings in Python.
 */
int native_compute_block_hash(
    const unsigned char* parent_hash, // 32 bytes or NULL
    const int* token_ids,
    int num_tokens,
    const char* model_name,
    const void* extra_keys_data,
    size_t extra_keys_size,
    unsigned char* out_hash // 32 bytes (allocated by caller)
) {
    CC_SHA256_CTX ctx;
    CC_SHA256_Init(&ctx);

    if (model_name && std::strlen(model_name) > 0) {
        CC_SHA256_Update(&ctx, model_name, std::strlen(model_name));
    }

    if (parent_hash) {
        // Chain with parent block hash
        CC_SHA256_Update(&ctx, parent_hash, 32);
    } else {
        // Root block seed
        CC_SHA256_Update(&ctx, "omlx-root", 9);
    }

    if (num_tokens > 0 && token_ids) {
        // Hash raw integer array directly (O(N) instead of O(Stringify(N)))
        CC_SHA256_Update(&ctx, token_ids, num_tokens * sizeof(int));
    }

    if (extra_keys_data && extra_keys_size > 0) {
        CC_SHA256_Update(&ctx, extra_keys_data, extra_keys_size);
    }

    CC_SHA256_Final(out_hash, &ctx);
    return 1;
}

/**
 * Fast native reader for Safetensors metadata header.
 * 
 * Only reads the minimal necessary bytes from disk to extract the JSON 
 * metadata, bypassing the full MLX loader when only metadata is needed 
 * (e.g. durante cache reconstruction).
 */
int native_read_safetensors_metadata(
    const char* path,
    char* out_json_buffer,
    size_t buffer_size,
    size_t* out_actual_len
) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return -1;

    try {
        uint64_t header_len = 0;
        if (std::fread(&header_len, 8, 1, f) != 1) {
            std::fclose(f);
            return 0;
        }

        if (header_len > buffer_size - 1) {
            std::fclose(f);
            return -2; // Buffer too small
        }

        if (std::fread(out_json_buffer, 1, header_len, f) != header_len) {
            std::fclose(f);
            return 0;
        }

        out_json_buffer[header_len] = '\0';
        if (out_actual_len) *out_actual_len = header_len;

        std::fclose(f);
        return 1;
    } catch (...) {
        std::fclose(f);
        return 0;
    }
}

} // extern "C"

