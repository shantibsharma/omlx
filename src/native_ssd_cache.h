#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <list>
#include <algorithm>
#include "mlx/mlx.h"

namespace mx = mlx::core;

namespace cmlx {

/**
 * NativeSSDCache
 * Manages offloading and loading of KV cache blocks to/from SSD.
 * Uses safetensors format for compatibility and performance.
 */
class NativeSSDCache {
public:
    struct Config {
        std::string cache_dir;
        size_t max_cache_size_bytes = 100LL * 1024 * 1024 * 1024; // 100GB
        int num_io_threads = 4;
    };

    NativeSSDCache(const Config& config);
    ~NativeSSDCache();

    /**
     * Save a KV block to SSD.
     * block_id: Unique identifier for the block (e.g., hash of tokens).
     * keys: mx::array containing keys.
     * values: mx::array containing values.
     */
    bool save_block(const std::string& block_id, const mx::array& keys, const mx::array& values);

    /**
     * Load a KV block from SSD.
     * Returns a pair of (keys, values).
     */
    std::pair<mx::array, mx::array> load_block(const std::string& block_id);

    /**
     * Check if a block exists on SSD.
     */
    bool has_block(const std::string& block_id) const;

    /**
     * Evict a block from SSD to maintain max size.
     */
    void evict_block(const std::string& block_id);

    /**
     * Clear all cached blocks.
     */
    void clear();

    // Stats
    size_t get_current_size_bytes() const;
    int get_block_count() const;

private:
    Config config;
    mutable std::mutex mtx;
    std::unordered_map<std::string, size_t> block_sizes;
    std::unordered_map<std::string, std::list<std::string>::iterator> lru_map;
    std::list<std::string> lru_list; // Front is LRU, back is MRU
    size_t current_size_bytes = 0;

    std::string get_block_path(const std::string& block_id) const;
    void scan_cache_dir();
    void touch_block_internal(const std::string& block_id);
};

} // namespace cmlx
