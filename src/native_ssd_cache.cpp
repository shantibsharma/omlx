#include "native_ssd_cache.h"
#include <filesystem>
#include <iostream>
#include <fstream>

namespace fs = std::filesystem;

namespace cmlx {

NativeSSDCache::NativeSSDCache(const Config& cfg) : config(cfg) {
    if (!fs::exists(config.cache_dir)) {
        fs::create_directories(config.cache_dir);
    }
    scan_cache_dir();
}

NativeSSDCache::~NativeSSDCache() {}

std::string NativeSSDCache::get_block_path(const std::string& block_id) const {
    // Implement hash-based subdirectory structure: cache_dir/ab/cd/block_id.safetensors
    if (block_id.length() < 4) {
        return (fs::path(config.cache_dir) / (block_id + ".safetensors")).string();
    }
    std::string prefix1 = block_id.substr(0, 2);
    std::string prefix2 = block_id.substr(2, 2);
    fs::path p = fs::path(config.cache_dir) / prefix1 / prefix2;
    return (p / (block_id + ".safetensors")).string();
}

void NativeSSDCache::touch_block_internal(const std::string& block_id) {
    if (lru_map.count(block_id)) {
        lru_list.erase(lru_map[block_id]);
    }
    lru_list.push_back(block_id);
    lru_map[block_id] = --lru_list.end();
}

bool NativeSSDCache::save_block(const std::string& block_id, const mx::array& keys, const mx::array& values) {
    std::string path = get_block_path(block_id);
    fs::path p(path);
    if (!fs::exists(p.parent_path())) {
        fs::create_directories(p.parent_path());
    }

    std::unordered_map<std::string, mx::array> data;
    data.emplace("k", keys);
    data.emplace("v", values);

    try {
        mx::save_safetensors(path, data);
        
        size_t size = fs::file_size(path);
        std::lock_guard<std::mutex> lock(mtx);
        
        if (block_sizes.count(block_id)) {
            current_size_bytes -= block_sizes[block_id];
        }
        
        // Eviction logic
        while (current_size_bytes + size > config.max_cache_size_bytes && !lru_list.empty()) {
            std::string lru_id = lru_list.front();
            std::string lru_path = get_block_path(lru_id);
            if (fs::exists(lru_path)) {
                current_size_bytes -= block_sizes[lru_id];
                fs::remove(lru_path);
            }
            lru_map.erase(lru_id);
            block_sizes.erase(lru_id);
            lru_list.pop_front();
            std::cout << "[NativeSSDCache] LRU Evicted: " << lru_id << std::endl;
        }

        block_sizes[block_id] = size;
        current_size_bytes += size;
        touch_block_internal(block_id);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[NativeSSDCache] Failed to save block " << block_id << ": " << e.what() << std::endl;
        return false;
    }
}

std::pair<mx::array, mx::array> NativeSSDCache::load_block(const std::string& block_id) {
    std::string path = get_block_path(block_id);
    try {
        auto [data, metadata] = mx::load_safetensors(path);
        {
            std::lock_guard<std::mutex> lock(mtx);
            touch_block_internal(block_id);
        }
        return {data.at("k"), data.at("v")};
    } catch (const std::exception& e) {
        std::cerr << "[NativeSSDCache] Failed to load block " << block_id << ": " << e.what() << std::endl;
        return {mx::array(0.0f), mx::array(0.0f)};
    }
}

bool NativeSSDCache::has_block(const std::string& block_id) const {
    return fs::exists(get_block_path(block_id));
}

void NativeSSDCache::evict_block(const std::string& block_id) {
    std::string path = get_block_path(block_id);
    if (fs::exists(path)) {
        size_t size = fs::file_size(path);
        fs::remove(path);
        std::lock_guard<std::mutex> lock(mtx);
        block_sizes.erase(block_id);
        current_size_bytes -= size;
    }
}

void NativeSSDCache::clear() {
    fs::remove_all(config.cache_dir);
    fs::create_directories(config.cache_dir);
    std::lock_guard<std::mutex> lock(mtx);
    block_sizes.clear();
    current_size_bytes = 0;
}

size_t NativeSSDCache::get_current_size_bytes() const {
    std::lock_guard<std::mutex> lock(mtx);
    return current_size_bytes;
}

int NativeSSDCache::get_block_count() const {
    std::lock_guard<std::mutex> lock(mtx);
    return block_sizes.size();
}

void NativeSSDCache::scan_cache_dir() {
    std::lock_guard<std::mutex> lock(mtx);
    block_sizes.clear();
    lru_list.clear();
    lru_map.clear();
    current_size_bytes = 0;

    if (!fs::exists(config.cache_dir)) return;

    struct CacheEntry {
        std::string id;
        size_t size;
        fs::file_time_type mtime;
    };
    std::vector<CacheEntry> entries;

    for (const auto& entry : fs::recursive_directory_iterator(config.cache_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
            entries.push_back({
                entry.path().stem().string(),
                entry.file_size(),
                fs::last_write_time(entry)
            });
        }
    }

    // Sort by modification time to reconstruct LRU order
    std::sort(entries.begin(), entries.end(), [](const CacheEntry& a, const CacheEntry& b) {
        return a.mtime < b.mtime;
    });

    for (const auto& e : entries) {
        block_sizes[e.id] = e.size;
        current_size_bytes += e.size;
        lru_list.push_back(e.id);
        lru_map[e.id] = --lru_list.end();
    }
    
    std::cout << "[NativeSSDCache] Scanned " << block_sizes.size() 
              << " blocks (" << current_size_bytes / (1024*1024) << " MB)" << std::endl;
}

} // namespace cmlx
