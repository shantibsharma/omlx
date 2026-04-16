#include <vector>
#include <unordered_map>
#include <list>
#include <mutex>
#include <algorithm>
#include <ctime>
#include <cstdint>
#include <string>
#include <cstring>
#include <iostream>

/**
 * oMLX Paged Cache Core v2
 * ========================
 * High-performance C++ implementation of KV cache block management.
 * 
 * Features:
 * - O(1) Block allocation and LRU eviction (doubly linked list)
 * - Prefix caching with 256-bit SHA256 hashes
 * - Multi-model cache isolation
 * - Thread-safe operations
 * - Memory pressure awareness
 */

// 256-bit hash represented as 4x64-bit integers
struct BlockHash256 {
    uint64_t h[4];

    bool operator==(const BlockHash256& other) const {
        return h[0] == other.h[0] && h[1] == other.h[1] && 
               h[2] == other.h[2] && h[3] == other.h[3];
    }
};

// Hash function for std::unordered_map
struct BlockHash256Hasher {
    size_t operator()(const BlockHash256& bh) const {
        // Use first 64 bits as hash seed
        return bh.h[0] ^ bh.h[1] ^ bh.h[2] ^ bh.h[3];
    }
};

struct CacheBlock {
    int32_t block_id;
    int32_t ref_count;
    double last_access;
    bool is_null;
    BlockHash256 block_hash;
    bool has_hash;
    bool in_free_list;

    // Doubly linked list pointers for LRU
    CacheBlock* prev_free;
    CacheBlock* next_free;
};

class PagedCacheCore {
private:
    std::vector<CacheBlock> blocks;
    std::unordered_map<BlockHash256, int32_t, BlockHash256Hasher> hash_to_id;
    
    // LRU Free Queue (doubly linked list)
    CacheBlock* head_free;
    CacheBlock* tail_free;
    int32_t num_free;
    
    std::mutex mtx;
    size_t bytes_per_block = 0;
    size_t current_cache_memory_bytes = 0;

    void remove_from_free_queue(CacheBlock* block) {
        if (!block->in_free_list) return;

        if (block->prev_free) block->prev_free->next_free = block->next_free;
        else head_free = block->next_free;
        
        if (block->next_free) block->next_free->prev_free = block->prev_free;
        else tail_free = block->prev_free;
        
        block->prev_free = nullptr;
        block->next_free = nullptr;
        block->in_free_list = false;
        num_free--;
    }

    void push_to_free_queue_tail(CacheBlock* block) {
        if (block->in_free_list) return;

        block->prev_free = tail_free;
        block->next_free = nullptr;
        if (tail_free) tail_free->next_free = block;
        else head_free = block;
        tail_free = block;
        block->in_free_list = true;
        num_free++;
    }

    void move_to_free_queue_tail(CacheBlock* block) {
        if (!block->in_free_list) {
            push_to_free_queue_tail(block);
        } else {
            remove_from_free_queue(block);
            push_to_free_queue_tail(block);
        }
    }

public:
    PagedCacheCore(int32_t max_blocks, int32_t initial_blocks, size_t block_size_bytes) 
        : head_free(nullptr), tail_free(nullptr), num_free(0), 
        bytes_per_block(block_size_bytes), current_cache_memory_bytes(0),
        model_weight_bytes(0) {
        blocks.resize(max_blocks);
        for (int32_t i = 0; i < max_blocks; ++i) {
            blocks[i].block_id = i;
            blocks[i].ref_count = 0;
            blocks[i].last_access = 0;
            blocks[i].is_null = (i == 0);
            blocks[i].has_hash = false;
            blocks[i].in_free_list = false;
            std::memset(&blocks[i].block_hash, 0, sizeof(BlockHash256));
            blocks[i].prev_free = nullptr;
            blocks[i].next_free = nullptr;

            if (i > 0 && i < initial_blocks) { // Block 0 is null_block, reserved
                push_to_free_queue_tail(&blocks[i]);
            }
        }
    }

    int32_t allocate_block() {
        std::lock_guard<std::mutex> lock(mtx);
        if (!head_free) return -1;

        CacheBlock* block = head_free;
        remove_from_free_queue(block);
        block->ref_count = 1;
        block->last_access = static_cast<double>(std::time(nullptr));

        current_cache_memory_bytes += bytes_per_block;

        return block->block_id;
    }

    void free_block(int32_t block_id) {
        std::lock_guard<std::mutex> lock(mtx);
        if (block_id <= 0 || block_id >= static_cast<int32_t>(blocks.size())) return;

        CacheBlock* block = &blocks[block_id];
        block->ref_count--;
        if (block->ref_count <= 0) {
            block->ref_count = 0;
            if (block->has_hash) {
                hash_to_id.erase(block->block_hash);
                block->has_hash = false;
                std::memset(&block->block_hash, 0, sizeof(BlockHash256));
            }
            push_to_free_queue_tail(block);
            current_cache_memory_bytes -= bytes_per_block;
        }
    }

    void register_block(int32_t block_id) {
        std::lock_guard<std::mutex> lock(mtx);
        if (block_id <= 0 || block_id >= static_cast<int32_t>(blocks.size())) return;
        CacheBlock* block = &blocks[block_id];
        if (block->ref_count == 0) {
            push_to_free_queue_tail(block);
        }
    }

    void set_model_weight_bytes(size_t bytes) {
        std::lock_guard<std::mutex> lock(mtx);
        model_weight_bytes = bytes;
    }

    size_t get_total_memory_usage() {
        std::lock_guard<std::mutex> lock(mtx);
        return current_cache_memory_bytes + model_weight_bytes;
    }

    int32_t find_by_hash(const uint64_t* hash_ptr) {
        std::lock_guard<std::mutex> lock(mtx);
        BlockHash256 hash;
        std::memcpy(&hash, hash_ptr, sizeof(BlockHash256));
        
        auto it = hash_to_id.find(hash);
        if (it != hash_to_id.end()) {
            int32_t id = it->second;
            blocks[id].last_access = static_cast<double>(std::time(nullptr));
            return id;
        }
        return -1;
    }

    void set_hash(int32_t block_id, const uint64_t* hash_ptr) {
        std::lock_guard<std::mutex> lock(mtx);
        if (block_id < 0 || block_id >= static_cast<int32_t>(blocks.size())) return;
        
        CacheBlock* block = &blocks[block_id];
        if (block->has_hash) {
            hash_to_id.erase(block->block_hash);
        }
        
        if (hash_ptr) {
            std::memcpy(&block->block_hash, hash_ptr, sizeof(BlockHash256));
            block->has_hash = true;
            hash_to_id[block->block_hash] = block_id;
        } else {
            block->has_hash = false;
            std::memset(&block->block_hash, 0, sizeof(BlockHash256));
        }
    }

    // Touch a block: mark as MRU. If it was free, it remains free but moved to tail.
    void touch_block(int32_t block_id) {
        std::lock_guard<std::mutex> lock(mtx);
        if (block_id < 0 || block_id >= static_cast<int32_t>(blocks.size())) return;
        
        CacheBlock* block = &blocks[block_id];
        block->last_access = static_cast<double>(std::time(nullptr));
        
        if (block->ref_count == 0 && !block->is_null) {
            move_to_free_queue_tail(block);
        }
    }

    // Allocate a specific block (used for prefix cache hits)
    bool allocate_specific_block(int32_t block_id) {
        std::lock_guard<std::mutex> lock(mtx);
        if (block_id <= 0 || block_id >= static_cast<int32_t>(blocks.size())) return false;
        
        CacheBlock* block = &blocks[block_id];
        if (block->ref_count > 0) {
            block->ref_count++;
            return true;
        }
        
        // Block was free, remove from free list
        remove_from_free_queue(block);
        block->ref_count = 1;
        block->last_access = static_cast<double>(std::time(nullptr));
        current_cache_memory_bytes += bytes_per_block;
        return true;
    }

    int32_t get_num_free() const {
        return num_free;
    }

    // Eviction candidate selection: returns a list of block_ids that are free (ref_count 0)
    // in LRU order (limit results by count).
    int32_t get_eviction_candidates(int32_t* out_ids, int32_t count) {
        std::lock_guard<std::mutex> lock(mtx);
        int32_t found = 0;
        CacheBlock* curr = head_free;
        while (curr && found < count) {
            if (!curr->is_null) {
                out_ids[found++] = curr->block_id;
            }
            curr = curr->next_free;
        }
        return found;
    }

private:
    size_t model_weight_bytes;
};

extern "C" {

static PagedCacheCore* g_cache_core = nullptr;

void cache_core_init(int32_t max_blocks, int32_t initial_blocks, size_t block_size_bytes) {
    if (g_cache_core) delete g_cache_core;
    g_cache_core = new PagedCacheCore(max_blocks, initial_blocks, block_size_bytes);
}

void cache_core_register_block(int32_t block_id) {
    if (g_cache_core) g_cache_core->register_block(block_id);
}

int32_t cache_core_allocate() {
    if (!g_cache_core) return -2;
    return g_cache_core->allocate_block();
}

void cache_core_free(int32_t block_id) {
    if (g_cache_core) g_cache_core->free_block(block_id);
}

int32_t cache_core_find_hash(const uint64_t* hash_ptr) {
    if (!g_cache_core) return -1;
    return g_cache_core->find_by_hash(hash_ptr);
}

void cache_core_set_hash(int32_t block_id, const uint64_t* hash_ptr) {
    if (g_cache_core) g_cache_core->set_hash(block_id, hash_ptr);
}

int32_t cache_core_get_free_count() {
    if (!g_cache_core) return 0;
    return g_cache_core->get_num_free();
}

void cache_core_touch(int32_t block_id) {
    if (g_cache_core) g_cache_core->touch_block(block_id);
}

int32_t cache_core_allocate_specific(int32_t block_id) {
    if (!g_cache_core) return 0;
    return g_cache_core->allocate_specific_block(block_id) ? 1 : 0;
}

int32_t cache_core_get_eviction_candidates(int32_t* out_ids, int32_t count) {
    if (!g_cache_core) return 0;
    return g_cache_core->get_eviction_candidates(out_ids, count);
}

void cache_core_set_model_weight_bytes(size_t bytes) {
    if (g_cache_core) g_cache_core->set_model_weight_bytes(bytes);
}

long long cache_core_get_total_usage() {
    if (!g_cache_core) return 0;
    return static_cast<long long>(g_cache_core->get_total_memory_usage());
}

} // extern "C"
