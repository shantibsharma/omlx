#include <vector>
#include <unordered_map>
#include <list>
#include <mutex>
#include <algorithm>
#include <ctime>
#include <cstdint>

/**
 * oMLX Paged Cache Core
 * =====================
 * C++ implementation of the KV cache block management metadata.
 * Reduces Python overhead in the high-frequency scheduler loop.
 */

struct CacheBlock {
    int32_t block_id;
    int32_t ref_count;
    double last_access;
    bool is_null;
    uint64_t block_hash; // Simplified hash for O(1) matching

    // Doubly linked list pointers for LRU
    CacheBlock* prev_free;
    CacheBlock* next_free;
};

class PagedCacheCore {
private:
    std::vector<CacheBlock> blocks;
    std::unordered_map<uint64_t, int32_t> hash_to_id;
    
    // LRU Free Queue (doubly linked list)
    CacheBlock* head_free;
    CacheBlock* tail_free;
    int32_t num_free;
    
    std::mutex mtx;

    void remove_from_free_queue(CacheBlock* block) {
        if (block->prev_free) block->prev_free->next_free = block->next_free;
        else head_free = block->next_free;
        
        if (block->next_free) block->next_free->prev_free = block->prev_free;
        else tail_free = block->prev_free;
        
        block->prev_free = nullptr;
        block->next_free = nullptr;
        num_free--;
    }

    void push_to_free_queue_tail(CacheBlock* block) {
        block->prev_free = tail_free;
        block->next_free = nullptr;
        if (tail_free) tail_free->next_free = block;
        else head_free = block;
        tail_free = block;
        num_free++;
    }

public:
    PagedCacheCore(int32_t max_blocks) : head_free(nullptr), tail_free(nullptr), num_free(0) {
        blocks.resize(max_blocks);
        for (int32_t i = 0; i < max_blocks; ++i) {
            blocks[i].block_id = i;
            blocks[i].ref_count = 0;
            blocks[i].last_access = 0;
            blocks[i].is_null = (i == 0);
            blocks[i].block_hash = 0;
            blocks[i].prev_free = nullptr;
            blocks[i].next_free = nullptr;
            
            if (i > 0) { // Block 0 is null_block, reserved
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
        return block->block_id;
    }

    void free_block(int32_t block_id) {
        std::lock_guard<std::mutex> lock(mtx);
        if (block_id <= 0 || block_id >= blocks.size()) return;
        
        CacheBlock* block = &blocks[block_id];
        block->ref_count--;
        if (block->ref_count <= 0) {
            block->ref_count = 0;
            if (block->block_hash != 0) {
                hash_to_id.erase(block->block_hash);
                block->block_hash = 0;
            }
            push_to_free_queue_tail(block);
        }
    }

    int32_t find_by_hash(uint64_t hash) {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = hash_to_id.find(hash);
        if (it != hash_to_id.end()) {
            int32_t id = it->second;
            blocks[id].last_access = static_cast<double>(std::time(nullptr));
            return id;
        }
        return -1;
    }

    void set_hash(int32_t block_id, uint64_t hash) {
        std::lock_guard<std::mutex> lock(mtx);
        if (block_id < 0 || block_id >= blocks.size()) return;
        
        CacheBlock* block = &blocks[block_id];
        if (block->block_hash != 0) {
            hash_to_id.erase(block->block_hash);
        }
        block->block_hash = hash;
        if (hash != 0) {
            hash_to_id[hash] = block_id;
        }
    }

    int32_t get_num_free() const {
        return num_free;
    }
};

// Singleton for simplicity in shared library
static PagedCacheCore* g_cache_core = nullptr;

extern "C" {

void cache_core_init(int32_t max_blocks) {
    if (g_cache_core) delete g_cache_core;
    g_cache_core = new PagedCacheCore(max_blocks);
}

int32_t cache_core_allocate() {
    if (!g_cache_core) return -2;
    return g_cache_core->allocate_block();
}

void cache_core_free(int32_t block_id) {
    if (g_cache_core) g_cache_core->free_block(block_id);
}

int32_t cache_core_find_hash(uint64_t hash) {
    if (!g_cache_core) return -1;
    return g_cache_core->find_by_hash(hash);
}

void cache_core_set_hash(int32_t block_id, uint64_t hash) {
    if (g_cache_core) g_cache_core->set_hash(block_id, hash);
}

int32_t cache_core_get_free_count() {
    if (!g_cache_core) return 0;
    return g_cache_core->get_num_free();
}

}
