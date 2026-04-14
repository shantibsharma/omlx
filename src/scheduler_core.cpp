#include <vector>
#include <unordered_map>
#include <string>
#include <mutex>
#include <atomic>
#include <thread>
#include <chrono>
#include <deque>
#include <algorithm>
#include <unordered_set>
#include <cstring>
#include "mlx/mlx.h"

namespace mx = mlx::core;

/**
 * oMLX Scheduler Core v2
 * ======================
 * Native C++ implementation of the scheduling decision logic and memory enforcement.
 * 
 * Features:
 * - High-frequency background memory monitoring (1ms resolution).
 * - Atomic memory pressure state for zero-latency Python checks.
 * - Explicit GPU synchronization for stability on M4 hardware.
 * - Native wait queue tracking mapping to Python's requests via `request_id`.
 */

struct RequestTrack {
    std::string id;
    int priority;
    long arrival_time_ms;
};

class SchedulerCore {
private:
    std::mutex mtx;
    std::atomic<bool> memory_pressure_soft{false};
    std::atomic<bool> memory_pressure_hard{false};
    
    float soft_limit_gb;
    float hard_limit_gb;
    
    float soft_threshold = 0.85f;
    float hard_threshold = 0.95f;
    
    std::atomic<bool> stop_monitor{false};
    std::thread monitor_thread;

    // Native Priority-aware Wait Queue
    std::deque<RequestTrack> waiting_queue;

    // Thread-safe MPSC abort queue (separate lock to avoid contention)
    std::mutex abort_mtx;
    std::unordered_set<std::string> pending_aborts;

    void monitor_loop() {
        while (!stop_monitor.load()) {
            size_t active_mem = mx::get_active_memory();
            float active_gb = static_cast<float>(active_mem) / (1024.0f * 1024.0f * 1024.0f);
            
            bool soft = (soft_limit_gb > 0 && active_gb > (soft_limit_gb * soft_threshold));
            bool hard = (hard_limit_gb > 0 && active_gb > (hard_limit_gb * hard_threshold));
            
            memory_pressure_soft.store(soft);
            memory_pressure_hard.store(hard);
            
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

public:
    SchedulerCore(float soft_limit, float hard_limit) 
        : soft_limit_gb(soft_limit), hard_limit_gb(hard_limit) {
        monitor_thread = std::thread(&SchedulerCore::monitor_loop, this);
    }

    ~SchedulerCore() {
        stop_monitor.store(true);
        if (monitor_thread.joinable()) {
            monitor_thread.join();
        }
    }

    void set_limits(float soft, float hard) {
        std::lock_guard<std::mutex> lock(mtx);
        soft_limit_gb = soft;
        hard_limit_gb = hard;
    }

    bool is_pressure_soft() {
        return memory_pressure_soft.load();
    }

    bool is_pressure_hard() {
        return memory_pressure_hard.load();
    }

    float get_current_memory_gb() {
        return static_cast<float>(mx::get_active_memory()) / (1024.0f * 1024.0f * 1024.0f);
    }

    void gpu_sync() {
        mx::synchronize();
    }

    // Task 3.1: Queue Management
    void waiting_append(const char* request_id, int priority) {
        std::lock_guard<std::mutex> lock(mtx);
        auto now = std::chrono::system_clock::now().time_since_epoch();
        long current_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
        waiting_queue.push_back({std::string(request_id), priority, current_ms});
    }

    void waiting_appendleft(const char* request_id, int priority) {
        std::lock_guard<std::mutex> lock(mtx);
        // Preempted requests maintain their original priority but jump the queue.
        waiting_queue.push_front({std::string(request_id), priority, 0}); // 0 arrival guarantees first processing
    }

    int waiting_popleft(char* out_id, size_t max_len) {
        std::lock_guard<std::mutex> lock(mtx);
        if (waiting_queue.empty()) return 0;
        std::string highest_id = waiting_queue.front().id;
        waiting_queue.pop_front();
        std::strncpy(out_id, highest_id.c_str(), max_len - 1);
        out_id[max_len - 1] = '\0';
        return 1;
    }

    void waiting_remove(const char* request_id) {
        std::lock_guard<std::mutex> lock(mtx);
        std::string id(request_id);
        waiting_queue.erase(
            std::remove_if(waiting_queue.begin(), waiting_queue.end(), 
                           [&id](const RequestTrack& r) { return r.id == id; }),
            waiting_queue.end());
    }

    int waiting_size() {
        std::lock_guard<std::mutex> lock(mtx);
        return waiting_queue.size();
    }

    void waiting_clear() {
        std::lock_guard<std::mutex> lock(mtx);
        waiting_queue.clear();
    }

    // --- MPSC Abort Queue ---
    void abort_enqueue(const char* request_id) {
        std::lock_guard<std::mutex> lock(abort_mtx);
        pending_aborts.insert(std::string(request_id));
    }

    int abort_dequeue(char* out_id, size_t max_len) {
        std::lock_guard<std::mutex> lock(abort_mtx);
        if (pending_aborts.empty()) return 0;
        auto it = pending_aborts.begin();
        std::strncpy(out_id, it->c_str(), max_len - 1);
        out_id[max_len - 1] = '\0';
        pending_aborts.erase(it);
        return 1;
    }

    int abort_has_pending() {
        std::lock_guard<std::mutex> lock(abort_mtx);
        return pending_aborts.empty() ? 0 : 1;
    }

    int abort_contains(const char* request_id) {
        std::lock_guard<std::mutex> lock(abort_mtx);
        return pending_aborts.count(std::string(request_id)) ? 1 : 0;
    }

    void abort_clear() {
        std::lock_guard<std::mutex> lock(abort_mtx);
        pending_aborts.clear();
    }
};

static SchedulerCore* g_scheduler_core = nullptr;

extern "C" {

void scheduler_core_init(float soft_limit_gb, float hard_limit_gb) {
    if (g_scheduler_core) delete g_scheduler_core;
    g_scheduler_core = new SchedulerCore(soft_limit_gb, hard_limit_gb);
}

int32_t scheduler_core_is_soft_critical() {
    if (!g_scheduler_core) return 0;
    return g_scheduler_core->is_pressure_soft() ? 1 : 0;
}

int32_t scheduler_core_is_hard_critical() {
    if (!g_scheduler_core) return 0;
    return g_scheduler_core->is_pressure_hard() ? 1 : 0;
}

float scheduler_core_get_memory_gb() {
    if (!g_scheduler_core) return 0.0f;
    return g_scheduler_core->get_current_memory_gb();
}

void scheduler_core_gpu_sync() {
    if (g_scheduler_core) g_scheduler_core->gpu_sync();
}

void scheduler_core_set_limits(float soft_gb, float hard_gb) {
    if (g_scheduler_core) g_scheduler_core->set_limits(soft_gb, hard_gb);
}

// Memory Queue Wrappers
void scheduler_core_waiting_append(const char* request_id, int priority) {
    if (g_scheduler_core) g_scheduler_core->waiting_append(request_id, priority);
}

void scheduler_core_waiting_appendleft(const char* request_id, int priority) {
    if (g_scheduler_core) g_scheduler_core->waiting_appendleft(request_id, priority);
}

int32_t scheduler_core_waiting_popleft(char* out_id, size_t max_len) {
    if (!g_scheduler_core) return 0;
    return g_scheduler_core->waiting_popleft(out_id, max_len);
}

void scheduler_core_waiting_remove(const char* request_id) {
    if (g_scheduler_core) g_scheduler_core->waiting_remove(request_id);
}

int32_t scheduler_core_waiting_size() {
    if (!g_scheduler_core) return 0;
    return g_scheduler_core->waiting_size();
}

void scheduler_core_waiting_clear() {
    if (g_scheduler_core) g_scheduler_core->waiting_clear();
}

// --- Abort Queue Wrappers ---
void scheduler_core_abort_enqueue(const char* request_id) {
    if (g_scheduler_core) g_scheduler_core->abort_enqueue(request_id);
}

int32_t scheduler_core_abort_dequeue(char* out_id, size_t max_len) {
    if (!g_scheduler_core) return 0;
    return g_scheduler_core->abort_dequeue(out_id, max_len);
}

int32_t scheduler_core_abort_has_pending() {
    if (!g_scheduler_core) return 0;
    return g_scheduler_core->abort_has_pending();
}

int32_t scheduler_core_abort_contains(const char* request_id) {
    if (!g_scheduler_core) return 0;
    return g_scheduler_core->abort_contains(request_id);
}

void scheduler_core_abort_clear() {
    if (g_scheduler_core) g_scheduler_core->abort_clear();
}

} // extern "C"
