#include <vector>
#include <unordered_map>
#include <string>
#include <mutex>
#include <atomic>
#include <thread>
#include <chrono>
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
 */

class SchedulerCore {
private:
    std::mutex mtx;
    std::atomic<bool> memory_pressure_critical{false};
    float memory_hard_limit_gb;
    float abort_threshold = 0.92f;
    
    std::atomic<bool> stop_monitor{false};
    std::thread monitor_thread;

    void monitor_loop() {
        while (!stop_monitor.load()) {
            size_t active_mem = mx::get_active_memory();
            float active_gb = static_cast<float>(active_mem) / (1024.0f * 1024.0f * 1024.0f);
            
            bool critical = (memory_hard_limit_gb > 0 && active_gb > (memory_hard_limit_gb * abort_threshold));
            memory_pressure_critical.store(critical);
            
            // Sleep for 1ms
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

public:
    SchedulerCore(float hard_limit_gb) : memory_hard_limit_gb(hard_limit_gb) {
        monitor_thread = std::thread(&SchedulerCore::monitor_loop, this);
    }

    ~SchedulerCore() {
        stop_monitor.store(true);
        if (monitor_thread.joinable()) {
            monitor_thread.join();
        }
    }

    void set_hard_limit(float limit_gb) {
        std::lock_guard<std::mutex> lock(mtx);
        memory_hard_limit_gb = limit_gb;
    }

    bool is_pressure_critical() {
        return memory_pressure_critical.load();
    }

    float get_current_memory_gb() {
        return static_cast<float>(mx::get_active_memory()) / (1024.0f * 1024.0f * 1024.0f);
    }

    void gpu_sync() {
        mx::synchronize();
    }
};

static SchedulerCore* g_scheduler_core = nullptr;

extern "C" {

void scheduler_core_init(float hard_limit_gb) {
    if (g_scheduler_core) delete g_scheduler_core;
    g_scheduler_core = new SchedulerCore(hard_limit_gb);
}

int32_t scheduler_core_is_critical() {
    if (!g_scheduler_core) return 0;
    return g_scheduler_core->is_pressure_critical() ? 1 : 0;
}

float scheduler_core_get_memory_gb() {
    if (!g_scheduler_core) return 0.0f;
    return g_scheduler_core->get_current_memory_gb();
}

void scheduler_core_gpu_sync() {
    if (g_scheduler_core) g_scheduler_core->gpu_sync();
}

void scheduler_core_set_limit(float limit_gb) {
    if (g_scheduler_core) g_scheduler_core->set_hard_limit(limit_gb);
}

} // extern "C"
