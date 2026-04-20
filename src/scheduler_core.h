#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <mutex>
#include <atomic>
#include <deque>
#include <unordered_set>
#include <thread>

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

    std::deque<RequestTrack> waiting_queue;

    std::mutex abort_mtx;
    std::unordered_set<std::string> pending_aborts;

    void monitor_loop();

public:
    SchedulerCore(float soft_limit, float hard_limit);
    ~SchedulerCore();

    void set_limits(float soft, float hard);
    bool is_pressure_soft();
    bool is_pressure_hard();
    float get_current_memory_gb();
    void gpu_sync();

    void waiting_append(const char* request_id, int priority);
    void waiting_appendleft(const char* request_id, int priority);
    int waiting_popleft(char* out_id, size_t max_len);
    void waiting_remove(const char* request_id);
    int waiting_size();
    void waiting_clear();

    void abort_enqueue(const char* request_id);
    int abort_dequeue(char* out_id, size_t max_len);
    int abort_has_pending();
    int abort_contains(const char* request_id);
    void abort_clear();
};
