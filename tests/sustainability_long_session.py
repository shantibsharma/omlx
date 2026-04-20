import asyncio
import httpx
import time
import statistics
import json
import os

# Configuration
URL = "http://localhost:8000/v1/chat/completions"
STATS_URL = "http://localhost:8000/admin/api/stats"
MODEL = "Qwen2.5-Coder-14B-Instruct-MLX-4bit"
CONCURRENT_REQUESTS = 4
ITERATIONS = 30 # Total waves of requests
PROMPT = "Explain the complex mathematical principles behind transformer architectures, specifically focusing on multi-head attention and positional encodings."
MAX_TOKENS = 64

async def get_stats(client):
    try:
        resp = await client.get(STATS_URL)
        if resp.status_code == 200:
            return resp.json()
    except:
        return None
    return None

async def send_request(client):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "stream": False
    }
    try:
        start = time.perf_counter()
        response = await client.post(URL, json=payload, timeout=60.0)
        end = time.perf_counter()
        if response.status_code == 200:
            return {"success": True, "latency": end - start}
    except Exception as e:
        return {"success": False, "error": str(e)}
    return {"success": False, "error": f"Status {response.status_code}"}

async def run_sustainability_test():
    print(f"♾️ Starting cMLX Sustainability & Long-Session Test")
    print(f"Model: {MODEL}")
    print(f"Duration: {ITERATIONS} waves of {CONCURRENT_REQUESTS} requests")
    print("-" * 50)

    async with httpx.AsyncClient() as client:
        # Warm up
        print("⏳ Initializing...")
        await send_request(client)
        
        start_test = time.perf_counter()
        latencies = []
        memory_history = []
        
        for i in range(ITERATIONS):
            tasks = [send_request(client) for _ in range(CONCURRENT_REQUESTS)]
            results = await asyncio.gather(*tasks)
            
            success_count = sum(1 for r in results if r["success"])
            wave_latencies = [r["latency"] for r in results if r["success"]]
            latencies.extend(wave_latencies)
            
            # Monitor memory via admin API
            stats = await get_stats(client)
            mem_active = 0
            if stats:
                mem_active = stats.get("metal_active_memory", 0) / (1024**3)
                memory_history.append(mem_active)
            
            print(f"Wave {i+1}/{ITERATIONS}: Success={success_count}/{CONCURRENT_REQUESTS}, Mem={mem_active:.2f}GB, Avg Latency={statistics.mean(wave_latencies) if wave_latencies else 0:.2f}s")
            
            # Small rest between waves to allow for cache cleanup/GC if needed
            await asyncio.sleep(0.5)

        end_test = time.perf_counter()
        total_duration = end_test - start_test
        
        print("-" * 50)
        print(f"✅ Sustainability Test Complete")
        print(f"Total time: {total_duration:.2f}s")
        print(f"Overall Avg Latency: {statistics.mean(latencies):.2f}s")
        
        if memory_history:
            initial_mem = memory_history[0]
            final_mem = memory_history[-1]
            peak_mem = max(memory_history)
            print(f"Memory Trend: Initial={initial_mem:.2f}GB, Peak={peak_mem:.2f}GB, Final={final_mem:.2f}GB")
            
            if final_mem > initial_mem * 1.1:
                print("⚠️ Warning: Significant memory growth detected.")
            else:
                print("💎 Stability: Memory usage remained stable throughout the session.")

if __name__ == "__main__":
    asyncio.run(run_sustainability_test())
