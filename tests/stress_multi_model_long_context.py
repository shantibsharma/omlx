import asyncio
import httpx
import time
import random
import statistics
import os

# Configuration
URL = "http://localhost:8000/v1/chat/completions"
STATS_URL = "http://localhost:8000/admin/api/stats"

# We use the three discovered models to trigger swapping
MODELS = [
    "Qwen2.5-Coder-14B-Instruct-MLX-4bit",
    "Qwen3-Coder-30B-A3B-Instruct-4bit",
    "gemma-4-26b-a4b-it-8bit"
]

# Large prompt to stress KV cache (approx 1.5k tokens repeated to make it massive if needed)
BASE_PROMPT = """
Explain the intricate details of the following technical topics in extreme depth. 
Provide code examples, mathematical proofs, and architectural diagrams in text format.
1. Distributed Paged Attention mechanisms in LLM inference.
2. Zero-copy memory mapping for SSD-based KV cache offloading.
3. Metal Performance Shaders and custom kernel optimization for Apple Silicon.
4. The evolution of quantization from INT8 to FP8 and binary weights.
5. Continuous batching and request prioritization in high-concurrency environments.
""" * 5 # ~2000 tokens per request to ensure significant KV cache pressure

CONCURRENCY_PER_MODEL = 2
WAVES = 6 # Total iterations of model switching
MAX_TOKENS = 32

async def get_stats(client):
    try:
        resp = await client.get(STATS_URL)
        if resp.status_code == 200:
            return resp.json()
    except:
        return None
    return None

async def send_request(client, model_id, request_id):
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": f"ID:{request_id} - {BASE_PROMPT}"}],
        "max_tokens": MAX_TOKENS,
        "stream": False
    }
    
    print(f"  [Req {request_id}] Sending to {model_id}...")
    start_time = time.perf_counter()
    try:
        # Long timeout for model loading/swapping
        response = await client.post(URL, json=payload, timeout=300.0)
        end_time = time.perf_counter()
        
        if response.status_code == 200:
            print(f"  [Req {request_id}] Success in {end_time - start_time:.2f}s")
            return {
                "success": True,
                "model": model_id,
                "latency": end_time - start_time
            }
        else:
            print(f"  [Req {request_id}] Failed with status {response.status_code}")
            return {"success": False, "error": f"Status {response.status_code}: {response.text}"}
    except Exception as e:
        print(f"  [Req {request_id}] Error: {str(e)}")
        return {"success": False, "error": str(e)}

async def run_extreme_stress_test():
    print(f"🌋 Starting cMLX Extreme Multi-Model & Long-Context Stress Test")
    print(f"Models: {MODELS}")
    print(f"Context Pressure: ~2,000 tokens per request")
    print(f"Waves: {WAVES} switching cycles")
    print("-" * 60)

    async with httpx.AsyncClient() as client:
        start_test = time.perf_counter()
        all_latencies = []
        
        for wave in range(WAVES):
            # Select a random model to switch to, or cycle through them
            # Cycling is better for testing LRU eviction
            target_model = MODELS[wave % len(MODELS)]
            print(f"🌊 Wave {wave+1}/{WAVES}: Stressing {target_model}...")
            
            tasks = [send_request(client, target_model, f"w{wave}_r{i}") for i in range(CONCURRENCY_PER_MODEL)]
            
            # Start a request for a DIFFERENT model in parallel to trigger memory pressure
            peer_model = MODELS[(wave + 1) % len(MODELS)]
            tasks.append(send_request(client, peer_model, f"w{wave}_peer"))
            
            results = await asyncio.gather(*tasks)
            
            success_count = sum(1 for r in results if r["success"])
            wave_latencies = [r["latency"] for r in results if r["success"]]
            all_latencies.extend(wave_latencies)
            
            # Get system metrics
            stats = await get_stats(client)
            if stats:
                active_mem = stats.get("metal_active_memory", 0) / (1024**3)
                cache_mem = stats.get("metal_cache_memory", 0) / (1024**3)
                ssd_blocks = stats.get("paged_ssd_cache_blocks", 0)
                print(f"  Result: {success_count}/{len(tasks)} OK | Mem: {active_mem:.1f}GB Active, {cache_mem:.1f}GB Cached | SSD Cache: {ssd_blocks} blocks")
            
            # Allow short settling time
            await asyncio.sleep(1)

        end_test = time.perf_counter()
        total_duration = end_test - start_test
        
        print("-" * 60)
        print(f"🏆 Extreme Stress Test Complete")
        print(f"Total time: {total_duration:.2f}s")
        if all_latencies:
            print(f"Avg Latency (inc. load/swap): {statistics.mean(all_latencies):.2f}s")
            print(f"Min Latency: {min(all_latencies):.2f}s | Max Latency: {max(all_latencies):.2f}s")
        else:
            print("❌ No requests succeeded.")
        
        # Check SSD cache growth
        stats = await get_stats(client)
        if stats:
            print(f"🏁 Final SSD Cache State: {stats.get('paged_ssd_cache_blocks', 0)} blocks used")
            print(f"🏁 Final Active Models: {stats.get('active_models', 'N/A')}")

if __name__ == "__main__":
    asyncio.run(run_extreme_stress_test())
