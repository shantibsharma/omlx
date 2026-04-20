import asyncio
import httpx
import time
import statistics
import json

# Configuration
URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen2.5-Coder-14B-Instruct-MLX-4bit"
CONCURRENT_REQUESTS = 10
TOTAL_REQUESTS = 20
PROMPT = "Write a comprehensive summary of the history of artificial intelligence from its early roots to the present day. Include key figures and milestones."
MAX_TOKENS = 128

async def send_request(client, request_id):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "stream": False
    }
    
    start_time = time.perf_counter()
    try:
        response = await client.post(URL, json=payload, timeout=120.0)
        end_time = time.perf_counter()
        
        if response.status_code == 200:
            data = response.json()
            tokens = data["usage"]["completion_tokens"]
            duration = end_time - start_time
            ttft = data["usage"].get("time_to_first_token", 0) # Non-streaming might not have this, but for internal tracking
            
            return {
                "success": True,
                "duration": duration,
                "tokens": tokens,
                "tps": tokens / duration if duration > 0 else 0
            }
        else:
            return {"success": False, "error": f"Status {response.status_code}: {response.text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

async def run_load_test():
    print(f"🔥 Starting cMLX Large Load Test")
    print(f"Model: {MODEL}")
    print(f"Concurrency: {CONCURRENT_REQUESTS}, Total Requests: {TOTAL_REQUESTS}")
    print(f"Prompt Size: {len(PROMPT.split())} words")
    print(f"Max Tokens per Req: {MAX_TOKENS}")
    print("-" * 40)

    async with httpx.AsyncClient() as client:
        # First request to ensure model is pre-loaded
        print("⏳ Warm-up: Loading model...")
        warmup = await send_request(client, "warmup")
        if not warmup["success"]:
            print(f"❌ Warm-up failed: {warmup['error']}")
            return

        print("🚀 Model loaded. Starting concurrent flood...")
        
        start_test = time.perf_counter()
        
        # Run in batches of CONCURRENT_REQUESTS
        all_results = []
        for i in range(0, TOTAL_REQUESTS, CONCURRENT_REQUESTS):
            tasks = []
            for j in range(CONCURRENT_REQUESTS):
                if i + j < TOTAL_REQUESTS:
                    tasks.append(send_request(client, f"req_{i+j}"))
            
            print(f"  Sending batch {i//CONCURRENT_REQUESTS + 1}...")
            batch_results = await asyncio.gather(*tasks)
            all_results.extend(batch_results)
            
        end_test = time.perf_counter()
        total_duration = end_test - start_test
        
        # Statistics
        successes = [r for r in all_results if r["success"]]
        failures = [r for r in all_results if not r["success"]]
        
        if not successes:
            print("❌ No successful requests.")
            return

        durations = [r["duration"] for r in successes]
        tps_values = [r["tps"] for r in successes]
        total_tokens = sum(r["tokens"] for r in successes)
        
        print("-" * 40)
        print(f"✅ Load Test Complete")
        print(f"Success: {len(successes)}/{TOTAL_REQUESTS}")
        if failures:
            print(f"Failures: {len(failures)}")
            for f in failures[:3]:
                print(f"  - {f['error']}")

        print(f"\n📊 Performance Metrics:")
        print(f"Total Wall Time: {total_duration:.2f}s")
        print(f"Total Tokens Generated: {total_tokens}")
        print(f"Average Throughput (Aggregate): {total_tokens / total_duration:.2f} tokens/s")
        print(f"Average Latency per Request: {statistics.mean(durations):.2f}s")
        print(f"Min Latency: {min(durations):.2f}s")
        print(f"Max Latency: {max(durations):.2f}s")
        print(f"Avg Tokens/s per Stream: {statistics.mean(tps_values):.2f} tokens/s")

if __name__ == "__main__":
    asyncio.run(run_load_test())
