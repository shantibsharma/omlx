import asyncio
import httpx
import time
import statistics
import os
import uuid

# Configuration
URL = "http://localhost:8000/v1/chat/completions"
STATS_URL = "http://localhost:8000/admin/api/stats"
MODEL = "Qwen3-Coder-30B-A3B-Instruct-4bit" # The heavy hitter

# Simulation of a codebase (approx 4k tokens per block)
CODEBASE_CONTEXT = """
// Imagine 4000 tokens of complex C++ and Python source code here.
// We are simulating the "Context" that Claude Code sends on every turn.
""" * 50 # Massive shared prefix (~8k-10k tokens)

CONCURRENT_SESSIONS = 3 # Simulating 3 developers using the same server
TURNS_PER_SESSION = 5
MAX_TOKENS = 64

async def get_stats(client):
    try:
        resp = await client.get(STATS_URL)
        if resp.status_code == 200:
            return resp.json()
    except:
        return None
    return None

async def developer_session(client, dev_id):
    session_prefix = f"/* Developer {dev_id} Private Context */\n"
    history = []
    
    print(f"👤 [Dev {dev_id}] Starting session...")
    
    for turn in range(TURNS_PER_SESSION):
        # Simulate growing conversation history
        history.append(f"Turn {turn}: How do I optimize this specific function?")
        
        # Construct the full prompt (Codebase + Session Prefix + History)
        # This stresses the Prefix Cache (shared part) and growing tail (private part)
        full_content = CODEBASE_CONTEXT + session_prefix + "\n".join(history)
        
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": full_content}],
            "max_tokens": MAX_TOKENS,
            "stream": False
        }
        
        print(f"  [Dev {dev_id}] Turn {turn+1}: Sending request ({len(full_content)//4} estimated tokens)...")
        
        start_time = time.perf_counter()
        try:
            response = await client.post(URL, json=payload, timeout=300.0)
            end_time = time.perf_counter()
            
            if response.status_code == 200:
                latency = end_time - start_time
                data = response.json()
                # Check if we hit the cache (latency will be low for the shared part)
                print(f"  [Dev {dev_id}] Turn {turn+1} Success: {latency:.2f}s")
                
                # Record response into history for next turn
                history.append(f"Assistant: {data['choices'][0]['message']['content'][:50]}...")
            else:
                print(f"  [Dev {dev_id}] Turn {turn+1} FAILED: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"  [Dev {dev_id}] Turn {turn+1} ERROR: {str(e)}")
            
        # Random delay between turns (simulating thinking/coding time)
        await asyncio.sleep(2)

async def run_claude_simulation():
    print(f"🚀 Starting cMLX Claude Code Simulation Stress Test")
    print(f"Target Model: {MODEL}")
    print(f"Concurrency: {CONCURRENT_SESSIONS} parallel developer loops")
    print(f"Context Depth: Large shared codebase context (~10k+ tokens)")
    print("-" * 60)

    # Clean up base path for a fresh test if needed
    os.system("rm -rf ./test_claude_base && mkdir -p ./test_claude_base/models")
    # Copy only the target model to save time and prevent accidental multi-model loading
    os.system(f"cp -r ~/.cmlx/models/{MODEL} ./test_claude_base/models/")

    # Start the server in the background
    # We use a strict memory limit to force SSD caching
    print("🛰️ Launching server...")
    server_cmd = f"CMLX_BASE_PATH=$(pwd)/test_claude_base .cmlxvnv/bin/python3 -m cmlx.server --model-dir ./test_claude_base/models --port 8000"
    server_process = await asyncio.create_subprocess_shell(server_cmd)
    
    await asyncio.sleep(20) # Wait for model load

    async with httpx.AsyncClient() as client:
        start_test = time.perf_counter()
        
        # Run all sessions concurrently
        await asyncio.gather(*[developer_session(client, i) for i in range(CONCURRENT_SESSIONS)])
            
        end_test = time.perf_counter()
        
        # Final system state
        stats = await get_stats(client)
        print("-" * 60)
        print(f"🏆 Simulation Complete in {end_test - start_test:.2f}s")
        if stats:
            print(f"📊 Final Metal Mem: {stats.get('metal_active_memory',0)/1024**3:.1f}GB")
            print(f"📊 SSD Blocks Used: {stats.get('paged_ssd_cache_blocks', 0)}")
            print(f"📊 Cache Hit Rate: {stats.get('cache_hit_rate', 0)*100:.1f}%")

    # Cleanup
    server_process.terminate()
    await server_process.wait()
    print("🛑 Server stopped.")

if __name__ == "__main__":
    asyncio.run(run_claude_simulation())
