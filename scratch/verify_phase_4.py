import asyncio
import time
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from cmlx.engine_pool import EnginePool
from cmlx.scheduler import SchedulerConfig
from cmlx.admin.benchmark import _generate_prompt

async def test_memory_stability(engine, duration_sec=60):
    print(f"\n--- Memory Stability Test ({duration_sec}s) ---")
    start_time = time.time()
    iterations = 0
    prompt = _generate_prompt(engine.tokenizer, 4096)
    
    import mlx.core as mx
    initial_mem = mx.get_active_memory()
    print(f"  Initial active memory: {initial_mem / 1e9:.2f} GB")
    
    while time.time() - start_time < duration_sec:
        async for _ in engine.stream_generate(prompt=prompt, max_tokens=128):
            pass
        iterations += 1
        if iterations % 5 == 0:
            print(f"  Iteration {iterations}, active memory: {mx.get_active_memory() / 1e9:.2f} GB")
    
    final_mem = mx.get_active_memory()
    print(f"  Final active memory: {final_mem / 1e9:.2f} GB")
    print(f"  Delta: {(final_mem - initial_mem) / 1e6:.2f} MB")
    print(f"Stability test complete: {iterations} iterations.")

async def test_ssd_latency(engine):
    print("\n--- SSD Swap Latency Test ---")
    # Access the scheduler via the underlying engine core
    scheduler = engine._engine.engine.scheduler
    ssd_manager = scheduler.paged_ssd_cache_manager
    
    if not ssd_manager:
        print("  SSD Manager not configured, skipping latency test.")
        return
        
    import mlx.core as mx
    # Create dummy KV data (list of tensors, e.g. 2 layers)
    dummy_data = [
        mx.zeros((8, 16, 128)), 
        mx.zeros((8, 16, 128))
    ]
    mx.eval(dummy_data)
    test_hash = b"test_latency_hash_v1"
    
    # Measure Save Latency (Enqueue)
    t0 = time.perf_counter()
    ssd_manager.save_block(test_hash, dummy_data, token_count=16)
    t1 = time.perf_counter()
    print(f"  SSD Save Enqueue Latency (16 tokens): {(t1 - t0)*1000:.2f} ms")
    
    # Measure Load Latency
    t0 = time.perf_counter()
    loaded_data = ssd_manager.load_block(test_hash)
    if loaded_data:
        mx.eval(loaded_data)
    t1 = time.perf_counter()
    print(f"  SSD Load Latency (16 tokens): {(t1 - t0)*1000:.2f} ms")
    
    # Clean up
    ssd_manager.delete_block(test_hash)

async def main():
    # Use a temp directory for SSD cache
    ssd_dir = "/tmp/cmlx_verify_ssd_cache"
    os.makedirs(ssd_dir, exist_ok=True)
    
    config = SchedulerConfig(
        max_num_seqs=32,
        paged_ssd_cache_dir=ssd_dir
    )
    pool = EnginePool(
        max_model_memory=40 * 1024**3,
        scheduler_config=config
    )
    pool.discover_models(model_dirs=[str(Path.home() / ".cmlx" / "models")])
    
    model_id = "Qwen3-Coder-30B-A3B-Instruct-4bit"
    print(f"Loading engine {model_id}...")
    engine = await pool.get_engine(model_id, force_lm=True)
    
    try:
        await test_ssd_latency(engine)
        await test_memory_stability(engine, duration_sec=60) 
    finally:
        await pool._unload_engine(model_id)
        # Clean up temp dir
        import shutil
        if os.path.exists(ssd_dir):
            shutil.rmtree(ssd_dir)

if __name__ == "__main__":
    asyncio.run(main())
