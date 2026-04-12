import asyncio
import time
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from omlx.engine_pool import EnginePool
from omlx.scheduler import SchedulerConfig

async def main():
    config = SchedulerConfig(
        max_num_seqs=32,
        prefill_step_size=512
    )
    pool = EnginePool(
        max_model_memory=40 * 1024**3,
        scheduler_config=config
    )
    pool.discover_models(model_dirs=[str(Path.home() / ".omlx" / "models")])
    
    model_id = "Qwen3-Coder-30B-A3B-Instruct-4bit"
    engine = await pool.get_engine(model_id, force_lm=True)
    
    # Launch 10 parallel long requests
    prompts = [f"Explain quantum physics in detail. Repeat 10 times. {' '.join(['test']*100)}"] * 10
    
    print(f"Launching {len(prompts)} parallel requests to stress memory...")
    
    tasks = []
    for i, p in enumerate(prompts):
        tasks.append(run_request(engine, p, i))
        
    start = time.time()
    await asyncio.gather(*tasks)
    end = time.time()
    
    print(f"Stress test complete in {end - start:.2f}s")
    await pool._unload_engine(model_id)

async def run_request(engine, prompt, idx):
    count = 0
    async for _ in engine.stream_generate(prompt=prompt, max_tokens=512):
        count += 1
        if count % 100 == 0:
            print(f"Request {idx}: {count} tokens")
    print(f"Request {idx} finished: {count} tokens")

if __name__ == "__main__":
    asyncio.run(main())
