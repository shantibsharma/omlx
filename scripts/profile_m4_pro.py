import asyncio
import time
import argparse
import sys
import os
import cProfile
import pstats
import io
from pathlib import Path

# Add the project root to sys.path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    import mlx.core as mx
except ImportError:
    print("Error: mlx not found.")
    sys.exit(1)

from cmlx.engine_pool import EnginePool
from cmlx.scheduler import SchedulerConfig
from cmlx.admin.benchmark import _generate_prompt

async def run_profile(engine, prompt, gen_len):
    print(f"Profiling inference with {len(prompt)} chars...")
    
    # We profile the entire stream_generate call
    async for _ in engine.stream_generate(
        prompt=prompt,
        max_tokens=gen_len,
        temperature=0.0
    ):
        pass

async def main():
    parser = argparse.ArgumentParser(description="Profile cMLX on M4 Pro")
    parser.add_argument("--model", type=str, default="Qwen3-Coder-30B-A3B-Instruct-4bit", help="Model ID to profile")
    parser.add_argument("--prompt-len", type=int, default=4096, help="Prompt length to test")
    parser.add_argument("--gen-len", type=int, default=32, help="Generation length")
    
    args = parser.parse_args()
    
    # Initialize EnginePool
    config = SchedulerConfig()
    pool = EnginePool(
        max_model_memory=40 * 1024 * 1024 * 1024,
        scheduler_config=config
    )
    pool.discover_models(model_dirs=[str(Path.home() / ".cmlx" / "models")])
    
    engine = await pool.get_engine(args.model, force_lm=True)
    tokenizer = engine.tokenizer
    
    # Warmup
    print("Warming up...")
    warmup_prompt = _generate_prompt(tokenizer, 128)
    async for _ in engine.stream_generate(prompt=warmup_prompt, max_tokens=16):
        pass
    
    prompt = _generate_prompt(tokenizer, args.prompt_len)
    
    print("\n--- Starting Profiling ---")
    pr = cProfile.Profile()
    pr.enable()
    
    await run_profile(engine, prompt, args.gen_len)
    
    pr.disable()
    print("--- Profiling Complete ---\n")
    
    s = io.StringIO()
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(50)  # Top 50 functions
    
    print(s.getvalue())
    
    # Also save to file
    with open("profile_results_m4_pro.txt", "w") as f:
        f.write(s.getvalue())
        
    await pool._unload_engine(args.model)

if __name__ == "__main__":
    asyncio.run(main())
