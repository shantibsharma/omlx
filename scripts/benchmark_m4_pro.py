import asyncio
import time
import argparse
import sys
import os
import json
from pathlib import Path

# Add the project root to sys.path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    import mlx.core as mx
    import numpy as np
except ImportError:
    print("Error: mlx or numpy not found. Please run this script in the correct environment.")
    sys.exit(1)

from cmlx.engine_pool import EnginePool
from cmlx.scheduler import SchedulerConfig
from cmlx.admin.benchmark import _generate_prompt, _run_single_test, _run_batch_test, _compute_single_metrics

async def main():
    parser = argparse.ArgumentParser(description="Baseline Benchmark for cMLX on M4 Pro")
    parser.add_argument("--model", type=str, default="Qwen3-Coder-30B-A3B-Instruct-4bit", help="Model ID to benchmark")
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[1024, 4096, 8192], help="Prompt lengths to test")
    parser.add_argument("--gen-len", type=int, default=128, help="Generation length")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8], help="Batch sizes to test")
    parser.add_argument("--output", type=str, default="benchmark_results_m4_pro.json", help="Output file for results")
    
    args = parser.parse_args()
    
    print(f"Starting baseline benchmark for model: {args.model}")
    print(f"Hardare: {mx.device_info()}")
    
    # Initialize EnginePool
    config = SchedulerConfig()
    # M4 Pro specific tweaks if needed
    pool = EnginePool(
        max_model_memory=40 * 1024 * 1024 * 1024, # 40GB limit for 48GB machine
        scheduler_config=config
    )
    pool.discover_models(model_dirs=[str(Path.home() / ".cmlx" / "models")])
    
    try:
        engine = await pool.get_engine(args.model, force_lm=True)
        print(f"Model loaded: {args.model}")
    except Exception as e:
        print(f"Failed to load engine: {e}")
        return

    results = {
        "model": args.model,
        "hardware": mx.device_info(),
        "single_request": [],
        "batch_inference": []
    }
    
    tokenizer = engine.tokenizer
    
    # Warmup
    print("Warming up...")
    warmup_prompt = _generate_prompt(tokenizer, 128)
    async for _ in engine.stream_generate(prompt=warmup_prompt, max_tokens=16):
        pass
    print("Warmup complete.")
    
    # Single Request Benchmarks
    print("\n--- Single Request Benchmarks ---")
    for pp_len in args.prompt_lengths:
        print(f"Testing Prompt Length: {pp_len}...")
        prompt = _generate_prompt(tokenizer, pp_len)
        
        metrics = await _run_single_test(
            engine=engine,
            prompt=prompt,
            max_tokens=args.gen_len,
            pp_len=pp_len
        )
        
        print(f"  TTFT: {metrics['ttft_ms']}ms")
        print(f"  Gen TPS: {metrics['gen_tps']}")
        print(f"  PP TPS: {metrics['processing_tps']}")
        
        results["single_request"].append({
            "prompt_length": pp_len,
            "metrics": metrics
        })
        
    # Batch Benchmarks
    print("\n--- Batch Inference Benchmarks ---")
    # Use 1024 as baseline prompt length for batches
    batch_base_len = 1024
    max_batch = max(args.batch_sizes)
    batch_prompts = [_generate_prompt(tokenizer, batch_base_len) for _ in range(max_batch)]
    
    for bs in args.batch_sizes:
        if bs == 1:
            # We already have bs=1 from single request if 1024 was included
            continue
            
        print(f"Testing Batch Size: {bs}...")
        batch_metrics = await _run_batch_test(
            engine=engine,
            prompts=batch_prompts[:bs],
            prompt_tokens=batch_base_len,
            max_tokens=args.gen_len,
            batch_size=bs
        )
        
        print(f"  Avg TTFT: {batch_metrics['avg_ttft_ms']}ms")
        print(f"  TG TPS (Aggregate): {batch_metrics['tg_tps']}")
        print(f"  PP TPS (Aggregate): {batch_metrics['pp_tps']}")
        
        results["batch_inference"].append({
            "batch_size": bs,
            "metrics": batch_metrics
        })
        
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nResults saved to {args.output}")
    
    # Clean up
    await pool._unload_engine(args.model)

if __name__ == "__main__":
    asyncio.run(main())
