import os
import sys
import torch
import importlib.util
import argparse
import pandas as pd
import time
import copy
import gc
from pathlib import Path

# Help reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set matmul precision for better performance on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

def load_module(path):
    """Dynamically load a Python module from a file path."""
    module_name = os.path.basename(path).replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def benchmark_op(model, inputs, warmup=10, rep=100, device_type='cuda'):
    """
    Benchmarks a model with given inputs.
    Returns average execution time in milliseconds.
    """
    # Warmup
    try:
        with torch.no_grad():
            for _ in range(warmup):
                model(*inputs)
        if device_type == 'cuda':
            torch.cuda.synchronize()
        elif device_type == 'mps':
            torch.mps.synchronize()
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError("OOM during warmup")
    except Exception as e:
        raise RuntimeError(f"Warmup failed: {e}")

    # Timing
    try:
        if device_type == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            with torch.no_grad():
                for _ in range(rep):
                    model(*inputs)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = start_event.elapsed_time(end_event)
        elif device_type == 'mps':
            start_time = time.time()
            with torch.no_grad():
                for _ in range(rep):
                    model(*inputs)
            torch.mps.synchronize()
            end_time = time.time()
            elapsed_time_ms = (end_time - start_time) * 1000
        else:
            # CPU timing
            start_time = time.time()
            with torch.no_grad():
                for _ in range(rep):
                    model(*inputs)
            end_time = time.time()
            elapsed_time_ms = (end_time - start_time) * 1000
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError("OOM during benchmark")
    
    avg_time_ms = elapsed_time_ms / rep
    return avg_time_ms

def main():
    parser = argparse.ArgumentParser(description="KernelBench: Eager vs torch.compile(Triton/Inductor)")
    parser.add_argument("--base-dir", type=str, default="KernelBench", help="Path to KernelBench directory")
    parser.add_argument("--levels", nargs='+', default=["level1", "level2", "level3"], help="Levels to run")
    parser.add_argument("--filter", type=str, default="", help="Filter files by name substring")
    parser.add_argument("--output", type=str, default="benchmark_results.csv", help="Output CSV path")
    parser.add_argument("--device", type=int, default=6, help="CUDA device index")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--rep", type=int, default=20, help="Benchmark iterations")
    
    args = parser.parse_args()

    # Device Setup
    if torch.cuda.is_available():
        device_str = f"cuda:{args.device}"
        try:
            device = torch.device(device_str)
            # Test if device index is valid
            torch.cuda.set_device(device)
            device_type = "cuda"
            # Check free memory
            free_mem, total_mem = torch.cuda.mem_get_info(device)
            print(f"Device: {torch.cuda.get_device_name(device)}")
            print(f"Memory: {free_mem / 1024**3:.2f} GB free / {total_mem / 1024**3:.2f} GB total")
        except Exception as e:
            print(f"Warning: CUDA device {args.device} not available ({e}). Falling back to CPU.")
            device = torch.device("cpu")
            device_type = "cpu"
    else:
        print("Warning: CUDA not available. Falling back to CPU.")
        device = torch.device("cpu")
        device_type = "cpu"
    
    print(f"Running benchmark on: {device} (Type: {device_type})")
    
    results = []
    
    if not os.path.exists(args.base_dir):
        print(f"Error: Base directory '{args.base_dir}' does not exist.")
        return

    for level in args.levels:
        level_path = os.path.join(args.base_dir, level)
        if not os.path.exists(level_path):
            print(f"Skipping {level_path}: directory not found.")
            continue
            
        print(f"\nScanning {level_path}...")
        files = sorted([f for f in os.listdir(level_path) if f.endswith(".py") and f != "__init__.py"])
        
        for filename in files:
            if args.filter and args.filter not in filename:
                continue
                
            file_path = os.path.join(level_path, filename)
            print(f"Processing {filename}...", end="", flush=True)
            
            # Aggressive cleanup before each file
            if device_type == "cuda":
                gc.collect()
                torch.cuda.empty_cache()
            
            error_msg = None
            eager_ms = None
            triton_ms = None
            speedup = None
            
            try:
                # 1. Load Module
                module = load_module(file_path)
                
                if not hasattr(module, "Model") or not hasattr(module, "get_inputs"):
                    print(" [SKIP: Missing Model/get_inputs]")
                    continue
                
                # 2. Prepare Init Inputs
                init_inputs = []
                if hasattr(module, "get_init_inputs"):
                    init_args = module.get_init_inputs()
                    if isinstance(init_args, (list, tuple)):
                        init_inputs = init_args
                    else:
                        init_inputs = [init_args]
                
                # 3. Prepare Runtime Inputs
                inputs_raw = module.get_inputs()
                if not isinstance(inputs_raw, (list, tuple)):
                    inputs_raw = [inputs_raw]
                
                # Move inputs to device once
                inputs = []
                try:
                    for x in inputs_raw:
                        if isinstance(x, torch.Tensor):
                            inputs.append(x.to(device))
                        else:
                            inputs.append(x)
                except torch.cuda.OutOfMemoryError:
                    print(" [Setup Fail: OOM]")
                    error_msg = "Setup: OOM"
                    # Cleanup immediately
                    del inputs
                    if 'inputs_raw' in locals(): del inputs_raw
                    if device_type == "cuda":
                        gc.collect()
                        torch.cuda.empty_cache()
                    # Record failure and continue to next file
                    results.append({
                        "Level": level,
                        "File": filename,
                        "Eager_ms": None,
                        "Triton_ms": None,
                        "Speedup": None,
                        "Error": error_msg
                    })
                    continue

                
                # --- PHASE 1: Benchmark Eager ---
                try:
                    # Instantiate specific model for Eager pass
                    model_eager = module.Model(*init_inputs).to(device)
                    model_eager.eval()
                    
                    eager_ms = benchmark_op(model_eager, inputs, warmup=args.warmup, rep=args.rep, device_type=device_type)
                    
                    # Cleanup Eager model to free memory and avoid state leakage
                    del model_eager
                    if device_type == "cuda":
                        gc.collect()
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    if "OOM" in str(e) or isinstance(e, torch.cuda.OutOfMemoryError):
                         print(" [Eager Fail: OOM]")
                         error_msg = "Eager: OOM"
                    else:
                         print(f" [Eager Fail: {e}]")
                         error_msg = f"Eager: {e}"
                    
                    # Try to cleanup
                    if 'model_eager' in locals(): del model_eager
                    if device_type == "cuda":
                        gc.collect()
                        torch.cuda.empty_cache()
                
                # --- PHASE 2: Benchmark Triton (Inductor) ---
                if eager_ms is not None:
                    try:
                        torch._dynamo.reset() # Reset dynamo cache
                        
                        # Re-instantiate a FRESH model for compilation
                        model_triton = module.Model(*init_inputs).to(device)
                        model_triton.eval()
                        
                        # Use 'reduce-overhead' to minimize Python dispatch time (uses CUDA Graphs)
                        # This is CRITICAL for accurate benchmarking of small kernels
                        compiled_model = torch.compile(
                            model_triton, 
                        )
                        
                        # JIT Compilation trigger (Warmup compile)
                        with torch.no_grad():
                            compiled_model(*inputs)
                            
                        triton_ms = benchmark_op(compiled_model, inputs, warmup=args.warmup, rep=args.rep, device_type=device_type)
                        
                        speedup = eager_ms / triton_ms if triton_ms > 0 else 0.0
                        print(f" Done. Eager: {eager_ms:.3f}ms, Triton: {triton_ms:.3f}ms, Speedup: {speedup:.2f}x")
                        
                        # Cleanup Triton model
                        del model_triton
                        del compiled_model
                        
                    except Exception as e:
                        if "OOM" in str(e) or isinstance(e, torch.cuda.OutOfMemoryError):
                            print(" [Triton Fail: OOM]")
                            err_text = "Triton: OOM"
                        else:
                            print(f" [Triton Fail: {e}]")
                            err_text = f"Triton: {e}"

                        if error_msg:
                            error_msg += f" | {err_text}"
                        else:
                            error_msg = err_text
                
            except Exception as e:
                print(f" [Setup Fail: {e}]")
                error_msg = f"Setup: {e}"
            
            finally:
                # Final cleanup after each file
                if 'inputs' in locals(): del inputs
                if 'inputs_raw' in locals(): del inputs_raw
                if device_type == "cuda":
                    gc.collect()
                    torch.cuda.empty_cache()

            results.append({
                "Level": level,
                "File": filename,
                "Eager_ms": eager_ms,
                "Triton_ms": triton_ms,
                "Speedup": speedup,
                "Error": error_msg
            })

    # Save Results
    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"\nBenchmarks completed. Results saved to {args.output}")
        
        print("\nSummary (Top Speedups):")
        # Filter out errors and sort
        valid_results = df.dropna(subset=['Speedup']).sort_values(by='Speedup', ascending=False)
        if not valid_results.empty:
            print(valid_results[['Level', 'File', 'Speedup']].head(10).to_string(index=False))
        else:
            print("No valid speedup data collected.")
    else:
        print("\nNo benchmarks ran.")

if __name__ == "__main__":
    main()
