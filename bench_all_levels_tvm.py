import os
import sys
import time
import importlib.util
import argparse
import glob
import multiprocessing
import csv
import traceback

# Note: We delay heavy imports (torch, tvm) to the worker process 
# or ensure they are used safely with 'spawn'.

def get_torch_tvm_imports():
    """
    Helper to import torch and tvm inside the worker process
    to avoid context contamination if using fork (though we try to use spawn).
    """
    import torch
    import numpy as np
    import gc
    import builtins
    
    try:
        import tvm
        from tvm import relax, runtime
        from tvm.relax.frontend.torch import from_fx
    except ImportError:
        raise ImportError("TVM (Unity) not found.")
        
    return torch, tvm, relax, runtime, from_fx, builtins, np, gc

def to_tvm_array(x, dev, tvm, runtime):
    """
    Convert numpy / torch tensor to TVM Tensor/NDArray on dev.
    """
    import numpy as _np
    
    # Convert to numpy, ensure C contiguous + float32
    x = _np.array(x, copy=False)
    if not x.flags["C_CONTIGUOUS"]:
        x = _np.ascontiguousarray(x)
    
    if x.dtype != _np.float32 and x.dtype != _np.int64 and x.dtype != _np.int32:
         x = x.astype("float32")

    rt = runtime

    # 0) New interface: tvm.runtime.tensor
    if hasattr(rt, "tensor"):
        try:
            return rt.tensor(x, device=dev)
        except TypeError:
            return rt.tensor(x, dev)

    # 1) Class interface: tvm.runtime.Tensor.from_numpy(...)
    TensorCls = getattr(rt, "Tensor", None)
    if TensorCls is not None:
        if hasattr(TensorCls, "from_numpy"):
            try:
                return TensorCls.from_numpy(x, dev)
            except TypeError:
                try:
                    return TensorCls.from_numpy(x)
                except TypeError:
                    pass
        try:
            return TensorCls(x, dev)
        except TypeError:
            try:
                return TensorCls(x)
            except TypeError:
                pass

    # 2) Old interface: tvm.nd.array(x, dev)
    if hasattr(tvm, "nd") and hasattr(tvm.nd, "array"):
        return tvm.nd.array(x, dev)

    # 3) Even older: tvm.runtime.ndarray.array(x, dev)
    ndarray_mod = getattr(rt, "ndarray", None)
    if ndarray_mod is not None and hasattr(ndarray_mod, "array"):
        return ndarray_mod.array(x, dev)

    raise RuntimeError("Could not create TVM NDArray/Tensor. Check TVM installation.")

def load_module_worker(path):
    """Dynamically load a Python module from a file path."""
    module_name = os.path.basename(path).replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def benchmark_eager(model, inputs, warmup, rep, device, torch):
    """Benchmark PyTorch Eager mode."""
    if device == "cuda":
        torch.cuda.synchronize()

    try:
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                model(*inputs)

        if device == "cuda":
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.time()

        # Run
        with torch.no_grad():
            for _ in range(rep):
                model(*inputs)

        if device == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
        else:
            elapsed_ms = (time.time() - start_time) * 1000

        return elapsed_ms / rep
    except Exception as e:
        print(f" [PyTorch Error] {e}")
        return None

def benchmark_tvm_relax(vm, inputs_tvm, dev, warmup, rep):
    """Benchmark TVM Relax VirtualMachine."""
    try:
        # Warmup
        for _ in range(warmup):
            vm["main"](*inputs_tvm)

        dev.sync()

        # Timing
        timer = vm.module.time_evaluator("main", dev, number=rep, repeat=1)
        prof_res = timer(*inputs_tvm)
        return prof_res.mean * 1000.0  # ms
    except Exception as e:
        print(f" [TVM Error] {e}")
        return None

def worker_entry_point(file_path, device_str, result_queue):
    """
    Worker function to run in a separate process.
    Executes the benchmark and puts the result dict into result_queue.
    """
    eager_ms = "N/A"
    tvm_ms = "N/A"
    speedup = "N/A"
    status_msg = "Success"
    
    try:
        # Import inside process
        torch, tvm, relax, runtime, from_fx, builtins, np, gc = get_torch_tvm_imports()
        
        module = load_module_worker(file_path)
        
        # Prepare init inputs
        init_inputs = module.get_init_inputs() if hasattr(module, "get_init_inputs") else []
        if not isinstance(init_inputs, (list, tuple)):
            init_inputs = [init_inputs]

        # Setup Devices
        if device_str == "cuda" and torch.cuda.is_available():
            torch_dev = torch.device("cuda")
            tvm_target = "cuda"
            try:
                tvm_dev = tvm.cuda(0)
            except:
                tvm_dev = runtime.device("cuda", 0)
        else:
            torch_dev = torch.device("cpu")
            tvm_target = "llvm"
            tvm_dev = tvm.cpu(0)

        # Build PyTorch Model
        model = module.Model(*init_inputs).to(torch_dev)
        model.eval()

        # Prepare Inputs
        inputs_raw = module.get_inputs()
        if not isinstance(inputs_raw, (list, tuple)):
            inputs_raw = [inputs_raw]
        inputs = [x.to(torch_dev) if isinstance(x, torch.Tensor) else x for x in inputs_raw]

        # 1. Benchmark Eager
        res_eager = benchmark_eager(model, inputs, 10, 100, device_str, torch)
        if res_eager is None:
            status_msg = "Eager Benchmark Failed"
        else:
            eager_ms = res_eager

        # === AGGRESSIVE MEMORY CLEANUP BEFORE TVM ===
        # Move inputs to CPU to free GPU memory
        inputs_cpu = [x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in inputs]
        
        # Delete GPU tensors
        del inputs
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # ============================================

        # 2. Compile with TVM Relax + DLight
        from torch import fx
        
        # Re-load model on CPU for tracing (to avoid GPU OOM during build if model is huge)
        # Or just trace a small dummy model? 
        # Actually, we need to rebuild the model for FX tracing.
        # Since we deleted 'model', let's re-instantiate it on CPU or GPU?
        # Ideally CPU to be safe, but FX might need it to be on the same device as inputs?
        # Usually FX tracing works on CPU models.
        
        # Re-instantiate model for FX tracing (CPU)
        try:
             # We need original init inputs again. 
             # Assuming get_init_inputs returns fresh ones or we can re-call it.
            init_inputs_cpu = module.get_init_inputs() if hasattr(module, "get_init_inputs") else []
            if not isinstance(init_inputs_cpu, (list, tuple)):
                init_inputs_cpu = [init_inputs_cpu]
            
            # Map init inputs to CPU if they are tensors (unlikely for init args but possible)
            # Usually init args are hyperparameters.
            
            model_for_trace = module.Model(*init_inputs_cpu) # Default device (CPU)
            model_for_trace.eval()
            
            # Prepare CPU inputs for tracing
            # We already have inputs_cpu
            inputs_for_trace = [torch.from_numpy(x) if isinstance(x, np.ndarray) else x for x in inputs_cpu]
            
            graph_module = fx.symbolic_trace(model_for_trace)
        except Exception as trace_err:
             raise RuntimeError(f"FX Tracing failed: {trace_err}")

        # === FIX: Replace .T attribute ===
        changed = False
        for node in graph_module.graph.nodes:
            is_getattr = False
            if node.op == 'call_function':
                if node.target is builtins.getattr:
                    is_getattr = True
                elif getattr(node.target, '__name__', '') == 'getattr':
                    is_getattr = True
            
            if (is_getattr and len(node.args) == 2 and node.args[1] == 'T'):
                # print(f"  [Fix] Replacing .T node: {node.name}")
                with graph_module.graph.inserting_after(node):
                    new_node = graph_module.graph.call_function(
                        torch.transpose, 
                        args=(node.args[0], -2, -1)
                    )
                node.replace_all_uses_with(new_node)
                changed = True
        
        if changed:
            graph_module.graph.lint()
            graph_module.recompile()
        # =================================

        # Prepare Input Info
        input_info = []
        for inp in inputs_cpu:
            if isinstance(inp, np.ndarray):
                shape = list(inp.shape)
                dtype = str(inp.dtype) # numpy dtype
                input_info.append((shape, dtype))
            else:
                pass
        
        # Import to Relax IR
        mod = from_fx(graph_module, input_info)
        target = tvm.target.Target(tvm_target)
        
        # DLight / Build logic
        ex = None
        if tvm_target == "cuda":
            try:
                import tvm.dlight as dl
                pipeline = tvm.transform.Sequential([
                    relax.transform.LegalizeOps(),
                    relax.transform.AnnotateTIROpPattern(),
                    relax.transform.FoldConstant(),
                    relax.transform.FuseOps(),
                    relax.transform.FuseTIR(),
                    dl.ApplyDefaultSchedule(
                        dl.gpu.Matmul(),
                        dl.gpu.GEMV(),
                        dl.gpu.Reduction(),
                        dl.gpu.GeneralReduction(),
                        dl.gpu.Fallback(),
                    ),
                ])
                with target, tvm.transform.PassContext(opt_level=3):
                    mod = pipeline(mod)
                    ex = relax.build(mod, target=target)
            except Exception as e:
                # print(f"  [Warning] DLight failed: {e}")
                with tvm.transform.PassContext(opt_level=3):
                    ex = relax.build(mod, target=target)
        else:
            with tvm.transform.PassContext(opt_level=3):
                ex = relax.build(mod, target=target)
        
        # Create VM
        vm = relax.VirtualMachine(ex, tvm_dev)
        
        # Prepare Inputs for TVM (Upload from CPU to GPU)
        inputs_tvm = [to_tvm_array(inp, tvm_dev, tvm, runtime) for inp in inputs_cpu if isinstance(inp, np.ndarray)]
        
        # Benchmark TVM
        res_tvm = benchmark_tvm_relax(vm, inputs_tvm, tvm_dev, 10, 100)
        
        if res_tvm is not None:
            tvm_ms = res_tvm
            if isinstance(eager_ms, (int, float)):
                speedup = eager_ms / tvm_ms if tvm_ms > 0 else 0
        else:
            if status_msg == "Success":
                status_msg = "TVM Run Failed"
            else:
                status_msg += " | TVM Run Failed"

    except Exception as e:
        # print(f" [Error in Worker] {e}")
        # traceback.print_exc()
        if status_msg == "Success":
            status_msg = f"Error: {str(e)}"
        else:
            status_msg += f" | Error: {str(e)}"
    
    # Send results back
    result_queue.put({
        "eager_ms": eager_ms,
        "tvm_ms": tvm_ms,
        "speedup": speedup,
        "status": status_msg
    })

def main():
    parser = argparse.ArgumentParser(description="Benchmark KernelBench Levels with TVM Relax + DLight (Multiprocess)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--base-dir", type=str, default="KernelBench", help="Base directory of KernelBench")
    parser.add_argument("--output", type=str, default="tvm_benchmark_results.csv", help="Output CSV file path")
    args = parser.parse_args()
    
    # Try to set start method to spawn for clean CUDA context
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Initialize output file
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Level", "File", "Eager Time (ms)", "TVM Time (ms)", "Speedup", "Status"])
        
        levels = ["level1", "level2", "level3"]
        
        for level in levels:
            level_path = os.path.join(args.base_dir, level)
            if not os.path.exists(level_path):
                print(f"Skipping {level_path}, not found.")
                continue
                
            print(f"\nScanning {level_path}...")
            files = sorted(glob.glob(os.path.join(level_path, "*.py")))
            
            for file_path in files:
                if "__init__.py" in file_path:
                    continue
                
                print(f"Processing: {file_path}")
                
                # Create Queue and Process
                queue = multiprocessing.Queue()
                p = multiprocessing.Process(target=worker_entry_point, args=(file_path, args.device, queue))
                
                p.start()
                p.join(timeout=600) # 10 minute timeout per kernel
                
                result = None
                if p.is_alive():
                    print("  [Timeout] Terminating process...")
                    p.terminate()
                    p.join()
                    status_msg = "Timeout"
                    eager_ms, tvm_ms, speedup = "N/A", "N/A", "N/A"
                else:
                    if p.exitcode != 0:
                        status_msg = f"Process Crashed (Exit Code: {p.exitcode})"
                        eager_ms, tvm_ms, speedup = "N/A", "N/A", "N/A"
                    else:
                        if not queue.empty():
                            result = queue.get()
                            eager_ms = result["eager_ms"]
                            tvm_ms = result["tvm_ms"]
                            speedup = result["speedup"]
                            status_msg = result["status"]
                        else:
                            status_msg = "No Result returned"
                            eager_ms, tvm_ms, speedup = "N/A", "N/A", "N/A"

                # Print result to console
                if result:
                    if isinstance(tvm_ms, (int, float)):
                         print(f"  Result: Eager={eager_ms:.4f}ms, TVM={tvm_ms:.4f}ms, Speedup={speedup:.2f}x")
                    else:
                         print(f"  Result: {status_msg}")
                else:
                    print(f"  Result: {status_msg}")

                # Write to CSV
                writer.writerow([
                    level, 
                    os.path.basename(file_path), 
                    f"{eager_ms:.4f}" if isinstance(eager_ms, float) else eager_ms,
                    f"{tvm_ms:.4f}" if isinstance(tvm_ms, float) else tvm_ms,
                    f"{speedup:.2f}" if isinstance(speedup, float) else speedup,
                    status_msg
                ])
                f.flush()

    print(f"\nAll benchmarks completed. Results saved to {args.output}")

if __name__ == "__main__":
    main()
