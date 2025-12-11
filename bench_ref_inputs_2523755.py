#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified benchmark script for NCU profiling.
Only runs the test kernel without correctness checking.
"""

import sys
import argparse
from pathlib import Path
import importlib.util

import torch
torch.backends.cudnn.enabled = False

def load_module(path: Path, name: str):
    """Dynamically load a Python module"""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--dump", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")

    # Load reference and get inputs
    ref_mod = load_module(args.reference, "ref_module")
    get_inputs = getattr(ref_mod, "get_inputs", None)
    if get_inputs is None:
        raise RuntimeError("Reference must define get_inputs()")

    inputs = get_inputs()
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    # Load test kernel
    test_mod = load_module(args.candidate, "test_module")
    ModelNew = getattr(test_mod, "ModelNew", None)
    if ModelNew is None:
        raise RuntimeError("Candidate must define class ModelNew")

    # Create model and move to device
    model = ModelNew().to(device).eval()
    inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]

    # Warmup
    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = model(*inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Run for profiling (NCU will capture kernels during this)
    with torch.inference_mode():
        for _ in range(args.repeat):
            output = model(*inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    print(f"[bench] Completed {args.repeat} iterations successfully")

if __name__ == "__main__":
    main()
