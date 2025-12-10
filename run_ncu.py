#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This module wraps three tasks:
1) Collect core metrics for specified CUDA kernels with Nsight Compute into CSV (`profile_bench`).
2) Extract and clean those metrics into a DataFrame from the CSV (`load_ncu_metrics`).
3) Convert the metrics table into a string suitable for inclusion in an LLM prompt (`metrics_to_prompt`).

Typical usage:
    from gpu_profile_utils import profile_bench, load_ncu_metrics, metrics_to_prompt

    kernel_names = extract_cuda_kernel_names(test_kernel)
    csv_path = profile_bench(kernel_names=kernel_names)
    df = load_ncu_metrics(csv_path, extra_keep=("Kernel Name",))
    prompt_block = metrics_to_prompt(df)
"""

import os
import re
import sys
import shutil
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Sequence, Union, Any, Dict
import json, math
import pandas as pd
import numpy as np


__all__ = [
    "METRICS",
    "METRIC_COLUMNS",
    "profile_bench",
    "load_ncu_metrics",
    "metrics_to_prompt",
]

# Triton-optimized: Top 4 most actionable metrics
# Each metric directly maps to Triton optimization parameters (BLOCK_M/N/K, num_warps, num_stages)
# Removed redundant result-only metrics (compute util, memory stalls) that can be inferred
METRICS = ",".join([
    # 1. Memory Bandwidth: DRAM throughput utilization
    #    → Triton: BLOCK_M/N/K sizing, data reuse strategy
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",

    # 2. Cache Efficiency: L2 cache hit rate (LTS = L2 Texture cache/shared memory subsystem)
    #    → Triton: Block-level data reuse, computation ordering, grid layout for L2 sharing
    #    Note: L1 cache (l1tex) is hard to control in Triton, L2 is more actionable
    "lts__t_sector_hit_rate.pct",

    # 3. Occupancy: Theoretical occupancy achieved
    #    → Triton: num_warps (2/4/8), block size, register pressure
    "sm__maximum_warps_per_active_cycle_pct",

    # 4. Memory Latency Hiding: Warp stalls due to memory dependency
    #    → Triton: num_stages (1/2/3/4) for software pipelining
    #    High stall rate (>30%) → increase num_stages to hide memory latency
    #    Low stall rate (<10%) → num_stages=1 is sufficient, saves registers
    "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct",
])

# Full metric set (23 metrics) - available but not used by default
METRICS_FULL = ",".join([
    "sm__cycles_active.avg",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__registers_per_thread",
    "sm__inst_executed.sum",
    "sm__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes.sum.per_second",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sector_hit_rate.pct",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "lts__t_sector_hit_rate.pct",
    "lts__throughput.avg.pct_of_peak_sustained_active",
    "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct",
    "smsp__sass_average_branch_targets_threads_uniform.pct",
])


# List version for convenient header selection
METRIC_COLUMNS: List[str] = [s.strip() for s in METRICS.split(",")]


def profile_bench(
    bench_py: str = "bench_ref_inputs.py",
    kernel_names: Optional[List[str]] = None,
    conda_bin: str = "/root/miniconda3/envs/CudaForge/bin",
    out_csv: Union[str, Path] = "ncu_temp.csv",
    repeat: int = 10,
    use_full_metrics: bool = False,  # New: option to use full 23-metric set
    device_idx: Optional[int] = None,  # GPU device index to use
    auto_sudo: bool = True,  # Automatically retry with sudo if permission denied
    ref_file: Optional[str] = None,  # Reference file path (default: ref_0.py)
    test_file: Optional[str] = None,  # Test file path (default: test_kernel_0.py)
) -> Path:
    ncu_bin = shutil.which("ncu") or "/usr/local/cuda/bin/ncu"
    csv_path = Path(out_csv).resolve()

    env = os.environ.copy()
    env["PATH"] = f"{conda_bin}:{env.get('PATH', '')}"

    # Set CUDA_VISIBLE_DEVICES to isolate GPU device
    if device_idx is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(device_idx)
        print(f"[ncu] Using GPU device {device_idx} (CUDA_VISIBLE_DEVICES={device_idx})")

    tmp_ncu_dir = Path.home() / "ncu-tmp"
    tmp_ncu_dir.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(tmp_ncu_dir)
    tmp_ext = tempfile.mkdtemp(prefix="torch_ext_")
    env["TORCH_EXTENSIONS_DIR"] = tmp_ext

    # Choose metric set based on parameter
    metrics_to_use = METRICS_FULL if use_full_metrics else METRICS

    # Build bench script arguments (use provided files or defaults)
    if ref_file is None:
        ref_file_path = Path.cwd() / "ref_0.py"
    else:
        ref_file_path = Path.cwd() / ref_file

    if test_file is None:
        test_file_path = Path.cwd() / "test_kernel_0.py"
    else:
        test_file_path = Path.cwd() / test_file

    cmd = [
        ncu_bin,
        "--csv",
        "--page=raw",
        "--target-processes=all",
        "--replay-mode=kernel",
        "--profile-from-start=on",
        f"--log-file={str(csv_path)}",
        f"--metrics={metrics_to_use}",
        sys.executable, bench_py,
        str(ref_file_path),  # reference model
        str(test_file_path),  # candidate model
        "--repeat", str(repeat),
    ]

    # Choose insertion strategy based on number of kernel names
    if kernel_names:
        names = sorted({k.strip() for k in kernel_names if k and k.strip()})
        if names:
            # Find the --metrics argument (handle both METRICS and METRICS_FULL)
            insert_pos = cmd.index(f"--metrics={metrics_to_use}")
            if len(names) == 1:
                # Single name: direct match
                cmd.insert(insert_pos, f"--kernel-name={names[0]}")
            else:
                # Multiple names: merge into a single regex
                pattern = "|".join(re.escape(k) for k in names)
                cmd.insert(insert_pos, f"--kernel-name=::regex:^({pattern})(\\(|$)")

    print("[ncu] running:", " ".join(cmd))
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True)

    # Print NCU output for debugging
    if proc.stdout:
        print("[ncu stdout]:", proc.stdout[:500])  # Print first 500 chars
    if proc.stderr:
        print("[ncu stderr]:", proc.stderr[:500])  # Print first 500 chars

    # Check if no kernels were profiled (likely due to kernel name mismatch)
    csv_content = csv_path.read_text() if csv_path.exists() else ""
    if "No kernels were profiled" in csv_content and kernel_names:
        print("\n⚠️  [ncu] No kernels were profiled with specified names.")
        print(f"Kernel names specified: {kernel_names}")
        print("Triton may have mangled the kernel names. Retrying without kernel name filter...")

        # Rebuild command without kernel name filter
        cmd_no_filter = [
            ncu_bin,
            "--csv",
            "--page=raw",
            "--kernel-name-base=demangled",
            "--target-processes=all",
            "--replay-mode=kernel",
            "--profile-from-start=on",
            f"--log-file={str(csv_path)}",
            f"--metrics={metrics_to_use}",
            "--launch-skip=0",
            "--launch-count=20",
            sys.executable, bench_py,
            str(ref_file),
            str(test_file),
            "--repeat", str(repeat),
        ]

        print("[ncu] retry running:", " ".join(cmd_no_filter))
        proc = subprocess.run(cmd_no_filter, env=env, text=True, capture_output=True)

        # Print full output for debugging
        if proc.stdout:
            print("[ncu retry stdout]:", proc.stdout)
        if proc.stderr:
            print("[ncu retry stderr]:", proc.stderr)
        print(f"[ncu retry] Return code: {proc.returncode}")

    # Re-read CSV content after retry to check for errors
    csv_content_final = csv_path.read_text() if csv_path.exists() else ""

    if proc.returncode != 0:
        error_msg = f"[ncu] Command failed with return code {proc.returncode}\n"
        sys.stderr.write(error_msg)
        sys.stderr.write(proc.stderr or "")

        # Check if it's a GPU performance counter permission issue
        # NCU writes errors to the CSV file with ==ERROR== prefix, not stderr!
        is_permission_error = (
            "ERR_NVGPUCTRPERM" in csv_content_final or
            "ERR_NVGPUCTRPERM" in (proc.stderr or "") or
            "perf_event_paranoid" in (proc.stderr or "") or
            "permission" in (proc.stderr or "").lower()
        )

        # Retry with sudo if it's a permission issue and auto_sudo is enabled
        if is_permission_error and auto_sudo:
            print("\n⚠️  NCU permission denied. Attempting to retry with sudo...")
            print("You may be prompted for your password.")

            # Build sudo command - use cmd_no_filter if it exists (from retry), otherwise use original cmd
            final_cmd = cmd_no_filter if 'cmd_no_filter' in locals() else cmd

            # Build sudo command
            sudo_cmd = ["sudo", "-E", "env", f"PATH={env.get('PATH', '')}"]

            # Preserve important environment variables
            if device_idx is not None:
                sudo_cmd.append(f"CUDA_VISIBLE_DEVICES={device_idx}")

            sudo_cmd.extend(final_cmd)

            print("[ncu] sudo command:", " ".join(sudo_cmd[:10]) + "...")  # Print first few args

            # Run with sudo (this will prompt for password interactively)
            proc_sudo = subprocess.run(sudo_cmd, env=env, text=True, capture_output=True)

            if proc_sudo.stdout:
                print("[ncu sudo stdout]:", proc_sudo.stdout[:500])
            if proc_sudo.stderr:
                print("[ncu sudo stderr]:", proc_sudo.stderr[:500])

            if proc_sudo.returncode == 0:
                print(f"✓ [ok] NCU succeeded with sudo! CSV written: {csv_path}")
                return csv_path
            else:
                print(f"✗ NCU failed even with sudo (return code {proc_sudo.returncode})")
                sys.stderr.write(proc_sudo.stderr or "")
        else:
            if is_permission_error:
                print("\n⚠️  NCU profiling failed due to permission issues.")
                print("Hint: Run the entire program with sudo:")
                print(f"  sudo -E env PATH=$PATH python main.py ...")

        # Don't exit immediately - let caller handle the error
        # Create an empty CSV to avoid file not found errors
        csv_path.write_text("", encoding="utf-8")
        return csv_path

    print(f"[ok] CSV written: {csv_path}")
    return csv_path



def load_ncu_metrics(
    csv_path: Union[str, Path] = "ncu_temp.csv",
    columns: Optional[Sequence[str]] = None,
    extra_keep: Optional[Sequence[str]] = ("Kernel Name",),
    coerce_numeric: bool = True,
    name_list: Optional[Sequence[str]] = None,  # New: multiple kernel names
    select: str = "last",                       # Selection policy when multiple rows per name
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, comment="=", low_memory=False)

    metric_cols = list(columns) if columns is not None else METRIC_COLUMNS
    keep_cols: List[str] = []
    if extra_keep:
        keep_cols.extend([c for c in extra_keep if c in df.columns])
    keep_cols.extend([c for c in metric_cols if c in df.columns])
    if not keep_cols:
        raise ValueError("No requested columns found in the CSV header.")

    sub = df[keep_cols].copy()

    # Drop the units row
    if len(sub) > 0:
        first_row_str = sub.iloc[0].astype(str).str.lower()
        unit_tokens = ("%", "inst", "cycle", "block", "register", "register/thread")
        if first_row_str.apply(lambda x: any(tok in x for tok in unit_tokens)).any():
            sub = sub.iloc[1:].reset_index(drop=True)

    # Coerce metrics to numeric
    if coerce_numeric:
        metric_in_sub = [c for c in metric_cols if c in sub.columns]
        sub[metric_in_sub] = (
            sub[metric_in_sub]
            .replace({",": "", "%": ""}, regex=True)
            .apply(pd.to_numeric, errors="coerce")
        )

    # ========== Extract by kernel name list ==========
    if name_list:
        results = []
        for name in name_list:
            # Use contains match instead of exact equality
            matched = sub[sub["Kernel Name"].astype(str).str.contains(name, regex=False, na=False)]
            if matched.empty:
                continue
            if len(matched) > 1:
                if select == "first":
                    row = matched.iloc[[0]]
                elif select == "last":
                    row = matched.iloc[[-1]]
                elif select == "max_cycles" and "sm__cycles_active.avg" in matched.columns:
                    row = matched.sort_values("sm__cycles_active.avg", ascending=False).head(1)
                else:
                    row = matched.iloc[[-1]]  # fallback
            else:
                row = matched
            results.append(row)

        if results:
            sub = pd.concat(results, ignore_index=True)
        else:
            sub = pd.DataFrame(columns=keep_cols)

    return sub


def metrics_to_prompt(
    df: pd.DataFrame,
    title: str = "Here are the GPU profiling metrics:",  # Placeholder, not emitted
    key_by: str = "Kernel Name",
    round_digits: Optional[int] = 3,
    compact: bool = False,
    keep_cols: Optional[List[str]] = None,
) -> str:
    """
    Return **only** the data section as a JSON string:
    {
      "<key>": { "<metric>": <value>, ... }  OR
      "<key>": [{...}, {...}]  # list if there are multiple rows for the same key
    }
    If the key column doesn't exist, return a list of rows: [ {col: val, ...}, ... ]
    """

    def _safe(v: Any) -> Any:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        if isinstance(v, (pd.Timestamp, pd.Timedelta, pd.Interval)):
            return str(v)
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, float) and math.isinf(v):
            return "inf" if v > 0 else "-inf"
        if isinstance(v, float) and round_digits is not None:
            return round(v, round_digits)
        return v

    # Empty table
    if df is None or df.empty:
        return "{}"

    cols = list(df.columns)

    # Round numeric columns
    if round_digits is not None:
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            df = df.copy()
            df[num_cols] = df[num_cols].round(round_digits)

    # If key column is missing, return a list of rows
    if key_by not in cols:
        rows = [{k: _safe(v) for k, v in rec.items()} for rec in df.to_dict(orient="records")]
        return json.dumps(rows, ensure_ascii=False, indent=None if compact else 2)

    # Determine value columns
    value_cols = [c for c in cols if c != key_by]
    if keep_cols is not None:
        value_cols = [c for c in value_cols if c in keep_cols]

    data: Dict[str, Any] = {}
    for rec in df[[key_by] + value_cols].to_dict(orient="records"):
        k = str(rec.pop(key_by))
        val_obj = {ck: _safe(cv) for ck, cv in rec.items()}
        if k in data:
            if isinstance(data[k], list):
                data[k].append(val_obj)
            else:
                data[k] = [data[k], val_obj]
        else:
            data[k] = val_obj

    return json.dumps(data, ensure_ascii=False, indent=None if compact else 2)



if __name__ == "__main__":
    # Simple self-check: doesn't force execution; only runs when this file is executed directly.
    # Note: `profile_bench` requires root privileges and an Nsight Compute environment.
    print("gpu_profile_utils module loaded. Import its functions in your main script.")
