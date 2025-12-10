# main.py
from __future__ import annotations
import argparse
import re
import random
import time
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from run_ncu import profile_bench, load_ncu_metrics, metrics_to_prompt
import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt

from agents.query_server import query_server
from prompts.generate_custom_cuda import build_seed_prompt, default_system_prompt
from utils.compile_and_run import compare_and_bench
from utils.kernel_io import extract_code_block, save_kernel_code, extract_json, extract_cuda_kernel_names
from scripts.individual import KernelIndividual  # adjust path if needed
from prompts.error import build_error_prompt
from prompts.optimization import build_optimization_prompt
from prompts.judger_repair import build_correctness_prompts
from prompts.judger_optimization import build_judger_optimization_prompts
_INVOCATION_SPLITTER = "Invoked with:"

def _sanitize_error_message(exc: Exception) -> str:
    """Strip pybind's large‑tensor printouts and keep only the key error text."""
    msg = str(exc)
    if _INVOCATION_SPLITTER in msg:
        msg = msg.split(_INVOCATION_SPLITTER, 1)[0].rstrip()
    return msg

# ------------------------- CLI -------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Single-LLM self-iterative kernel generation/optimization")
    p.add_argument(
        "arch_py",
        type=Path,
        help="Path to a single task .py file OR a directory containing many tasks (.py)",
    )
    p.add_argument("--gpu", default="Quadro RTX 6000", help="GPU name in prompt spec")
    p.add_argument("--server_type", default="local", help="LLM provider (local, openai, deepseek, vllm, etc.)")
    p.add_argument("--server_address", default="localhost", help="LLM server address (for vllm/sglang)")
    p.add_argument("--server_port", type=int, default=8000, help="LLM server port (for vllm/sglang)")
    p.add_argument("--model_name", default="deepseek", help="LLM model")
    p.add_argument("--round", "-G", type=int, default=10, help="Number of generations per task")
    p.add_argument("--work_dir", type=Path, default=Path("run"), help="Output root directory")
    p.add_argument("--device", type=int, default=0, help="CUDA device index for benchmarking")
    p.add_argument("--warmup", type=int, default=5, help="Warm-up iterations")
    p.add_argument("--repeat", type=int, default=20, help="Timed iterations per benchmark")
    p.add_argument("--tol", type=float, default=1e-3, help="Max |err| tolerated")
    p.add_argument("--max_tokens", type=int, default=16384, help="LLM max new tokens")
    p.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    p.add_argument("--top_p", type=float, default=1.0, help="LLM top_p")
    # multi-task controls
    p.add_argument("--first_n", type=int, default=0, help="When arch_py is a directory, take the first N tasks (sorted)")
    p.add_argument("--num_tasks", type=int, default=1, help="When sampling, how many tasks to pick (if >0 and first_n=0)")
    p.add_argument("--shuffle_seed", type=int, default=0, help="Random seed for sampling (0 = time)")
    
    p.add_argument("--subproc_id", type=int, default=0, help="Identifier for sub-process (e.g., when running multiple in parallel)")

    # Multi-stage optimization parameters
    p.add_argument("--num_steps", type=int, default=4, help="Number of optimization steps (stages)")
    p.add_argument("--max_repair_attempts", type=int, default=3, help="Max repair attempts per failure")

    return p


# ---------------------- optimization stages configuration -----------------
# Four-stage progressive CUDA optimization strategy (ordered by impact and dependency):
# Stage 1: Grid & Parallel - Foundation (affects occupancy)
# Stage 2: Memory Coalescing - **MOST CRITICAL** (10-100x impact, must fix before tiling)
# Stage 3: Block Tiling - Data reuse (depends on grid config, affects L2)
# Stage 4: Memory Hierarchy - Advanced optimizations (shared memory, L2, double buffering)
#
# **CRITICAL**: Stages must be executed in THIS order to avoid breaking previous optimizations:
# - Stage 2 MUST come before Stage 3 (coalescing breaks if tiling changes access patterns incorrectly)
# - Stage 3 sets tile sizes that Stage 4's shared memory optimizations depend on
OPTIMIZATION_STAGES = [
    {"name": "grid_and_parallel",
     "description": "Optimize grid/block dimensions for SM utilization"},

    {"name": "memory_coalescing",
     "description": "Fix global memory access patterns for coalesced access"},

    {"name": "block_tiling",
     "description": "Tune block tile sizes (BLOCK_M/N/K) for optimal register/shared memory balance and data reuse"},

    {"name": "memory_hierarchy",
     "description": "Optimize memory hierarchy: shared memory (eliminate bank conflicts), L2 cache, and advanced techniques (double buffering, prefetching)"},
]

# ---------------------- early exit criteria per stage -----------------
# Define metric thresholds for skipping optimization stages when metrics are already optimal
# Note: These are suggestions based on empirical data. Adjust based on your workload.
# Comparisons: ">=" for metrics where higher is better, "<=" for metrics where lower is better
STAGE_EXIT_CRITERIA = {
    "grid_and_parallel": {
        "metrics": ["sm__maximum_warps_per_active_cycle_pct"],
        "thresholds": [90.0],  # SM occupancy > 90% -> skip grid optimization
        "comparisons": [">="],  # Higher is better
        "operator": "and",
        "description": "SM occupancy already optimal (>90%)"
    },

    "memory_coalescing": {
        # **NEW STAGE** - CRITICAL for CUDA performance
        # Skip if global memory access is already well-coalesced
        "metrics": ["l1tex__t_sectors_pipe_lsu_mem_global_op_ld.avg.pct_of_peak_sustained_elapsed"],
        "thresholds": [85.0],  # >85% means memory access is well-coalesced
        "comparisons": [">="],  # Higher is better
        "operator": "and",
        "description": "Global memory access already well-coalesced (>85%)"
    },

    "block_tiling": {
        "metrics": ["sm__maximum_warps_per_active_cycle_pct"],
        "thresholds": [85.0],  # Good occupancy suggests block config is decent
        "comparisons": [">="],  # Higher is better
        "operator": "and",
        "description": "Block configuration already efficient (occupancy >85%)"
    },

    "memory_hierarchy": {
        # **MERGED** from memory_access + advanced_memory
        # Skip if ALL of these conditions are met (memory hierarchy already optimal):
        # 1. Low bank conflicts (<100)
        # 2. High L2 cache hit rate (>90%)
        # 3. Low memory stalls (<10%)
        "metrics": [
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",  # Bank conflicts
            "lts__t_sector_hit_rate.pct",  # L2 cache hit rate
            "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct"  # Memory stalls
        ],
        "thresholds": [100.0, 90.0, 10.0],
        "comparisons": ["<=", ">=", "<="],  # Low conflicts, high L2, low stalls
        "operator": "and",  # All conditions must be met
        "description": "Memory hierarchy already near-optimal (conflicts <100, L2 >90%, stalls <10%)"
    }
}

# ---------------------- naming helpers -----------------
def _slugify_tag(text: str, max_len: int = 80) -> str:
    """Collapse a string into a filesystem-friendly slug."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    slug = re.sub(r"_+", "_", slug)
    if max_len > 0:
        slug = slug[:max_len]
    return slug or "unknown"


def _build_run_tag(server_type: str, model_name: str) -> str:
    server_tag = _slugify_tag(server_type)
    model_tag = _slugify_tag(model_name)
    return f"{server_tag}_{model_tag}"


# ---------------------- small utils --------------------
def _last_n_lines(text: str, n: int = 150) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:]) if len(lines) > n else text


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_full_cuda_source(text: str) -> str:
    """Extract CUDA source from a Python or markdown-like file.

    Order:
      1) ```cuda ... ``` fenced code
      2) source = \"\"\" ... \"\"\"
      3) fallback: raw text
    """
    m = re.search(r"```cuda\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"source\s*=\s*([\"']{3})(.*?)(?:\1)", text, flags=re.DOTALL)
    if m:
        return m.group(2).strip()
    return text.strip()


def _build_history_block(code_dir: Path, keep_last: int = 10) -> str:
    """Collect the CUDA `source` of the most recent *keep_last* kernel files from code_dir."""
    if not code_dir.exists():
        return "## Existing kernels\n(None yet)\n"

    files: List[Path] = sorted(
        list(code_dir.glob("*.py")) + list(code_dir.glob("*.cu")),
        key=lambda p: p.stat().st_mtime,
    )[-keep_last:]

    if not files:
        return "## Existing kernels\n(None yet)\n"

    snippets: List[str] = []
    for idx, p in enumerate(files, 1):
        try:
            cuda_src = _extract_full_cuda_source(_read_text(p))
        except Exception:
            cuda_src = "(failed to read/extract)"
        snippets.append(f"### Kernel {idx} · {p.name}\n```cuda\n{cuda_src}\n```")

    return "## Existing kernels\n" + "\n\n".join(snippets) + "\n"


# ------------------- LLM & eval steps ------------------
def _make_llm_caller(args):

    def call_llm(
        prompt: str,
        sys_prompt: Optional[str] = None,
        log_path: Optional[Path] = None,
        call_type: str = "unknown",
        round_idx: int = -1,
    ) -> str:
        sp = default_system_prompt if sys_prompt is None else sys_prompt
        res = query_server(
            prompt=prompt,
            system_prompt=sp,
            server_type=args.server_type,
            model_name=args.model_name,
        max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            server_address=args.server_address,
            server_port=args.server_port,
            log_path=str(log_path) if log_path else None,
            call_type=call_type,
            round_idx=round_idx,
        )
        if isinstance(res, list):
            return res[0] if res else ""
        return str(res)
    return call_llm


def _llm_to_kernel(
    prompt: str,
    code_dir: Path,
    call_llm,
    io_dir: Path,
    round_idx,
    sys_prompt: Optional[str] = None,   # New: optional system prompt
    log_path: Optional[Path] = None,
    call_type: str = "unknown",
    max_retries: int = 1,  # Retry once on truncation
) -> KernelIndividual:
    """LLM → code → save → KernelIndividual (no evaluation)."""

    for retry_idx in range(max_retries + 1):
        # Modify prompt on retry to encourage conciseness
        current_prompt = prompt
        if retry_idx > 0:
            print(f"⚠️  Retry {retry_idx}/{max_retries}: Using concise prompt to avoid truncation...")
            concise_instruction = "\n**CRITICAL: Output Length Limit**\nYour previous response was truncated. Generate a CONCISE implementation with minimal comments.\n\n"
            current_prompt = concise_instruction + prompt

        raw = call_llm(
            current_prompt,
            sys_prompt=sys_prompt,
            log_path=log_path,
            call_type=f"{call_type}_retry{retry_idx}" if retry_idx > 0 else call_type,
            round_idx=round_idx * 100 + retry_idx,
        )

        reply_file = io_dir / f"{round_idx}_raw_reply_retry{retry_idx}.txt"
        reply_file.write_text(raw, encoding="utf-8")

        # Check if output indicates truncation
        is_truncated = "Warning: Output truncated due to max_tokens limit" in raw

        try:
            code = extract_code_block(raw) or raw  # fallback
            # If we got valid code, break out of retry loop
            if not is_truncated:
                break
            elif retry_idx < max_retries:
                print(f"⚠️  Output truncated, retrying with concise prompt...")
                continue
        except RuntimeError as e:
            if "No ``` code block found" in str(e) and retry_idx < max_retries:
                print(f"⚠️  Failed to extract code (attempt {retry_idx + 1}/{max_retries + 1})")
                continue
            else:
                # Last retry failed or different error
                raise

    path = save_kernel_code(code, code_dir)
    ind = KernelIndividual(code)
    ind.code_path = path  # type: ignore[attr-defined]
    return ind

# ================== Top-level worker: MUST live at module top level, not inside another function ==================
def _bench_worker_entry(test_py: str,
                        ref_py: str,
                        device_idx: int,
                        warmup: int,
                        repeat: int,
                        tol: float,
                        conn) -> None:
    """
    Subprocess entry: set GPU, call compare_and_bench, and send result or error
    back to the parent via a Pipe. Note: we pass string paths here to avoid
    non-picklable objects.

    NOTE: CUDA_VISIBLE_DEVICES must be set BEFORE this process starts.
    The parent process should pass it via the environment when spawning.
    """
    import torch
    import gc
    from pathlib import Path
    from utils.compile_and_run import CompilationError, AccuracyError

    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(device_idx)

        res = compare_and_bench(
            ref_py=Path(ref_py),
            test_py=Path(test_py),
            device_idx=device_idx,
            warmup=warmup,
            repeat=repeat,
            tol=tol,
        )
        conn.send(("ok", res))
    except Exception as e:
        # Clean the error message if helper is available; otherwise fall back to str(e)
        try:
            cleaned = _sanitize_error_message(e)
            msg = _last_n_lines(cleaned)
        except Exception:
            msg = str(e)

        if isinstance(e, CompilationError):
            err_type = "CompilationError"
        elif isinstance(e, AccuracyError):
            err_type = "AccuracyError"
        else:
            err_type = e.__class__.__name__

        conn.send(("err", {"type": err_type, "message": msg}))
    finally:
        # Try to sync at the end so errors surface within this round
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize(device_idx)
            except Exception:
                pass
        try:
            conn.close()
        except Exception:
            pass


# ================== Keep original behavior: _bench_and_score (uses spawn + top-level worker) ==================
def _bench_and_score(
    ind: KernelIndividual,
    *,
    ref_py: Path,
    device_idx: int,
    warmup: int,
    repeat: int,
    tol: float,
    phase: str = "seed",
    metrics_dir: Path | None = None,
) -> None:
    """
    Benchmark and update the individual's metrics/score; on exception, fill in
    failure info and save metrics (if a directory is provided).
    Same functionality as the original version, but runs compare_and_bench in a
    **spawned subprocess** to isolate the CUDA context.
    """
    import torch
    from multiprocessing import get_context

    ctx = get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)

    # Only pass picklable arguments (e.g., string paths)
    p = ctx.Process(
        target=_bench_worker_entry,
        args=(
            str(ind.code_path),  # type: ignore[attr-defined]
            str(ref_py),
            device_idx,
            warmup,
            repeat,
            tol,
            child_conn,
        ),
    )
    p.start()
    # Parent does not use the child end
    try:
        child_conn.close()
    except Exception:
        pass

    # Wait for child and receive the payload
    p.join()
    payload = parent_conn.recv() if parent_conn.poll() else None
    try:
        parent_conn.close()
    except Exception:
        pass

    # —— Update metrics/score based on child payload (same logic as before) ——
    if isinstance(payload, tuple) and len(payload) == 2 and payload[0] in ("ok", "err"):
        tag, data = payload
        if tag == "ok":
            metrics = data
            metrics["runnable"] = True
            metrics["phase"] = phase
            speedup = metrics["ref_latency_ms"]["avg"] / max(1e-9, metrics["test_latency_ms"]["avg"])
            metrics["score"] = speedup

            ind.metrics = metrics
            ind.score = speedup
            print(f"[{phase}] score={speedup:.4f}")

        else:
            err_type = "RuntimeError"
            message = data
            if isinstance(data, dict):
                err_type = data.get("type", err_type) or err_type
                message = data.get("message", message)

            if not isinstance(message, str):
                message = str(message)

            print(f"\033[91mTest Error ({err_type}):\033[0m {message}")
            ind.metrics = {
                "runnable": False,
                "phase": phase,
                "error_type": err_type,
                "message": message,
            }
            ind.score = float("-inf")
            print(f"[{phase}] failed. See metrics.message for details.")
    else:
        # Subprocess exited unexpectedly with no payload
        ind.metrics = {
            "runnable": False,
            "phase": phase,
            "error_type": "SubprocessCrashed",
            "message": "subprocess exited unexpectedly (no payload received)",
        }
        ind.score = float("-inf")
        print(f"[{phase}] failed. Subprocess crashed.")

    # —— As before: try to save metrics regardless of success/failure —— 
    if metrics_dir is not None:
        try:
            saved = ind.save_metrics(metrics_dir)
            print(f"[{phase}] metrics saved to: {saved}")
        except Exception as save_exc:
            print(f"[{phase}] WARNING: failed to save metrics: {save_exc}")

    # Light cleanup in parent (not required, but safer)
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize(device_idx)
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass



def _should_skip_stage(stage_name: str, metrics_df) -> tuple[bool, str]:
    """
    Check if optimization stage should be skipped based on current NCU metrics.

    Args:
        stage_name: Name of the optimization stage
        metrics_df: pandas DataFrame with NCU profiling metrics

    Returns:
        (should_skip, reason): Tuple of bool and descriptive string
    """
    # Check if stage has exit criteria defined
    if stage_name not in STAGE_EXIT_CRITERIA:
        return (False, "")

    criteria = STAGE_EXIT_CRITERIA[stage_name]
    metric_names = criteria["metrics"]
    thresholds = criteria["thresholds"]
    comparisons = criteria.get("comparisons", [">="] * len(metric_names))  # Default to ">="
    operator = criteria.get("operator", "and")
    description = criteria.get("description", "")

    # Empty metrics -> don't skip
    if metrics_df is None or metrics_df.empty:
        return (False, "")

    # Check each metric-threshold pair
    conditions_met = []
    metric_values = []

    for metric_name, threshold, comparison in zip(metric_names, thresholds, comparisons):
        if metric_name not in metrics_df.columns:
            # Metric not available -> condition not met
            conditions_met.append(False)
            metric_values.append(f"{metric_name}=N/A")
            continue

        # Get the metric value (use first row if multiple kernels)
        value = metrics_df[metric_name].iloc[0]

        # Check if value is numeric and meets threshold
        try:
            value_float = float(value)

            # Apply comparison operator
            if comparison == ">=":
                met = value_float >= threshold
                symbol = ">="
            elif comparison == "<=":
                met = value_float <= threshold
                symbol = "<="
            elif comparison == ">":
                met = value_float > threshold
                symbol = ">"
            elif comparison == "<":
                met = value_float < threshold
                symbol = "<"
            else:
                met = value_float >= threshold  # Default fallback
                symbol = ">="

            conditions_met.append(met)
            metric_values.append(f"{metric_name}={value_float:.2f}% ({symbol}{threshold})")
        except (ValueError, TypeError):
            conditions_met.append(False)
            metric_values.append(f"{metric_name}=invalid")

    # Evaluate based on operator
    if operator == "and":
        should_skip = all(conditions_met)
    elif operator == "or":
        should_skip = any(conditions_met)
    else:
        should_skip = False

    if should_skip:
        metrics_str = ", ".join(metric_values)
        reason = f"{description} [{metrics_str}]"
        return (True, reason)

    return (False, "")


# ---------------------- task helpers -------------------
def _collect_tasks(maybe_dir: Path) -> List[Path]:
    """If a directory, return all .py files (sorted); if a file, return [file]."""
    if maybe_dir.is_file():
        return [maybe_dir]
    if maybe_dir.is_dir():
        return sorted([p for p in maybe_dir.rglob("*.py") if p.is_file()])
    raise FileNotFoundError(f"{maybe_dir} not found")


def _pick_first_n(tasks: List[Path], n: int) -> List[Path]:
    n = max(1, min(max(n, 0), len(tasks)))
    return tasks[:n]


def _sample_tasks(all_tasks: List[Path], k: int, seed: int | None) -> List[Path]:
    if not all_tasks:
        raise RuntimeError("No .py tasks found.")
    k = max(1, min(k, len(all_tasks)))
    if seed is None or seed == 0:
        seed = int(time.time())
    rng = random.Random(seed)
    return rng.sample(all_tasks, k)


def _plot_scores(save_path: Path, scores: List[float], err_flags: List[bool], title: str):
    """Plot per-round score curve; mark error rounds with an 'x'."""
    xs = list(range(len(scores)))
    plt.figure()
    plt.plot(xs, scores, marker="o")
    for x, y, bad in zip(xs, scores, err_flags):
        if bad:
            plt.scatter([x], [y], marker="x")
    plt.xlabel("Round")
    plt.ylabel("Speedup (ref/test)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def _append_usage_totals(log_path: Path) -> Dict[str, int]:
    """Append a totals row to usage.csv and return the summed token counts."""
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    if not log_path.exists():
        return totals

    with log_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not fieldnames or not rows:
        return totals

    for row in rows:
        if row.get("call_type") == "sum" or row.get("timestamp") == "Total":
            continue
        for key in totals:
            try:
                totals[key] += int(row.get(key, 0) or 0)
            except (TypeError, ValueError):
                continue

    total_row = {fn: "" for fn in fieldnames}
    for key, value in totals.items():
        if key in total_row:
            total_row[key] = str(value)

    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(total_row)

    return totals


# --------------------- single-task run -----------------
def _run_single_task(task_path: Path, args, batch_dir: Path) -> Dict[str, Any]:
    import os

    # Use PID to ensure unique temporary files when running multiple processes
    proc_id = os.getpid()

    # --- per-task directories under the SAME batch_dir
    task_root = (batch_dir / task_path.stem).resolve()
    code_dir = task_root / "code"
    eval_dir = task_root / "evaluation"
    fig_dir = task_root / "figures"
    io_dir = eval_dir / "llm_io"

    code_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    io_dir.mkdir(parents=True, exist_ok=True)
    log_path = task_root / "usage.csv"

    # === Create test kernel file path (ref uses task_path directly) ===
    root_dir = Path(__file__).resolve().parent
    test_kernel = root_dir / f"test_kernel_{proc_id}.py"

    call_llm = _make_llm_caller(args)

    current_kernel: Optional[KernelIndividual] = None
    best_kernel: Optional[KernelIndividual] = None
    best_score: float = float("-inf")

    scores: List[float] = []
    err_flags: List[bool] = []
    last_score_for_curve = 0.0  # default baseline for plotting on early failures

    # ====== Step 1: Generate Seed Program (outside loop) ======
    print("[Seed] Generating the initial kernel ...")
    seed_prompt = build_seed_prompt(arch_path=task_path, gpu_name=args.gpu)
    prompt_file = io_dir / "seed_prompt.txt"
    prompt_file.write_text(seed_prompt, encoding="utf-8")
    current_kernel = _llm_to_kernel(seed_prompt, code_dir, call_llm, io_dir,
                                    0, log_path=log_path, call_type="seed")
    _bench_and_score(
        current_kernel,
        ref_py=task_path,
        device_idx=args.device,
        warmup=args.warmup,
        repeat=args.repeat,
        tol=args.tol,
        phase="seed",
        metrics_dir=eval_dir,
    )

    # Record seed score
    runnable = bool(getattr(current_kernel, "metrics", {}).get("runnable", False))
    this_score = current_kernel.score if (current_kernel.score is not None and runnable) else None
    if this_score is not None:
        last_score_for_curve = this_score
        scores.append(this_score)
        err_flags.append(False)
        best_score = this_score
        best_kernel = current_kernel
        with open(test_kernel, "w") as f:
            f.write(best_kernel.code)
    else:
        scores.append(0.0)
        err_flags.append(True)

    # ====== Step 2: Repair seed if not runnable (max 3 attempts) ======
    repair_attempt = 0
    max_repair = 5
    error_history_list = []  # Track previous repair attempts
    while not runnable and repair_attempt < max_repair:
        repair_attempt += 1
        print(f"[Seed Repair {repair_attempt}/{max_repair}] Repairing seed kernel...")
        error_log = _last_n_lines(getattr(current_kernel, "metrics", {}).get("message", ""))

        # Use error history from PREVIOUS attempts (not including current)
        error_history = "\n\n".join(error_history_list[-3:]) if error_history_list else ""

        repair_prompt = build_error_prompt(
            old_code=current_kernel.code,
            error_log=error_log,
            problem=None,
            gpu_name=args.gpu,
            error_history=error_history,
            arch_path=task_path,
        )
        prompt_file = io_dir / f"seed_repair_{repair_attempt}_prompt.txt"
        prompt_file.write_text(repair_prompt, encoding="utf-8")
        current_kernel = _llm_to_kernel(repair_prompt, code_dir, call_llm, io_dir,
                                        repair_attempt, log_path=log_path, call_type="seed_repair")
        _bench_and_score(
            current_kernel,
            ref_py=task_path,
            device_idx=args.device,
            warmup=args.warmup,
            repeat=args.repeat,
            tol=args.tol,
            phase="seed_repair",
            metrics_dir=eval_dir,
        )

        runnable = bool(getattr(current_kernel, "metrics", {}).get("runnable", False))
        this_score = current_kernel.score if (current_kernel.score is not None and runnable) else None
        if this_score is not None:
            last_score_for_curve = this_score
            scores.append(this_score)
            err_flags.append(False)
            # Only update best if this score is better
            if this_score > best_score:
                best_score = this_score
                best_kernel = current_kernel
                with open(test_kernel, "w") as f:
                    f.write(best_kernel.code)
        else:
            scores.append(last_score_for_curve)
            err_flags.append(True)
            # Add current error to history for next repair attempt (keep last 5 lines)
            if error_log.strip():
                error_lines = error_log.splitlines()
                last_lines = "\n".join(error_lines[-5:]) if len(error_lines) > 5 else error_log
                error_history_list.append(f"Attempt {repair_attempt}:\n{last_lines}")

    if not runnable:
        print(f"[ERROR] Failed to repair seed kernel after {max_repair} attempts. Stopping.")
    else:
        # ====== Step 3: Four-stage Optimization Loop ======
        print("\n[Optimization] Starting 4-stage optimization process...")
        stage_round = 0
        previous_stage_score = best_score  # Track score before each stage

        for stage_idx in range(len(OPTIMIZATION_STAGES)):
            stage = OPTIMIZATION_STAGES[stage_idx]
            stage_name = stage["name"]
            stage_description = stage["description"]

            print(f"\n{'='*80}")
            print(f"[Stage {stage_idx + 1}/{len(OPTIMIZATION_STAGES)}] {stage_name}")
            print(f"Description: {stage_description}")
            print(f"{'='*80}")

            # Check if optimization is needed based on performance gain from previous stage
            print(f"Current best: {best_score:.4f}")

            # Record the score before this stage starts
            score_before_stage = best_score

            # Check if best_kernel is available and runnable
            if best_kernel is None:
                print(f"[ERROR] No best kernel available for stage {stage_idx + 1}. Skipping.")
                continue

            is_runnable = bool(getattr(best_kernel, "metrics", {}).get("runnable", False))
            if not is_runnable:
                print(f"[ERROR] Best kernel is not runnable at stage {stage_idx + 1}. Skipping.")
                continue

            # Perform optimization for this stage (single optimization with full NCU profiling)
            stage_round += 1
            print(f"[Stage {stage_idx + 1}] Optimizing {stage_name}...")

            # Step 1: Profile the current best_kernel to get NCU metrics
            with open(test_kernel, "w") as f:
                f.write(best_kernel.code)

            # Create bench script for this process (copy from template)
            import shutil
            bench_template = root_dir / "bench_ref_inputs_template.py"
            bench_script = root_dir / f"bench_ref_inputs_{proc_id}.py"

            if not bench_template.exists():
                raise FileNotFoundError(
                    f"Benchmark template not found: {bench_template}\n"
                    "Please ensure bench_ref_inputs_template.py exists in the project root."
                )

            # Always copy the template to create the process-specific bench script
            shutil.copy(bench_template, bench_script)
            print(f"Created benchmark script: {bench_script}")

            kernel_names = extract_cuda_kernel_names(test_kernel)
            print(f"Detected kernel names: {kernel_names}")
            csv_path = profile_bench(
                bench_py=f"bench_ref_inputs_{proc_id}.py",
                kernel_names=kernel_names,  # Add kernel name filter
                out_csv=f"ncu_temp_{proc_id}.csv",
                device_idx=args.device,
                ref_file=str(task_path),  # Use original task file, consistent with _bench_and_score
                test_file=f"test_kernel_{proc_id}.py")
            metrics_df = load_ncu_metrics(csv_path, extra_keep=("Kernel Name",),
                                          name_list=kernel_names, select="last")
            metrics_block = metrics_to_prompt(metrics_df)

            # Check if stage should be skipped based on metrics
            should_skip, skip_reason = _should_skip_stage(stage_name, metrics_df)
            if should_skip:
                print(f"\n⏩ [Stage {stage_idx + 1}] SKIPPED: {skip_reason}")
                print(f"   Current metrics already meet optimization goals for this stage.")
                print(f"   Proceeding to next stage...\n")
                continue

            # Step 2: Build optimization prompt with NCU metrics and stage description
            # history_block = _build_history_block(code_dir, keep_last=0)
            opt_prompt = build_optimization_prompt(
                arch_path=best_kernel.code_path,  # type: ignore[union-attr]
                gpu_name=args.gpu,
                ncu_metrics=metrics_block,  # Now contains actual profiling data
                history_block=None,#history_block,
                stage_name=stage_name,
                stage_description=stage_description,
                failure_analysis="",
            )
            prompt_file = io_dir / f"stage{stage_idx + 1}_{stage_name}_prompt.txt"
            prompt_file.write_text(opt_prompt, encoding="utf-8")

            # Step 3: Generate optimized kernel
            current_kernel = _llm_to_kernel(opt_prompt, code_dir, call_llm, io_dir, stage_round,
                                            log_path=log_path, call_type=f"stage{stage_idx + 1}_{stage_name}")
            _bench_and_score(
                current_kernel,
                ref_py=task_path,
                device_idx=args.device,
                warmup=args.warmup,
                repeat=args.repeat,
                tol=args.tol,
                phase=f"stage{stage_idx + 1}_{stage_name}",
                metrics_dir=eval_dir,
            )

            # Step 4: Check if optimization succeeded
            runnable = bool(getattr(current_kernel, "metrics", {}).get("runnable", False))
            this_score = current_kernel.score if (current_kernel.score is not None and runnable) else None

            # Step 5: If optimization failed, try repair (max 2 attempts)
            if not runnable:
                print(f"[Stage {stage_idx + 1}] Optimization failed, attempting repair...")
                repair_attempt = 0
                max_repair = 3
                stage_error_history_list = []  # Track previous repair attempts in this stage
                while not runnable and repair_attempt < max_repair:
                    repair_attempt += 1
                    stage_round += 1
                    print(f"[Stage {stage_idx + 1} Repair {repair_attempt}/{max_repair}]")
                    error_log = _last_n_lines(getattr(current_kernel, "metrics", {}).get("message", ""))

                    # Use error history from PREVIOUS attempts (not including current)
                    stage_error_history = "\n\n".join(stage_error_history_list[-3:]) if stage_error_history_list else ""

                    repair_prompt = build_error_prompt(
                        old_code=current_kernel.code,
                        error_log=error_log,
                        problem=None,
                        gpu_name=args.gpu,
                        error_history=stage_error_history,
                        arch_path=task_path,
                    )
                    prompt_file = io_dir / f"stage{stage_idx + 1}_repair_{repair_attempt}_prompt.txt"
                    prompt_file.write_text(repair_prompt, encoding="utf-8")
                    current_kernel = _llm_to_kernel(repair_prompt, code_dir, call_llm, io_dir,
                                                    stage_round, log_path=log_path, call_type=f"stage{stage_idx + 1}_repair")
                    _bench_and_score(
                        current_kernel,
                        ref_py=task_path,
                        device_idx=args.device,
                        warmup=args.warmup,
                        repeat=args.repeat,
                        tol=args.tol,
                        phase=f"stage{stage_idx + 1}_repair",
                        metrics_dir=eval_dir,
                    )

                    runnable = bool(getattr(current_kernel, "metrics", {}).get("runnable", False))
                    this_score = current_kernel.score if (current_kernel.score is not None and runnable) else None
                    if this_score is not None:
                        last_score_for_curve = this_score
                        scores.append(this_score)
                        err_flags.append(False)
                        if this_score > best_score:
                            best_score = this_score
                            best_kernel = current_kernel
                            with open(test_kernel, "w") as f:
                                f.write(best_kernel.code)
                            print(f"[Stage {stage_idx + 1} Repair] New best score: {best_score:.4f}")
                    else:
                        scores.append(last_score_for_curve)
                        err_flags.append(True)
                        # Add current error to history for next repair attempt (keep last 5 lines)
                        if error_log.strip():
                            error_lines = error_log.splitlines()
                            last_lines = "\n".join(error_lines[-5:]) if len(error_lines) > 5 else error_log
                            stage_error_history_list.append(f"Attempt {repair_attempt}:\n{last_lines}")

                if not runnable:
                    print(f"[Stage {stage_idx + 1}] Failed to repair after {max_repair} attempts. Keeping best_kernel unchanged.")
                    # Don't update best_kernel, keep the previous one
                    previous_stage_score = score_before_stage
                    continue
                else:
                    # Repair succeeded, skip Step 6 (already handled in repair loop)
                    print(f"[Stage {stage_idx + 1}] Repair succeeded.")
                    previous_stage_score = score_before_stage
                    continue

            # Step 6: Update best_kernel if improved (only for optimization, not repair)
            if this_score is not None:
                last_score_for_curve = this_score
                scores.append(this_score)
                err_flags.append(False)
                if this_score > best_score:
                    best_score = this_score
                    best_kernel = current_kernel
                    with open(test_kernel, "w") as f:
                        f.write(best_kernel.code)
                    print(f"[Stage {stage_idx + 1}] New best score: {best_score:.4f} (improved from {score_before_stage:.4f})")
                else:
                    print(f"[Stage {stage_idx + 1}] Score: {this_score:.4f} (not better than best: {best_score:.4f})")
                    print(f"[Stage {stage_idx + 1}] Keeping previous best_kernel")
            else:
                scores.append(last_score_for_curve)
                err_flags.append(True)
                print(f"[Stage {stage_idx + 1}] Optimization produced non-runnable kernel. Keeping best_kernel unchanged.")

            # Update previous_stage_score for next iteration
            previous_stage_score = score_before_stage

    # plot per-task curve
    fig_path = fig_dir / f"{task_path.stem}_score.png"
    _plot_scores(fig_path, scores, err_flags, title=f"{task_path.stem} (best={best_score:.4f})")
    print(f"[{task_path.name}] Figure saved to: {fig_path}")

    usage_totals = _append_usage_totals(log_path)

    return {
        "task": str(task_path),
        "best_score": float(best_score) if best_score != float("-inf") else 0.0,
        "best_runnable": bool(getattr(best_kernel, "metrics", {}).get("runnable", False)) if best_kernel else False,
        "task_dir": str(task_root),
        "figure": str(fig_path),
        "input_tokens_sum": usage_totals["input_tokens"],
        "output_tokens_sum": usage_totals["output_tokens"],
        "total_tokens_sum": usage_totals["total_tokens"],
    }


# --------------------- summary saving ------------------
def _save_global_summary(batch_dir: Path, summary: List[Dict[str, Any]], avg_speedup: float, accuracy: float, total_tokens_sum: float) -> None:
    """Save summary.json and summary.csv under the batch_dir."""
    batch_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    out_json = {
        "avg_speedup": avg_speedup,
        "accuracy": accuracy,
        "total_tokens_sum": total_tokens_sum,
        "num_tasks": len(summary),
        "tasks": summary,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    (batch_dir / "summary.json").write_text(json.dumps(out_json, indent=2), encoding="utf-8")

    # CSV
    csv_path = batch_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "best_score", "best_runnable", "task_dir", "figure"])
        for s in summary:
            writer.writerow([s["task"], f'{s["best_score"]:.6f}', int(
                bool(s["best_runnable"])), s["task_dir"], s["figure"]])
        writer.writerow([])
        writer.writerow(["avg_speedup", f"{avg_speedup:.6f}"])
        writer.writerow(["accuracy", f"{accuracy:.6f}"])
        writer.writerow(["total_tokens_sum", f"{int(total_tokens_sum)}"])

    print(f"[GLOBAL] Saved: {batch_dir/'summary.json'}")
    print(f"[GLOBAL] Saved: {csv_path}")


# --------------------------- main ----------------------
def main():
    args = _build_arg_parser().parse_args()

    all_tasks = _collect_tasks(args.arch_py)

    # ---- Create ONE batch folder for this run ----
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = _build_run_tag(args.server_type, args.model_name)
    # batch name hints: single file uses file stem; directory uses 'batch'
    if args.arch_py.is_file():
        batch_name = f"{stamp}_{args.arch_py.stem}_{run_tag}"
    else:
        # include sampling info for traceability
        pick_note = f"first{args.first_n}" if (args.first_n and args.first_n >
                                               0) else f"num{args.num_tasks}_seed{args.shuffle_seed}"
        batch_name = f"{stamp}_batch_{pick_note}_{run_tag}"
    batch_dir = (args.work_dir / batch_name).resolve()
    batch_dir.mkdir(parents=True, exist_ok=True)
    print(f"[BATCH] Output folder: {batch_dir}")

    # single file → run once (still inside the same batch folder)
    if args.arch_py.is_file():
        res = _run_single_task(all_tasks[0], args, batch_dir=batch_dir)
        summary = [res]
        avg_speedup = res["best_score"]
        accuracy = 1.0 if res["best_runnable"] else 0.0
        total_tokens_sum = res.get("total_tokens_sum", 0)
        print(f"[SUMMARY] {res}")
        print(f"[GLOBAL] Avg speedup={avg_speedup:.4f}, Accuracy={accuracy:.4f}")

        _save_global_summary(batch_dir, summary, avg_speedup, accuracy, total_tokens_sum)
        return

    # directory: first_n takes precedence; else optionally sample
    if args.first_n and args.first_n > 0:
        picked = _pick_first_n(all_tasks, args.first_n)
        print(f"[Task Picker] Found {len(all_tasks)} tasks, taking first {len(picked)} (sorted).")
    else:
        picked = _sample_tasks(all_tasks, args.num_tasks, args.shuffle_seed)
        print(f"[Task Picker] Found {len(all_tasks)} tasks, sampled {len(picked)} with seed={args.shuffle_seed}.")

    summary: List[Dict[str, Any]] = []
    for i, task in enumerate(picked, 1):
        print(f"\n===== [{i}/{len(picked)}] Running task: {task} =====")
        res = _run_single_task(task, args, batch_dir=batch_dir)
        summary.append(res)

    # global summary using each task's best kernel
    if summary:
        avg_speedup = sum(s["best_score"] for s in summary) / len(summary)
        accuracy = sum(1 for s in summary if s["best_runnable"]) / len(summary)
        total_tokens_sum = sum(int(s.get("total_tokens_sum", 0) or 0) for s in summary)
        print("\n===== SUMMARY =====")
        for s in summary:
            print(f"{s['task']}: best_score={s['best_score']:.4f}  runnable={s['best_runnable']}  fig={s['figure']}")
        print(f"\n[GLOBAL] Avg speedup={avg_speedup:.4f}, Accuracy={accuracy:.4f}")

        # ---- save under the SAME batch folder ----
        _save_global_summary(batch_dir, summary, avg_speedup, accuracy, total_tokens_sum)
    else:
        print("No tasks were run.")


if __name__ == "__main__":
	main()