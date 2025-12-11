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

# Import operator categorization system
from config.operator_categories_v3 import (
    classify_operator,
    OPERATOR_CATEGORIES,
    get_key_ncu_metrics,
    check_early_exit,
    should_skip_stage,
)
from utils.ncu_context_builder import (
    build_enhanced_ncu_context,
    extract_metrics_from_df,
    CORE_NCU_METRICS,
)

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
    p.add_argument("--warmup", type=int, default=2, help="Warm-up iterations")
    p.add_argument("--repeat", type=int, default=5, help="Timed iterations per benchmark")
    p.add_argument("--tol", type=float, default=1e-3, help="Max |err| tolerated")
    p.add_argument("--max_tokens", type=int, default=20000, help="LLM max new tokens")
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
# Four-stage progressive optimization strategy (ordered by dependency):
# 1. Grid & Parallel: Configure work distribution across SMs
# 2. Block Tiling: Determine BLOCK_M/N/K (affects all subsequent optimizations)
# 3. Memory Access: Optimize L2 cache based on block configuration
# 4. Advanced Memory: Fine-tune cache policies and instruction-level optimizations
OPTIMIZATION_STAGES = [
    {"name": "grid_and_parallel", "description": "Optimize grid layout and parallel work distribution across SMs."},
    {"name": "block_tiling", "description": "Tune BLOCK_M/N/K sizes for optimal register/memory balance."},
    {"name": "memory_access", "description": "Optimize L2 cache utilization and memory access patterns."},
    {"name": "advanced_memory", "description": "Fine-tune cache policies and instruction-level optimizations."},
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
        "operator": "and",  # All conditions must be met
        "description": "SM occupancy already optimal (>90%)"
    },
    "block_tiling": {
        "metrics": ["sm__maximum_warps_per_active_cycle_pct"],
        "thresholds": [85.0],  # Good occupancy suggests block config is decent
        "comparisons": [">="],  # Higher is better
        "operator": "and",
        "description": "Block configuration already efficient (occupancy >85%)"
    },
    "memory_access": {
        # Skip if BOTH conditions are met (memory access is already good)
        "metrics": [
            "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct",  # Memory stalls
            "dram__throughput.avg.pct_of_peak_sustained_elapsed"  # DRAM utilization
        ],
        "thresholds": [10.0, 85.0],  # Low stalls (<10%) AND high DRAM (>85%)
        "comparisons": ["<=", ">="],  # Lower stalls better, higher DRAM better
        "operator": "and",  # Both conditions must be met
        "description": "Memory access already optimized (stalls <10% and DRAM >85%)"
    },
    "advanced_memory": {
        # Skip if memory is already very efficient
        "metrics": [
            "lts__t_sector_hit_rate.pct",  # L2 cache hit rate
            "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct"  # Memory stalls
        ],
        "thresholds": [95.0, 5.0],  # High L2 (>95%) OR very low stalls (<5%)
        "comparisons": [">=", "<="],  # Higher L2 better, lower stalls better
        "operator": "or",  # Any condition met -> advanced tuning won't help
        "description": "Memory system already near-optimal (L2 >95% or stalls <5%)"
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
            # device_idx is relative to CUDA_VISIBLE_DEVICES
            # If CUDA_VISIBLE_DEVICES=3, then device_idx=0 refers to GPU 3
            torch.cuda.set_device(device_idx)

            # Enable expandable segments to reduce memory fragmentation
            import os
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

            # Clear any CUDA cache from parent process
            torch.cuda.empty_cache()
            gc.collect()

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
        # Clean up CUDA resources and sync before subprocess exits
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize(device_idx)
                torch.cuda.empty_cache()
                gc.collect()
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
    pytorch_baseline_ms: Optional[float] = None,
) -> None:
    """
    Benchmark and update the individual's metrics/score; on exception, fill in
    failure info and save metrics (if a directory is provided).
    Same functionality as the original version, but runs compare_and_bench in a
    **spawned subprocess** to isolate the CUDA context.
    """
    import os
    from multiprocessing import get_context

    # IMPORTANT: Do NOT import torch or call any CUDA functions here!
    # Doing so will initialize CUDA context BEFORE we set CUDA_VISIBLE_DEVICES,
    # which will lock the GPU to the wrong device.

    ctx = get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)

    # Prepare environment for subprocess: set CUDA_VISIBLE_DEVICES
    # This MUST be set before the subprocess starts to take effect
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device_idx)

    # Only pass picklable arguments (e.g., string paths)
    # Note: multiprocessing.Process doesn't support env parameter directly,
    # so we need to use a wrapper or modify the global environment temporarily
    # The best approach is to set it globally before spawning
    old_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_idx)

    try:
        p = ctx.Process(
            target=_bench_worker_entry,
            args=(
                str(ind.code_path),  # type: ignore[attr-defined]
                str(ref_py),
                0,  # After setting CUDA_VISIBLE_DEVICES, device index is always 0
                warmup,
                repeat,
                tol,
                child_conn,
            ),
        )
        p.start()
    finally:
        # Restore original CUDA_VISIBLE_DEVICES
        if old_cuda_visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_visible
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

            # Use fixed PyTorch baseline if provided, otherwise use dynamic reference
            if pytorch_baseline_ms is not None:
                speedup = pytorch_baseline_ms / max(1e-9, metrics["test_latency_ms"]["avg"])
                metrics["pytorch_baseline_ms"] = pytorch_baseline_ms
            else:
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

    # NOTE: Do NOT clean up CUDA in parent process!
    # Importing torch here will initialize CUDA on the wrong device.
    # Let the subprocess handle all CUDA operations.

    # Force garbage collection to release any Python references
    import gc
    gc.collect()


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

    # === Classify operator and get category-specific configuration ===
    task_name = task_path.stem  # e.g., "56_Matmul_Sigmoid_Sum"
    level = "level2" if "/level2/" in str(task_path) else "level1"
    category = classify_operator(task_name, level)
    category_config = OPERATOR_CATEGORIES[category]

    print(f"\n{'='*80}")
    print(f"📂 Operator Category: {category}")
    print(f"   Description: {category_config['description']}")
    print(f"   Total Stages: {len(category_config['stages'])}")
    if category_config.get('early_exit_enabled'):
        print(f"   Early Exit: Enabled")
    print(f"{'='*80}\n")

    # Save category metadata
    category_metadata = {
        "category": category,
        "level": level,
        "task_name": task_name,
        "num_stages": len(category_config['stages']),
        "early_exit_enabled": category_config.get('early_exit_enabled', False),
    }
    (task_root / "category_metadata.json").write_text(
        json.dumps(category_metadata, indent=2)
    )

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

    # ====== Step 0: Benchmark PyTorch Baseline ONCE ======
    print(f"\n{'='*80}")
    print("[Baseline] Benchmarking PyTorch performance")
    print(f"{'='*80}")

    # Create a dummy kernel that wraps the reference Model as ModelNew
    # This allows us to use the existing benchmarking infrastructure
    ref_module_name = task_path.stem
    dummy_code = f'''import sys
import importlib.util
from pathlib import Path

# Import the reference Model dynamically (module name may start with digit)
ref_path = Path("{task_path}").resolve()

spec = importlib.util.spec_from_file_location("{ref_module_name}", ref_path)
ref_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref_module)

# Create ModelNew as an alias to Model for baseline benchmarking
ModelNew = ref_module.Model
'''

    dummy_kernel = KernelIndividual(dummy_code)
    # Save the dummy kernel to a temporary file
    dummy_kernel_path = code_dir / "baseline_dummy.py"
    dummy_kernel_path.write_text(dummy_code, encoding="utf-8")
    dummy_kernel.code_path = dummy_kernel_path

    _bench_and_score(
        dummy_kernel,
        ref_py=task_path,
        device_idx=args.device,
        warmup=args.warmup,
        repeat=args.repeat,
        tol=args.tol,
        phase="pytorch_baseline",
        metrics_dir=eval_dir,
    )

    # Extract and save PyTorch baseline latency
    pytorch_baseline_ms = None
    if dummy_kernel.metrics and "ref_latency_ms" in dummy_kernel.metrics:
        pytorch_baseline_ms = dummy_kernel.metrics["ref_latency_ms"]["avg"]
        print(f"✅ PyTorch baseline: {pytorch_baseline_ms:.4f} ms")
        print(f"   All subsequent scores will be calculated as: {pytorch_baseline_ms:.4f} / triton_time")

        # Save baseline info
        baseline_info = {
            "pytorch_latency_ms": pytorch_baseline_ms,
            "timestamp": datetime.now().isoformat(),
            "warmup": args.warmup,
            "repeat": args.repeat,
        }
        (task_root / "pytorch_baseline.json").write_text(
            json.dumps(baseline_info, indent=2)
        )
    else:
        print("⚠️  Warning: Failed to benchmark PyTorch baseline, will use dynamic comparison")
    print(f"{'='*80}\n")

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
        pytorch_baseline_ms=pytorch_baseline_ms,
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
            pytorch_baseline_ms=pytorch_baseline_ms,
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
        # ====== Step 3: Category-specific Optimization Loop ======
        optimization_stages = category_config["stages"]
        print(f"\n[Optimization] Starting {len(optimization_stages)}-stage optimization ({category})...")
        stage_round = 0

        # Initialize Stage 1 baseline tracking (for comparison in later stages)
        stage1_score = None
        stage1_metrics_dict = {}

        for stage_idx, stage_config in enumerate(optimization_stages):
            stage_name = stage_config["name"]
            stage_description = stage_config["description"]
            stage_focus = stage_config["focus"]

            print(f"\n{'='*80}")
            print(f"[Stage {stage_idx + 1}/{len(optimization_stages)}] {stage_description}")
            print(f"Stage Name: {stage_name}")
            print(f"Focus: {stage_focus}")
            print(f"Category: {category}")
            print(f"{'='*80}")

            # Check if optimization is needed based on performance gain from previous stage
            print(f"Current best: {best_score:.4f}")

            # Record the score before this stage starts
            score_before_stage = best_score

            # === Prepare operator metadata ===
            op_metadata = {
                "op_type": task_name.lower(),
                "score": best_score,
            }
            # Try to extract kernel_size for Conv operators
            try:
                task_code = task_path.read_text()
                import re
                ks_match = re.search(r'kernel_size\s*=\s*(\d+)', task_code)
                if ks_match:
                    op_metadata["kernel_size"] = int(ks_match.group(1))
            except:
                pass

            # === Early Exit Check ===
            # Skip early exit check for Stage 0 (give it a chance to optimize)
            # Only check early exit from Stage 1 onwards
            if stage_idx >= 1:
                should_exit, exit_reason = check_early_exit(
                    category=category,
                    stage_id=stage_idx,
                    performance_score=best_score,
                    op_metadata=op_metadata,
                )

                if should_exit:
                    print(f"\n⛔ {'='*70}")
                    print(f"⛔ EARLY EXIT TRIGGERED")
                    print(f"⛔ Reason: {exit_reason}")
                    print(f"⛔ Current best score: {best_score:.4f}")
                    print(f"⛔ Skipping remaining stages.")
                    print(f"⛔ {'='*70}\n")
                    break

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
            bench_template = root_dir / "bench_ref_inputs_template.py"
            bench_script = root_dir / f"bench_ref_inputs_{proc_id}.py"
            if bench_template.exists():
                import shutil
                shutil.copy(bench_template, bench_script)
            else:
                raise FileNotFoundError(f"Bench template not found: {bench_template}")

            kernel_names = extract_cuda_kernel_names(test_kernel)
            print(f"Detected kernel names: {kernel_names}")
            csv_path = profile_bench(
                bench_py=f"bench_ref_inputs_{proc_id}.py",
                out_csv=f"ncu_temp_{proc_id}.csv",
                device_idx=args.device,
                ref_file=str(task_path),  # Use original task file, consistent with _bench_and_score
                test_file=f"test_kernel_{proc_id}.py")
            metrics_df = load_ncu_metrics(csv_path, extra_keep=("Kernel Name",),
                                          name_list=kernel_names, select="last")

            # === Filter NCU metrics to only key metrics for this stage ===
            key_metrics = get_key_ncu_metrics(category, stage_idx)

            print(f"\n📊 Key Metrics for {stage_name}:")
            for metric_name, ncu_metric in key_metrics.items():
                print(f"   • {metric_name}: {ncu_metric}")

            # Filter metrics_df to only include key metrics (by column)
            if not metrics_df.empty:
                # Get available key metrics (columns that exist in the dataframe)
                available_key_metrics = [m for m in key_metrics.values() if m in metrics_df.columns]

                # Also include CORE metrics for baseline comparison (if available)
                available_core_metrics = [m for m in CORE_NCU_METRICS if m in metrics_df.columns and m not in available_key_metrics]

                # Keep "Kernel Name" and other non-metric columns
                extra_cols = [c for c in metrics_df.columns if not c.startswith(('sm', 'dram', 'l1', 'l2', 'lts', 'smsp'))]

                # Combine: extra columns + key metrics + core metrics
                keep_cols = extra_cols + available_key_metrics + available_core_metrics

                if available_key_metrics or available_core_metrics:
                    filtered_df = metrics_df[keep_cols].copy()
                    print(f"   Filtered: {len(metrics_df.columns)} → {len(filtered_df.columns)} columns")
                    print(f"   Stage-specific: {', '.join(available_key_metrics)}")
                    if available_core_metrics:
                        print(f"   Core metrics: {', '.join(available_core_metrics)}")
                else:
                    print(f"⚠️  Warning: No key metrics found in dataframe, using all metrics")
                    filtered_df = metrics_df

                # Extract metrics to dictionary for enhanced context
                current_metrics_dict = extract_metrics_from_df(
                    filtered_df, kernel_names[0] if kernel_names else None
                )

                # Build enhanced NCU context with baseline comparison
                metrics_block = build_enhanced_ncu_context(
                    current_metrics=current_metrics_dict,
                    category=category,
                    stage_name=stage_name,
                    current_score=best_score,
                    baseline_metrics=stage1_metrics_dict if stage_idx >= 1 and stage1_metrics_dict else None,
                    baseline_score=stage1_score if stage_idx >= 1 and stage1_score else None,
                )
            else:
                metrics_block = "No NCU metrics available"

            # === Skip Stage Check (category-specific) ===
            should_skip_cat, skip_reason_cat = should_skip_stage(
                category=category,
                stage_id=stage_idx,
                op_metadata=op_metadata,
            )

            if should_skip_cat:
                print(f"\n⏩ {'='*70}")
                print(f"⏩ STAGE SKIPPED")
                print(f"⏩ Reason: {skip_reason_cat}")
                print(f"⏩ Proceeding to next stage...")
                print(f"⏩ {'='*70}\n")
                continue

            # Check if stage should be skipped based on NCU metrics (original logic)
            should_skip, skip_reason = _should_skip_stage(stage_name, metrics_df)
            if should_skip:
                print(f"\n⏩ [Stage {stage_idx + 1}] SKIPPED (NCU): {skip_reason}")
                print(f"   Current metrics already meet optimization goals for this stage.")
                print(f"   Proceeding to next stage...\n")
                continue

            # Step 2: Build optimization prompt with NCU metrics and stage description
            # history_block = _build_history_block(code_dir, keep_last=0)
            opt_prompt = build_optimization_prompt(
                arch_path=best_kernel.code_path,  # type: ignore[union-attr]
                gpu_name=args.gpu,
                ncu_metrics=metrics_block,  # Filtered key metrics
                history_block=None,#history_block,
                stage_name=stage_name,
                stage_description=stage_description,
                failure_analysis="",
                # === Pass category information ===
                category=category,
                stage_id=stage_idx,
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
                pytorch_baseline_ms=pytorch_baseline_ms,
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
                        pytorch_baseline_ms=pytorch_baseline_ms,
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

                            # Save Stage 1 baseline if repair succeeded
                            if stage_idx == 0 and stage1_score is None:
                                stage1_score = best_score
                                # Use metrics extracted before this stage (current_metrics_dict may exist)
                                if 'current_metrics_dict' in locals() and current_metrics_dict:
                                    stage1_metrics_dict = current_metrics_dict.copy()
                                    print(f"✅ Stage 1 baseline saved (from repair): score={stage1_score:.4f}, {len(stage1_metrics_dict)} metrics")
                                else:
                                    print(f"✅ Stage 1 baseline saved (from repair): score={stage1_score:.4f}, no metrics available")
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
                    continue
                else:
                    # Repair succeeded, skip Step 6 (already handled in repair loop)
                    print(f"[Stage {stage_idx + 1}] Repair succeeded.")
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

                # Save Stage 1 baseline for future comparison (only if Stage 1 succeeds)
                if stage_idx == 0 and stage1_score is None:
                    stage1_score = best_score
                    stage1_metrics_dict = current_metrics_dict.copy()
                    print(f"✅ Stage 1 baseline saved: score={stage1_score:.4f}, {len(stage1_metrics_dict)} metrics")

                # Check for performance regression after stage 1+
                # If current stage caused >10% degradation from previous best, consider early exit
                if stage_idx >= 1 and this_score < score_before_stage * 0.90:
                    degradation_pct = (1 - this_score / score_before_stage) * 100
                    print(f"\n⚠️  [Stage {stage_idx + 1}] Performance degraded by {degradation_pct:.1f}%")
                    print(f"   Previous best: {score_before_stage:.4f} → Current: {this_score:.4f}")

                    # If degradation > 15% and we're past stage 2, consider exiting
                    if degradation_pct > 15 and stage_idx >= 2:
                        print(f"⛔ Significant regression detected. Stopping further optimization.")
                        print(f"⛔ Keeping best kernel from Stage {stage_idx}: score={best_score:.4f}")
                        break
                    else:
                        print(f"   Continuing with next stage (will keep best kernel so far)")
            else:
                scores.append(last_score_for_curve)
                err_flags.append(True)
                print(f"[Stage {stage_idx + 1}] Optimization produced non-runnable kernel. Keeping best_kernel unchanged.")

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
