# main.py
from __future__ import annotations
import argparse
import re
import random
import time
import json
import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from run_ncu import profile_bench, load_ncu_metrics, metrics_to_prompt
import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt

from agents.query_server import query_server
from prompts.generate_custom_cuda import build_seed_prompt, default_system_prompt, MODEL_SINGLE, MODEL_FUSION, MODEL_NETWORK
from utils.compile_and_run import compare_and_bench
from utils.kernel_io import extract_code_block, save_kernel_code, extract_json, extract_cuda_kernel_names
from scripts.individual import KernelIndividual  # adjust path if needed
from prompts.error import build_error_prompt
from prompts.optimization import build_optimization_prompt
from prompts.judger_optimization import build_judger_optimization_prompts
_INVOCATION_SPLITTER = "Invoked with:"


# ---------------------- Beam Search Data Structure -----------------
@dataclass
class BeamCandidate:
    """A candidate in beam search, holding kernel and its evaluation results."""
    kernel: Any  # KernelIndividual
    score: float = float("-inf")
    metrics_df: Any = None  # pandas DataFrame with NCU metrics
    ncu_block: str = ""  # NCU metrics formatted for prompt
    runnable: bool = False

    def __lt__(self, other):
        """For sorting: higher score = better."""
        return self.score < other.score

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
    p.add_argument("--tol", type=float, default=1, help="Absolute tolerance for accuracy check")
    p.add_argument("--rtol", type=float, default=1, help="Relative tolerance")
    p.add_argument("--max_tokens", type=int, default=16000, help="LLM max new tokens")
    p.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    p.add_argument("--top_p", type=float, default=1.0, help="LLM top_p")
    # multi-task controls
    p.add_argument("--first_n", type=int, default=0, help="When arch_py is a directory, take the first N tasks (sorted)")
    p.add_argument("--start_idx", type=int, default=0, help="Start file number prefix for task selection (inclusive). E.g., --start_idx=30 selects files starting with 30_xxx.py")
    p.add_argument("--end_idx", type=int, default=0, help="End file number prefix for task selection (inclusive). 0 means no upper limit. E.g., --end_idx=40 includes 40_xxx.py")
    p.add_argument("--num_tasks", type=int, default=1, help="When sampling, how many tasks to pick (if >0 and first_n=0)")
    p.add_argument("--shuffle_seed", type=int, default=0, help="Random seed for sampling (0 = time)")
    
    p.add_argument("--subproc_id", type=int, default=0, help="Identifier for sub-process (e.g., when running multiple in parallel)")

    # Multi-stage optimization parameters
    p.add_argument("--num_steps", type=int, default=4, help="Number of optimization steps (stages)")
    p.add_argument("--max_repair_attempts", type=int, default=3, help="Max repair attempts per failure")

    # Search parameters
    p.add_argument("--num_seeds", type=int, default=2, help="Number of seed candidates to generate (default 2)")
    p.add_argument("--elimination_threshold", type=float, default=0.02, help="Soft elimination threshold τ: keep both if gap ≤ τ (default 0.02 = 2%)")

    # Legacy beam search parameters (deprecated, kept for backward compatibility)
    p.add_argument("--beam_width", type=int, default=1, help="[Deprecated] Use --num_seeds instead")
    p.add_argument("--beam_temperature", type=float, default=0.5, help="[Deprecated]")

    # Fusion operator flag
    p.add_argument("--fusion", action="store_true", help="Enable fusion operator mode (for level2 multi-op kernels)")

    # Model type (level1/2/3)
    p.add_argument("--model", default=MODEL_SINGLE,
                   choices=[MODEL_SINGLE, MODEL_FUSION, MODEL_NETWORK],
                   help="Model type: single (level1), fusion (level2), network (level3)")

    return p


# ---------------------- optimization stages configuration -----------------
# Three-stage progressive optimization strategy (ordered by dependency):
# 1. Grid & Parallel: Configure work distribution across SMs
# 2. Block Tiling: Determine BLOCK_M/N/K (affects all subsequent optimizations)
# 3. Memory & Tuning: Optimize memory access and fine-tune num_stages/num_warps
OPTIMIZATION_STAGES = [
    {"name": "grid_and_parallel", "description": "Optimize grid layout and parallel work distribution across SMs."},
    {"name": "block_tiling", "description": "Tune BLOCK_M/N/K sizes for optimal register/memory balance."},
    {"name": "memory_and_tuning", "description": "Optimize memory access patterns and fine-tune num_stages/num_warps."},
]

# ---------------------- early exit criteria (post-hoc) -----------------
# Post-hoc early stopping: stop optimization if no significant improvement
# for consecutive stages. This replaces the pre-judgment approach which
# was unreliable (predicting whether a stage would help based on metrics).
#
# New approach: Always try optimization, then check if it helped.
# If no improvement for N consecutive stages, stop early.
STAGE_EXIT_CRITERIA = {}  # Disabled: pre-judgment early stop removed

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
        temperature_override: Optional[float] = None,  # For beam search diversity
    ) -> str:
        sp = default_system_prompt if sys_prompt is None else sys_prompt
        # Use override temperature if provided (for beam search), otherwise use args.temperature
        temp = temperature_override if temperature_override is not None else args.temperature
        res = query_server(
            prompt=prompt,
            system_prompt=sp,
            server_type=args.server_type,
            model_name=args.model_name,
            max_tokens=args.max_tokens,
            temperature=temp,
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
    temperature_override: Optional[float] = None,  # For beam search diversity
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
            temperature_override=temperature_override,
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
                        rtol: float,
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
            rtol=rtol,
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
    rtol: float = 1e-2,
    phase: str = "seed",
    metrics_dir: Path | None = None,
    cached_baseline_ms: Optional[float] = None,  # Cached baseline latency (if available)
) -> Optional[float]:
    """
    Benchmark and update the individual's metrics/score; on exception, fill in
    failure info and save metrics (if a directory is provided).
    Same functionality as the original version, but runs compare_and_bench in a
    **spawned subprocess** to isolate the CUDA context.

    Args:
        cached_baseline_ms: If provided, use this cached PyTorch baseline latency instead
                           of the one from current benchmark. This improves consistency.

    Returns:
        The PyTorch baseline latency (avg ms) if this was the first run, None otherwise.
        This allows caching the baseline for subsequent stages.
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
                rtol,
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
    try:
        payload = parent_conn.recv() if parent_conn.poll() else None
    except (EOFError, BrokenPipeError) as e:
        # Child process crashed (e.g., Triton compiler assertion failure)
        payload = ("err", {"error_type": "ProcessCrashed", "message": f"Child process crashed: {e}"})
    try:
        parent_conn.close()
    except Exception:
        pass

    # —— Update metrics/score based on child payload (same logic as before) ——
    returned_baseline_ms = None  # To cache for future runs

    # Handle case where payload is None (child crashed without sending data)
    if payload is None:
        payload = ("err", {"error_type": "ProcessCrashed", "message": "Child process crashed without sending results"})

    if isinstance(payload, tuple) and len(payload) == 2 and payload[0] in ("ok", "err"):
        tag, data = payload
        if tag == "ok":
            metrics = data
            metrics["runnable"] = True
            metrics["phase"] = phase

            # Use cached baseline if provided, otherwise use current benchmark result
            if cached_baseline_ms is not None:
                baseline_ms = cached_baseline_ms
                metrics["pytorch_baseline_ms"] = baseline_ms  # Record which baseline was used
            else:
                baseline_ms = metrics["ref_latency_ms"]["avg"]
                returned_baseline_ms = baseline_ms  # Return for caching

            speedup = baseline_ms / max(1e-9, metrics["test_latency_ms"]["avg"])
            metrics["score"] = speedup

            ind.metrics = metrics
            ind.score = speedup
            print(f"[{phase}] score={speedup:.4f} (baseline={baseline_ms:.4f}ms)")

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

    return returned_baseline_ms  # Return baseline for caching (None if failed or cached was used)


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


# ---------------------- Beam Search Helper Functions -------------------
def _profile_kernel_ncu(
    kernel: Any,
    test_kernel_path: Path,
    bench_script: str,
    task_path: Path,
    device_idx: int,
    model_type: str,
    proc_id: int,
) -> tuple[Any, str]:
    """Profile a kernel with NCU and return (metrics_df, metrics_block)."""
    # Write kernel to test file
    with open(test_kernel_path, "w") as f:
        f.write(kernel.code)

    kernel_names = extract_cuda_kernel_names(test_kernel_path)

    metrics_block = "No NCU metrics available"
    metrics_df = None

    if model_type != MODEL_NETWORK:
        try:
            csv_path = profile_bench(
                bench_py=bench_script,
                out_csv=f"ncu_temp_{proc_id}.csv",
                device_idx=device_idx,
                ref_file=str(task_path),
                test_file=str(test_kernel_path.name),
            )
            metrics_df = load_ncu_metrics(
                csv_path, extra_keep=("Kernel Name",),
                name_list=kernel_names, select="last"
            )
            metrics_block = metrics_to_prompt(metrics_df)
        except Exception as e:
            print(f"\033[93mWarning: NCU profiling failed: {e}\033[0m")

    return metrics_df, metrics_block


def _optimize_stage_single(
    base_kernel: Any,
    stage_idx: int,
    stage_name: str,
    stage_description: str,
    ncu_block: str,
    args,
    call_llm,
    code_dir: Path,
    io_dir: Path,
    log_path: Path,
    stage_round: int,
    temperature: Optional[float] = None,
    beam_idx: int = 0,
) -> Any:
    """Generate a single optimized kernel for a stage.

    Args:
        base_kernel: The kernel to optimize from
        stage_idx: Stage index (0-3)
        stage_name: Stage name
        stage_description: Stage description
        ncu_block: NCU metrics formatted as string
        args: CLI arguments
        call_llm: LLM caller function
        code_dir: Directory to save kernel code
        io_dir: Directory to save prompts/responses
        log_path: Path to usage log
        stage_round: Current round number
        temperature: Temperature override for LLM (for beam search diversity)
        beam_idx: Beam candidate index (for logging)

    Returns:
        KernelIndividual: The optimized kernel (not yet benchmarked)
    """
    opt_prompt = build_optimization_prompt(
        arch_path=base_kernel.code_path,
        gpu_name=args.gpu,
        ncu_metrics=ncu_block,
        stage_name=stage_name,
        stage_description=stage_description,
        fusion=args.fusion,
        model=args.model,
    )

    # Save prompt with beam index if using beam search
    suffix = f"_beam{beam_idx}" if beam_idx > 0 else ""
    prompt_file = io_dir / f"stage{stage_idx + 1}_{stage_name}{suffix}_prompt.txt"
    prompt_file.write_text(opt_prompt, encoding="utf-8")

    call_type = f"stage{stage_idx + 1}_{stage_name}"
    if beam_idx > 0:
        call_type += f"_beam{beam_idx}"

    kernel = _llm_to_kernel(
        opt_prompt, code_dir, call_llm, io_dir, stage_round,
        log_path=log_path, call_type=call_type,
        temperature_override=temperature,
    )

    return kernel


# ---------------------- task helpers -------------------
def _natural_sort_key(path: Path) -> tuple:
    """Extract leading number from filename for natural sorting (e.g., 2_xxx < 10_xxx < 31_xxx)."""
    import re
    name = path.stem
    match = re.match(r'^(\d+)', name)
    return (int(match.group(1)), name) if match else (float('inf'), name)

def _collect_tasks(maybe_dir: Path) -> List[Path]:
    """If a directory, return all .py files (naturally sorted by leading number); if a file, return [file]."""
    if maybe_dir.is_file():
        return [maybe_dir]
    if maybe_dir.is_dir():
        return sorted([p for p in maybe_dir.rglob("*.py") if p.is_file()], key=_natural_sort_key)
    raise FileNotFoundError(f"{maybe_dir} not found")


def _pick_first_n(tasks: List[Path], n: int) -> List[Path]:
    n = max(1, min(max(n, 0), len(tasks)))
    return tasks[:n]


def _pick_range(tasks: List[Path], start_idx: int, end_idx: int) -> List[Path]:
    """Pick tasks by file number prefix (not list index).

    Args:
        tasks: List of task paths (should be sorted by _natural_sort_key)
        start_idx: Start file number prefix (inclusive)
        end_idx: End file number prefix (inclusive). If 0, no upper limit.

    Returns:
        Tasks whose leading number is in range [start_idx, end_idx]
    """
    result = []
    for task in tasks:
        match = re.match(r'^(\d+)', task.stem)
        if match:
            file_num = int(match.group(1))
            # Check if file number is within range
            if file_num >= start_idx:
                if end_idx <= 0 or file_num <= end_idx:
                    result.append(task)
    return result


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


# --------------------- Tee Logger for Console + File -----------------
class TeeLogger:
    """Write to both console and file simultaneously."""
    def __init__(self, file_path: Path, stream):
        self.file = open(file_path, "w", encoding="utf-8")
        self.stream = stream
        self.encoding = getattr(stream, 'encoding', 'utf-8')

    def write(self, message):
        self.stream.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()


# --------------------- single-task run -----------------
def _run_single_task(task_path: Path, args, batch_dir: Path) -> Dict[str, Any]:
    import os
    import sys

    # Use PID to ensure unique temporary files when running multiple processes
    proc_id = os.getpid()

    # --- per-task directories under the SAME batch_dir
    task_root = (batch_dir / task_path.stem).resolve()
    task_root.mkdir(parents=True, exist_ok=True)  # Create task root first!

    code_dir = task_root / "code"
    eval_dir = task_root / "evaluation"
    fig_dir = task_root / "figures"
    io_dir = eval_dir / "llm_io"

    code_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    io_dir.mkdir(parents=True, exist_ok=True)
    log_path = task_root / "usage.csv"

    # === Setup console log file ===
    console_log_path = task_root / "console.log"
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    tee_stdout = TeeLogger(console_log_path, old_stdout)
    tee_stderr = TeeLogger(task_root / "console_stderr.log", old_stderr)
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr

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

    # ====== Step 1: Generate Multiple Seed Programs ======
    num_seeds = args.num_seeds
    elimination_threshold = args.elimination_threshold
    pytorch_baseline_ms = None

    print(f"[Seed] Generating {num_seeds} seed candidates...")
    seed_prompt = build_seed_prompt(arch_path=task_path, gpu_name=args.gpu, fusion=args.fusion, model=args.model)
    prompt_file = io_dir / "seed_prompt.txt"
    prompt_file.write_text(seed_prompt, encoding="utf-8")

    # Generate multiple seeds (rely on model's inherent randomness for diversity)
    seed_candidates: List[BeamCandidate] = []
    for seed_idx in range(num_seeds):
        print(f"[Seed {seed_idx + 1}/{num_seeds}] Generating...")

        current_kernel = _llm_to_kernel(
            seed_prompt, code_dir, call_llm, io_dir,
            seed_idx, log_path=log_path, call_type=f"seed_{seed_idx}",
        )

        # Benchmark seed
        baseline_result = _bench_and_score(
            current_kernel,
            ref_py=task_path,
            device_idx=args.device,
            warmup=args.warmup,
            repeat=args.repeat,
            tol=args.tol,
            rtol=args.rtol,
            phase=f"seed_{seed_idx}",
            metrics_dir=eval_dir,
            cached_baseline_ms=pytorch_baseline_ms,
        )
        if pytorch_baseline_ms is None and baseline_result is not None:
            pytorch_baseline_ms = baseline_result

        runnable = bool(getattr(current_kernel, "metrics", {}).get("runnable", False))
        this_score = current_kernel.score if (current_kernel.score is not None and runnable) else 0.0

        if runnable and this_score > 0:
            seed_candidates.append(BeamCandidate(
                kernel=current_kernel,
                score=this_score,
                runnable=True,
            ))
            last_score_for_curve = this_score
            scores.append(this_score)
            err_flags.append(False)
            if this_score > best_score:
                best_score = this_score
                best_kernel = current_kernel
                with open(test_kernel, "w") as f:
                    f.write(best_kernel.code)
            print(f"[Seed {seed_idx + 1}] Score: {this_score:.4f} ✓")
        else:
            scores.append(0.0)
            err_flags.append(True)
            # Store failed seed for potential repair
            seed_candidates.append(BeamCandidate(
                kernel=current_kernel,
                score=0.0,
                runnable=False,
            ))
            print(f"[Seed {seed_idx + 1}] Failed, will attempt repair...")

    # ====== Step 2: Repair failed seeds ======
    max_repair_per_seed = 5
    for seed_idx, candidate in enumerate(seed_candidates):
        if candidate.runnable:
            continue  # Already working

        repair_attempt = 0
        error_history_list = []
        current_kernel = candidate.kernel

        while not candidate.runnable and repair_attempt < max_repair_per_seed:
            repair_attempt += 1
            print(f"[Seed {seed_idx + 1} Repair {repair_attempt}/{max_repair_per_seed}] Repairing...")
            error_log = _last_n_lines(getattr(current_kernel, "metrics", {}).get("message", ""))

            # Build error history from previous attempts (concise format)
            error_history = "\n\n".join(error_history_list[-3:]) if error_history_list else ""

            # Generate repair directly (no separate analysis step)
            repair_prompt = build_error_prompt(
                old_code=current_kernel.code,
                error_log=error_log,
                problem=None,  # No analysis step
                gpu_name=args.gpu,
                error_history=error_history,
                arch_path=task_path,
            )
            current_kernel = _llm_to_kernel(
                repair_prompt, code_dir, call_llm, io_dir,
                seed_idx * 100 + repair_attempt, log_path=log_path,
                call_type=f"seed_{seed_idx}_repair",
            )

            baseline_result = _bench_and_score(
                current_kernel,
                ref_py=task_path,
                device_idx=args.device,
                warmup=args.warmup,
                repeat=args.repeat,
                tol=args.tol,
                rtol=args.rtol,
                phase=f"seed_{seed_idx}_repair",
                metrics_dir=eval_dir,
                cached_baseline_ms=pytorch_baseline_ms,
            )
            if pytorch_baseline_ms is None and baseline_result is not None:
                pytorch_baseline_ms = baseline_result

            runnable = bool(getattr(current_kernel, "metrics", {}).get("runnable", False))
            this_score = current_kernel.score if (current_kernel.score is not None and runnable) else None

            if this_score is not None:
                # Update candidate with repaired kernel
                seed_candidates[seed_idx] = BeamCandidate(
                    kernel=current_kernel,
                    score=this_score,
                    runnable=True,
                )
                last_score_for_curve = this_score
                scores.append(this_score)
                err_flags.append(False)
                if this_score > best_score:
                    best_score = this_score
                    best_kernel = current_kernel
                    with open(test_kernel, "w") as f:
                        f.write(best_kernel.code)
                print(f"[Seed {seed_idx + 1} Repair] Score: {this_score:.4f} ✓")
                break
            else:
                scores.append(last_score_for_curve)
                err_flags.append(True)
                # Build concise error summary for history
                if error_log.strip():
                    # Extract just the key error info (last 3 lines typically contain the actual error)
                    error_lines = error_log.strip().splitlines()
                    key_lines = error_lines[-3:] if len(error_lines) > 3 else error_lines
                    error_summary = " | ".join(line.strip() for line in key_lines if line.strip())
                    error_history_list.append(f"[Attempt {repair_attempt}] {error_summary[:200]}")

    # Filter to only runnable seeds
    runnable_seeds = [c for c in seed_candidates if c.runnable]
    if not runnable_seeds:
        print(f"[ERROR] No runnable seeds after repair. Stopping.")
    else:
        # ====== Step 3: Three-stage Optimization with Soft Elimination ======
        # Strategy: Multi-Seed + Stage1 Elimination + Greedy Stage2/3
        # - All seeds → Stage 1 (parallel optimization)
        # - After Stage 1: soft elimination based on gap threshold
        #   - gap > τ: keep only top-1
        #   - gap ≤ τ: keep top-2, eliminate after Stage 2
        # - Stage 2: optimize remaining candidates
        # - Stage 3: greedy on single best

        print(f"\n[Optimization] Starting 3-stage optimization with Soft Elimination...")
        print(f"  - {len(runnable_seeds)} seed(s) → Stage 1")
        print(f"  - Elimination threshold τ = {elimination_threshold:.1%}")
        print(f"  - Stage 2/3: greedy on selected candidate(s)")

        stage_round = 0
        score_history = [best_score]

        # Initialize beam with all runnable seeds
        beam: List[BeamCandidate] = runnable_seeds.copy()

        # Setup bench script once
        bench_template_source = root_dir / "bench_ref_inputs_template.py"
        bench_script = root_dir / f"bench_ref_inputs_{proc_id}.py"
        if not bench_script.exists() or proc_id != 0:
            import shutil
            if bench_template_source.exists():
                shutil.copy(bench_template_source, bench_script)
            else:
                raise FileNotFoundError(f"Bench template not found: {bench_template_source}")

        for stage_idx in range(len(OPTIMIZATION_STAGES)):
            stage = OPTIMIZATION_STAGES[stage_idx]
            stage_name = stage["name"]
            stage_description = stage["description"]

            print(f"\n{'='*80}")
            print(f"[Stage {stage_idx + 1}/{len(OPTIMIZATION_STAGES)}] {stage_name}")
            print(f"Description: {stage_description}")
            print(f"Current candidates: {len(beam)}, best score: {best_score:.4f}")
            print(f"{'='*80}")

            # Filter beam to only runnable candidates
            beam = [c for c in beam if c.runnable]
            if not beam:
                print(f"[ERROR] No runnable candidates at stage {stage_idx + 1}. Stopping.")
                break

            # Stage 3 (last stage): always greedy, use single best
            if stage_idx == len(OPTIMIZATION_STAGES) - 1 and len(beam) > 1:
                beam = [max(beam)]
                print(f"[Stage {stage_idx + 1}] Final stage: using best candidate (score={beam[0].score:.4f})")

            # Profile all candidates in beam to get NCU metrics
            print(f"[Stage {stage_idx + 1}] Profiling {len(beam)} candidate(s)...")
            for candidate in beam:
                if candidate.ncu_block:  # Already profiled
                    continue
                metrics_df, metrics_block = _profile_kernel_ncu(
                    candidate.kernel, test_kernel, f"bench_ref_inputs_{proc_id}.py",
                    task_path, args.device, args.model, proc_id
                )
                candidate.metrics_df = metrics_df
                candidate.ncu_block = metrics_block

            # Generate new candidates: one optimization per beam candidate
            new_candidates: List[BeamCandidate] = []
            stage_round += 1

            for beam_idx, base_candidate in enumerate(beam):
                print(f"[Stage {stage_idx + 1}] Optimizing candidate {beam_idx + 1}/{len(beam)}...")

                # Generate optimized kernel
                new_kernel = _optimize_stage_single(
                    base_kernel=base_candidate.kernel,
                    stage_idx=stage_idx,
                    stage_name=stage_name,
                    stage_description=stage_description,
                    ncu_block=base_candidate.ncu_block,
                    args=args,
                    call_llm=call_llm,
                    code_dir=code_dir,
                    io_dir=io_dir,
                    log_path=log_path,
                    stage_round=stage_round + beam_idx,
                    beam_idx=beam_idx,
                )

                # Benchmark the new kernel
                _bench_and_score(
                    new_kernel,
                    ref_py=task_path,
                    device_idx=args.device,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    tol=args.tol,
                    rtol=args.rtol,
                    phase=f"stage{stage_idx + 1}_{stage_name}_cand{beam_idx}",
                    metrics_dir=eval_dir,
                    cached_baseline_ms=pytorch_baseline_ms,
                )

                # Check if optimization succeeded
                is_runnable = bool(getattr(new_kernel, "metrics", {}).get("runnable", False))
                new_score = new_kernel.score if (new_kernel.score is not None and is_runnable) else float("-inf")

                # If optimization failed, try repair (max 1 attempt per candidate)
                if not is_runnable:
                    print(f"  Candidate {beam_idx + 1}: failed, attempting repair...")
                    error_log = _last_n_lines(getattr(new_kernel, "metrics", {}).get("message", ""))

                    # Generate repair directly (no separate analysis step)
                    repair_prompt = build_error_prompt(
                        old_code=new_kernel.code,
                        error_log=error_log,
                        problem=None,  # No analysis step
                        gpu_name=args.gpu,
                        error_history="",  # Stage repair is single-attempt, no history
                        arch_path=task_path,
                    )
                    prompt_file = io_dir / f"stage{stage_idx + 1}_repair_cand{beam_idx}_prompt.txt"
                    prompt_file.write_text(repair_prompt, encoding="utf-8")

                    new_kernel = _llm_to_kernel(
                        repair_prompt, code_dir, call_llm, io_dir,
                        stage_round + beam_idx + 100,
                        log_path=log_path,
                        call_type=f"stage{stage_idx + 1}_repair_beam{beam_idx}",
                    )

                    # Re-benchmark repaired kernel
                    _bench_and_score(
                        new_kernel,
                        ref_py=task_path,
                        device_idx=args.device,
                        warmup=args.warmup,
                        repeat=args.repeat,
                        tol=args.tol,
                        rtol=args.rtol,
                        phase=f"stage{stage_idx + 1}_{stage_name}_repair_beam{beam_idx}",
                        metrics_dir=eval_dir,
                        cached_baseline_ms=pytorch_baseline_ms,
                    )

                    is_runnable = bool(getattr(new_kernel, "metrics", {}).get("runnable", False))
                    new_score = new_kernel.score if (new_kernel.score is not None and is_runnable) else float("-inf")

                # Only add to candidates if runnable (skip failed ones)
                if is_runnable:
                    new_candidates.append(BeamCandidate(
                        kernel=new_kernel,
                        score=new_score,
                        runnable=True,
                        ncu_block="",  # Will be profiled in next stage if needed
                    ))
                    last_score_for_curve = new_score
                    scores.append(new_score)
                    err_flags.append(False)
                    print(f"  Candidate {beam_idx + 1}: score={new_score:.4f} ✓")
                else:
                    scores.append(last_score_for_curve)
                    err_flags.append(True)
                    print(f"  Candidate {beam_idx + 1}: failed (after repair attempt) ✗")

            # Select top-k candidates for next stage
            # Include previous beam candidates to allow "no improvement" path
            all_candidates = new_candidates + beam
            all_candidates = [c for c in all_candidates if c.runnable]

            if not all_candidates:
                print(f"[Stage {stage_idx + 1}] All candidates failed. Keeping previous beam.")
                continue

            # Sort by score (descending) and apply soft elimination
            all_candidates.sort(reverse=True)

            # Soft elimination logic:
            # Stage 1: Keep both if gap <= threshold, else keep top-1
            # Stage 2+: Always keep top-1 (greedy)
            if stage_idx == 0 and len(all_candidates) >= 2:
                top1_score = all_candidates[0].score
                top2_score = all_candidates[1].score
                if top1_score > 0:
                    gap = (top1_score - top2_score) / top1_score
                else:
                    gap = float("inf")  # Cannot compare with non-positive scores

                if gap <= args.elimination_threshold:
                    # Soft elimination: keep both candidates
                    beam = all_candidates[:2]
                    print(f"[Stage {stage_idx + 1}] Soft elimination: gap={gap:.4f} <= τ={args.elimination_threshold}, keeping 2 candidates")
                else:
                    # Hard elimination: keep only top-1
                    beam = all_candidates[:1]
                    print(f"[Stage {stage_idx + 1}] Hard elimination: gap={gap:.4f} > τ={args.elimination_threshold}, keeping top-1")
            else:
                # Stage 2+: always greedy (keep top-1)
                beam = all_candidates[:1]

            # Update global best
            if beam[0].score > best_score:
                best_score = beam[0].score
                best_kernel = beam[0].kernel
                with open(test_kernel, "w") as f:
                    f.write(best_kernel.code)
                print(f"[Stage {stage_idx + 1}] ★ New best score: {best_score:.4f}")
            else:
                print(f"[Stage {stage_idx + 1}] Best in beam: {beam[0].score:.4f} (global best: {best_score:.4f})")

            # Log beam status
            print(f"[Stage {stage_idx + 1}] Beam after selection: {[f'{c.score:.4f}' for c in beam]}")

            score_history.append(best_score)

    # plot per-task curve
    fig_path = fig_dir / f"{task_path.stem}_score.png"
    _plot_scores(fig_path, scores, err_flags, title=f"{task_path.stem} (best={best_score:.4f})")
    print(f"[{task_path.name}] Figure saved to: {fig_path}")

    usage_totals = _append_usage_totals(log_path)

    # === Restore stdout/stderr and close log files ===
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    tee_stdout.close()
    tee_stderr.close()
    print(f"[{task_path.name}] Console log saved to: {console_log_path}")

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

    # Calculate additional statistics for failed tasks
    total_tasks = len(summary)
    successful_tasks = sum(1 for s in summary if s["best_runnable"])
    failed_tasks = total_tasks - successful_tasks

    # JSON - Enhanced with failure statistics
    out_json = {
        "avg_speedup": avg_speedup,
        "accuracy": accuracy,
        "total_tokens_sum": total_tokens_sum,
        "num_tasks": total_tasks,
        "successful_tasks": successful_tasks,
        "failed_tasks": failed_tasks,
        "tasks": summary,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    (batch_dir / "summary.json").write_text(json.dumps(out_json, indent=2), encoding="utf-8")

    # CSV - Handle missing 'figure' field for failed tasks
    csv_path = batch_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "best_score", "best_runnable", "task_dir", "figure", "status", "error"])
        for s in summary:
            writer.writerow([
                s["task"],
                f'{s["best_score"]:.6f}',
                int(bool(s["best_runnable"])),
                s["task_dir"],
                s.get("figure", "N/A"),
                s.get("status", "completed"),
                s.get("error", "")[:200]  # Truncate long error messages
            ])
        writer.writerow([])
        writer.writerow(["avg_speedup", f"{avg_speedup:.6f}"])
        writer.writerow(["accuracy", f"{accuracy:.6f}"])
        writer.writerow(["successful_tasks", str(successful_tasks)])
        writer.writerow(["failed_tasks", str(failed_tasks)])
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
        if args.start_idx > 0 or args.end_idx > 0:
            end_note = args.end_idx if args.end_idx > 0 else "end"
            pick_note = f"range{args.start_idx}to{end_note}"
        elif args.first_n and args.first_n > 0:
            pick_note = f"first{args.first_n}"
        else:
            pick_note = f"num{args.num_tasks}_seed{args.shuffle_seed}"
        batch_name = f"{stamp}_batch_{pick_note}_{run_tag}"
    batch_dir = (args.work_dir / batch_name).resolve()
    batch_dir.mkdir(parents=True, exist_ok=True)
    print(f"[BATCH] Output folder: {batch_dir}")

    # single file → run once (still inside the same batch folder)
    if args.arch_py.is_file():
        try:
            res = _run_single_task(all_tasks[0], args, batch_dir=batch_dir)
            print(f"[SUCCESS] Task completed successfully")
        except Exception as e:
            error_msg = str(e)
            print(f"\033[91m[ERROR] Task failed: {all_tasks[0].name}\033[0m")
            print(f"\033[91mError details: {error_msg}\033[0m")
            # Create a failed result entry
            res = {
                "task": str(all_tasks[0]),
                "best_score": 0.0,
                "best_runnable": False,
                "task_dir": str((batch_dir / all_tasks[0].stem).resolve()),
                "error": error_msg,
                "status": "failed"
            }

        summary = [res]
        avg_speedup = res["best_score"]
        accuracy = 1.0 if res["best_runnable"] else 0.0
        total_tokens_sum = res.get("total_tokens_sum", 0)
        print(f"[SUMMARY] {res}")
        print(f"[GLOBAL] Avg speedup={avg_speedup:.4f}, Accuracy={accuracy:.4f}")

        _save_global_summary(batch_dir, summary, avg_speedup, accuracy, total_tokens_sum)
        return

    # directory: priority order - range selection > first_n > random sampling
    if args.start_idx > 0 or args.end_idx > 0:
        # Range selection has highest priority (by file number prefix, not list index)
        picked = _pick_range(all_tasks, args.start_idx, args.end_idx)
        end_display = args.end_idx if args.end_idx > 0 else "∞"
        print(f"[Task Picker] Found {len(all_tasks)} tasks, selecting by file number prefix [{args.start_idx}, {end_display}] = {len(picked)} tasks.")
    elif args.first_n and args.first_n > 0:
        picked = _pick_first_n(all_tasks, args.first_n)
        print(f"[Task Picker] Found {len(all_tasks)} tasks, taking first {len(picked)} (sorted).")
    else:
        picked = _sample_tasks(all_tasks, args.num_tasks, args.shuffle_seed)
        print(f"[Task Picker] Found {len(all_tasks)} tasks, sampled {len(picked)} with seed={args.shuffle_seed}.")

    summary: List[Dict[str, Any]] = []
    for i, task in enumerate(picked, 1):
        print(f"\n===== [{i}/{len(picked)}] Running task: {task} =====")
        try:
            res = _run_single_task(task, args, batch_dir=batch_dir)
            summary.append(res)
            print(f"[SUCCESS] Task {i}/{len(picked)} completed: {task.name}")
        except Exception as e:
            error_msg = str(e)
            print(f"\033[91m[ERROR] Task {i}/{len(picked)} failed: {task.name}\033[0m")
            print(f"\033[91mError details: {error_msg}\033[0m")
            # Record failed task in summary
            failed_res = {
                "task": str(task),
                "best_score": 0.0,
                "best_runnable": False,
                "task_dir": str((batch_dir / task.stem).resolve()),
                "error": error_msg,
                "status": "failed"
            }
            summary.append(failed_res)
            print(f"[CONTINUE] Moving to next task...")

    # global summary using each task's best kernel
    if summary:
        # Calculate statistics
        total_tasks = len(summary)
        successful_tasks = sum(1 for s in summary if s["best_runnable"])
        failed_tasks = total_tasks - successful_tasks
        avg_speedup = sum(s["best_score"] for s in summary) / len(summary)
        accuracy = successful_tasks / total_tasks
        total_tokens_sum = sum(int(s.get("total_tokens_sum", 0) or 0) for s in summary)

        # Print detailed summary
        print("\n===== SUMMARY =====")
        print(f"Total tasks: {total_tasks}")
        print(f"Successful tasks: {successful_tasks}")
        print(f"Failed tasks: {failed_tasks}")
        print("\n[Task Details]")
        for s in summary:
            status_mark = "✓" if s["best_runnable"] else "✗"
            print(f"{status_mark} {s['task']}: best_score={s['best_score']:.4f}  runnable={s['best_runnable']}  fig={s.get('figure', 'N/A')}")

        # List failed tasks with error details
        if failed_tasks > 0:
            print("\n[FAILED TASKS DETAILS]")
            for s in summary:
                if not s["best_runnable"]:
                    task_name = Path(s["task"]).name
                    error_info = s.get("error", "Unknown error")
                    print(f"  - {task_name}: {error_info[:150]}...")

        print(f"\n[GLOBAL] Avg speedup={avg_speedup:.4f}, Accuracy={accuracy:.4f} ({successful_tasks}/{total_tasks})")

        # ---- save under the SAME batch folder ----
        _save_global_summary(batch_dir, summary, avg_speedup, accuracy, total_tokens_sum)
    else:
        print("No tasks were run.")


if __name__ == "__main__":
	main()
