#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# 你现有的 compare_and_bench.py 里定义的函数/异常
from compare_and_bench import compare_and_bench, CompilationError


def _find_reference_file(kernelbench_level_dir: Path, task_name: str) -> Optional[Path]:
    """
    在 KernelBench/levelX 下找 baseline torch 文件。
    兼容：
      - 完全同名：<task_name>.py
      - 可能存在尾部下划线差异：task_name.rstrip('_')
      - 可能是前缀/模糊匹配：glob(task_name*.py)
    """
    if not kernelbench_level_dir.exists():
        return None

    direct = kernelbench_level_dir / f"{task_name}.py"
    if direct.exists():
        return direct

    # 去掉末尾 '_' 再试
    t2 = task_name.rstrip("_")
    direct2 = kernelbench_level_dir / f"{t2}.py"
    if direct2.exists():
        return direct2

    # 模糊匹配
    cands = sorted(kernelbench_level_dir.glob(f"{task_name}*.py"))
    if not cands and t2 != task_name:
        cands = sorted(kernelbench_level_dir.glob(f"{t2}*.py"))

    return cands[0] if cands else None


def _format_speedup(x: float) -> str:
    if not math.isfinite(x) or x <= 0:
        return "N/A"
    return f"{x:.2f}"


def _iter_tasks(results_level_dir: Path) -> List[Tuple[str, Path]]:
    """
    返回 [(task_name, code_dir), ...]
    其中 code_dir = results/levelX/<task>/code
    """
    tasks: List[Tuple[str, Path]] = []
    if not results_level_dir.exists():
        return tasks

    for task_dir in sorted(results_level_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        code_dir = task_dir / "code"
        if code_dir.is_dir():
            tasks.append((task_dir.name, code_dir))
    return tasks


def _bench_one_task(
    task_name: str,
    ref_py: Path,
    code_dir: Path,
    device_idx: int,
    warmup: int,
    repeat: int,
    tol: float,
    rtol: float,
    seed: int,
    *,
    verbose: bool = False,
    log_dir: Path | None = None,
    level: str | None = None,
) -> Tuple[bool, Optional[Dict[str, Any]], List[Tuple[Path, str]], bool]:
    """
    尝试跑该 task 的所有候选，返回：
      - success: 是否至少一个候选跑通且精度通过
      - best_result: 最快的那份结果（test avg latency 最小）
      - errors: [(candidate_path, error_type_or_msg), ...]
      - cuda_corrupted: CUDA context 是否被破坏（需要停止所有后续 tasks）
    """
    import math
    import torch

    best: Optional[Dict[str, Any]] = None
    errors: List[Tuple[Path, str]] = []
    cuda_corrupted = False

    cand_files = sorted(code_dir.glob("*.py"))
    if not cand_files:
        msg = f"No .py files in {code_dir}"
        if verbose:
            print(f"[FAIL][{level}/{task_name}] {msg}")
        return False, None, [(code_dir, msg)], False

    if verbose:
        print(f"\n=== [TASK][{level}/{task_name}] ===")
        print(f"  ref:  {ref_py}")
        print(f"  code: {code_dir}")
        print(f"  candidates: {len(cand_files)}")

    # 用于写日志的目录
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

    # helper: 在同进程内提前检测 GPU 是否已被污染
    def _pre_sync(tag: str) -> bool:
        if not torch.cuda.is_available():
            return True
        try:
            torch.cuda.synchronize(device_idx)
            return True
        except Exception as e:
            msg = f"[GPU DIRTY] pre-sync failed ({tag}): {repr(e)}"
            if verbose:
                print(f"     [ERR] {msg}")
            errors.append((Path(tag), msg))
            # CUDA illegal access 这类基本不可恢复，继续跑意义不大
            return False

    def _post_sync(tag: str) -> None:
        if not torch.cuda.is_available():
            return
        # post-sync 目的：把异步错误“归因到当前 candidate”
        torch.cuda.synchronize(device_idx)

    for i, cand_py in enumerate(cand_files, 1):
        if verbose:
            print(f"  -> ({i}/{len(cand_files)}) run candidate: {cand_py.name}")

        # -------- pre-sync：把上一个 candidate 的异步错误提前暴露 --------
        if not _pre_sync(f"{level}/{task_name} before {cand_py.name}"):
            if verbose:
                print("     [HINT] GPU context likely corrupted (e.g., illegal memory access). "
                      "Stop remaining candidates for this task and all subsequent tasks.")
            cuda_corrupted = True
            break

        try:
            res = compare_and_bench(
                ref_py=ref_py,
                test_py=cand_py,
                device_idx=device_idx,
                warmup=warmup,
                repeat=repeat,
                tol=tol,
                rtol=rtol,
                seed=seed,
            )

            # -------- post-sync：把当前 candidate 的异步错误就地抛出 --------
            _post_sync(f"{level}/{task_name} after {cand_py.name}")

            test_avg = float(res["test_latency_ms"]["avg"])
            ref_avg = float(res["ref_latency_ms"]["avg"])
            spd = ref_avg / test_avg if test_avg > 0 else float("nan")

            if verbose:
                print(f"     [OK] test_avg={test_avg:.4f} ms, ref_avg={ref_avg:.4f} ms, speedup={spd:.2f}x")

            if best is None or test_avg < float(best["test_latency_ms"]["avg"]):
                best = res

        except CompilationError as e:
            full = str(e)
            short = f"CompilationError: {full[:400].replace(chr(10), ' ')}"
            errors.append((cand_py, short))

            if log_dir is not None:
                tag = f"{level}_{task_name}__{cand_py.stem}__compile.log"
                (log_dir / tag).write_text(full, encoding="utf-8", errors="ignore")

            if verbose:
                print(f"     [ERR] {short}")
                if log_dir is not None:
                    print(f"           saved full log -> {log_dir / tag}")

            continue

        except RuntimeError as e:
            # compare_and_bench 会塞完整 traceback 到 RuntimeError(str)
            full = str(e)

            # 这里再做一次 post-sync（有时错误在 compare_and_bench 内部被吞到返回后才爆）
            try:
                _post_sync(f"{level}/{task_name} post-sync exception after {cand_py.name}")
            except Exception as sync_e:
                full = full + "\n\n[Post-sync raised]\n" + repr(sync_e)

            short = f"RuntimeError: {full[:400].replace(chr(10), ' ')}"
            errors.append((cand_py, short))

            if log_dir is not None:
                tag = f"{level}_{task_name}__{cand_py.stem}__runtime.log"
                (log_dir / tag).write_text(full, encoding="utf-8", errors="ignore")

            if verbose:
                print(f"     [ERR] {short}")
                if ("illegal memory access" in full.lower()) or ("cudaerrorillegaladdress" in full.lower()):
                    print("           [HINT] Detected illegal memory access; CUDA context is now corrupted. "
                          "Stopping all remaining tasks.")
                if log_dir is not None:
                    print(f"           saved full log -> {log_dir / tag}")

            # 如果检测到 illegal memory access，标记 CUDA 被破坏并停止
            if ("illegal memory access" in full.lower()) or ("cudaerrorillegaladdress" in full.lower()):
                cuda_corrupted = True
                break

            continue

        except Exception as e:
            full = repr(e)
            short = f"Exception: {full[:400]}"
            errors.append((cand_py, short))

            if log_dir is not None:
                tag = f"{level}_{task_name}__{cand_py.stem}__exception.log"
                (log_dir / tag).write_text(full, encoding="utf-8", errors="ignore")

            if verbose:
                print(f"     [ERR] {short}")
                if log_dir is not None:
                    print(f"           saved -> {log_dir / tag}")
            continue

    if best is None and verbose:
        print(f"=== [TASK][{level}/{task_name}] NO PASSING CANDIDATE ===")

    return (best is not None), best, errors, cuda_corrupted


def _write_md(
    out_path: Path,
    rows: List[Tuple[str, str, str]],
    title: str = "KernelBench Results",
) -> None:
    """
    rows: [(task_name, speedup_str, ok_mark), ...]
    """
    lines = []
    lines.append(f"# {title}\n")
    lines.append("| Task | Speedup | Pass |\n")
    lines.append("|---|---:|:---:|\n")
    for task, spd, ok in rows:
        lines.append(f"| {task} | {spd} | {ok} |\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")


def _save_intermediate_results(
    out_base: Path,
    rows_by_level: Dict[str, List[Tuple[str, str, str]]],
    best_json: Dict[str, Any],
    err_json: Dict[str, Any],
    save_json_path: Path | None,
    save_errors_path: Path | None,
    levels: List[str],
    interrupted: bool = False,
) -> None:
    """
    保存中间结果（在 CUDA corruption 或正常结束时调用）
    """
    # 写 markdown
    if out_base.suffix.lower() == ".md":
        stem = out_base.stem
        parent = out_base.parent
    else:
        stem = out_base.name
        parent = out_base

    suffix = ".interrupted" if interrupted else ""
    for level in levels:
        if level not in rows_by_level or not rows_by_level[level]:
            continue
        out_md = parent / f"{stem}.{level}{suffix}.md"
        _write_md(
            out_md,
            rows_by_level[level],
            title=f"KernelBench {level} Best-of-Code Summary{' (INTERRUPTED)' if interrupted else ''}",
        )
        print(f"[OK] Wrote markdown: {out_md}")

    # 保存 json
    if save_json_path is not None and best_json:
        save_json_path.parent.mkdir(parents=True, exist_ok=True)
        if interrupted:
            final_path = save_json_path.parent / f"{save_json_path.stem}{suffix}{save_json_path.suffix}"
        else:
            final_path = save_json_path
        final_path.write_text(
            json.dumps(best_json, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[OK] Wrote best json: {final_path}")

    if save_errors_path is not None and err_json:
        save_errors_path.parent.mkdir(parents=True, exist_ok=True)
        if interrupted:
            final_path = save_errors_path.parent / f"{save_errors_path.stem}{suffix}{save_errors_path.suffix}"
        else:
            final_path = save_errors_path
        final_path.write_text(
            json.dumps(err_json, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[OK] Wrote errors json: {final_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernelbench", type=Path, required=True,
                    help="KernelBench root (contains level1/level2/level3)")
    ap.add_argument("--results", type=Path, required=True,
                    help="results root (contains level1/level2/level3)")
    ap.add_argument("--out", type=Path, default=Path("summary.md"),
                    help="output markdown base name, e.g. summary.md")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--tol", type=float, default=1e-1)
    ap.add_argument("--rtol", type=float, default=1e-1)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--save_json", type=Path, default=None,
                    help="optional: save best results json per task")
    ap.add_argument("--save_errors", type=Path, default=None,
                    help="optional: save errors json")
    ap.add_argument("--verbose", action="store_true",
                help="print per-task and per-candidate errors to console")
    ap.add_argument("--log_dir", type=Path, default=Path("run/task_logs"),
                help="save full error logs per candidate here")
    ap.add_argument("--levels", type=str, nargs='+', default=["level1", "level2", "level3"],
                choices=["level1", "level2", "level3"],
                help="which levels to run (default: all)")
    ap.add_argument("--skip-tasks", type=int, default=0,
                help="skip first N tasks (for resuming after CUDA corruption)")
    args = ap.parse_args()

    # ========== 每个 level 一张表 ==========
    rows_by_level: Dict[str, List[Tuple[str, str, str]]] = {
        level: [] for level in args.levels
    }

    best_json: Dict[str, Any] = {}
    err_json: Dict[str, Any] = {}

    task_counter = 0  # 用于跳过前 N 个任务
    cuda_context_corrupted = False  # 全局 CUDA corruption 标志

    for level in args.levels:
        if cuda_context_corrupted:
            break  # CUDA 已破坏，停止所有后续 levels

        kb_level = args.kernelbench / level
        rs_level = args.results / level
        tasks = _iter_tasks(rs_level)

        for task_name, code_dir in tasks:
            # -------- 跳过前 N 个任务（用于断点续跑） --------
            if task_counter < args.skip_tasks:
                task_counter += 1
                if args.verbose:
                    print(f"[SKIP] Skipping task {task_counter}: {level}/{task_name}")
                continue
            task_counter += 1
            ref_py = _find_reference_file(kb_level, task_name)
            task_key = f"{level}/{task_name}"

            # -------- baseline 缺失 --------
            if ref_py is None:
                rows_by_level[level].append((task_name, "N/A", "❌"))
                err_json[task_key] = {
                    "ref_missing": True,
                    "code_dir": str(code_dir),
                }
                continue

            # -------- 跑所有 candidate，取最优 --------
            ok, best, errors, cuda_corrupted = _bench_one_task(
                task_name=task_name,
                ref_py=ref_py,
                code_dir=code_dir,
                device_idx=args.device,
                warmup=args.warmup,
                repeat=args.repeat,
                tol=args.tol,
                rtol=args.rtol,
                seed=args.seed,
                verbose=args.verbose,
                log_dir=args.log_dir,
                level=level,
            )

            # -------- 检测 CUDA corruption --------
            if cuda_corrupted:
                cuda_context_corrupted = True
                print("\n" + "="*80)
                print("[CRITICAL] CUDA context corrupted. Saving current results and stopping.")
                print(f"[INFO] Completed {task_counter} tasks before corruption.")
                print(f"[INFO] To resume, use: --skip-tasks {task_counter}")
                print("="*80 + "\n")
                # 保存当前结果
                _save_intermediate_results(
                    out_base=args.out,
                    rows_by_level=rows_by_level,
                    best_json=best_json,
                    err_json=err_json,
                    save_json_path=args.save_json,
                    save_errors_path=args.save_errors,
                    levels=args.levels,
                    interrupted=True,
                )
                # 退出
                return

            if ok and best is not None:
                ref_avg = float(best["ref_latency_ms"]["avg"])
                test_avg = float(best["test_latency_ms"]["avg"])
                speedup = ref_avg / test_avg if test_avg > 0 else float("nan")

                rows_by_level[level].append(
                    (task_name, _format_speedup(speedup), "✅")
                )

                best_json[task_key] = {
                    "task": task_name,
                    "level": level,
                    "ref": str(ref_py),
                    "best_candidate": str(best["candidate_file"]),
                    "speedup": speedup,
                    "ref_avg_ms": ref_avg,
                    "test_avg_ms": test_avg,
                    "align_stats": best.get("align_stats", None),
                    "max_abs_err": best.get("max_abs_err", None),
                    "mean_abs_err": best.get("mean_abs_err", None),
                }
            else:
                rows_by_level[level].append((task_name, "N/A", "❌"))

            if errors:
                err_json[task_key] = {
                    "ref": str(ref_py),
                    "code_dir": str(code_dir),
                    "errors": [(str(p), msg) for p, msg in errors],
                }

    # ========== 正常结束，保存最终结果 ==========
    if args.verbose:
        print(f"\n[INFO] All tasks completed successfully. Total tasks: {task_counter}")

    _save_intermediate_results(
        out_base=args.out,
        rows_by_level=rows_by_level,
        best_json=best_json,
        err_json=err_json,
        save_json_path=args.save_json,
        save_errors_path=args.save_errors,
        levels=args.levels,
        interrupted=False,
    )


if __name__ == "__main__":
    main()