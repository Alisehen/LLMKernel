#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced NCU Metrics Context Builder

Builds rich context from NCU metrics including:
- Current stage metrics
- Baseline comparison (Stage 1)
- Bottleneck analysis
- Optimization potential estimation
"""

from typing import Dict, Any, Optional, List
import json


def analyze_bottleneck(metrics: Dict[str, float], category: str) -> Dict[str, Any]:
    """
    Analyze performance bottleneck from NCU metrics.

    Args:
        metrics: NCU metric values (e.g., {"dram_throughput": 88.26, ...})
        category: Operator category

    Returns:
        Bottleneck analysis dictionary
    """
    bottleneck = {
        "type": "unknown",
        "severity": "unknown",
        "description": "",
        "recommendation": "",
    }

    # Get key metrics
    dram_throughput = metrics.get("dram__throughput.avg.pct_of_peak_sustained_elapsed", 0)
    compute_throughput = metrics.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", 0)
    memory_coalescing = metrics.get("smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct", 0)
    l2_hit_rate = metrics.get("lts__t_sector_hit_rate.pct", 0)
    warp_occupancy = metrics.get("sm__warps_active.avg.pct_of_peak_sustained_active", 0)

    # Bottleneck classification rules
    if dram_throughput > 80:
        if memory_coalescing < 70:
            bottleneck = {
                "type": "memory_access_pattern",
                "severity": "high",
                "description": f"High DRAM throughput ({dram_throughput:.1f}%) but poor coalescing ({memory_coalescing:.1f}%)",
                "recommendation": "Improve memory access pattern: use stride-based access, align to 128-byte boundaries",
            }
        elif compute_throughput < 5:
            bottleneck = {
                "type": "memory_bandwidth",
                "severity": "medium",
                "description": f"DRAM bandwidth saturated ({dram_throughput:.1f}%), compute underutilized ({compute_throughput:.1f}%)",
                "recommendation": "Memory-bound kernel. Consider data reuse (shared memory) or algorithm improvements",
            }
        else:
            bottleneck = {
                "type": "balanced",
                "severity": "low",
                "description": f"DRAM throughput {dram_throughput:.1f}%, compute {compute_throughput:.1f}% - reasonably balanced",
                "recommendation": "Already efficient. Further optimization may have limited impact",
            }
    elif compute_throughput > 60:
        bottleneck = {
            "type": "compute_bound",
            "severity": "medium",
            "description": f"High compute throughput ({compute_throughput:.1f}%), low memory ({dram_throughput:.1f}%)",
            "recommendation": "Compute-bound. Focus on algorithmic improvements or use tensor cores if applicable",
        }
    elif l2_hit_rate < 70:
        bottleneck = {
            "type": "cache_locality",
            "severity": "high",
            "description": f"Low L2 hit rate ({l2_hit_rate:.1f}%), poor data locality",
            "recommendation": "Improve spatial/temporal locality: adjust block tiling, enable pipelining (num_stages)",
        }
    elif warp_occupancy < 50:
        bottleneck = {
            "type": "low_occupancy",
            "severity": "high",
            "description": f"Low warp occupancy ({warp_occupancy:.1f}%), GPU underutilized",
            "recommendation": "Reduce register usage, tune num_warps, or increase grid size",
        }
    else:
        bottleneck = {
            "type": "suboptimal",
            "severity": "medium",
            "description": f"Moderate utilization: DRAM {dram_throughput:.1f}%, Compute {compute_throughput:.1f}%",
            "recommendation": "Room for improvement. Profile deeper to identify specific bottleneck",
        }

    return bottleneck


def estimate_optimization_potential(
    category: str,
    stage_name: str,
    current_score: float,
    current_metrics: Dict[str, float],
    baseline_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Estimate optimization potential for the next stage.

    Args:
        category: Operator category
        stage_name: Current/next stage name
        current_score: Current speedup score
        current_metrics: Current NCU metrics
        baseline_metrics: Stage 1 baseline metrics (if available)

    Returns:
        Potential estimation dictionary
    """
    potential = {
        "level": "unknown",  # low, medium, high
        "expected_gain": "unknown",
        "confidence": "low",  # low, medium, high
        "reasoning": "",
        "should_proceed": True,
    }

    # Category-specific rules
    if category == "Activation":
        # Element-wise activations: very limited potential
        if current_score < 1.0:
            potential = {
                "level": "very_low",
                "expected_gain": "<5%",
                "confidence": "high",
                "reasoning": "Element-wise activation already slower than PyTorch. Further optimization unlikely to help.",
                "should_proceed": False,
            }
        elif current_score < 1.1:
            potential = {
                "level": "low",
                "expected_gain": "<3%",
                "confidence": "high",
                "reasoning": "Element-wise ops have minimal optimization space. Best to stop here.",
                "should_proceed": False,
            }

    elif category == "Normalization":
        dram = current_metrics.get("dram__throughput.avg.pct_of_peak_sustained_elapsed", 0)
        coalescing = current_metrics.get("smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct", 0)

        if "shared_memory" in stage_name.lower():
            # Shared memory stage for Norm
            if dram > 85 and coalescing > 95:
                potential = {
                    "level": "very_low",
                    "expected_gain": "<5%",
                    "confidence": "high",
                    "reasoning": f"Memory already efficient (DRAM {dram:.1f}%, coalescing {coalescing:.1f}%). "
                                 "Norm ops have limited data reuse. Shared memory won't help much.",
                    "should_proceed": False,
                }
        elif "welford" in stage_name.lower() or "algorithm" in stage_name.lower():
            # Algorithm improvements
            potential = {
                "level": "medium",
                "expected_gain": "5-15%",
                "confidence": "medium",
                "reasoning": "Welford algorithm reduces memory reads by ~33%. Expected 5-15% speedup for Norm ops.",
                "should_proceed": True,
            }

    elif category == "Conv":
        kernel_size = current_metrics.get("kernel_size", 3)  # Default 3

        if kernel_size <= 3 and current_score < 0.5:
            potential = {
                "level": "very_low",
                "expected_gain": "<10%",
                "confidence": "high",
                "reasoning": f"Small kernel conv ({kernel_size}x{kernel_size}) with score {current_score:.2f}. "
                             "cuDNN is highly optimized for small kernels. Hard to compete.",
                "should_proceed": False,
            }
        elif "shared_memory" in stage_name.lower():
            if kernel_size >= 5:
                potential = {
                    "level": "high",
                    "expected_gain": "20-40%",
                    "confidence": "medium",
                    "reasoning": f"Large kernel ({kernel_size}x{kernel_size}) has significant data reuse. "
                                 "Shared memory can reduce global loads substantially.",
                    "should_proceed": True,
                }
            else:
                potential = {
                    "level": "low",
                    "expected_gain": "<10%",
                    "confidence": "medium",
                    "reasoning": f"Small kernel ({kernel_size}x{kernel_size}) has limited reuse. "
                                 "Shared memory overhead may exceed benefit.",
                    "should_proceed": False,
                }

    elif category == "MatMul":
        compute_throughput = current_metrics.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", 0)
        warp_occupancy = current_metrics.get("sm__warps_active.avg.pct_of_peak_sustained_active", 0)

        if "pipelining" in stage_name.lower():
            if compute_throughput < 40:
                potential = {
                    "level": "high",
                    "expected_gain": "15-30%",
                    "confidence": "high",
                    "reasoning": f"Low compute throughput ({compute_throughput:.1f}%). "
                                 "Software pipelining (num_stages) can hide memory latency.",
                    "should_proceed": True,
                }
        elif "warp" in stage_name.lower():
            if warp_occupancy < 60:
                potential = {
                    "level": "medium",
                    "expected_gain": "10-20%",
                    "confidence": "medium",
                    "reasoning": f"Low occupancy ({warp_occupancy:.1f}%). "
                                 "Tuning num_warps can improve SM utilization.",
                    "should_proceed": True,
                }

    # General rule: if score already > 2.0, diminishing returns
    if current_score > 2.0:
        potential["level"] = "low"
        potential["expected_gain"] = "<5%"
        potential["confidence"] = "high"
        potential["reasoning"] += f" Current score {current_score:.2f} already excellent. Diminishing returns likely."
        potential["should_proceed"] = True  # Still proceed but with caution

    return potential


def compare_with_baseline(
    current_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    key_metrics: List[str],
) -> Dict[str, Any]:
    """
    Compare current metrics with baseline (Stage 1).

    Args:
        current_metrics: Current stage metrics
        baseline_metrics: Stage 1 baseline metrics
        key_metrics: List of metric names to compare

    Returns:
        Comparison analysis
    """
    comparison = {
        "improved": [],
        "degraded": [],
        "unchanged": [],
        "summary": "",
    }

    for metric in key_metrics:
        if metric not in current_metrics or metric not in baseline_metrics:
            continue

        curr_val = current_metrics[metric]
        base_val = baseline_metrics[metric]

        # Determine if higher is better (most metrics, except stalls)
        higher_is_better = "stall" not in metric.lower() and "latency" not in metric.lower()

        # Calculate change percentage
        if base_val != 0:
            change_pct = ((curr_val - base_val) / base_val) * 100
        else:
            change_pct = 0

        # Classify change (>5% threshold for significance)
        if abs(change_pct) < 5:
            comparison["unchanged"].append({
                "metric": metric,
                "current": curr_val,
                "baseline": base_val,
                "change_pct": change_pct,
            })
        elif (higher_is_better and change_pct > 0) or (not higher_is_better and change_pct < 0):
            comparison["improved"].append({
                "metric": metric,
                "current": curr_val,
                "baseline": base_val,
                "change_pct": abs(change_pct),
            })
        else:
            comparison["degraded"].append({
                "metric": metric,
                "current": curr_val,
                "baseline": base_val,
                "change_pct": abs(change_pct),
            })

    # Generate summary
    if comparison["degraded"]:
        worst = max(comparison["degraded"], key=lambda x: x["change_pct"])
        comparison["summary"] = f"⚠️ Regression detected: {worst['metric']} degraded by {worst['change_pct']:.1f}%"
    elif comparison["improved"]:
        best = max(comparison["improved"], key=lambda x: x["change_pct"])
        comparison["summary"] = f"✅ Improvement: {best['metric']} improved by {best['change_pct']:.1f}%"
    else:
        comparison["summary"] = "Metrics unchanged from baseline"

    return comparison


def build_enhanced_ncu_context(
    current_metrics: Dict[str, float],
    category: str,
    stage_name: str,
    current_score: float,
    baseline_metrics: Optional[Dict[str, float]] = None,
    baseline_score: Optional[float] = None,
) -> str:
    """
    Build enhanced NCU metrics context for LLM prompt.

    Args:
        current_metrics: Current stage NCU metrics
        category: Operator category
        stage_name: Current stage name
        current_score: Current speedup score
        baseline_metrics: Stage 1 baseline metrics
        baseline_score: Stage 1 baseline score

    Returns:
        Formatted context string for LLM prompt
    """
    context_parts = []

    # 1. Current Metrics
    context_parts.append("# Current NCU Metrics")
    context_parts.append("```json")
    context_parts.append(json.dumps(current_metrics, indent=2))
    context_parts.append("```")
    context_parts.append("")

    # 2. Baseline Comparison (if available)
    if baseline_metrics and baseline_score is not None:
        context_parts.append("# Baseline Comparison (Stage 1)")
        context_parts.append(f"**Stage 1 Score**: {baseline_score:.4f}x")
        context_parts.append(f"**Current Score**: {current_score:.4f}x")
        context_parts.append(f"**Gain from Stage 1**: {((current_score / baseline_score - 1) * 100):.1f}%")
        context_parts.append("")

        # Detailed metric comparison
        key_metrics = list(current_metrics.keys())
        comparison = compare_with_baseline(current_metrics, baseline_metrics, key_metrics)

        context_parts.append(f"**Comparison Summary**: {comparison['summary']}")
        context_parts.append("")

        if comparison["improved"]:
            context_parts.append("**Improved Metrics**:")
            for item in comparison["improved"]:
                context_parts.append(
                    f"  • {item['metric']}: {item['baseline']:.2f} → {item['current']:.2f} (+{item['change_pct']:.1f}%)"
                )
            context_parts.append("")

        if comparison["degraded"]:
            context_parts.append("**Degraded Metrics** (⚠️ Attention needed):")
            for item in comparison["degraded"]:
                context_parts.append(
                    f"  • {item['metric']}: {item['baseline']:.2f} → {item['current']:.2f} (-{item['change_pct']:.1f}%)"
                )
            context_parts.append("")

    # 3. Bottleneck Analysis
    bottleneck = analyze_bottleneck(current_metrics, category)
    context_parts.append("# Bottleneck Analysis")
    context_parts.append(f"**Type**: {bottleneck['type']}")
    context_parts.append(f"**Severity**: {bottleneck['severity']}")
    context_parts.append(f"**Description**: {bottleneck['description']}")
    context_parts.append(f"**Recommendation**: {bottleneck['recommendation']}")
    context_parts.append("")

    # 4. Optimization Potential
    potential = estimate_optimization_potential(
        category, stage_name, current_score, current_metrics, baseline_metrics
    )
    context_parts.append("# Optimization Potential for This Stage")
    context_parts.append(f"**Potential Level**: {potential['level']}")
    context_parts.append(f"**Expected Gain**: {potential['expected_gain']}")
    context_parts.append(f"**Confidence**: {potential['confidence']}")
    context_parts.append(f"**Reasoning**: {potential['reasoning']}")

    if not potential["should_proceed"]:
        context_parts.append("")
        context_parts.append("⛔ **RECOMMENDATION: Consider skipping this stage**")
        context_parts.append("The analysis suggests this optimization is unlikely to yield significant improvements.")
        context_parts.append("You may proceed if you have a specific idea, but manage expectations.")

    context_parts.append("")

    return "\n".join(context_parts)


def extract_metrics_from_df(metrics_df, kernel_name: Optional[str] = None) -> Dict[str, float]:
    """
    Extract metrics from pandas DataFrame into a dictionary.

    Args:
        metrics_df: Pandas DataFrame with NCU metrics
        kernel_name: Specific kernel name to filter (if multiple kernels)

    Returns:
        Dictionary of metric name -> value
    """
    if metrics_df is None or metrics_df.empty:
        return {}

    # If multiple kernels, filter by name
    if kernel_name and "Kernel Name" in metrics_df.columns:
        metrics_df = metrics_df[metrics_df["Kernel Name"] == kernel_name]

    if metrics_df.empty:
        return {}

    # Take first row (or average if multiple invocations)
    row = metrics_df.iloc[0]

    # Convert to dictionary, excluding non-metric columns
    metrics = {}
    for col in metrics_df.columns:
        if col not in ["Kernel Name", "ID", "Invocations"]:
            try:
                metrics[col] = float(row[col])
            except (ValueError, TypeError):
                pass

    return metrics


# Core metrics that should be available for all stages (for baseline comparison)
CORE_NCU_METRICS = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",  # Compute utilization
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",  # Memory bandwidth
    "lts__t_sector_hit_rate.pct",  # L2 cache hit rate
    "sm__warps_active.avg.pct_of_peak_sustained_active",  # Warp occupancy
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",  # Memory coalescing
]
