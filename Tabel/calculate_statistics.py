#!/usr/bin/env python3
"""
Kernel Benchmark 统计分析脚本

计算指标 (所有指标基于全部样本，失败记为0):
- 成功率 (Success Rate)
- 平均加速比 (Mean Speedup)
- P75, P50, P25
- Fast1: 加速比 > 1.0x 的任务比例
"""

import numpy as np
from pathlib import Path

def calculate_statistics(summary_file='summary_table2.md'):
    """从summary_table.md计算统计指标，失败的记为0分"""

    summary_path = Path(__file__).parent / summary_file

    # 读取summary_table.md
    with open(summary_path, 'r') as f:
        lines = f.readlines()

    all_scores = []  # 所有 kernel 的分数（失败记为 0）
    total_count = 0
    success_count = 0

    for line in lines[2:]:  # 跳过header
        parts = line.strip().split('|')
        if len(parts) < 4:
            continue

        # 检查是否是有效行（包含 kernel 数据）
        # 支持两种格式：
        # 格式1: | # | Kernel | Speedup | Status | Ref | Triton | (8 parts, score在parts[3])
        # 格式2: | 算子 | best_score | 状态 | (5 parts, score在parts[2])
        if len(parts) >= 8:
            # 格式1: level1/2/3 的详细格式
            score_str = parts[3].strip()
        elif len(parts) >= 5:
            # 格式2: summary_table.md 的简化格式
            score_str = parts[2].strip()
        else:
            continue

        # 跳过非数据行（header, separator等）
        if not score_str or score_str in ['Speedup', 'best_score', '---------', '---']:
            continue

        total_count += 1

        try:
            score = float(score_str)
        except:
            score = 0.0

        all_scores.append(score)
        if score > 0:
            success_count += 1

    if total_count == 0:
        print(f"No valid data found in {summary_file}")
        return None

    all_scores = np.array(all_scores)

    # 计算统计指标（全部基于 all_scores，失败记为0）
    success_rate = success_count / total_count
    mean_speedup = np.mean(all_scores)
    p50 = np.median(all_scores)
    p75 = np.percentile(all_scores, 75)
    p25 = np.percentile(all_scores, 25)
    fast1_count = int(np.sum(all_scores > 1.0))

    # 打印结果
    print("=" * 50)
    print(f"Kernel Benchmark Statistics: {summary_file}")
    print("=" * 50)
    print(f"成功率: {success_count}/{total_count} ({success_rate*100:.1f}%)")
    print(f"平均加速比: {mean_speedup:.4f}x")
    print(f"P75: {p75:.4f}x")
    print(f"P50: {p50:.4f}x")
    print(f"P25: {p25:.4f}x")
    print(f"Fast1: {fast1_count}/{total_count} ({fast1_count/total_count*100:.1f}%)")
    print("=" * 50)

    return {
        'total': total_count,
        'success_count': success_count,
        'success_rate': success_rate,
        'mean': mean_speedup,
        'p75': p75,
        'p50': p50,
        'p25': p25,
        'fast1_count': fast1_count,
        'fast1_ratio': fast1_count / total_count,
    }

if __name__ == '__main__':
    import sys
    stats = calculate_statistics()
