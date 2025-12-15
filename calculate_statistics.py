#!/usr/bin/env python3
"""
Kernel Benchmark 统计分析脚本

计算指标:
- 平均加速比 (Mean Speedup)
- P75, P50 (中位数)
- Fast1: 加速比 > 1.0x 的任务比例
"""

import numpy as np
from pathlib import Path

def calculate_statistics(summary_file='summary_table2.md'):
    """从summary_table.md计算统计指标"""

    summary_path = Path(__file__).parent / summary_file

    # 读取summary_table.md
    with open(summary_path, 'r') as f:
        lines = f.readlines()

    scores = []
    for line in lines[2:]:  # 跳过header
        parts = line.strip().split('|')
        if len(parts) < 3:
            continue
        score_str = parts[2].strip()
        try:
            score = float(score_str)
            scores.append(score)
        except:
            continue

    scores = np.array(scores)

    # 计算统计指标
    mean_speedup = np.mean(scores)
    p50 = np.median(scores)
    p75 = np.percentile(scores, 75)
    p25 = np.percentile(scores, 25)
    fast1_count = np.sum(scores > 1.0)
    total_count = len(scores)

    # 打印结果
    print("=" * 50)
    print("Kernel Benchmark Statistics")
    print("=" * 50)
    print(f"平均加速比: {mean_speedup:.4f}x")
    print(f"P75: {p75:.4f}x")
    print(f"P50: {p50:.4f}x")
    print(f"P25: {p25:.4f}x")
    print(f"Fast1: {fast1_count}/{total_count} ({fast1_count/total_count*100:.1f}%)")
    print("=" * 50)

    return {
        'mean': mean_speedup,
        'p75': p75,
        'p50': p50,
        'p25': p25,
        'fast1_count': fast1_count,
        'fast1_ratio': fast1_count / total_count,
        'total': total_count
    }

if __name__ == '__main__':
    stats = calculate_statistics()
