#!/usr/bin/env python3
"""
Kernel Benchmark 统计分析脚本

计算指标:
- 成功率 (Success Rate)
- 平均加速比 (Mean Speedup) - 失败的记为 0
- P75, P50 (中位数) - 仅成功的
- Fast1: 加速比 > 1.0x 的任务比例
"""

import numpy as np
from pathlib import Path

def calculate_statistics(summary_file='summary_table3.md'):
    """从summary_table.md计算统计指标"""

    summary_path = Path(__file__).parent / summary_file

    # 读取summary_table.md
    with open(summary_path, 'r') as f:
        lines = f.readlines()

    all_scores = []  # 所有 kernel 的分数（失败记为 0）
    success_scores = []  # 仅成功的分数
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
            status_str = parts[4].strip()
        elif len(parts) >= 5:
            # 格式2: summary_table.md 的简化格式
            score_str = parts[2].strip()
            status_str = parts[3].strip()
        else:
            continue

        # 跳过非数据行（header, separator等）
        if not score_str or score_str in ['Speedup', 'best_score', '---------', '---']:
            continue

        total_count += 1

        try:
            score = float(score_str)
            all_scores.append(score)
            success_scores.append(score)
            success_count += 1
        except:
            # 失败的 kernel，记为 0
            all_scores.append(0.0)

    if total_count == 0:
        print(f"No valid data found in {summary_file}")
        return None

    all_scores = np.array(all_scores)
    success_scores = np.array(success_scores) if success_scores else np.array([0.0])

    # 计算统计指标
    success_rate = success_count / total_count
    mean_speedup_all = np.mean(all_scores)  # 失败记为 0
    mean_speedup_success = np.mean(success_scores) if success_count > 0 else 0.0
    p50 = np.median(success_scores) if success_count > 0 else 0.0
    p75 = np.percentile(success_scores, 75) if success_count > 0 else 0.0
    p25 = np.percentile(success_scores, 25) if success_count > 0 else 0.0
    fast1_count = np.sum(success_scores > 1.0) if success_count > 0 else 0

    # 打印结果
    print("=" * 50)
    print(f"Kernel Benchmark Statistics: {summary_file}")
    print("=" * 50)
    print(f"成功率: {success_count}/{total_count} ({success_rate*100:.1f}%)")
    print(f"平均加速比 (所有): {mean_speedup_all:.4f}x")
    print(f"平均加速比 (仅成功): {mean_speedup_success:.4f}x")
    print(f"P75: {p75:.4f}x")
    print(f"P50: {p50:.4f}x")
    print(f"P25: {p25:.4f}x")
    print(f"Fast1: {fast1_count}/{success_count} ({fast1_count/max(1,success_count)*100:.1f}% of successful)")
    print("=" * 50)

    return {
        'total': total_count,
        'success_count': success_count,
        'success_rate': success_rate,
        'mean_all': mean_speedup_all,
        'mean_success': mean_speedup_success,
        'p75': p75,
        'p50': p50,
        'p25': p25,
        'fast1_count': fast1_count,
        'fast1_ratio': fast1_count / max(1, success_count),
    }

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        stats = calculate_statistics(sys.argv[1])
    else:
        # 默认计算所有 level
        for level in ['level1', 'level2', 'level3']:
            fname = f'summary_table_{level}.md'
            if (Path(__file__).parent / fname).exists():
                print()
                stats = calculate_statistics(fname)
