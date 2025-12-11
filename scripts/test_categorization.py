#!/usr/bin/env python3
"""
测试算子分类系统

使用方法:
  python scripts/test_categorization.py
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.operator_categories_v2 import (
    classify_operator,
    OPERATOR_CATEGORIES,
    get_stage_config,
    build_stage_prompt_section,
    get_key_ncu_metrics,
    check_early_exit,
)


def test_classification():
    """测试算子分类功能"""
    print("="*80)
    print("测试算子分类")
    print("="*80)

    test_cases = [
        # (op_name, level, expected_category)
        ("1_Square_matrix_multiplication_", "level1", "Compute-Intensive"),
        ("67_conv_standard_1D", "level1", "Memory-Intensive"),
        ("19_ReLU", "level1", "Memory-Intensive"),
        ("56_Matmul_Sigmoid_Sum", "level2", "Fusion-Compute"),
        ("1_Conv2D_ReLU_BiasAdd", "level2", "Fusion-Memory"),
        ("12_Gemm_Multiply_LeakyReLU", "level2", "Fusion-Compute"),
    ]

    passed = 0
    failed = 0

    for op_name, level, expected in test_cases:
        actual = classify_operator(op_name, level)
        status = "✅" if actual == expected else "❌"
        if actual == expected:
            passed += 1
        else:
            failed += 1

        print(f"{status} {op_name:40s} ({level:6s}) -> {actual:20s} (expected: {expected})")

    print(f"\n结果: {passed} passed, {failed} failed")
    return failed == 0


def test_stage_configs():
    """测试每个类别的stage配置"""
    print("\n" + "="*80)
    print("测试Stage配置")
    print("="*80)

    for category, config in OPERATOR_CATEGORIES.items():
        print(f"\n【{category}】")
        print(f"  描述: {config['description']}")
        print(f"  算子数: {config['count']}")
        print(f"  Stage数: {len(config['stages'])}")

        for i, stage in enumerate(config['stages']):
            print(f"\n  Stage {i+1}: {stage['name']}")
            print(f"    描述: {stage['description']}")
            print(f"    关注: {stage['focus']}")
            print(f"    关键指标数: {len(stage['key_metrics'])}")

            # 列出关键指标
            for metric_name in stage['key_metrics']:
                print(f"      • {metric_name}")


def test_prompt_generation():
    """测试prompt生成"""
    print("\n" + "="*80)
    print("测试Prompt生成")
    print("="*80)

    # 测试Compute-Intensive的stage1 prompt
    category = "Compute-Intensive"
    stage_id = 0

    prompt_section = build_stage_prompt_section(category, stage_id)

    print(f"\n【示例】{category} - Stage {stage_id+1}")
    print("-"*80)
    print(prompt_section[:500])  # 只显示前500字符
    print("...")
    print("-"*80)


def test_key_metrics():
    """测试关键指标提取"""
    print("\n" + "="*80)
    print("测试关键指标提取")
    print("="*80)

    test_cases = [
        ("Compute-Intensive", 0),
        ("Compute-Intensive", 1),
        ("Memory-Intensive", 0),
        ("Fusion-Compute", 2),
    ]

    for category, stage_id in test_cases:
        metrics = get_key_ncu_metrics(category, stage_id)
        stage_config = get_stage_config(category, stage_id)

        print(f"\n{category} - {stage_config['name']}")
        print(f"  关键指标:")
        for name, ncu_metric in metrics.items():
            print(f"    • {name:30s} -> {ncu_metric}")


def test_early_exit():
    """测试early exit逻辑"""
    print("\n" + "="*80)
    print("测试Early Exit逻辑")
    print("="*80)

    test_cases = [
        # (category, stage_id, score, metadata, should_exit)
        ("Memory-Intensive", 0, 0.2, {"op_type": "conv", "kernel_size": 3}, True),
        ("Memory-Intensive", 0, 0.5, {"op_type": "conv", "kernel_size": 3}, False),
        ("Fusion-Compute", 0, 0.4, {"op_type": "matmul"}, True),
        ("Fusion-Compute", 0, 0.6, {"op_type": "matmul"}, False),
    ]

    for category, stage_id, score, metadata, expected_exit in test_cases:
        should_exit, reason = check_early_exit(category, stage_id, score, metadata)

        status = "✅" if should_exit == expected_exit else "❌"
        print(f"{status} {category:20s} stage{stage_id} score={score:.1f} -> exit={should_exit} (expected={expected_exit})")
        if should_exit:
            print(f"   Reason: {reason}")


def test_full_workflow():
    """测试完整工作流程"""
    print("\n" + "="*80)
    print("测试完整工作流程")
    print("="*80)

    # 模拟一个Matmul_Sigmoid_Sum算子的优化流程
    op_name = "56_Matmul_Sigmoid_Sum"
    level = "level2"

    print(f"\n算子: {op_name} ({level})")

    # Step 1: 分类
    category = classify_operator(op_name, level)
    print(f"  1. 分类: {category}")

    # Step 2: 获取stages
    category_config = OPERATOR_CATEGORIES[category]
    stages = category_config['stages']
    print(f"  2. Stage数: {len(stages)}")

    # Step 3: 遍历每个stage
    for stage_id, stage_config in enumerate(stages):
        print(f"\n  Stage {stage_id+1}: {stage_config['name']}")

        # 获取关键指标
        metrics = get_key_ncu_metrics(category, stage_id)
        print(f"     关键指标数: {len(metrics)}")

        # 生成prompt section
        prompt = build_stage_prompt_section(category, stage_id)
        print(f"     Prompt长度: {len(prompt)} chars")

        # 检查是否应该early exit (假设一些分数)
        test_score = 0.8 + stage_id * 0.1
        should_exit, reason = check_early_exit(
            category, stage_id, test_score, {"op_type": "matmul"}
        )
        if should_exit:
            print(f"     ⛔ Early Exit: {reason}")
            break
        else:
            print(f"     ✅ Continue (score={test_score:.2f})")


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("算子分类系统测试")
    print("="*80 + "\n")

    tests = [
        ("分类功能", test_classification),
        ("Stage配置", test_stage_configs),
        ("Prompt生成", test_prompt_generation),
        ("关键指标", test_key_metrics),
        ("Early Exit", test_early_exit),
        ("完整流程", test_full_workflow),
    ]

    results = []
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, "✅ 通过"))
        except Exception as e:
            results.append((name, f"❌ 失败: {e}"))

    # 汇总结果
    print("\n" + "="*80)
    print("测试汇总")
    print("="*80)
    for name, result in results:
        print(f"  {name:20s} {result}")


if __name__ == "__main__":
    main()
