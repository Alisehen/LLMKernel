#!/usr/bin/env python3
"""Test script for early exit mechanism based on NCU metrics."""

import pandas as pd
from pathlib import Path
import sys

# Add parent dir to path to import from main
sys.path.insert(0, str(Path(__file__).parent))

# Import the functions we need
from main import _should_skip_stage, STAGE_EXIT_CRITERIA

def test_should_skip_stage():
    """Test the early exit logic with different metric scenarios."""

    print("=" * 80)
    print("Testing Early Exit Mechanism (Updated with Memory Stalls)")
    print("=" * 80)

    # Test Case 1: grid_and_parallel with high SM occupancy (should skip)
    print("\n[Test 1] grid_and_parallel with SM occupancy = 92%")
    metrics_df = pd.DataFrame({
        "Kernel Name": ["test_kernel"],
        "sm__maximum_warps_per_active_cycle_pct": [92.0]
    })
    should_skip, reason = _should_skip_stage("grid_and_parallel", metrics_df)
    print(f"  Result: should_skip={should_skip}, reason='{reason}'")
    assert should_skip == True, "Should skip when SM occupancy > 90%"
    print("  ✓ PASS")

    # Test Case 2: grid_and_parallel with low SM occupancy (should NOT skip)
    print("\n[Test 2] grid_and_parallel with SM occupancy = 75%")
    metrics_df = pd.DataFrame({
        "Kernel Name": ["test_kernel"],
        "sm__maximum_warps_per_active_cycle_pct": [75.0]
    })
    should_skip, reason = _should_skip_stage("grid_and_parallel", metrics_df)
    print(f"  Result: should_skip={should_skip}, reason='{reason}'")
    assert should_skip == False, "Should NOT skip when SM occupancy < 90%"
    print("  ✓ PASS")

    # Test Case 3: block_tiling with good occupancy (should skip)
    print("\n[Test 3] block_tiling with SM occupancy = 88%")
    metrics_df = pd.DataFrame({
        "Kernel Name": ["test_kernel"],
        "sm__maximum_warps_per_active_cycle_pct": [88.0]
    })
    should_skip, reason = _should_skip_stage("block_tiling", metrics_df)
    print(f"  Result: should_skip={should_skip}, reason='{reason}'")
    assert should_skip == True, "Should skip when SM occupancy > 85%"
    print("  ✓ PASS")

    # Test Case 4: memory_access with low stalls + high DRAM (should skip, AND condition)
    print("\n[Test 4] memory_access with stalls = 8%, DRAM = 88%")
    metrics_df = pd.DataFrame({
        "Kernel Name": ["test_kernel"],
        "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct": [8.0],
        "dram__throughput.avg.pct_of_peak_sustained_elapsed": [88.0]
    })
    should_skip, reason = _should_skip_stage("memory_access", metrics_df)
    print(f"  Result: should_skip={should_skip}, reason='{reason}'")
    assert should_skip == True, "Should skip when stalls <10% AND DRAM >85%"
    print("  ✓ PASS")

    # Test Case 5: memory_access with low stalls but low DRAM (should NOT skip)
    print("\n[Test 5] memory_access with stalls = 8%, DRAM = 70%")
    metrics_df = pd.DataFrame({
        "Kernel Name": ["test_kernel"],
        "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct": [8.0],
        "dram__throughput.avg.pct_of_peak_sustained_elapsed": [70.0]
    })
    should_skip, reason = _should_skip_stage("memory_access", metrics_df)
    print(f"  Result: should_skip={should_skip}, reason='{reason}'")
    assert should_skip == False, "Should NOT skip when DRAM < 85% (AND condition)"
    print("  ✓ PASS")

    # Test Case 6: memory_access with high stalls (should NOT skip)
    print("\n[Test 6] memory_access with stalls = 35%, DRAM = 90%")
    metrics_df = pd.DataFrame({
        "Kernel Name": ["test_kernel"],
        "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct": [35.0],
        "dram__throughput.avg.pct_of_peak_sustained_elapsed": [90.0]
    })
    should_skip, reason = _should_skip_stage("memory_access", metrics_df)
    print(f"  Result: should_skip={should_skip}, reason='{reason}'")
    assert should_skip == False, "Should NOT skip when stalls > 10% (needs optimization)"
    print("  ✓ PASS")

    # Test Case 7: advanced_memory with very high L2 (should skip, OR condition)
    print("\n[Test 7] advanced_memory with L2 = 96%, stalls = 20%")
    metrics_df = pd.DataFrame({
        "Kernel Name": ["test_kernel"],
        "lts__t_sector_hit_rate.pct": [96.0],
        "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct": [20.0]
    })
    should_skip, reason = _should_skip_stage("advanced_memory", metrics_df)
    print(f"  Result: should_skip={should_skip}, reason='{reason}'")
    assert should_skip == True, "Should skip when L2 > 95% (OR condition)"
    print("  ✓ PASS")

    # Test Case 8: advanced_memory with very low stalls (should skip, OR condition)
    print("\n[Test 8] advanced_memory with L2 = 80%, stalls = 3%")
    metrics_df = pd.DataFrame({
        "Kernel Name": ["test_kernel"],
        "lts__t_sector_hit_rate.pct": [80.0],
        "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct": [3.0]
    })
    should_skip, reason = _should_skip_stage("advanced_memory", metrics_df)
    print(f"  Result: should_skip={should_skip}, reason='{reason}'")
    assert should_skip == True, "Should skip when stalls < 5% (OR condition)"
    print("  ✓ PASS")

    # Test Case 9: advanced_memory with moderate metrics (should NOT skip)
    print("\n[Test 9] advanced_memory with L2 = 85%, stalls = 15%")
    metrics_df = pd.DataFrame({
        "Kernel Name": ["test_kernel"],
        "lts__t_sector_hit_rate.pct": [85.0],
        "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct": [15.0]
    })
    should_skip, reason = _should_skip_stage("advanced_memory", metrics_df)
    print(f"  Result: should_skip={should_skip}, reason='{reason}'")
    assert should_skip == False, "Should NOT skip when both metrics are moderate"
    print("  ✓ PASS")

    # Test Case 10: Empty DataFrame (should NOT skip)
    print("\n[Test 10] Empty metrics DataFrame")
    metrics_df = pd.DataFrame()
    should_skip, reason = _should_skip_stage("grid_and_parallel", metrics_df)
    print(f"  Result: should_skip={should_skip}, reason='{reason}'")
    assert should_skip == False, "Should NOT skip with empty metrics"
    print("  ✓ PASS")

    # Test Case 11: Missing metric column (should NOT skip)
    print("\n[Test 11] Missing metric column")
    metrics_df = pd.DataFrame({
        "Kernel Name": ["test_kernel"],
        "some_other_metric": [99.0]
    })
    should_skip, reason = _should_skip_stage("grid_and_parallel", metrics_df)
    print(f"  Result: should_skip={should_skip}, reason='{reason}'")
    assert should_skip == False, "Should NOT skip when metric column missing"
    print("  ✓ PASS")

    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)


def print_configuration():
    """Print the current exit criteria configuration."""
    print("\n" + "=" * 80)
    print("Current Exit Criteria Configuration")
    print("=" * 80)

    for stage_name, criteria in STAGE_EXIT_CRITERIA.items():
        print(f"\n[{stage_name}]")
        print(f"  Description: {criteria['description']}")
        print(f"  Metrics: {criteria['metrics']}")
        print(f"  Thresholds: {criteria['thresholds']}")
        print(f"  Operator: {criteria['operator']}")


if __name__ == "__main__":
    print_configuration()
    test_should_skip_stage()

    print("\n" + "=" * 80)
    print("Summary: Early exit mechanism is working correctly!")
    print("The optimization stages will automatically skip when metrics")
    print("indicate that further optimization is unlikely to help.")
    print("=" * 80)
