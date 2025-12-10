#!/bin/bash
# Test NCU profiling with different configurations

set -e  # Exit on error

echo "=========================================="
echo "NCU Profiling Test Suite"
echo "=========================================="

# Check perf_event_paranoid setting
echo ""
echo "1. Checking perf_event_paranoid setting:"
PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid)
echo "   Current value: $PARANOID"
if [ "$PARANOID" -le 2 ]; then
    echo "   ✓ Should be OK for NCU (value <= 2)"
else
    echo "   ✗ May block NCU (value > 2)"
    echo "   Run: sudo sysctl -w kernel.perf_event_paranoid=2"
fi

# Check NCU version
echo ""
echo "2. NCU version:"
ncu --version | head -3

# Check if CUDA_VISIBLE_DEVICES is set
echo ""
echo "3. Environment:"
echo "   CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "   Python: $(which python)"
echo "   Current dir: $(pwd)"

# Test 1: Run kernel without NCU
echo ""
echo "=========================================="
echo "Test 1: Run kernel WITHOUT NCU (baseline)"
echo "=========================================="
CUDA_VISIBLE_DEVICES=3 python test_ncu.py

# Test 2: Simple NCU test (list kernels)
echo ""
echo "=========================================="
echo "Test 2: NCU list kernels only (no metrics)"
echo "=========================================="
CUDA_VISIBLE_DEVICES=3 ncu --list-kernels python test_ncu.py 2>&1 | tail -20

# Test 3: Profile with basic metrics
echo ""
echo "=========================================="
echo "Test 3: NCU profile with basic metrics"
echo "=========================================="
CUDA_VISIBLE_DEVICES=3 ncu \
    --csv \
    --log-file ncu_test_output.csv \
    --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
    --launch-count 1 \
    python test_ncu.py

# Check if CSV was created
echo ""
if [ -f ncu_test_output.csv ]; then
    echo "✓ CSV file created successfully"
    echo "  File size: $(wc -c < ncu_test_output.csv) bytes"
    echo ""
    echo "  First 10 lines of CSV:"
    head -10 ncu_test_output.csv
else
    echo "✗ CSV file NOT created"
fi

# Test 4: Profile with kernel name filter
echo ""
echo "=========================================="
echo "Test 4: NCU with kernel name filter"
echo "=========================================="
CUDA_VISIBLE_DEVICES=3 ncu \
    --csv \
    --log-file ncu_test_output2.csv \
    --kernel-name simple_add_kernel \
    --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_sector_hit_rate.pct \
    --launch-count 1 \
    python test_ncu.py

echo ""
if [ -f ncu_test_output2.csv ]; then
    echo "✓ CSV file 2 created successfully"
    echo "  File size: $(wc -c < ncu_test_output2.csv) bytes"
    echo ""
    echo "  Content preview:"
    head -20 ncu_test_output2.csv
else
    echo "✗ CSV file 2 NOT created"
fi

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
