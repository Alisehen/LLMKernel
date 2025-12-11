#!/bin/bash
# Wrapper script to run main.py with NCU profiling enabled (using sudo)
# This script preserves your conda environment and all settings

# Usage: ./run_with_ncu.sh task.py [additional args...]

if [ $# -eq 0 ]; then
    echo "Usage: $0 <task_file> [additional arguments]"
    echo "Example: $0 KernelBench/level1/2_Standard_matrix_multiplication_.py --device 3"
    exit 1
fi

# Get the task file (first argument)
TASK_FILE="$1"
shift  # Remove first argument, keep the rest

# Default arguments
DEFAULT_ARGS=(
    "--device" "0"  # Note: Use 0 because CUDA_VISIBLE_DEVICES will map GPU 3 to 0
    "--round" "4"
    "--max_repair_attempts" "3"
    "--profile_iters_per_step" "2"
)

echo "=========================================="
echo "Running with NCU Profiling (sudo mode)"
echo "=========================================="
echo "Task: $TASK_FILE"
echo "GPU: Physical GPU 3 (appears as device 0 inside program)"
echo "Args: ${DEFAULT_ARGS[@]} $@"
echo ""

# Run with sudo, preserving environment variables
sudo -E env \
    PATH="$PATH" \
    PYTHONPATH="$PYTHONPATH" \
    CUDA_VISIBLE_DEVICES=3 \
    python main.py "$TASK_FILE" "${DEFAULT_ARGS[@]}" "$@"

# Fix file ownership after sudo execution
echo ""
echo "Fixing file ownership..."
sudo chown -R $USER:$USER run/ 2>/dev/null || true

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
