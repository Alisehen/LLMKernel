#!/usr/bin/env python3
"""Test script for Triton error pattern detection."""

from prompts.error import _detect_triton_error_patterns

# Test case 1: atomic_add on block_ptr
error_log_1 = """
AttributeError: 'block_type' object has no attribute 'primitive_bitwidth'
triton.compiler.errors.CompilationError: at 47:4:
    tl.atomic_add(c_block_ptr, acc)
"""

code_1 = """
c_block_ptr = tl.make_block_ptr(...)
tl.atomic_add(c_block_ptr, acc)
"""

print("=" * 80)
print("Test 1: atomic_add on block_ptr")
print("=" * 80)
result = _detect_triton_error_patterns(error_log_1, code_1)
print(result)

# Test case 2: unsupported tensor indexing
error_log_2 = """
ValueError: unsupported tensor index: int32[]
triton.compiler.errors.CompilationError: at 44:22:
    val = acc[i, j]
"""

code_2 = """
for i in range(BLOCK_M):
    for j in range(BLOCK_N):
        val = acc[i, j]
"""

print("\n" + "=" * 80)
print("Test 2: Unsupported tensor indexing")
print("=" * 80)
result = _detect_triton_error_patterns(error_log_2, code_2)
print(result)

# Test case 3: simultaneous multiple comparison
error_log_3 = """
triton.compiler.errors.UnsupportedLanguageConstruct: at 34:25:
    mask=(rm + row_offsets)[:, None] < M & (k + col_offsets)[None, :] < K,
simultaneous multiple comparison is not supported
"""

code_3 = """
mask=(rm + row_offsets)[:, None] < M & (k + col_offsets)[None, :] < K
"""

print("\n" + "=" * 80)
print("Test 3: Simultaneous multiple comparison")
print("=" * 80)
result = _detect_triton_error_patterns(error_log_3, code_3)
print(result)

# Test case 4: BLOCK_M/BLOCK_N confusion
error_log_4 = "The kernel has 4096 threads per block"
code_4 = """
# BLOCK_M=64, BLOCK_N=64 create 4096 threads per block
BLOCK_M = 64
BLOCK_N = 64
"""

print("\n" + "=" * 80)
print("Test 4: BLOCK_M/BLOCK_N confusion")
print("=" * 80)
result = _detect_triton_error_patterns(error_log_4, code_4)
print(result)

# Test case 5: No error pattern detected
error_log_5 = "Some unrelated error"
code_5 = "some normal code"

print("\n" + "=" * 80)
print("Test 5: No pattern detected")
print("=" * 80)
result = _detect_triton_error_patterns(error_log_5, code_5)
print(f"Result: '{result}' (should be empty)")
