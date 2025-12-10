# Error History Implementation

## Overview
Implemented error history tracking for repair prompts to prevent LLM from repeating the same mistakes.

## Implementation Strategy
Chose **Option 2: Direct concatenation of error logs** for speed and simplicity, as requested by user: "希望结合速度与效果"

## Changes Made

### 1. prompts/error.py
- **Line 19-49**: Updated `COMPILE_ERROR` template to include `$ERROR_HISTORY` placeholder
- **Line 21-22**: Added "History Error:" section at the beginning
- **Line 44**: Added instruction: "Learn from previous repair attempts to avoid repeating the same mistakes"
- **Line 260-266**: Added error history formatting in `build_error_prompt()`
  ```python
  history_section = ""
  if error_history and error_history.strip():
      history_section = f"""Previous Repair Attempts (avoid repeating these errors):
  {error_history.strip()}

  """
  ```
- **Line 269**: Pass formatted `ERROR_HISTORY` to template

### 2. main.py - Seed Repair (lines 585-605)
- **Line 585**: Added `error_history_list = []` to track errors
- **Lines 591-593**: Collect current error with attempt number, limit to 200 chars
  ```python
  if error_log.strip():
      error_history_list.append(f"Attempt {repair_attempt}: {error_log[:200]}")
  ```
- **Line 596**: Keep only last 3 errors to control prompt length
- **Line 603**: Pass `error_history` to `build_error_prompt()`

### 3. main.py - Stage Repair (lines 726-746)
- **Line 726**: Added `stage_error_history_list = []` to track stage-specific errors
- **Lines 733-735**: Collect current error with attempt number, limit to 200 chars
- **Line 738**: Keep only last 3 errors to control prompt length
- **Line 745**: Pass `stage_error_history` to `build_error_prompt()`

## Key Design Decisions

1. **Error Limit**: Cap at 200 characters per error to prevent prompt bloat
2. **History Window**: Keep only last 3 errors to balance context vs prompt length
3. **Separate Tracking**:
   - `error_history_list` for seed repair
   - `stage_error_history_list` for each stage repair (resets per stage)
4. **Format**: `"Attempt N: [first 200 chars of error]"`

## Expected Benefits

1. **Avoid Repetition**: LLM can see previous error patterns and try different approaches
2. **Fast**: No additional LLM call for summarization
3. **Controlled Size**: Limited to 3×200 = 600 chars max per repair session
4. **Clear Context**: Numbered attempts make pattern recognition easier

## Example Error History Format
```
Attempt 1: TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'
  File "test.py", line 42, in forward
    result = x + None

Attempt 2: AttributeError: 'Tensor' object has no attribute 'atomic_add'
  Use tl.atomic_add() instead of tensor.atomic_add()
```

## Testing
- [x] Syntax validation passed
- [ ] Integration test with actual repair scenario (pending user testing)
