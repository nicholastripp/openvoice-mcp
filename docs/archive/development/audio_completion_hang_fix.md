# Audio Completion Hang Fix

## Summary

Fixed an application hang that occurred after the second audio response in multi-turn conversations. The hang was caused by recursive task creation in the audio completion detection mechanism, which could lead to event loop issues and memory problems.

## Root Cause

The issue occurred in the `_delayed_completion_check()` method in `src/audio/playback.py`:

1. **Recursive Task Creation**: When audio was still buffered, the method would recursively create new asyncio tasks using `asyncio.create_task(self._delayed_completion_check())`
2. **Threading Issues**: The completion check runs in a separate thread with its own event loop, and recursive task creation could cause threading conflicts
3. **State Management**: The `_completion_check_retries` counter wasn't properly initialized for each response
4. **Stack Growth**: Recursive calls could lead to stack overflow or memory issues

## Changes Made

### 1. Replaced Recursion with Loop (src/audio/playback.py)
- Modified `_delayed_completion_check()` to use a loop instead of recursive task creation
- This prevents stack growth and event loop issues
- Added proper retry counting with a maximum of 10 attempts

### 2. Fixed State Initialization (src/audio/playback.py)
- Added `self._completion_check_retries = 0` in `start_response()` method
- Ensures clean state for each new response
- Added reset in `_notify_completion()` for extra safety

### 3. Enhanced Error Handling (src/audio/playback.py)
- Added comprehensive error handling in the thread runner
- Added traceback logging for better debugging
- Protected event loop closure with try/except

## Technical Details

### Previous Flow (Problematic)
1. Audio response starts playing
2. OpenAI finishes sending audio
3. `_delayed_completion_check()` starts
4. If buffer not empty, creates new task recursively
5. Multiple recursive tasks could pile up
6. Eventually causes hang/crash

### New Flow (Fixed)
1. Audio response starts playing
2. OpenAI finishes sending audio
3. `_delayed_completion_check()` starts with loop
4. Loops with exponential backoff up to 10 times
5. Cleanly completes or times out
6. No recursive task creation

## Expected Behavior

With these fixes:
1. First audio response plays and completes normally
2. System transitions to multi-turn listening
3. User asks follow-up question
4. Second audio response plays and completes normally
5. System transitions to multi-turn listening again
6. No hang or crash occurs

## Testing Notes

Test multi-turn conversations with:
- Multiple back-and-forth exchanges
- Function calls that return audio responses
- Rapid follow-up questions
- Long audio responses
- Network delays or interruptions

Each scenario should complete without hanging.