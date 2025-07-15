# Multi-Turn Audio Completion Hang Fix (v2)

## Summary

Fixed a critical issue where the application would hang after the second audio response in multi-turn conversations. The hang was caused by the audio completion notification flag not being reset between responses, preventing the completion callback from firing on subsequent responses.

## Root Cause

The issue had multiple contributing factors:

1. **Completion Flag Not Reset**: The `_completion_notified` flag was set to True after the first response but wasn't properly reset for the second response
2. **Blocked Completion Callback**: When the second audio underrun triggered `_notify_completion()`, it returned early because the flag was still True
3. **Stuck Session State**: The session remained in AUDIO_PLAYING state indefinitely
4. **Infinite Deferral Loop**: The watchdog timer tried to force completion but kept deferring because the state was AUDIO_PLAYING

## Changes Made

### 1. Added Force Parameter to Completion (src/audio/playback.py)
- Added `force` parameter to `_notify_completion()` method
- When `force=True`, bypasses the duplicate notification check
- Used by watchdog timer for emergency completion

### 2. Enhanced Watchdog Timer (src/main.py)
- Watchdog now resets `_completion_notified` flag before forcing completion
- Calls `_notify_completion(force=True)` to ensure it works
- Added deferral counter to prevent infinite loops (max 5 deferrals)
- After max deferrals, forces immediate state cleanup

### 3. Improved Fallback Session End (src/main.py)
- Forces audio playback state cleanup when stuck
- Clears audio queue and resets all flags
- Forces state transition out of AUDIO_PLAYING if needed

### 4. Reset Response Tracking (src/main.py)
- Added `response_active = False` reset when transitioning to MULTI_TURN_LISTENING
- Ensures clean state for each new response

## Technical Details

### Problem Flow (Before Fix)
1. First response plays and completes normally
2. `_completion_notified` set to True
3. System transitions to MULTI_TURN_LISTENING
4. User asks follow-up question
5. Second response starts but `_completion_notified` still True
6. Audio underrun occurs, calls `_notify_completion()`
7. Method returns early due to duplicate check
8. Session stuck in AUDIO_PLAYING state
9. Watchdog tries to help but keeps deferring
10. Infinite loop of deferrals

### Solution Flow (After Fix)
1. First response plays and completes normally
2. System transitions to MULTI_TURN_LISTENING
3. Response tracking flags reset including `response_active`
4. User asks follow-up question
5. Second response starts with clean state
6. If audio underrun occurs and completion fails:
   - Watchdog detects stuck state
   - Resets `_completion_notified` flag
   - Calls `_notify_completion(force=True)`
   - Tracks deferrals to prevent infinite loops
   - Forces cleanup after max deferrals

## Expected Behavior

With these fixes:
- Multi-turn conversations work reliably with multiple exchanges
- Audio completion is properly detected for all responses
- If completion detection fails, watchdog forces recovery
- No infinite loops or hangs occur
- System recovers gracefully from stuck states

## Testing Notes

Test scenarios:
1. Multiple back-and-forth conversations (3+ turns)
2. Rapid follow-up questions
3. Long audio responses that might have underruns
4. Function calls in multi-turn mode
5. Network interruptions during responses

Each scenario should complete without hanging.