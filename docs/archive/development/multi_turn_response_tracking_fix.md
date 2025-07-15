# Multi-Turn Response Tracking Fix

## Summary

Fixed an issue where `start_response()` wasn't being called for the second response in multi-turn conversations, causing the `_completion_notified` flag to remain True and blocking audio completion notifications.

## Root Cause

The issue occurred because:

1. **Conditional start_response()**: The `start_response()` method was only called when `self.response_active` was False in `_on_audio_response()`
2. **Response ID Not Tracked**: The system didn't track response IDs to detect when a new response started
3. **Flag Not Reset**: Without `start_response()` being called, `_completion_notified` remained True from the first response
4. **Blocked Completion**: The second response's underrun couldn't trigger completion because the flag was still True

## Changes Made

### 1. Added Response ID Tracking (src/main.py)
- Added `self._current_response_id = None` to track the current response
- Detects when a new response ID is received in `_on_response_created()`

### 2. Force start_response() for New Responses (src/main.py)
- When a new response ID is detected, always call `start_response()`
- This happens in `_on_response_created()` before any audio is received
- Ensures every response starts with clean state

### 3. Enhanced Logging (src/main.py & src/audio/playback.py)
- Added logging when new response is detected
- Added logging when `start_response()` is called
- Added print statement to confirm completion flag reset

## Technical Details

### Previous Flow (Problematic)
1. First response: `response_active = False`, so `start_response()` called
2. First response completes, `_completion_notified = True`
3. Transition to MULTI_TURN_LISTENING
4. Second response starts
5. `response_active` might still be True, so `start_response()` NOT called
6. `_completion_notified` still True from first response
7. Underrun occurs but completion blocked

### New Flow (Fixed)
1. First response: `response.created` event sets response ID
2. New response detected, `start_response()` called
3. First response completes normally
4. Second response: `response.created` event with new ID
5. New response detected, `start_response()` called again
6. `_completion_notified` reset to False
7. Underrun can properly trigger completion

## Expected Behavior

With these fixes:
- Every response gets its own `start_response()` call
- The completion flag is properly reset for each response
- Multi-turn conversations work reliably
- No more hangs after the second response

## Testing Notes

The fix adds clear logging to track:
- When new responses are detected
- When `start_response()` is called
- When the completion flag is reset

Look for these log messages:
- "New response detected: resp_XXX (previous: resp_YYY)"
- "Forcing start_response for new response ID"
- "*** AUDIO PLAYBACK: start_response() called - completion_notified reset to False ***"