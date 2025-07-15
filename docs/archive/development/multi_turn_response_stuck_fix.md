# Multi-turn Response Stuck Fix

## Summary

Fixed issues causing the system to hang after receiving audio responses. The main problems were missing response.done event handling, unhandled error types, and infinite recursion in completion checks.

## Root Causes

1. **Missing response.done tracking** - System didn't track when OpenAI finished processing, leading to uncertain completion states
2. **"invalid_request_error" treated as unhandled** - This common error was ending sessions unnecessarily
3. **Infinite recursion in delayed completion checks** - Audio playback kept scheduling completion checks indefinitely
4. **No fallback for missing response.done events** - System would hang if OpenAI never sent response.done

## Changes Made

### 1. Added "invalid_request_error" Handler (src/main.py:1383-1386)
- Added specific case for "invalid_request_error" that doesn't end session
- This prevents "Unhandled OpenAI error type" logging for recoverable errors

### 2. Enhanced response.done Event Handling (src/main.py:1407-1424)
- Added tracking for response.done receipt with `response_done_received` flag
- Added tracking for audio responses with `_audio_response_received` flag
- Added fallback handler for responses without audio
- System can now handle text-only or empty responses properly

### 3. Fixed Recursive Completion Check (src/audio/playback.py:605-668)
- Added retry counter to prevent infinite recursion (max 10 retries)
- Implemented exponential backoff for retries (2s * retry count, max 10s)
- Force completion after max retries to prevent hanging

### 4. Response State Tracking (src/main.py)
- Added instance variables to track response state
- Reset tracking variables at session start
- Use flags to determine if audio completion should wait

## Technical Details

### Response Tracking Flow:
1. Session starts: `response_done_received = False`, `_audio_response_received = False`
2. When audio arrives: `_audio_response_received = True`
3. When response.done arrives: `response_done_received = True`
4. If response.done but no audio: Schedule non-audio completion check
5. Audio completion uses these flags to determine proper behavior

### Completion Check Improvements:
```python
# Before: Infinite recursion
asyncio.create_task(self._delayed_completion_check())

# After: Limited retries with backoff
self._completion_check_retries += 1
if self._completion_check_retries >= 10:
    self._notify_completion()  # Force completion
else:
    backoff_delay = min(2.0 * self._completion_check_retries, 10.0)
    await asyncio.sleep(backoff_delay)
    asyncio.create_task(self._delayed_completion_check())
```

## Expected Behavior

1. System properly handles "invalid_request_error" without ending session
2. Response completion is tracked and handled for both audio and non-audio responses
3. Audio playback completion checks have finite retries with exponential backoff
4. System doesn't hang waiting for events that may never arrive
5. Multi-turn conversations continue working even with network delays or errors

## Testing Notes

With these fixes:
- The "Unhandled OpenAI error type: invalid_request_error" message should not appear
- System should log "RESPONSE.DONE RECEIVED" when OpenAI completes
- Audio completion checks should retry up to 10 times with increasing delays
- System should force completion after ~55 seconds worst case (sum of backoff delays)
- Multi-turn conversations should be more resilient to network issues