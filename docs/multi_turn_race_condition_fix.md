# Multi-turn Race Condition and Session Management Fix

## Summary

Fixed multiple race conditions and session management issues that were preventing multi-turn conversations from working properly.

## Issues Fixed

### 1. Session Ending on "conversation_already_has_active_response" Error
- **Problem**: Error handler was returning early but session was still being ended
- **Cause**: The error handler had a fall-through to `_end_session()` for unhandled errors
- **Fix**: Added proper return statements and only end session for truly unrecoverable errors

### 2. Duplicate response.create After Function Calls
- **Problem**: After sending function call results, we were calling response.create again
- **Cause**: OpenAI client was automatically requesting responses after function outputs
- **Fix**: Removed automatic response.create calls - let OpenAI handle the flow

### 3. Session Ending During Audio Playback
- **Problem**: Session could end while audio was still playing, breaking multi-turn
- **Cause**: No protection against ending session during active audio states
- **Fix**: Added state checks to defer session end until audio completes

### 4. Audio Completion Handler Timeout
- **Problem**: Handler was timing out after 5 seconds for longer responses
- **Cause**: Timeout was too short for longer audio responses
- **Fix**: Increased timeout from 5s to 30s

## Technical Details

### Error Handler Fix
```python
# Before: Fall-through to _end_session()
elif error_type == 'connection_error':
    # handle connection error
    
# End session on unrecoverable errors
await self._end_session()

# After: Proper error categorization
elif error_type == 'connection_error':
    # handle connection error
    return

# Only end for truly unrecoverable errors
unrecoverable_errors = ['authentication_error', 'permission_error', 'not_found_error']
if error_type in unrecoverable_errors:
    await self._end_session()
```

### Session Protection
```python
# Added protection in _end_session()
if self.session_state in [SessionState.RESPONDING, SessionState.AUDIO_PLAYING]:
    self.logger.warning(f"Attempted to end session during {self.session_state.value} - deferring")
    # Schedule session end after audio completes
    return
```

### Function Call Flow
```python
# Removed automatic response creation
await self._send_event(event)
# Don't automatically request response - let OpenAI handle the flow
```

## Expected Behavior

1. Session continues after "conversation_already_has_active_response" errors
2. No duplicate response requests after function calls
3. Session waits for audio to complete before ending
4. Longer audio responses don't timeout
5. Multi-turn conversations work properly

## Testing Notes

With these fixes:
- Function calls should work without causing errors
- Audio should play to completion before session ends
- Multi-turn mode should activate after audio completes
- No more race conditions or invalid state transitions