# Multi-turn Follow-up Conversation Fix

## Summary

Fixed multiple issues preventing follow-up conversations in multi-turn mode. The main problems were audio being blocked during IDLE state check, generic "error" type not being handled, and response tracking flags not being reset between turns.

## Issues Fixed

### 1. Audio Blocked During Multi-turn Listening
- **Problem**: Audio was being blocked when `session_state == SessionState.IDLE`, preventing audio during MULTI_TURN_LISTENING
- **Fix**: Removed the IDLE state check, now only checking if session is active
- **Impact**: Audio now flows to OpenAI during multi-turn listening state

### 2. Generic "error" Type Not Handled
- **Problem**: OpenAI was sending errors with type "error" which wasn't in our handled list
- **Fix**: Added handler for generic "error" type that checks error code for more details
- **Impact**: "Unhandled OpenAI error type: error" messages are now properly handled

### 3. Response Tracking Not Reset
- **Problem**: Response tracking flags weren't reset when entering multi-turn listening
- **Fix**: Reset `_response_create_sent`, `response_done_received`, and `_audio_response_received` flags
- **Impact**: Each turn starts fresh without stale state from previous turn

### 4. Enhanced State Transition Logging
- **Problem**: Difficult to debug state transitions
- **Fix**: Added logging when entering MULTI_TURN_LISTENING state
- **Impact**: Clear visibility of when system is ready for follow-up questions

## Technical Details

### Audio Flow Fix (src/main.py:805)
```python
# Before: Blocked audio during IDLE
if not self.session_active or self.session_state == SessionState.IDLE:

# After: Only check session active
if not self.session_active:
```

### Error Handler Enhancement (src/main.py:1432-1441)
```python
elif error_type == 'error':
    # Generic error type - check the error code for more details
    if error_code == 'conversation_already_has_active_response':
        self.logger.warning("Generic error with active response code - treating as duplicate")
        self._response_create_sent = False
        return
    else:
        self.logger.warning(f"Generic error type: {error_message} (code: {error_code})")
        # Don't end session for generic errors
        return
```

### Response Tracking Reset (src/main.py:1084-1087)
```python
# Reset response tracking for next turn
self._response_create_sent = False
self.response_done_received = False
self._audio_response_received = False
```

## Expected Behavior

1. After initial response completes, system transitions to MULTI_TURN_LISTENING
2. User sees: "*** SESSION ENTERING MULTI-TURN LISTENING STATE - SPEAK YOUR FOLLOW-UP QUESTION ***"
3. Audio from user flows to OpenAI without being blocked
4. System properly handles follow-up questions without duplicate response errors
5. Each turn starts with clean response tracking state

## Root Cause Analysis

The primary issue was that audio was being blocked during MULTI_TURN_LISTENING because of the IDLE state check. This prevented OpenAI from receiving the user's follow-up questions. Combined with the unhandled "error" type and stale response tracking flags, the system couldn't properly handle multi-turn conversations.

## Testing Notes

With these fixes:
- Audio should flow to OpenAI during MULTI_TURN_LISTENING state
- No more "Unhandled OpenAI error type: error" messages
- Follow-up questions should work without "conversation_already_has_active_response" errors
- Clear state transition logging shows when system is ready for follow-up