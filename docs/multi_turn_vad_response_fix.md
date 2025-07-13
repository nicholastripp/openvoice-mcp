# Multi-turn VAD Response Fix

## Summary

Fixed critical issues preventing multi-turn conversations from working with server VAD mode. The main problems were duplicate response creation, improper error parsing, timeout failures, and missing response state tracking.

## Root Causes

1. **Duplicate Response Creation** - Manual response.create after speech_stopped conflicted with server VAD's automatic response generation
2. **Nested Error Structure** - OpenAI sends errors as `{type: 'error', error: {...}}` but handler expected direct fields
3. **Audio Completion Timeout** - 30-second timeout was too long and caused session termination
4. **Missing Response State Tracking** - No handler for response.created event to properly track active responses

## Changes Made

### 1. Fixed Error Parsing (src/openai_client/realtime.py:599)
- Changed from passing `error_info` (wrapper) to `error_data` (actual error)
- This ensures error handlers receive the expected error structure

### 2. Removed Manual Response Creation (src/main.py:1389-1414)
- Commented out manual response.create after speech_stopped
- Server VAD automatically creates responses when speech stops
- This prevents "conversation_already_has_active_response" errors

### 3. Fixed Audio Completion Timeout (src/main.py:1013-1033)
- Reduced timeout from 30s to 5s
- Added specific handling for TimeoutError
- Removed fallback session end on errors
- Continues operation instead of terminating session

### 4. Added Response State Tracking (src/main.py:1485-1499)
- Added handler for response.created event
- Sets response_active = True when response starts
- Properly transitions to RESPONDING state
- Resets flags when response completes or fails

### 5. Improved Connection Checks (src/main.py:1048-1054)
- Check if OpenAI client is connected before VAD updates
- Handle errors gracefully without failing completion handler
- Prevents hanging when WebSocket is closed

## Technical Details

### Server VAD Behavior
With server VAD enabled:
1. Audio is continuously streamed to OpenAI
2. OpenAI detects when user stops speaking
3. OpenAI automatically creates a response
4. Manual response.create causes duplicate response errors

### Response Lifecycle
1. User speaks → speech_stopped event
2. OpenAI automatically creates response → response.created event
3. Response generates audio → audio chunks received
4. Response completes → response.done event
5. Audio plays → audio completion handler
6. System transitions to MULTI_TURN_LISTENING

### Error Handling Flow
```
OpenAI Error: {type: 'error', error: {type: 'invalid_request_error', ...}}
↓
Extract nested error data
↓
Pass to error handler: {type: 'invalid_request_error', ...}
↓
Handle specific error type
```

## Expected Behavior

1. User speaks, system detects speech stop
2. Server VAD automatically creates response (no manual call)
3. Response state properly tracked via response.created
4. Audio plays without timeout errors
5. System transitions to multi-turn listening
6. Follow-up questions work without errors

## Testing Notes

With these fixes:
- No more "conversation_already_has_active_response" errors
- Audio completion completes within 5 seconds
- Session continues even if errors occur
- Response state properly tracked throughout lifecycle
- Multi-turn conversations work reliably with server VAD