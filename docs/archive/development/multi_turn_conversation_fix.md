# Multi-turn Conversation Fix

## Summary

Fixed multi-turn conversations that were ending prematurely after the initial response. The issue was caused by a response fallback timer creating duplicate `response.create` requests, leading to errors that ended the session.

## Root Cause

1. **Response Fallback Timer Race Condition**
   - A 2-second fallback timer was creating duplicate `response.create` requests
   - When OpenAI responded quickly, both the server VAD and fallback timer would trigger
   - This caused "conversation_already_has_active_response" errors

2. **Error Handling Ended Session**
   - The "conversation_already_has_active_response" error was ending the session
   - This prevented the multi-turn conversation flow from ever being reached

3. **Invalid State Transitions**
   - After session ended, lingering OpenAI events tried to transition to invalid states
   - This caused "Invalid state transition" warnings in the logs

## Changes Made

### 1. Removed Response Fallback Timer (src/main.py)
- Removed `response_fallback_task` and all related logic
- Removed `_response_creation_fallback()` method entirely
- Server VAD reliably triggers responses, so the fallback timer was unnecessary

### 2. Updated Error Handling
- "conversation_already_has_active_response" errors no longer end the session
- These errors are now ignored as they're expected in some race conditions

### 3. Enhanced State Validation
- Added check to prevent state transitions when session is not active
- This prevents invalid transitions from lingering events after session ends

### 4. Audio Completion Flow Verified
- Confirmed `_on_audio_response_done()` calls `audio_playback.end_response()`
- Confirmed audio completion callback triggers `_handle_audio_completion()`
- Confirmed multi-turn transition logic in `_handle_audio_completion()`

## Expected Behavior

1. User says wake word and asks initial question
2. OpenAI responds with audio
3. After audio playback completes:
   - If multi-turn enabled: transitions to MULTI_TURN_LISTENING state
   - User can ask follow-up questions without wake word
   - Session continues until timeout or end phrase
4. No duplicate response errors
5. No invalid state transitions

## Testing Notes

The system should now:
- Allow multiple conversation turns without wake word
- Properly transition states after each response
- Not end session on expected race condition errors
- Handle state transitions gracefully

## Configuration

Multi-turn conversation settings in `config.yaml`:
```yaml
session:
  conversation_mode: "multi_turn"  # Enable multi-turn
  multi_turn_timeout: 30.0         # Seconds to wait for follow-up
  multi_turn_max_turns: 10         # Maximum conversation turns
  multi_turn_end_phrases:          # Phrases to end conversation
    - "goodbye"
    - "stop"
    - "that's all"
    - "thank you"
    - "bye"
```