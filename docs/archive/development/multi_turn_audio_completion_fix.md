# Multi-turn Audio Completion Error Fix

## Summary

Fixed error in audio completion handler that was preventing multi-turn conversations from working. The handler was throwing an unhandled exception, causing fallback session end instead of transitioning to multi-turn listening.

## Root Cause

1. **Indentation Error**: The multi-turn logic was incorrectly indented, causing it to execute outside the `if conversation_mode == "multi_turn"` block
2. **Missing Error Handling**: The `_handle_audio_completion()` method lacked proper exception handling
3. **No Detailed Logging**: Errors were caught but not logged with full traceback

## Changes Made

### 1. Enhanced Error Logging (src/main.py)
- Added traceback logging in `_on_audio_playback_complete()` 
- Now logs full exception details when audio completion fails
- Added print statements for immediate visibility

### 2. Added Session State Validation
- Check if session is still active before processing
- Validate state is AUDIO_PLAYING or RESPONDING
- Return early if conditions aren't met

### 3. Fixed Indentation Bug
- Moved all multi-turn logic inside the `if conversation_mode == "multi_turn"` block
- This was the main bug - code was executing regardless of mode

### 4. Added Defensive Programming
- Use `getattr()` with defaults for all config access
- Added try/except wrapper around entire method
- Log conversation mode and session state for debugging

## Technical Details

### Before (Bug):
```python
if conversation_mode == "multi_turn" and self.session_active:
    # Check end phrases
    if self.last_user_input and self._contains_end_phrases(self.last_user_input):
        # ...
        return

# THIS WAS EXECUTING FOR ALL MODES!        
# Increment conversation turn count
self.conversation_turn_count += 1
# ... rest of multi-turn logic ...
```

### After (Fixed):
```python
if conversation_mode == "multi_turn" and self.session_active:
    # Check end phrases
    if self.last_user_input and self._contains_end_phrases(self.last_user_input):
        # ...
        return
    
    # NOW CORRECTLY INSIDE THE IF BLOCK
    # Increment conversation turn count
    self.conversation_turn_count += 1
    # ... rest of multi-turn logic ...
```

## Expected Behavior

1. Audio playback completes successfully
2. Handler logs: "*** AUDIO COMPLETION - MODE: multi_turn, ACTIVE: True ***"
3. System transitions to MULTI_TURN_LISTENING state
4. User can ask follow-up questions without wake word
5. No errors or fallback session ends

## Debugging Output

The system now logs:
- Full exception tracebacks if errors occur
- Conversation mode and session state on every audio completion
- Clear state transitions with visual banners

## Testing Notes

With these fixes:
- Multi-turn conversations should work when enabled in config
- Single-turn mode should continue to work as before
- Any errors will be clearly logged with full details
- No more silent failures in audio completion handler