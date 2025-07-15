# Audio Playback and Multi-turn Conversation Fix

## Summary

Fixed two critical issues:
1. Audio playback "stepping on itself" due to buffer underruns
2. Multi-turn conversations not working after initial response

## Changes Made

### 1. Audio Buffering Improvements (src/audio/playback.py)

#### Increased Buffer Tolerance
- Increased `max_consecutive_underruns` from 5 to 20 to handle network jitter
- This prevents premature audio completion due to temporary network delays

#### Added Pre-buffering
- Added `pre_buffering` flag and `pre_buffer_threshold` (300ms)
- Audio playback now waits until sufficient audio is buffered before starting
- Prevents the "stepped on" audio issue where words get cut off

#### Improved Completion Detection
- Added `_delayed_completion_check()` method that waits for audio buffer to drain
- Called when OpenAI finishes sending audio via `end_response()`
- Ensures audio playback completion callback is properly triggered

#### Reset Counters on New Response
- Reset underrun counters in `start_response()` for clean state

### 2. Response Handling Improvements (src/main.py)

#### Added Response Done Handler
- Added `_on_response_done()` handler for "response.done" event
- Cancels the response fallback timer to prevent duplicate response.create calls
- Prevents "conversation_already_has_active_response" errors

#### Improved Error Handling
- Added specific handling for "conversation_already_has_active_response" error
- Session no longer ends on this error (it's expected from our fallback timer)

### 3. Multi-turn Conversation Fix

The multi-turn conversation should now work because:
1. Audio playback completion is properly detected
2. The completion callback triggers `_handle_audio_completion()`
3. This checks for multi-turn mode and transitions to MULTI_TURN_LISTENING
4. Session doesn't end prematurely due to errors

## Technical Details

### Buffer Underrun Issue
- Audio was arriving in chunks from OpenAI
- Playback was starting too early without sufficient buffering
- When buffer ran empty, audio would cut out mid-word
- Solution: Pre-buffer 300ms of audio before starting playback

### Multi-turn Issue
- Audio completion callback wasn't being triggered
- Underrun detection was forcing completion too early
- "conversation_already_has_active_response" errors were ending the session
- Solution: Better completion detection and error handling

## Testing Notes

The system should now:
1. Play audio smoothly without cutting off words
2. Properly detect when audio playback completes
3. Transition to multi-turn listening mode after response
4. Allow follow-up questions within the timeout period
5. Not end session due to fallback timer errors