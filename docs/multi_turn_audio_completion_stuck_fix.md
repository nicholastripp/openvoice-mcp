# Multi-turn Audio Completion Stuck Fix

## Summary

Fixed audio playback completion detection that was preventing multi-turn conversations. The issue was that the delayed completion check wasn't working properly in the audio thread context, causing the system to get stuck in AUDIO_PLAYING state.

## Root Cause

1. **Async task creation in audio thread** - `asyncio.create_task()` doesn't work properly outside the main event loop
2. **High underrun threshold** - Required 20+ underruns before forcing completion
3. **Missing simple fallback** - No quick detection when OpenAI stops streaming

## Changes Made

### 1. Fixed Delayed Completion Check (src/audio/playback.py:273-289)
- Run completion check in separate thread with its own event loop
- Avoids event loop context issues
- Falls back to immediate completion on error

### 2. Added Simple Underrun Completion (src/audio/playback.py:407-412)
- Completes playback after just 1 underrun if:
  - OpenAI has stopped streaming
  - Audio queue is empty
- Much faster than waiting for 20+ underruns

### 3. Enhanced Logging (src/audio/playback.py)
- Added logging when OpenAI finishes sending
- Log delayed completion check start
- Log when playback completes

## Technical Details

### Threading Fix
```python
def run_completion_check():
    """Run completion check in a separate thread with its own event loop"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._delayed_completion_check())
    except Exception as e:
        self.logger.error(f"Error in completion check thread: {e}")
        self._notify_completion()
    finally:
        loop.close()

completion_thread = threading.Thread(target=run_completion_check, daemon=True)
completion_thread.start()
```

### Simple Completion Detection
```python
# If OpenAI stopped streaming and we have any underrun, complete
if self.underrun_count >= 1 and not self.openai_streaming_active and self.audio_queue.empty():
    self.logger.info(f"Underrun with empty queue and OpenAI done - completing playback")
    self._notify_completion()
    return
```

## Expected Behavior

1. Audio response plays completely
2. First underrun triggers completion check
3. If OpenAI done + queue empty → immediate completion
4. Audio completion callback fires
5. System transitions to MULTI_TURN_LISTENING
6. User can ask follow-up questions

## Testing Notes

With these fixes:
- Audio completion detected after first underrun (not 20+)
- Delayed completion check runs in separate thread
- No more stuck AUDIO_PLAYING state
- Multi-turn conversations should work reliably