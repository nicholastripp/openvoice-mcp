# Audio Buffering and Multi-turn Conversation Fix v2

## Summary

Enhanced audio buffering strategy and completion detection to fix:
1. Audio playback underruns causing "stepped on" words
2. Multi-turn conversations ending prematurely

## Root Cause Analysis

### Audio Underrun Issue
- Pre-buffering threshold of 300ms was insufficient for network variability
- Buffer filling strategy wasn't aggressive enough when buffer was empty
- Underrun detection was forcing completion even when OpenAI was still streaming

### Multi-turn Conversation Issue
- Completion detection was timing out and forcing session end
- System wasn't tracking whether OpenAI was still actively streaming audio
- Delayed completion check was waiting too long (5 seconds) causing timeouts

## Changes Made

### 1. Increased Pre-buffer Threshold (src/audio/playback.py)
- Changed from 300ms to 500ms pre-buffering
- Gives more time for initial audio chunks to accumulate
- Reduces chance of starting playback with insufficient buffer

### 2. More Aggressive Buffer Filling
- When buffer is empty and OpenAI is streaming, fill up to 50% of max buffer
- Changed low buffer multiplier from 1.5x to 2x target
- Ensures buffer builds up faster when critically low

### 3. OpenAI Streaming State Tracking
- Added `openai_streaming_active` flag
- Set to True when response starts, False when OpenAI sends response.audio.done
- Added `last_chunk_received_time` to track when audio chunks arrive
- Prevents premature completion when OpenAI is still sending audio

### 4. Improved Underrun Handling
- Consecutive underrun limit only applies when OpenAI has stopped streaming
- Added time-based check: if no chunks for 3 seconds, force completion
- Logs chunk reception for debugging network issues

### 5. Enhanced Completion Detection
- Reduced delayed completion check timeout from 5s to 3s
- Only forces completion if buffer is actually empty
- Reschedules check if audio is still buffered
- Checks time since last chunk received before timeout-based completion

## Technical Details

### Buffer Management Strategy
```
Pre-buffer: 500ms (24,000 samples at 48kHz)
Min buffer: 200ms (9,600 samples)
Target buffer: 500ms (24,000 samples)  
Max buffer: 2s (96,000 samples)

When buffer < min:
  - Fill target = 2x normal target
  - If buffer empty and streaming: fill to 50% of max
```

### Completion Detection Logic
1. Wait for OpenAI to signal end of streaming
2. Give 0.5s delay for final chunks
3. Monitor buffer drain for up to 3s
4. Only force completion if:
   - Buffer and queue are empty
   - No chunks received for 2+ seconds
   - OpenAI has stopped streaming

### Multi-turn Conversation Flow
1. Audio playback completes → triggers completion callback
2. Completion callback checks session configuration
3. If multi-turn enabled → transition to MULTI_TURN_LISTENING
4. Session continues waiting for follow-up questions

## Expected Behavior

1. **Audio Playback**: 
   - Smooth playback without word cutoffs
   - Better handling of network jitter
   - Pre-buffers 500ms before starting

2. **Multi-turn Conversation**:
   - Properly detects audio completion
   - Transitions to listening for follow-up
   - Doesn't end session prematurely

## Debugging

Enable DEBUG logging to see:
- Audio chunk reception timing
- Buffer fill levels
- Underrun occurrences
- Completion detection steps

```bash
# In config.yaml
system:
  log_level: "DEBUG"
```