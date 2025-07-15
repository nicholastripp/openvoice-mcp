# Multi-turn Audio Completion Notification Fix

## Summary

Fixed audio playback completion notification that was being blocked by the `is_response_active` check, preventing the session from transitioning out of AUDIO_PLAYING state. Also fixed the session timeout handler's retry behavior that was causing repeated timeout attempts.

## Root Causes

1. **Blocked Completion Notification** - `_notify_completion()` checked `if self.is_response_active:` but this flag was already False when called
2. **No Duplicate Prevention** - No mechanism to prevent duplicate completion notifications
3. **Session Timeout Retry Loop** - Session timeout handler kept retrying to end session during audio playback without rate limiting

## Changes Made

### 1. Removed is_response_active Check (src/audio/playback.py:291-330)
- Removed the `if self.is_response_active:` check that was blocking notifications
- Completion callbacks now always fire when `_notify_completion()` is called
- Added logging to track when `is_response_active` changes

### 2. Added Completion Notification Tracking (src/audio/playback.py)
- Added `_completion_notified` flag to prevent duplicate notifications
- Check flag at start of `_notify_completion()` and return if already notified
- Reset flag in `start_response()` for each new audio response
- Initialize flag in `__init__` method

### 3. Fixed Session Timeout Behavior (src/main.py:476-494)
- Skip timeout checks during RESPONDING and AUDIO_PLAYING states
- Added rate limiting to prevent rapid retry attempts (5 second minimum between attempts)
- Track last timeout attempt time to enforce rate limit

### 4. Enhanced Logging
- Log when completion callbacks are being notified and how many
- Log when `is_response_active` changes in various places
- Log when skipping timeout checks due to audio activity

## Technical Details

### Completion Flow
1. Audio underrun detected with OpenAI done → `_notify_completion()` called
2. Check `_completion_notified` flag - if True, skip (prevents duplicates)
3. Set `_completion_notified = True`
4. Log and print completion message
5. Call all registered callbacks (main.py `_on_audio_playback_complete`)
6. Session transitions from AUDIO_PLAYING to MULTI_TURN_LISTENING

### Session Timeout Protection
- During audio playback states, timeout handler returns early
- Prevents "SESSION TIMEOUT" messages during active audio
- Rate limits attempts to prevent rapid retries if end_session is deferred

## Expected Behavior

With these fixes:
1. Audio completion callbacks fire reliably when audio finishes
2. No duplicate completion notifications
3. Session properly transitions to MULTI_TURN_LISTENING
4. No session timeout retry loops during audio playback
5. Follow-up questions work in multi-turn mode

## Testing Notes

The system should now:
- Show "*** AUDIO PLAYBACK COMPLETED - NOTIFYING CALLBACKS ***" in logs
- Transition from AUDIO_PLAYING to MULTI_TURN_LISTENING after audio
- Not show repeated "SESSION TIMEOUT" messages during playback
- Accept and process follow-up questions properly