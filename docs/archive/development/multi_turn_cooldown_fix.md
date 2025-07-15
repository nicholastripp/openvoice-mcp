# Multi-turn COOLDOWN State Fix

## Summary

Fixed invalid state transitions and COOLDOWN state issues that were preventing multi-turn conversations from working properly. The system was incorrectly entering COOLDOWN state even in multi-turn mode and then failing to transition properly.

## Root Causes

1. **COOLDOWN in Multi-turn Mode** - System was scheduling session end with COOLDOWN state even when in multi-turn mode
2. **Invalid State Transition** - When ending session from COOLDOWN state, session_active was set to False before transitioning to IDLE
3. **Deferred Session End** - When audio was playing, system deferred session end but still scheduled COOLDOWN transition

## Changes Made

### 1. Fixed _end_session to Skip COOLDOWN in Multi-turn (src/main.py:573-578)
- Check if multi-turn mode is active
- Skip scheduling session end with COOLDOWN for multi-turn mode
- Let audio completion handler manage multi-turn transitions

### 2. Fixed _schedule_session_end Multi-turn Check (src/main.py:1616-1618)
- Added early return if in multi-turn mode
- Prevents COOLDOWN state transition in multi-turn conversations
- Logs warning if called incorrectly

### 3. Fixed State Transition Order (src/main.py:584-590)
- Transition to IDLE state BEFORE setting session_active = False
- Prevents "Invalid state transition: cooldown -> idle" error
- Ensures valid state transitions during session cleanup

### 4. Added Debug Logging (src/main.py:571-573)
- Log call stack when session end is deferred
- Helps identify what's trying to end session during audio
- Better debugging for future issues

## Technical Details

### State Transition Rules
- COOLDOWN can transition to: IDLE, LISTENING
- State transitions are validated against session_active flag
- If session_active is False and current state is not IDLE, transitions are rejected

### Multi-turn Flow
1. Audio response completes
2. System transitions to MULTI_TURN_LISTENING (not COOLDOWN)
3. User can ask follow-up questions
4. No session end is scheduled

### Single-turn Flow  
1. Audio response completes
2. If response_cooldown_delay > 0, transition to COOLDOWN
3. Wait for cooldown period
4. Auto-end session

## Expected Behavior

With these fixes:
- Multi-turn mode never enters COOLDOWN state
- No "Invalid state transition" errors
- Session properly transitions to MULTI_TURN_LISTENING after responses
- Follow-up questions work without hanging
- Single-turn mode still uses COOLDOWN as configured

## Testing Notes

The system should now:
1. Complete initial audio response
2. Transition directly to MULTI_TURN_LISTENING (no COOLDOWN)
3. Accept follow-up questions without errors
4. Not show "Invalid state transition" messages
5. Not hang after audio responses