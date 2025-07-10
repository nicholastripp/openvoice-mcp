# Wake Word Aggressive Reset Fix

## Problem
The wake word model was stuck returning the value `5.0768717e-06` for all predictions, even with valid audio input. The stuck state detection was working (showing `[STUCK]` in logs) but the model reset was not being triggered due to cooldown timers.

## Root Cause
1. The `_should_reset_model()` function was checking cooldown BEFORE checking for stuck state
2. This prevented the model from resetting even when stuck was detected
3. The model continued returning the stuck value indefinitely

## Fixes Applied

### 1. Bypass Cooldown for Stuck States (line 942-944)
- Moved stuck state check BEFORE cooldown check in `_should_reset_model()`
- Stuck states now bypass the cooldown timer completely
- Added logging to show when cooldown is bypassed

### 2. Force Immediate Reset for Critical Value (line 731-737)
- Added special handling for the specific stuck value `5.0768717e-06`
- When detected, forces immediate reset by setting `last_model_reset_time = 0`
- Logs as CRITICAL error for visibility

### 3. Reduced Thresholds for Faster Response
- Stuck detection threshold: 5 → 3 predictions (line 77)
- Minimum reset cooldown: 10s → 2s (line 80)
- Critical stuck value detection: only needs 2 occurrences (line 997)

### 4. Enhanced Logging
- Added warning when model is stuck but reset is blocked (line 662)
- Added info log when stuck state bypasses cooldown (line 943)
- Added debug log when reset is blocked by cooldown (line 948)

### 5. Alternative Model Suggestion
- Added comment in config.yaml suggesting to try "alexa" model if stuck persists
- This helps if the hey_jarvis model file itself is corrupted

## Expected Behavior
1. Model will detect stuck state after 2-3 predictions of `5.0768717e-06`
2. Reset will trigger immediately, bypassing any cooldown timers
3. Logs will show "[RESET] Resetting wake word model - Reason: critical_stuck_value"
4. Model should recover and start returning varied predictions

## Testing
The user should test with these changes and watch for:
- "CRITICAL STUCK VALUE 5.0768717e-06 DETECTED!" messages
- "[RESET]" log entries showing the model is actually resetting
- Varied prediction values after reset

If the model continues to return stuck values after reset, the model file itself may be corrupted and the user should try the "alexa" wake word model instead.