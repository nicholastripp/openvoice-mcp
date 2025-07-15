# Multi-turn Function Call Response Fix

## Summary

Fixed a regression where users weren't receiving audio responses after initial questions that triggered function calls (e.g., Home Assistant queries). The system was sending function outputs but OpenAI wasn't generating follow-up audio responses, causing the system to incorrectly transition to multi-turn listening without ever playing a response.

## Root Cause

When OpenAI responds with a function call:
1. The system correctly executes the function and sends the output back
2. But OpenAI doesn't automatically generate a response after receiving function output in server VAD mode
3. The system incorrectly transitioned to MULTI_TURN_LISTENING without waiting for/requesting a response
4. Users never heard the answer to their question

## Changes Made

### 1. Added response.create After Function Output (src/openai_client/realtime.py)
- Modified `_send_function_result()` to request a response after sending function output
- Modified `_send_function_error()` to request a response after sending function errors
- Added 100ms delay before response.create to ensure function output is processed
- Added logging to track the function → output → response flow

### 2. Added Function Response Tracking (src/openai_client/realtime.py)
- Added `waiting_for_function_response` flag to track when expecting response after function
- Set flag to True when sending function output
- Clear flag when response.done is received
- Added specific logging for responses after function calls

### 3. Enhanced Function Call Logging
- Log when function call is received with function name
- Log when creating response after function output
- Log when function response is completed
- Better visibility into the function call flow

## Technical Details

### Function Call Flow (Fixed)
1. User: "What's the temperature in the office?"
2. OpenAI: Generates function_call response → `control_home_assistant`
3. System: Executes function, gets temperature (e.g., 72°F)
4. System: Sends function output to OpenAI
5. System: **Sends response.create to request audio response** (NEW)
6. OpenAI: Generates audio response "The temperature in the office is 72 degrees"
7. System: Plays audio response
8. System: Transitions to MULTI_TURN_LISTENING after audio completes

### Why Manual response.create is Needed
- In server VAD mode, OpenAI doesn't automatically generate responses after function outputs
- The system must explicitly request a response using response.create
- This ensures users hear the results of their queries

## Expected Behavior

With these fixes:
1. User asks question requiring function call
2. Function executes and returns data
3. OpenAI generates audio response with the result
4. User hears the answer
5. System transitions to multi-turn mode for follow-up questions

## Testing Notes

Test with queries that trigger function calls:
- "What's the temperature in [room]?"
- "Turn on the [device]"
- "What's the status of [device]?"

Each should result in:
- Function execution
- Audio response with the result
- Ability to ask follow-up questions