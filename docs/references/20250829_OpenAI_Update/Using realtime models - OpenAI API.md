---
title: "Using realtime models - OpenAI API"
source: "https://platform.openai.com/docs/guides/realtime-models-prompting"
author:
published:
created: 2025-08-29
description: "Learn how to use OpenAI realtime models and prompting effectively."
tags:
  - "clippings"
---
Use realtime models and prompting effectively.

Realtime models are post-trained for specific customer use cases. In response to your feedback, the latest speech-to-speech model works differently from previous models. Use this guide to understand and get the most out of it.

Our most advanced speech-to-speech model is [gpt-realtime](https://platform.openai.com/docs/models/gpt-realtime).

This model shows improvements in following complex instructions, calling tools, and producing speech that sounds natural and expressive. For more information, see the announcement blog post.

After you initiate a session over [WebRTC](https://platform.openai.com/docs/guides/realtime-webrtc), [WebSocket](https://platform.openai.com/docs/guides/realtime-websocket), or [SIP](https://platform.openai.com/docs/guides/realtime-sip), the client and model are connected. The server will send a [session.created](https://platform.openai.com/docs/api-reference/realtime-server-events/session/created) event to confirm. Now it's a matter of prompting.

1. Create a basic audio prompt in [the dashboard](https://platform.openai.com/audio/realtime).
	If you don't know where to start, experiment with the prompt fields until you find something interesting. You can always manage, iterate on, and version your prompts later.
2. Update your realtime session to use the prompt you created. Provide its prompt ID in a `session.update` client event:

```javascript
1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

24

25

26

27

28

29

30

31

32

33

const event = {
  type: "session.update",
  session: {
      type: "realtime",
      model: "gpt-realtime",
      // Lock the output to audio (add "text" if you also want text)
      output_modalities: ["audio"],
      audio: {
        input: {
          format: "pcm16",
          turn_detection: { type: "semantic_vad", create_response: true }
        },
        output: {
          format: "g711_ulaw",
          voice: "alloy",
          speed: 1.0
        }
      },
      // Use a server-stored prompt by ID. Optionally pin a version and pass variables.
      prompt: {
        id: "pmpt_123",          // your stored prompt ID
        // version: "89",        // optional: pin a specific version
        variables: {
          city: "Paris"          // example variable used by your prompt
        }
      },
      // You can still set direct session fields; these override prompt fields if they overlap:
      instructions: "Speak clearly and briefly. Confirm understanding before taking actions."
  },
};

// WebRTC data channel and WebSocket both have .send()
dataChannel.send(JSON.stringify(event));
```

When the session's updated, the server emits a [session.updated](https://platform.openai.com/docs/api-reference/realtime-server-events/session/updated) event with the new state of the session. You can update the session any time.

To update the session mid-call (to swap prompt version or variables, or override instructions), send the update over the same data channel you're using:

```
1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

// Example: switch to a specific prompt version and change a variable
dc.send(JSON.stringify({
  type: "session.update",
  session: {
    prompt: {
      id: "pmpt_123",
      version: "89",
      variables: {
        city: "Berlin"
      }
    }
  }
}));

// Example: override instructions (note: direct session fields take precedence over Prompt fields)
dc.send(JSON.stringify({
  type: "session.update",
  session: {
    instructions: "Speak faster and keep answers under two sentences."
  }
}));
```

## Prompting gpt-realtime

Here are top tips for prompting the realtime speech-to-speech model. For a more in-depth guide to prompting, see the realtime prompting cookbook.

- **Iterate relentlessly**. Small wording changes can make or break behavior.
	Example: Swapping “inaudible” → “unintelligible” improved noisy input handling.
- **Use bullets over paragraphs**. Clear, short bullets outperform long paragraphs.
- **Guide with examples**. The model strongly follows onto sample phrases.
- **Be precise**. Ambiguity and conflicting instructions degrade performance, similar to GPT-5.
- **Control language**. Pin output to a target language if you see drift.
- **Reduce repetition**. Add a variety rule to reduce robotic phrasing.
- **Use all caps for emphasis**: Capitalize key rules to makes them stand out to the model.
- **Convert non-text rules to text**: The model responds better to clearly written text.
	Example: Instead of writing, "IF x > 3 THEN ESCALATE", write, "IF MORE THAN THREE FAILURES THEN ESCALATE."

Organize your prompt to help the model understand context and stay consistent across turns.

Use clear, labeled sections in your system prompt so the model can find and follow them. Keep each section focused on one thing.

```
1

2

3

4

5

6

7

8

# Role & Objective        — who you are and what “success” means
# Personality & Tone      — the voice and style to maintain
# Context                 — retrieved context, relevant info
# Reference Pronunciations — phonetic guides for tricky words
# Tools                   — names, usage rules, and preambles
# Instructions / Rules    — do’s, don’ts, and approach
# Conversation Flow       — states, goals, and transitions
# Safety & Escalation     — fallback and handoff logic
```

This format also makes it easier for you to iterate and modify problematic sections.

To make this system prompt your own, add domain-specific sections (e.g., Compliance, Brand Policy) and remove sections you don’t need. In each section, provide instructions and other information for the model to respond correctly. See specifics below.

Here are 10 tips for creating effective, consistently performing prompts with gpt-realtime. These are just an overview. For more details and full system prompt examples, see the realtime prompting cookbook.

The new realtime model is very good at instruction following. However, that also means small wording changes or unclear instructions can shift behavior in meaningful ways. Inspect and iterate on your system prompt to try different phrasing and fix instruction contradictions.

In one experiment we ran, changing the word "inaudible" to "unintelligble" in instructions for handling noisy inputs significantly improved the model's performance.

After your first attempt at a system prompt, have an LLM review it for ambiguity or conflicts.

Realtime models follow short bullet points better than long paragraphs.

Before (harder to follow):

```
When you can’t clearly hear the user, don’t proceed. If there’s background noise or you only caught part of the sentence, pause and ask them politely to repeat themselves in their preferred language, and make sure you keep the conversation in the same language as the user.
```

After (easier to follow):

```
1

2

3

4

5

Only respond to clear audio or text.

If audio is unclear/partial/noisy/silent, ask for clarification in \`{preferred_language}\`.

Continue in the same language as the user if intelligible.
```

The realtime model is good at following instructions on how to handle unclear audio. Spell out what to do when audio isn’t usable.

```
1

2

3

4

5

6

7

8

9

10

## Unclear audio 
- Always respond in the same language the user is speaking in, if intelligible.
- Only respond to clear audio or text. 
- If the user's audio is not clear (e.g., ambiguous input/background noise/silent/unintelligible) or if you did not fully hear or understand the user, ask for clarification using {preferred_language} phrases.

Sample clarification phrases (parameterize with {preferred_language}):

- “Sorry, I didn’t catch that—could you say it again?”
- “There’s some background noise. Please repeat the last part.”
- “I only heard part of that. What did you say after ___?”
```

If you see the model switching languages in an unhelpful way, add a dedicated "Language" section in your prompt. Make sure it doesn’t conflict with other rules. By default, mirroring the user’s language works well.

Here's a simple way to mirror the user's language:

```
1

2

3

## Language
Language matching: Respond in the same language as the user unless directed otherwise.
For non-English, start with the same standard accent/dialect the user uses.
```

Here's an example of an English-only constraint:

```
1

2

3

4

## Language
- The conversation will be only in English.
- Do not respond in any other language, even if the user asks.
- If the user speaks another language, politely explain that support is limited to English.
```

In a language teaching application, your language and conversation sections might look like this:

```
1

2

3

4

5

6

## Language
### Explanations
Use English when explaining grammar, vocabulary, or cultural context.

### Conversation
Speak in French when conducting practice, giving examples, or engaging in dialogue.
```

You can also control dialect for a more consistent personality:

```
## Language
Response only in argentine spanish.
```

The model learns style from examples. Give short, varied samples for common conversation moments.

For example, you might give this high-level shape of conversation flow to the model:

```
Greeting → Discover → Verify → Diagnose → Resolve → Confirm/Close. Advance only when criteria in each phase are met.
```

And then provide prompt guidance for each section. For example, here's how you might instruct for the greeting section:

```
1

2

3

4

5

6

7

8

9

10

11

12

13

## Conversation flow — Greeting
Goal: Set tone and invite the reason for calling.

How to respond:
- Identify as ACME Internet Support.
- Keep it brief; invite the caller’s goal.

Sample phrases (vary, don’t always reuse):
- “Thanks for calling ACME Internet—how can I help today?”
- “You’ve reached ACME Support. What’s going on with your service?”
- “Hi there—tell me what you’d like help with.”

Exit when: Caller states an initial goal or symptom.
```

If responses sound repetitive or robotic, include an explicit variety instruction. This can sometimes happen when using sample phrases.

Like many LLMs, using capitalization for important rules can help the model to understand and follow those rules. It's also helpful to convert non-text rules (such as numerical conditions) into text before capitalization.

Instead of:

```
## Rules
- If [func.return_value] > 0, respond 1 to the user.
```

Use:

```
## Rules
- IF [func.return_value] IS BIGGER THAN 0, RESPOND 1 TO THE USER.
```

The model's use of tools can alter the experience—how much they rely on user confirmation vs. taking action, what they say while they make the tool call, which rules they follow for each specific tool, etc.

One way to prompt for tool usage is to use preambles. Good preambles instruct the model to give the user some feedback about what it's doing before it makes the tool call, so the user always knows what's going on.

Here's an example:

```
# Tools
- Before any tool call, say one short line like “I’m checking that now.” Then call the tool immediately.
```

You can include sample phrases for preambles to add variety and better tailor to your use case.

There are several other ways to improve the model's behavior when performing tool calls and keeping the conversation going with the user. Ideally, the model is calling the right tools proactively, checking for confirmation for any important write actions, and keeping the user informed along the way. For more specifics, see the realtime prompting cookbook.

LLMs are great at finding what's going wrong in your prompt. Use ChatGPT or the API to get a model's review of your current realtime prompt and get help improving it.

Whether your prompt is working well or not, here's a prompt you can run to get a model's review:

```
1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

24

25

## Role & Objective  
You are a **Prompt-Critique Expert**.
Examine a user-supplied LLM prompt and surface any weaknesses following the instructions below.

## Instructions
Review the prompt that is meant for an LLM to follow and identify the following issues:
- Ambiguity: Could any wording be interpreted in more than one way?
- Lacking Definitions: Are there any class labels, terms, or concepts that are not defined that might be misinterpreted by an LLM?
- Conflicting, missing, or vague instructions: Are directions incomplete or contradictory?
- Unstated assumptions: Does the prompt assume the model has to be able to do something that is not explicitly stated?

## Do **NOT** list issues of the following types:
- Invent new instructions, tool calls, or external information. You do not know what tools need to be added that are missing.
- Issues that you are not sure about.

## Output Format

# Issues
- Numbered list; include brief quote snippets.

# Improvements
- Numbered list; provide the revised lines you would change and how you would changed them.

# Revised Prompt
- Revised prompt where you have applied all your improvements surgically with minimal edits to the original prompt
```

Use this template as a starting point for troubleshooting a recurring issue:

```
1

2

3

4

5

6

7

8

9

10

Here's my current prompt to an LLM:
[BEGIN OF CURRENT PROMPT]
{CURRENT_PROMPT}
[END OF CURRENT PROMPT]
 
But I see this issue happening from the LLM:
[BEGIN OF ISSUE]
{ISSUE}
[END OF ISSUE]
Can you provide some variants of the prompt so that the model can better understand the constraints to alleviate the issue?
```

Two frustrating user experiences are slow, mechanical voice agents and the inability to escalate. Help users faster by providing instructions in your system prompt for speed and escalation.

In the personality and tone section of your system prompt, add pacing instructions to get the model to quicken its support:

```
1

2

3

4

5

6

7

8

9

10

11

12

# Personality & Tone
## Personality
Friendly, calm and approachable expert customer service assistant.

## Tone
Tone: Warm, concise, confident, never fawning.

## Length
2–3 sentences per turn.

## Pacing
Deliver your audio response fast, but do not sound rushed. Do not modify the content of your response, only increase speaking speed for the same response.
```

Often with realtime voice agents, having a reliable way to escalate to a human is important. In a safety and escalation section, modify the instructions on WHEN to escalate depending on your use case. Here's an example:

```
1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

# Safety & Escalation
When to escalate (no extra troubleshooting):
- Safety risk (self-harm, threats, harassment)
- User explicitly asks for a human
- Severe dissatisfaction (e.g., “extremely frustrated,” repeated complaints, profanity)
- **2** failed tool attempts on the same task **or** **3** consecutive no-match/no-input events
- Out-of-scope or restricted (e.g., real-time news, financial/legal/medical advice)

What to say at the same time of calling the escalate_to_human tool (MANDATORY):
- “Thanks for your patience—**I’m connecting you with a specialist now**.”
- Then call the tool: \`escalate_to_human\`

Examples that would require escalation:
- “This is the third time the reset didn’t work. Just get me a person.”
- “I am extremely frustrated!”
```

This guide is long but not exhaustive! For more in a specific area, see the following resources:

- Realtime prompting cookbook: Full prompt examples and a deep dive into when and how to use them
- [Inputs and outputs](https://platform.openai.com/docs/guides/realtime-inputs-outputs): Text and audio input requirements and output options
- [Managing conversations](https://platform.openai.com/docs/guides/realtime-conversations): Learn to manage a conversation for the duration of a realtime session
- [Webhooks and server-side controls](https://platform.openai.com/docs/guides/realtime-server-controls): Create a sideband channel to separate sensitive server-side logic from an untrusted client
- [Function calling](https://platform.openai.com/docs/guides/realtime-function-calling): How to call functions in your realtime app
- [MCP servers](https://platform.openai.com/docs/guides/realtime-mcp): How to use MCP servers to access additional tools in realtime apps
- [Realtime transcription](https://platform.openai.com/docs/guides/realtime-transcription): How to transcribe audio with the Realtime API
- Voice agents: A quickstart for building a voice agent with the Agents SDK