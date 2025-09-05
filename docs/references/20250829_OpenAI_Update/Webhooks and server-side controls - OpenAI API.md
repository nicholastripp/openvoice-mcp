---
title: "Webhooks and server-side controls - OpenAI API"
source: "https://platform.openai.com/docs/guides/realtime-server-controls"
author:
published:
created: 2025-08-29
description: "Learn how to use webhooks and server-side controls with the Realtime API."
tags:
  - "clippings"
---
Use webhooks and server-side controls with the Realtime API.

The Realtime API allows clients to connect directly to the API server via WebRTC or SIP. However, you'll most likely want tool use and other business logic to reside on your application server to keep this logic private and client-agnostic.

Keep tool use, business logic, and other details secure on the server side by connecting over a “sideband” control channel. We now have sideband options for both SIP and WebRTC connections.

## With WebRTC

When [establishing a peer connection](https://platform.openai.com/docs/guides/realtime-webrtc), you receive an SDP response from the Realtime API to configure the connection. If you used the sample code from the WebRTC guide, that looks something like this:

The SDP response will contain a `Location` header that has a unique call ID that can be used on the server to establish a WebSocket connection to that same Realtime session.

On a server, you can then [listen for events and configure the session](https://platform.openai.com/docs/guides/realtime-conversations) just as you would from the client, using the ID from this URL:

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

import WebSocket from "ws";

// You'll need to get the call ID from the browser to your
// server somehow:
const callId = "rtc_u1_9c6574da8b8a41a18da9308f4ad974ce";

// Connect to a WebSocket for the in-progress call
const url = "wss://api.openai.com/v1/realtime?call_id=" + callId;
const ws = new WebSocket(url, {
    headers: {
        Authorization: "Bearer " + process.env.OPENAI_API_KEY,
    },
});

ws.on("open", function open() {
    console.log("Connected to server.");

    // Send client events over the WebSocket once connected
    ws.send(
        JSON.stringify({
            type: "session.update",
            session: {
                type: "realtime",
                instructions: "Be extra nice today!",
            },
        })
    );
});

// Listen for and parse server events
ws.on("message", function incoming(message) {
    console.log(JSON.parse(message.toString()));
});
```

In this way, you are able to add tools, monitor sessions, and carry out business logic on the server instead of needing to configure those actions on the client.

### With SIP

1. A user connects to OpenAI via phone over SIP.
2. OpenAI sends a webhook to your application’s backend webhook URL, notifying your app of the state of the session.
1. The application server opens a WebSocket connection to the Realtime API using the `call_id` value provided in the webhook. This `call_id` looks like this: `wss://api.openai.com/v1/realtime?call_id={callId}`. The WebSocket connection will live for the life of the SIP call.