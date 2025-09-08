---
title: "Realtime API with SIP - OpenAI API"
source: "https://platform.openai.com/docs/guides/realtime-sip"
author:
published:
created: 2025-08-29
description: "Learn how to connect to the Realtime API using SIP."
tags:
  - "clippings"
---
Connect to the Realtime API using SIP.

SIP is a protocol used to make phone calls over the internet. With SIP and the Realtime API you can direct incoming phone calls to the API.

## Overview

If you want to connect a phone number to the Realtime API, use a SIP trunking provider (e.g., Twilio). This is a service that converts your phone call to IP traffic. After you purchase a phone number from your SIP trunking provider, follow the instructions below.

Start by creating a [webhook](https://platform.openai.com/docs/guides/webhooks) for incoming calls, at platform.openai.com. Then, point your SIP trunk at the OpenAI SIP endpoint, using the project ID for which you configured the webhook, e.g., `sip:$PROJECT_ID@sip.api.openai.com;transport=tls`. To find your `$PROJECT_ID`, go to your \[settings\] > **General**. The page displays the project ID. It should have a `proj_` prefix.

When OpenAI receives SIP traffic associated with your project, the webhook that you configured will be fired. The event fired will be a [`realtime.call.incoming`](https://platform.openai.com/docs/api-reference/webhook_events/realtime/call/incoming) event.

This webhook lets you accept or reject the call. When accepting the call, you'll provide the configuration (instructions, voice, etc) for the Realtime API session. Once established, you can set up a web socket and monitor the session as usual. The APIs to accept, reject, and monitor the call are documented below.

## Connection details

URIs used for interacting with Realtime API and SIP:

| **SIP URI** | `sip:$PROJECT_ID@sip.api.openai.com;transport=tls` |
| --- | --- |
| **Accept URI** | `https://api.openai.com/v1/realtime/calls/$CALL_ID/accept` |
| **Reject URI** | `https://api.openai.com/v1/realtime/calls/$CALL_ID/reject` |
| **Refer URI** | `https://api.openai.com/v1/realtime/calls/$CALL_ID/refer` |
| **Events URI** | `wss://api.openai.com/v1/realtime?call_id=$CALL_ID` |

Find your `$CALL_ID` in the `call_id` field in data object present in the webhook. See an example in the next section.

The following is an example of a `realtime.call.incoming` handler. It accepts the call and then logs all the events from the Realtime API.

Python

python

```python
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

34

35

36

37

38

39

40

41

42

43

44

45

46

47

48

49

50

51

52

53

54

55

56

57

58

59

60

61

62

63

64

65

66

67

68

69

70

71

72

73

74

75

76

77

from flask import Flask, request, Response, jsonify, make_response
from openai import OpenAI, InvalidWebhookSignatureError
import asyncio
import json
import os
import requests
import time
import threading
import websockets

app = Flask(__name__)
client = OpenAI(
    webhook_secret=os.environ["OPENAI_WEBHOOK_SECRET"]
)

AUTH_HEADER = {
    "Authorization": "Bearer " + os.getenv("OPENAI_API_KEY")
}

call_accept = {
    "type": "realtime",
    "instructions": "You are a support agent.",
    "model": "gpt-4o-realtime-preview-2024-12-17",
}

response_create = {
    "type": "response.create",
    "response": {
        "instructions": (
            "Say to the user 'Thank you for calling, how can I help you'"
        )
    },
}

async def websocket_task(call_id):
    try:
        async with websockets.connect(
            "wss://api.openai.com/v1/realtime?call_id=" + call_id,
            additional_headers=AUTH_HEADER,
        ) as websocket:
            await websocket.send(json.dumps(response_create))

            while True:
                response = await websocket.recv()
                print(f"Received from WebSocket: {response}")
    except Exception as e:
        print(f"WebSocket error: {e}")

@app.route("/", methods=["POST"])
def webhook():
    try:
        event = client.webhooks.unwrap(request.data, request.headers)

        if event.type == "realtime.call.incoming":
            requests.post(
                "https://api.openai.com/v1/realtime/calls/"
                + event.data.call_id
                + "/accept",
                headers={**AUTH_HEADER, "Content-Type": "application/json"},
                json=call_accept,
            )
            threading.Thread(
                target=lambda: asyncio.run(
                    websocket_task(event.data.call_id)
                ),
                daemon=True,
            ).start()
            return Response(status=200)
    except InvalidWebhookSignatureError as e:
        print("Invalid signature", e)
        return Response("Invalid signature", status=400)

if __name__ == "__main__":
    app.run(port=8000)
```

It's also possible to redirect the call to another number. During the call, make a POST to the `refer` endpoint:

| **URL** | `https://api.openai.com/v1/realtime/calls/$CALL_ID/refer` |
| --- | --- |
| **Payload** | JSON with one key **`target_uri`**      This is the value used in the Refer-To. You can use Tel-URI for example `tel:+14152909007` |
| **Headers** | **`Authorization: Bearer YOUR_API_KEY`**      Substitute `YOUR_API_KEY` with a [standard API key](https://platform.openai.com/settings/organization/api-keys) |

Now that you've connected over SIP, use the left navigation or click into these pages to start building your realtime application.

- [Using realtime models](https://platform.openai.com/docs/guides/realtime-models-prompting)
- [Managing conversations](https://platform.openai.com/docs/guides/realtime-conversations)
- [Webhooks and server-side controls](https://platform.openai.com/docs/guides/realtime-server-controls)
- [Realtime transcription](https://platform.openai.com/docs/guides/realtime-transcription)