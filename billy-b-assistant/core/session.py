import asyncio
import json
import base64
import websockets.legacy.client
import numpy as np
from scipy.signal import resample
import sounddevice as sd
import time
import os
import re
import core.audio as audio
import core.button
from core.mqtt import mqtt_publish
from core.config import (
    OPENAI_API_KEY,
    VOICE,
    TEXT_ONLY_MODE,
    DEBUG_MODE,
    CHUNK_MS,
    INSTRUCTIONS,
    MIC_TIMEOUT_SECONDS,
    SILENCE_THRESHOLD,
    update_persona_ini,
    PERSONALITY,
    BACKSTORY
)
from core.movements import move_tail_async, move_head, stop_all_motors

TOOLS = [
    {
        "name": "update_personality",
        "type": "function",
        "description": "Adjusts Billy's personality traits",
        "parameters": {
            "type": "object",
            "properties": {
                trait: {"type": "integer", "minimum": 0, "maximum": 100}
                for trait in vars(PERSONALITY)
            }
        }
    },
    {
        "name": "play_song",
        "type": "function",
        "description": "Plays a special Billy song based on a given name.",
        "parameters": {
            "type": "object",
            "properties": {
                "song": {"type": "string"}
            },
            "required": ["song"]
        }
    }
]

class BillySession:
    def __init__(self, interrupt_event=None):
        self.ws = None
        self.loop = None
        self.audio_buffer = bytearray()
        self.committed = False
        self.first_text = True
        self.full_response_text = ""
        self.last_mic_activity = [time.time()]
        self.session_active = asyncio.Event()
        self.user_spoke_after_assistant = False
        self.allow_mic_input = True
        self.interrupt_event = interrupt_event or asyncio.Event()
        self.mic_stream = None

    async def start(self):
        self.loop = asyncio.get_running_loop()
        print("\n⏱️ Session starting...")
        mqtt_publish("billy/status", "listening")

        self.audio_buffer.clear()
        self.committed = False
        self.first_text = True
        self.full_response_text = ""
        self.last_mic_activity[0] = time.time()
        self.session_active.set()
        self.user_spoke_after_assistant = False

        if self.ws is None:
            uri = "wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview"
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "openai-beta": "realtime=v1",
            }
            self.ws = await websockets.legacy.client.connect(uri, extra_headers=headers)
            await self.ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "voice": VOICE,
                    "modalities": ["text"] if TEXT_ONLY_MODE else ["audio", "text"],
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "turn_detection": {"type": "server_vad"},
                    "instructions": INSTRUCTIONS,
                    "tools": TOOLS
                }
            }))

        if not TEXT_ONLY_MODE:
            audio.playback_done_event.clear()
            audio.ensure_playback_worker_started(CHUNK_MS)

        await self.run_stream()

    def mic_callback(self, indata, *_):
        if not self.allow_mic_input or not self.session_active.is_set():
            return
        samples = indata[:, 0]
        rms = np.sqrt(np.mean(np.square(samples.astype(np.float32))))
        if rms > SILENCE_THRESHOLD:
            self.last_mic_activity[0] = time.time()
            self.user_spoke_after_assistant = True

        audio.send_mic_audio(self.ws, samples, self.loop)

    async def run_stream(self):
        if not TEXT_ONLY_MODE and audio.playback_done_event.is_set():
            await asyncio.to_thread(audio.playback_done_event.wait)

        print("🎙️ Mic stream active. Say something...")
        mqtt_publish("billy/state", "listening")
        asyncio.create_task(self.mic_timeout_checker())

        try:
            self.mic_stream = sd.InputStream(
                samplerate=audio.MIC_RATE,
                device=audio.MIC_DEVICE_INDEX,
                channels=audio.MIC_CHANNELS,
                dtype='int16',
                blocksize=audio.CHUNK_SIZE,
                callback=self.mic_callback
            )
            self.mic_stream.start()

            async for message in self.ws:
                if not self.session_active.is_set():
                    print("🚪 Session marked as inactive, stopping stream loop.")
                    break
                data = json.loads(message)
                if DEBUG_MODE:
                    print(f"\n🔁 Raw message: {data} ")
                await self.handle_message(data)

        except Exception as e:
            print(f"❌ Error opening mic input: {e}")
            self.session_active.clear()

        finally:
            if self.mic_stream:
                try:
                    self.mic_stream.stop()
                    self.mic_stream.close()
                except Exception as e:
                    print(f"⚠️ Error closing mic stream: {e}")
                self.mic_stream = None
            print("🎙️ Mic stream closed.")
            await self.post_response_handling()

    async def handle_message(self, data):
        if not TEXT_ONLY_MODE and data["type"] in ("response.audio", "response.audio.delta"):
            if not self.committed:
                await self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                self.committed = True
            audio_b64 = data.get("audio") or data.get("delta")
            if audio_b64:
                audio_chunk = base64.b64decode(audio_b64)
                self.audio_buffer.extend(audio_chunk)
                self.last_mic_activity[0] = time.time()
                audio.playback_queue.put(audio_chunk)

                if self.interrupt_event.is_set():
                    print("⛔ Assistant turn interrupted. Stopping response playback.")
                    while not audio.playback_queue.empty():
                            try:
                                audio.playback_queue.get_nowait()
                                audio.playback_queue.task_done()
                            except Exception:
                                break

                    self.session_active.clear()
                    self.interrupt_event.clear()
                    return

        if data["type"] in ("response.audio_transcript.delta", "response.text.delta") and "delta" in data:
            self.allow_mic_input = False
            if self.first_text:
                print("\n🐟 Billy: ", end='', flush=True)
                mqtt_publish("billy/state", "speaking")
                self.first_text = False
            print(data["delta"], end='', flush=True)
            self.full_response_text += data["delta"]

        if data["type"] == "response.function_call_arguments.done":
            if data.get("name") == "update_personality":
                args = json.loads(data["arguments"])
                changes = []
                for trait, val in args.items():
                    if hasattr(PERSONALITY, trait) and isinstance(val, int):
                        setattr(PERSONALITY, trait, val)
                        update_persona_ini(trait, val)
                        changes.append((trait, val))
                if changes:
                    print("\n🎛️ Personality updated via function_call:")
                    for trait, val in changes:
                        print(f"  - {trait.capitalize()}: {val}%")
                    print("\n🧠 New Instructions:\n")
                    print(PERSONALITY.generate_prompt())

                    self.user_spoke_after_assistant = True
                    self.full_response_text = ""
                    self.last_mic_activity[0] = time.time()

                    confirmation_text = " ".join([f"Okay, {trait} is now set to {val}%." for trait, val in changes])
                    await self.ws.send(json.dumps({
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": confirmation_text}]
                        }
                    }))
                    await self.ws.send(json.dumps({"type": "response.create"}))

            elif data.get("name") == "play_song":
                args = json.loads(data["arguments"])
                song_name = args.get("song")
                if song_name:
                    print(f"\n🎵 Assistant requested to play song: {song_name} ")
                    await self.stop_session()
                    await asyncio.sleep(1.0)
                    await audio.play_song(song_name)
                    return

        if data["type"] == "response.done":
            print("\n✿ Assistant response complete.")

            if not TEXT_ONLY_MODE:
                await asyncio.to_thread(audio.playback_queue.join)

                if len(self.audio_buffer) > 0:
                    print(f"💾 Saving audio buffer ({len(self.audio_buffer)} bytes)")
                    audio.rotate_and_save_response_audio(self.audio_buffer)
                else:
                    print("⚠️ Audio buffer was empty, skipping save.")

                self.allow_mic_input = True
                self.audio_buffer.clear()

    async def mic_timeout_checker(self):
        print("🛡️ Mic timeout checker active")
        last_tail_move = 0

        while self.session_active.is_set():
            now = time.time()
            idle_seconds = now - max(self.last_mic_activity[0], audio.last_played_time)
            timeout_offset = 2

            if idle_seconds - timeout_offset > 0.5:
                elapsed = idle_seconds - timeout_offset
                progress = min(elapsed / MIC_TIMEOUT_SECONDS, 1.0)
                bar_len = 20
                filled = int(bar_len * progress)
                bar = '█' * filled + '-' * (bar_len - filled)
                print(f"\r👂 {MIC_TIMEOUT_SECONDS}s timeout: [{bar}] {elapsed:.1f}s", end='', flush=True)

                if now - last_tail_move > 1.0:
                    move_tail_async(duration=0.2)
                    last_tail_move = now

                if elapsed > MIC_TIMEOUT_SECONDS:
                    print(f"\n⏱️ No mic activity for {MIC_TIMEOUT_SECONDS}s. Ending input...")
                    await self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                    self.session_active.clear()
                    break

            await asyncio.sleep(0.5)

    async def post_response_handling(self):
        print(f"🧠 Full response: {self.full_response_text.strip()} ")

        core.button.is_active = False

        if not self.session_active.is_set():
            print("🚪 Session inactive after timeout or interruption. Not restarting.")
            mqtt_publish("billy/state", "idle")
            stop_all_motors()
            print("🕐 Waiting for button press...")
            return

        if re.search(r"[a-zA-Z]\?\s*$", self.full_response_text.strip()) and self.user_spoke_after_assistant:
            print("🔁 Follow-up detected. Restarting...\n")
            await self.start()
        else:
            print("🛑 No follow-up. Ending session.")
            mqtt_publish("billy/state", "idle")
            stop_all_motors()
            await self.ws.close()
            print("🕐 Waiting for button press...")

    async def stop_session(self):
        print("🛑 Stopping session for song playback...")
        self.session_active.clear()

        if self.mic_stream:
            try:
                self.mic_stream.stop()
                self.mic_stream.close()
                await asyncio.sleep(0.2)
            except Exception as e:
                print(f"⚠️ Error closing mic stream: {e}")
            self.mic_stream = None

        if self.ws:
            await self.ws.close()
            self.ws = None

        stop_all_motors()
        await asyncio.sleep(1)
