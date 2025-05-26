from gpiozero import Button
import time
import random
import asyncio
import threading
import core.audio
import core.config
from core.movements import move_head
from core.session import BillySession

# Button and session globals
is_active = False
session_thread = None
interrupt_event = threading.Event()
session_instance = None
last_button_time = 0
button_debounce_delay = 0.5  # seconds debounce

# Setup hardware button
button = Button(core.config.BUTTON_PIN, pull_up=True)

def is_billy_speaking():
    """Return True if Billy is playing audio (wake-up or response)."""
    if not core.audio.playback_done_event.is_set():
        return True
    if not core.audio.playback_queue.empty():
        return True
    return False

def on_button():
    global is_active, session_thread, interrupt_event, session_instance, last_button_time

    now = time.time()
    if now - last_button_time < button_debounce_delay:
        return  # Ignore very quick repeat presses (debounce)
    last_button_time = now

    if not button.is_pressed:
        return

    if is_active:
        print("🔁 Button pressed during active session.")
        interrupt_event.set()
        core.audio.stop_playback()

        if session_instance:
            try:
                print("🛑 Stopping active session...")
                asyncio.run_coroutine_threadsafe(
                    session_instance.stop_session(),
                    session_instance.loop
                )
                print("✅ Session stopped.")
            except Exception as e:
                print(f"⚠️ Error stopping session: {e}")
        is_active = False
        return

    is_active = True
    interrupt_event = threading.Event()  # Fresh event for each session
    print("🎤 Button pressed. Listening...")

    def run_session():
        global session_instance, is_active
        try:
            move_head("on")
            core.audio.ensure_playback_worker_started(core.config.CHUNK_MS)

            clip = core.audio.play_random_wake_up_clip()
            if clip:
                print(f"🐟 Enqueuing wake-up clip: {clip} ")
                core.audio.playback_queue.join()

            session_instance = BillySession(interrupt_event=interrupt_event)
            asyncio.run(session_instance.start())
        finally:
            move_head("off")
            is_active = False
            print("🕐 Waiting for button press...")

    session_thread = threading.Thread(target=run_session, daemon=True)
    session_thread.start()

def start_loop():
    core.audio.detect_devices(debug=core.config.DEBUG_MODE)
    button.when_pressed = on_button
    print("🎦 Ready. Press button to start a voice session. Press Ctrl+C to quit.")
    print("🕐 Waiting for button press...")
    while True:
        time.sleep(0.1)
