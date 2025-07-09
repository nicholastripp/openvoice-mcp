#!/usr/bin/env python3
"""
Home Assistant Realtime Voice Assistant

A standalone Raspberry Pi voice assistant that provides natural, low-latency 
conversations for Home Assistant control using OpenAI's Realtime API.
"""
import asyncio
import argparse
import signal
import sys
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config, AppConfig
from personality import PersonalityProfile
from utils.logger import setup_logging, get_logger
from openai_client.realtime import OpenAIRealtimeClient
from ha_client.conversation import HomeAssistantConversationClient
from audio.capture import AudioCapture
from audio.playback import AudioPlayback
from function_bridge import FunctionCallBridge
from wake_word.detector import WakeWordDetector


class VoiceAssistant:
    """Main voice assistant application"""
    
    def __init__(self, config: AppConfig, personality: PersonalityProfile):
        self.config = config
        self.personality = personality
        self.logger = get_logger("VoiceAssistant")
        self.running = False
        self._shutdown_event = asyncio.Event()
        
        # Store reference to event loop for thread-safe async calls
        self.loop = asyncio.get_event_loop()
        
        # Components (will be initialized later)
        self.openai_client: Optional[OpenAIRealtimeClient] = None
        self.ha_client: Optional[HomeAssistantConversationClient] = None
        self.audio_capture: Optional[AudioCapture] = None
        self.audio_playback: Optional[AudioPlayback] = None
        self.function_bridge: Optional[FunctionCallBridge] = None
        self.wake_word_detector: Optional[WakeWordDetector] = None
        
        # Session state
        self.session_active = False
        self.last_activity = asyncio.get_event_loop().time()
        self.session_start_time = None
        self.vad_timeout_task = None
    
    async def start(self) -> None:
        """Start the voice assistant"""
        self.logger.info("Starting Home Assistant Realtime Voice Assistant")
        
        try:
            # Initialize components
            await self._initialize_components()
            
            # Start main loop
            self.running = True
            await self._main_loop()
            
        except Exception as e:
            self.logger.error(f"Error starting assistant: {e}", exc_info=True)
            raise
    
    async def stop(self) -> None:
        """Stop the voice assistant"""
        self.logger.info("Stopping voice assistant...")
        self.running = False
        self._shutdown_event.set()
        
        # Cleanup components
        await self._cleanup_components()
        
        self.logger.info("Voice assistant stopped")
    
    async def _initialize_components(self) -> None:
        """Initialize all components"""
        self.logger.info("Initializing components...")
        
        # Initialize Home Assistant client
        self.logger.info("Initializing Home Assistant client...")
        self.ha_client = HomeAssistantConversationClient(self.config.home_assistant)
        await self.ha_client.start()
        
        # Initialize function bridge
        self.function_bridge = FunctionCallBridge(self.ha_client)
        
        # Initialize OpenAI client
        self.logger.info("Initializing OpenAI client...")
        personality_prompt = self.personality.generate_prompt()
        self.openai_client = OpenAIRealtimeClient(self.config.openai, personality_prompt)
        
        # Register function handlers
        for func_def in self.function_bridge.get_function_definitions():
            # Create a wrapper function that calls the bridge with the correct arguments
            def create_wrapper(func_name):
                async def function_wrapper(arguments):
                    return await self.function_bridge.handle_function_call(func_name, arguments)
                return function_wrapper
            
            self.openai_client.register_function(
                name=func_def["name"],
                handler=create_wrapper(func_def["name"]),
                description=func_def["description"],
                parameters=func_def["parameters"]
            )
        
        # Setup OpenAI event handlers
        self._setup_openai_handlers()
        
        # Connect to OpenAI
        success = await self.openai_client.connect()
        if not success:
            raise RuntimeError("Failed to connect to OpenAI Realtime API")
        
        # Initialize audio components
        self.logger.info("Initializing audio components...")
        self.audio_capture = AudioCapture(self.config.audio)
        self.audio_playback = AudioPlayback(self.config.audio)
        
        await self.audio_capture.start()
        await self.audio_playback.start()
        
        # Initialize wake word detector
        if self.config.wake_word.enabled:
            self.logger.info("Initializing wake word detector...")
            self.wake_word_detector = WakeWordDetector(self.config.wake_word)
            await self.wake_word_detector.start()
            
            # Setup wake word detection callback
            self.wake_word_detector.add_detection_callback(self._on_wake_word_detected)
            
            # Setup audio handlers for wake word detection
            self.audio_capture.add_callback(self._on_audio_captured_for_wake_word)
        else:
            # If wake word disabled, setup direct audio capture (development mode)
            self.audio_capture.add_callback(self._on_audio_captured_direct)
        
        self.logger.info("All components initialized successfully")
    
    async def _cleanup_components(self) -> None:
        """Cleanup all components"""
        components = [
            ("wake_word_detector", self.wake_word_detector),
            ("audio_capture", self.audio_capture),
            ("audio_playback", self.audio_playback),
            ("openai_client", self.openai_client),
            ("ha_client", self.ha_client)
        ]
        
        for name, component in components:
            if component and hasattr(component, 'stop'):
                try:
                    self.logger.debug(f"Stopping {name}")
                    await component.stop()
                except Exception as e:
                    self.logger.warning(f"Error stopping {name}: {e}")
            elif component and hasattr(component, 'disconnect'):
                try:
                    self.logger.debug(f"Disconnecting {name}")
                    await component.disconnect()
                except Exception as e:
                    self.logger.warning(f"Error disconnecting {name}: {e}")
    
    async def _main_loop(self) -> None:
        """Main application loop"""
        self.logger.info("Voice assistant started and listening...")
        
        try:
            while self.running:
                # Check connection health
                await self._check_connections()
                
                # Handle session timeouts
                await self._handle_session_timeout()
                
                # TODO: Handle wake word detection
                # For now, we'll assume always listening (for development)
                
                # Wait briefly before next iteration
                await asyncio.sleep(0.1)
                
                # Check for shutdown
                if self._shutdown_event.is_set():
                    break
                    
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}", exc_info=True)
            raise
    
    async def _check_connections(self) -> None:
        """Check and maintain connections"""
        # Check OpenAI connection
        if self.openai_client and self.openai_client.state.value == "failed":
            self.logger.warning("OpenAI connection failed, attempting reconnect...")
            await self.openai_client.connect()
    
    async def _handle_session_timeout(self) -> None:
        """Handle session timeouts"""
        if not self.session_active:
            return
            
        current_time = asyncio.get_event_loop().time()
        timeout = self.config.session.timeout
        
        if current_time - self.last_activity > timeout:
            self.logger.info("Session timeout, ending session")
            await self._end_session()
    
    async def _start_session(self) -> None:
        """Start a voice session"""
        if self.session_active:
            self.logger.warning("Attempted to start session but already active")
            return
            
        self.session_active = True
        self.last_activity = asyncio.get_event_loop().time()
        self.session_start_time = asyncio.get_event_loop().time()
        
        # Clear any existing audio queue
        if self.audio_playback:
            self.audio_playback.clear_queue()
            self.logger.debug("Cleared audio playback queue")
        
        # Reset conversation context
        if self.function_bridge:
            self.function_bridge.reset_conversation()
            self.logger.debug("Reset conversation context")
        
        # Start VAD timeout fallback
        self.vad_timeout_task = asyncio.create_task(self._vad_timeout_handler())
        
        self.logger.info("Voice session started - ready to receive audio input")
        print("*** VOICE SESSION ACTIVE - SPEAK YOUR QUESTION ***")
    
    async def _end_session(self) -> None:
        """End the current voice session"""
        if not self.session_active:
            return
            
        self.session_active = False
        
        # Cancel VAD timeout task if it exists
        if self.vad_timeout_task and not self.vad_timeout_task.done():
            self.vad_timeout_task.cancel()
            self.vad_timeout_task = None
        
        # Note: With server VAD enabled, manual commit_audio() calls are not needed
        # and will cause "input_audio_buffer_commit_empty" errors
        self.logger.debug("Session ended - server VAD handles audio buffer automatically")
        
        self.logger.info("Voice session ended")
    
    def _setup_openai_handlers(self) -> None:
        """Setup OpenAI event handlers"""
        if not self.openai_client:
            return
            
        # Audio response handler
        self.openai_client.on("audio_response", self._on_audio_response)
        
        # Audio response complete handler
        self.openai_client.on("audio_response_done", self._on_audio_response_done)
        
        # Speech stopped handler (user stopped talking)
        self.openai_client.on("speech_stopped", self._on_speech_stopped)
        
        # Error handler
        self.openai_client.on("error", self._on_openai_error)
    
    async def _on_audio_captured_for_wake_word(self, audio_data: bytes) -> None:
        """Handle captured audio for wake word detection"""
        if not self.session_active:
            # Send audio to wake word detector with proper sample rate
            if self.wake_word_detector:
                # Audio from capture is at the device sample rate (before resampling to 24kHz)
                device_sample_rate = self.config.audio.sample_rate
                self.wake_word_detector.process_audio(audio_data, input_sample_rate=device_sample_rate)
                # Debug: Log that audio is being sent to wake word detector
                if hasattr(self, '_wake_word_debug_counter'):
                    self._wake_word_debug_counter += 1
                else:
                    self._wake_word_debug_counter = 1
                    
                if self._wake_word_debug_counter % 100 == 0:  # Every 100 chunks
                    self.logger.debug(f"Sent {self._wake_word_debug_counter} audio chunks to wake word detector")
        else:
            # During active session, send audio to OpenAI
            await self._send_audio_to_openai(audio_data)
    
    async def _on_audio_captured_direct(self, audio_data: bytes) -> None:
        """Handle captured audio directly (development mode without wake word)"""
        if not self.session_active:
            # Start session on any audio (development mode)
            await self._start_session()
        
        # Send audio to OpenAI
        await self._send_audio_to_openai(audio_data)
    
    async def _send_audio_to_openai(self, audio_data: bytes) -> None:
        """Send audio data to OpenAI and update activity"""
        # Update activity timestamp
        self.last_activity = asyncio.get_event_loop().time()
        
        # Send audio to OpenAI
        if self.openai_client:
            await self.openai_client.send_audio(audio_data)
            # Debug: Log audio being sent to OpenAI
            if hasattr(self, '_openai_audio_counter'):
                self._openai_audio_counter += 1
            else:
                self._openai_audio_counter = 1
                self.logger.info("Started sending audio to OpenAI")
                print("*** STARTED SENDING AUDIO TO OPENAI ***")
                
            if self._openai_audio_counter % 50 == 0:  # Every 50 chunks
                self.logger.debug(f"Sent {self._openai_audio_counter} audio chunks to OpenAI")
        else:
            self.logger.error("No OpenAI client available to send audio!")
            print("*** ERROR: NO OPENAI CLIENT FOR AUDIO ***")
    
    async def _on_audio_response(self, audio_data: bytes) -> None:
        """Handle audio response from OpenAI"""
        self.logger.info(f"Received audio response from OpenAI: {len(audio_data)} bytes")
        print(f"*** AUDIO RESPONSE RECEIVED: {len(audio_data)} bytes ***")
        
        if self.audio_playback:
            self.audio_playback.play_audio(audio_data)
            self.logger.debug("Audio data sent to playback system")
            print("*** AUDIO SENT TO PLAYBACK - LISTEN FOR RESPONSE ***")
        else:
            self.logger.error("No audio playback system available!")
            print("*** ERROR: NO AUDIO PLAYBACK SYSTEM ***")
    
    async def _on_audio_response_done(self, _) -> None:
        """Handle completion of audio response"""
        self.logger.info("Audio response playback completed")
        print("*** AUDIO RESPONSE COMPLETED ***")
    
    async def _on_speech_stopped(self, event_data) -> None:
        """Handle user speech stopped"""
        self.logger.info("User speech stopped - server VAD will automatically trigger response")
        print("*** USER STOPPED SPEAKING - SERVER VAD PROCESSING ***")
        
        # Cancel VAD timeout since speech was properly detected
        if self.vad_timeout_task and not self.vad_timeout_task.done():
            self.vad_timeout_task.cancel()
            self.vad_timeout_task = None
            self.logger.debug("VAD timeout cancelled - speech properly detected")
        
        # Note: With server VAD enabled, OpenAI automatically commits the audio buffer
        # when speech stops. Manual commit_audio() calls cause "input_audio_buffer_commit_empty" errors
        # because the server has already processed and committed the buffer.
        self.logger.debug("Server VAD handling audio commit automatically - no manual commit needed")
        print("*** SERVER VAD WILL HANDLE RESPONSE - WAITING FOR OPENAI ***")
    
    async def _on_openai_error(self, error_data: dict) -> None:
        """Handle OpenAI errors"""
        self.logger.error(f"OpenAI error: {error_data}")
        
        # End session on error
        await self._end_session()
    
    async def _vad_timeout_handler(self) -> None:
        """Handle VAD timeout - force response if speech_stopped never fires"""
        try:
            # Wait for VAD timeout (10 seconds after session start)
            await asyncio.sleep(10.0)
            
            if self.session_active:
                self.logger.warning("VAD timeout - forcing response generation (speech_stopped never received)")
                print("*** VAD TIMEOUT - FORCING RESPONSE GENERATION ***")
                
                # Force OpenAI to generate a response
                if self.openai_client:
                    try:
                        # Create a response request to trigger OpenAI
                        response_event = {
                            "type": "response.create"
                        }
                        await self.openai_client._send_event(response_event)
                        self.logger.info("Forced response creation due to VAD timeout")
                    except Exception as e:
                        self.logger.error(f"Error forcing response after VAD timeout: {e}")
                        
        except asyncio.CancelledError:
            # Task was cancelled because speech was properly detected
            pass
        except Exception as e:
            self.logger.error(f"Error in VAD timeout handler: {e}")
    
    def _on_wake_word_detected(self, model_name: str, confidence: float) -> None:
        """Handle wake word detection"""
        self.logger.info(f"Wake word '{model_name}' detected with confidence {confidence:.6f}")
        print(f"*** WAKE WORD DETECTED: {model_name} (confidence: {confidence:.6f}) ***")
        print(f"*** STARTING VOICE SESSION - LISTEN FOR RESPONSE ***")
        
        # Check current session state
        if self.session_active:
            self.logger.warning("Wake word detected but session already active - ignoring")
            print("*** SESSION ALREADY ACTIVE - IGNORING WAKE WORD ***")
            return
        
        # Ensure OpenAI connection
        if not self.openai_client or self.openai_client.state.value != "connected":
            self.logger.warning("OpenAI client not connected, attempting connection...")
            print("*** OPENAI NOT CONNECTED - ATTEMPTING CONNECTION ***")
            asyncio.run_coroutine_threadsafe(self._ensure_openai_connection(), self.loop)
        else:
            self.logger.info(f"OpenAI client connected, state: {self.openai_client.state.value}")
            print(f"*** OPENAI CONNECTED (state: {self.openai_client.state.value}) ***")
        
        # Start voice session
        self.logger.info("Starting voice session from wake word detection")
        print("*** STARTING VOICE SESSION ***")
        asyncio.run_coroutine_threadsafe(self._start_session(), self.loop)
        
        # Optional: Play acknowledgment sound or provide visual feedback
        # This could be implemented later
    
    async def _ensure_openai_connection(self) -> None:
        """Ensure OpenAI client is connected"""
        if self.openai_client:
            try:
                await self.openai_client.connect()
                self.logger.info("OpenAI client connected for wake word session")
            except Exception as e:
                self.logger.error(f"Failed to connect OpenAI client: {e}")


def setup_signal_handlers(assistant: VoiceAssistant) -> None:
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        print(f"\\nReceived signal {signum}. Shutting down...")
        asyncio.create_task(assistant.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Home Assistant Realtime Voice Assistant"
    )
    parser.add_argument(
        "--config", 
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--persona",
        default="config/persona.ini", 
        help="Path to personality file"
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Override log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as daemon (no console output)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override log level if specified
        if args.log_level:
            config.system.log_level = args.log_level
        
        # Override daemon mode if specified
        if args.daemon:
            config.system.daemon = True
        
        # Setup logging
        logger = setup_logging(
            level=config.system.log_level,
            log_file=config.system.log_file,
            console=not config.system.daemon
        )
        
        # Load personality
        personality = PersonalityProfile(args.persona)
        
        logger.info("Configuration loaded successfully")
        logger.info(f"OpenAI Model: {config.openai.model}")
        logger.info(f"OpenAI Voice: {config.openai.voice}")
        logger.info(f"HA URL: {config.home_assistant.url}")
        logger.info(f"Assistant Name: {personality.backstory.name}")
        
        # Create and start assistant
        assistant = VoiceAssistant(config, personality)
        
        # Setup signal handlers
        setup_signal_handlers(assistant)
        
        # Start the assistant
        await assistant.start()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\\nPlease create configuration files:")
        print(f"  cp config/config.yaml.example {args.config}")
        print(f"  cp config/persona.ini.example {args.persona}")
        sys.exit(1)
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\\nShutdown requested by user")
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Ensure we're running with Python 3.9+
    if sys.version_info < (3, 9):
        print("Error: Python 3.9 or higher is required")
        sys.exit(1)
    
    # Run the main function
    asyncio.run(main())