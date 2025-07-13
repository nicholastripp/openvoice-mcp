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
from enum import Enum

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
from wake_word import create_wake_word_detector


class SessionState(Enum):
    """Session state enumeration"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    AUDIO_PLAYING = "audio_playing"
    COOLDOWN = "cooldown"
    MULTI_TURN_LISTENING = "multi_turn_listening"


class VoiceAssistant:
    """Main voice assistant application"""
    
    def __init__(self, config: AppConfig, personality: PersonalityProfile):
        print("DEBUG: VoiceAssistant.__init__() called", flush=True)
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
        self.session_state = SessionState.IDLE
        self.session_active = False
        self.last_activity = asyncio.get_event_loop().time()
        self.session_start_time = None
        self.vad_timeout_task = None
        self.response_active = False
        self.response_end_task = None
        
        # Multi-turn conversation state
        self.conversation_turn_count = 0
        self.multi_turn_timeout_task = None
        self.last_user_input = None
        
        # Session watchdog for stuck session detection
        self.last_state_change = asyncio.get_event_loop().time()
        self.max_state_duration = 60.0  # 60 seconds max in any state
        self.response_start_time = None
        
        # Periodic cleanup task
        self.cleanup_task = None
        self.cleanup_interval = 30.0  # Check every 30 seconds
    
    async def start(self) -> None:
        """Start the voice assistant"""
        print("DEBUG: VoiceAssistant.start() called", flush=True)
        self.logger.info("Starting Home Assistant Realtime Voice Assistant")
        
        try:
            print("DEBUG: About to call _initialize_components()", flush=True)
            # Initialize components
            await self._initialize_components()
            
            print("DEBUG: Components initialized, creating cleanup task", flush=True)
            # Start periodic cleanup task
            self.cleanup_task = asyncio.create_task(self._periodic_cleanup())
            
            print("DEBUG: Starting main loop", flush=True)
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
        
        # Cancel cleanup task
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
        
        # Cleanup components
        await self._cleanup_components()
        
        self.logger.info("Voice assistant stopped")
    
    def _transition_to_state(self, new_state: SessionState) -> None:
        """Transition to a new session state with validation and logging"""
        if self.session_state != new_state:
            old_state = self.session_state
            
            # Validate state transition
            if not self._validate_state_transition(old_state, new_state):
                self.logger.warning(f"Invalid state transition: {old_state.value} -> {new_state.value}")
                return
            
            self.session_state = new_state
            self.last_state_change = asyncio.get_event_loop().time()
            
            # Track response timing
            if new_state == SessionState.RESPONDING:
                self.response_start_time = self.last_state_change
            
            # Log multi-turn timeout task state during transitions
            if self.multi_turn_timeout_task:
                task_state = "running" if not self.multi_turn_timeout_task.done() else "done/cancelled"
                self.logger.info(f"Multi-turn timeout task state during transition: {task_state}")
                print(f"*** MULTI-TURN TIMEOUT TASK STATE: {task_state.upper()} ***")
            
            # Enhanced logging with more context
            self.logger.info(f"Session state: {old_state.value} -> {new_state.value} (session_active: {self.session_active}, response_active: {self.response_active})")
            print(f"*** SESSION STATE: {old_state.value.upper()} -> {new_state.value.upper()} (session_active: {self.session_active}, response_active: {self.response_active}) ***")
            
            # Enhanced state visibility with visual banner
            state_banner = {
                SessionState.IDLE: "[IDLE] IDLE - Waiting for wake word",
                SessionState.LISTENING: "[LISTEN] LISTENING - Speak your question",
                SessionState.PROCESSING: "[PROCESS] PROCESSING - Analyzing speech",
                SessionState.RESPONDING: "[RESPOND] RESPONDING - Generating answer",
                SessionState.AUDIO_PLAYING: "[PLAY] PLAYING - Response audio",
                SessionState.COOLDOWN: "[PAUSE] COOLDOWN - Session ending",
                SessionState.MULTI_TURN_LISTENING: "[LOOP] MULTI-TURN - Ask follow-up"
            }
            
            if new_state in state_banner:
                print(f"\n{'='*60}")
                print(f"STATE: {state_banner[new_state]}")
                print(f"{'='*60}\n")
            
            # Log additional context for specific transitions
            if new_state == SessionState.IDLE:
                self.logger.info(f"Session entering IDLE state - audio streaming should be blocked")
                print("*** SESSION ENTERING IDLE STATE - AUDIO STREAMING SHOULD BE BLOCKED ***")
            elif new_state == SessionState.LISTENING:
                self.logger.info(f"Session entering LISTENING state - audio streaming enabled")
                print("*** SESSION ENTERING LISTENING STATE - AUDIO STREAMING ENABLED ***")
            elif new_state == SessionState.RESPONDING:
                self.logger.info(f"Session entering RESPONDING state - audio streaming will be blocked")
                print("*** SESSION ENTERING RESPONDING STATE - AUDIO STREAMING WILL BE BLOCKED ***")
    
    def _validate_state_transition(self, old_state: SessionState, new_state: SessionState) -> bool:
        """Validate that a state transition is allowed"""
        # If not in an active session, only allow transitions from IDLE
        if not self.session_active and old_state != SessionState.IDLE:
            self.logger.debug(f"Invalid transition from {old_state.value} to {new_state.value} - session not active")
            print(f"Invalid state transition: {old_state.value} -> {new_state.value}")
            return False
        
        # Define valid transitions
        valid_transitions = {
            SessionState.IDLE: [SessionState.LISTENING, SessionState.COOLDOWN],
            SessionState.LISTENING: [SessionState.PROCESSING, SessionState.IDLE, SessionState.MULTI_TURN_LISTENING],
            SessionState.PROCESSING: [SessionState.RESPONDING, SessionState.IDLE, SessionState.LISTENING],
            SessionState.RESPONDING: [SessionState.AUDIO_PLAYING, SessionState.IDLE, SessionState.MULTI_TURN_LISTENING],
            SessionState.AUDIO_PLAYING: [SessionState.IDLE, SessionState.MULTI_TURN_LISTENING, SessionState.COOLDOWN],
            SessionState.MULTI_TURN_LISTENING: [SessionState.PROCESSING, SessionState.IDLE, SessionState.LISTENING],
            SessionState.COOLDOWN: [SessionState.IDLE, SessionState.LISTENING]
        }
        
        allowed_transitions = valid_transitions.get(old_state, [])
        is_valid = new_state in allowed_transitions
        
        if not is_valid:
            self.logger.debug(f"Invalid transition from {old_state.value} to {new_state.value}. Allowed: {[s.value for s in allowed_transitions]}")
            print(f"Invalid state transition: {old_state.value} -> {new_state.value}")
        
        return is_valid
    
    async def _generate_device_aware_personality(self) -> str:
        """Generate personality prompt with device information"""
        # Start with base personality
        base_prompt = self.personality.generate_prompt()
        
        try:
            # Get device information from Home Assistant
            self.logger.info("Fetching device information from Home Assistant...")
            states = await self.ha_client.rest_client.get_states()
            
            # Group devices by domain for better organization
            device_groups = {}
            for state in states:
                entity_id = state.get("entity_id", "")
                domain = entity_id.split(".")[0] if "." in entity_id else "unknown"
                
                if domain not in device_groups:
                    device_groups[domain] = []
                
                # Include entity with friendly name if available
                friendly_name = state.get("attributes", {}).get("friendly_name", entity_id)
                device_groups[domain].append({
                    "entity_id": entity_id,
                    "name": friendly_name,
                    "state": state.get("state", "unknown")
                })
            
            # Create device context
            device_context = "\n\nAvailable devices in your smart home:\n"
            
            # Prioritize common device types
            priority_domains = ["light", "switch", "climate", "media_player", "cover", "lock", "sensor"]
            
            for domain in priority_domains:
                if domain in device_groups:
                    devices = device_groups[domain][:10]  # Limit to 10 devices per domain
                    device_context += f"\n{domain.title()}s ({len(devices)} available):\n"
                    for device in devices:
                        device_context += f"  - {device['name']} ({device['entity_id']}) - {device['state']}\n"
            
            # Add other domains (limited)
            other_domains = [d for d in device_groups.keys() if d not in priority_domains and d != "unknown"]
            for domain in other_domains[:5]:  # Limit to 5 other domains
                devices = device_groups[domain][:5]  # Limit to 5 devices per domain
                device_context += f"\n{domain.title()}s ({len(devices)} available):\n"
                for device in devices:
                    device_context += f"  - {device['name']} ({device['entity_id']}) - {device['state']}\n"
            
            # Add helpful instructions
            device_context += "\nWhen users ask about controlling devices, you can help them by using the control_home_assistant function with natural language commands."
            
            # Combine base prompt with device context
            enhanced_prompt = base_prompt + device_context
            
            self.logger.info(f"Generated device-aware personality with {len(states)} total entities across {len(device_groups)} domains")
            self.logger.debug(f"Device context preview: {device_context[:200]}...")
            
            return enhanced_prompt
            
        except Exception as e:
            self.logger.error(f"Failed to fetch device information: {e}")
            # Fall back to base personality if device fetch fails
            return base_prompt
    
    async def _initialize_components(self) -> None:
        """Initialize all components"""
        print("DEBUG: _initialize_components() started", flush=True)
        self.logger.info("Initializing components...")
        
        # Check for wake word only mode early
        wake_word_only_mode = self.config.wake_word.enabled and getattr(self.config.wake_word, 'test_mode', False)
        print(f"DEBUG: wake_word_only_mode = {wake_word_only_mode}", flush=True)
        
        if wake_word_only_mode:
            self.logger.info("WAKE WORD TEST MODE: Skipping Home Assistant and OpenAI initialization")
            print("*** WAKE WORD TEST MODE: MINIMAL INITIALIZATION ***")
            print(f"*** TEST MODE VALUE: {self.config.wake_word.test_mode} ***")
            self.ha_client = None
            self.function_bridge = None
            self.openai_client = None
        else:
            print("DEBUG: About to initialize Home Assistant client", flush=True)
            # Initialize Home Assistant client
            self.logger.info("Initializing Home Assistant client...")
            self.ha_client = HomeAssistantConversationClient(self.config.home_assistant)
            await self.ha_client.start()
            print("DEBUG: Home Assistant client initialized", flush=True)
            
            # Initialize function bridge
            self.function_bridge = FunctionCallBridge(self.ha_client)
        
        if not wake_word_only_mode:
            print("DEBUG: About to initialize OpenAI client", flush=True)
            # Initialize OpenAI client with device-aware personality
            self.logger.info("Initializing OpenAI client...")
            personality_prompt = await self._generate_device_aware_personality()
            self.openai_client = OpenAIRealtimeClient(self.config.openai, personality_prompt)
            print("DEBUG: OpenAI client created", flush=True)
        
        if not wake_word_only_mode:
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
            
            print("DEBUG: About to connect to OpenAI", flush=True)
            # Connect to OpenAI
            success = await self.openai_client.connect()
            if not success:
                raise RuntimeError("Failed to connect to OpenAI Realtime API")
            print("DEBUG: OpenAI connection successful", flush=True)
        
        print("DEBUG: About to initialize audio components", flush=True)
        # Initialize audio components
        self.logger.info("Initializing audio components...")
        self.audio_capture = AudioCapture(self.config.audio)
        self.audio_playback = AudioPlayback(self.config.audio)
        
        print("DEBUG: Starting audio capture", flush=True)
        await self.audio_capture.start()
        print("DEBUG: Starting audio playback", flush=True)
        await self.audio_playback.start()
        print("DEBUG: Audio components started", flush=True)
        
        # Setup audio completion callback
        self.audio_playback.add_completion_callback(self._on_audio_playback_complete)
        
        # Initialize wake word detector
        if self.config.wake_word.enabled:
            print("DEBUG: Wake word enabled, creating detector", flush=True)
            self.logger.info("Initializing wake word detector...")
            self.wake_word_detector = create_wake_word_detector(self.config.wake_word)
            print("DEBUG: About to start wake word detector", flush=True)
            await self.wake_word_detector.start()
            print("DEBUG: Wake word detector started", flush=True)
            
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
                # Check connection health with error handling
                try:
                    await self._check_connections()
                except Exception as e:
                    self.logger.error(f"Error checking connections: {e}")
                    # Continue loop - connection issues are handled elsewhere
                
                # Handle session timeouts with error handling
                try:
                    await self._handle_session_timeout()
                except Exception as e:
                    self.logger.error(f"Error handling session timeout: {e}")
                    # Try to recover by ending the session
                    try:
                        await self._end_session()
                    except:
                        pass
                
                # Check for stuck sessions (watchdog)
                try:
                    await self._check_stuck_session()
                except Exception as e:
                    self.logger.error(f"Error checking stuck session: {e}")
                    # Try to recover by ending the session
                    try:
                        await self._end_session()
                    except:
                        pass
                
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
        
        # Check for stuck multi-turn listening sessions
        if self.session_state == SessionState.MULTI_TURN_LISTENING:
            time_in_multi_turn = current_time - self.last_state_change
            max_multi_turn_duration = self.config.session.multi_turn_timeout * 1.5  # 1.5x the timeout
            
            if time_in_multi_turn > max_multi_turn_duration:
                self.logger.error(f"Multi-turn session stuck for {time_in_multi_turn:.1f}s (max: {max_multi_turn_duration:.1f}s) - forcing session end")
                print(f"*** MULTI-TURN SESSION STUCK FOR {time_in_multi_turn:.1f}S - FORCING SESSION END ***")
                await self._end_session()
                return
        
        if current_time - self.last_activity > timeout:
            self.logger.info(f"Session timeout after {timeout}s of inactivity, ending session")
            print(f"*** SESSION TIMEOUT AFTER {timeout}S - ENDING SESSION ***")
            await self._end_session()
    
    async def _check_stuck_session(self) -> None:
        """Check for stuck sessions and force recovery"""
        if not self.session_active:
            return
            
        current_time = asyncio.get_event_loop().time()
        time_in_state = current_time - self.last_state_change
        
        # Check if we've been in the same state too long
        if time_in_state > self.max_state_duration:
            self.logger.error(f"Session stuck in {self.session_state.value} state for {time_in_state:.1f}s - forcing recovery")
            print(f"*** SESSION STUCK IN {self.session_state.value.upper()} STATE FOR {time_in_state:.1f}S - FORCING RECOVERY ***")
            
            # Force session end
            await self._end_session()
            return
        
        # Special check for audio responses
        if (self.session_state == SessionState.RESPONDING or 
            self.session_state == SessionState.AUDIO_PLAYING) and \
           self.response_start_time:
            
            response_duration = current_time - self.response_start_time
            max_response_time = 45.0  # 45 seconds max response time
            
            if response_duration > max_response_time:
                self.logger.error(f"Audio response stuck for {response_duration:.1f}s - forcing completion")
                print(f"*** AUDIO RESPONSE STUCK FOR {response_duration:.1f}S - FORCING COMPLETION ***")
                
                # Force audio completion
                if self.audio_playback and self.audio_playback.is_response_active:
                    self.audio_playback._notify_completion()
                else:
                    # Fallback: end session directly
                    await self._fallback_session_end()
    
    async def _start_session(self) -> None:
        """Start a voice session"""
        if self.session_active:
            self.logger.warning("Attempted to start session but already active")
            return
            
        self.session_active = True
        self.last_activity = asyncio.get_event_loop().time()
        self.session_start_time = asyncio.get_event_loop().time()
        
        # Transition to listening state
        self._transition_to_state(SessionState.LISTENING)
        
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
        self.logger.info("Started VAD timeout task (5s) to handle cases where no speech is detected")
        print("*** VAD TIMEOUT TASK STARTED - WILL END SESSION IF NO SPEECH ***")
        
        # Set initial VAD to be less sensitive to prevent premature triggers
        if self.openai_client:
            await self.openai_client.update_vad_settings(threshold=0.5, silence_duration_ms=1000)
            self.logger.info("Set initial VAD settings: threshold=0.5, silence_duration=1000ms")
            print("*** INITIAL VAD SETTINGS: LESS SENSITIVE TO PREVENT PREMATURE TRIGGERS ***")
            
            # Schedule VAD adjustment after initial period
            asyncio.create_task(self._adjust_vad_after_delay())
        
        self.logger.info("Voice session started - ready to receive audio input")
        print("*** VOICE SESSION ACTIVE - SPEAK YOUR QUESTION ***")
        
    
    async def _end_session(self) -> None:
        """End the current voice session with comprehensive cleanup"""
        if not self.session_active:
            return
            
        self.logger.info(f"Ending session (current state: {self.session_state.value})")
        print(f"*** ENDING SESSION (STATE: {self.session_state.value.upper()}) ***")
        
        self.session_active = False
        self.response_active = False
        
        # Transition to idle state
        self._transition_to_state(SessionState.IDLE)
        
        # Cancel VAD timeout task if it exists
        if self.vad_timeout_task and not self.vad_timeout_task.done():
            self.vad_timeout_task.cancel()
            self.vad_timeout_task = None
            self.logger.info("Cancelled VAD timeout task during session end")
            print("*** VAD TIMEOUT TASK CANCELLED DURING SESSION END ***")
        
        # Cancel response end task if it exists
        if self.response_end_task and not self.response_end_task.done():
            self.response_end_task.cancel()
            self.response_end_task = None
            self.logger.debug("Cancelled response end task")
        
        
        # Cancel multi-turn timeout task if it exists
        if self.multi_turn_timeout_task and not self.multi_turn_timeout_task.done():
            self.multi_turn_timeout_task.cancel()
            self.multi_turn_timeout_task = None
            self.logger.info("Cancelled multi-turn timeout task during session end")
            print("*** MULTI-TURN TIMEOUT TASK CANCELLED DURING SESSION END ***")
        
        # Reset multi-turn conversation state
        self.conversation_turn_count = 0
        self.last_user_input = None
        
        # Reset audio streaming counters for clean logging
        if hasattr(self, '_openai_audio_counter'):
            self.logger.info(f"Session ended - sent {self._openai_audio_counter} audio chunks to OpenAI")
            self._openai_audio_counter = 0
        if hasattr(self, '_blocked_audio_counter'):
            self.logger.info(f"Session ended - blocked {self._blocked_audio_counter} audio chunks from OpenAI")
            self._blocked_audio_counter = 0
        
        # Clear audio playback queue to prevent stuck audio
        if self.audio_playback:
            try:
                self.audio_playback.clear_queue()
                self.logger.debug("Cleared audio playback queue")
            except Exception as e:
                self.logger.warning(f"Error clearing audio queue: {e}")
        
        # Reset wake word detector state for clean next detection
        if self.wake_word_detector:
            try:
                # Reset any stuck states in wake word detection
                self.wake_word_detector.reset_audio_buffers()
                self.logger.debug("Wake word detector reset during session cleanup")
            except Exception as e:
                self.logger.warning(f"Error resetting wake word detector: {e}")
        
        # Reset OpenAI VAD settings to enhanced values for better speech detection
        if self.openai_client:
            try:
                # Use enhanced settings that work better for speech detection
                await self.openai_client.update_vad_settings(threshold=0.2, silence_duration_ms=800)
                self.logger.debug("OpenAI VAD settings reset to enhanced values (threshold=0.2)")
            except Exception as e:
                self.logger.warning(f"Error resetting OpenAI VAD settings: {e}")
        
        # Reset audio counters
        if hasattr(self, '_openai_audio_counter'):
            self._openai_audio_counter = 0
        if hasattr(self, '_mute_debug_counter'):
            self._mute_debug_counter = 0
        if hasattr(self, '_mute_debug_counter_direct'):
            self._mute_debug_counter_direct = 0
        if hasattr(self, '_response_blocked_counter'):
            self._response_blocked_counter = 0
        
        # Note: With server VAD enabled, manual commit_audio() calls are not needed
        # and will cause "input_audio_buffer_commit_empty" errors
        self.logger.debug("Session ended - server VAD handles audio buffer automatically")
        
        self.logger.info("Voice session ended with complete cleanup")
        print("*** VOICE SESSION ENDED - READY FOR WAKE WORD ***")
        
        # Log audio capture state to verify it's still working
        if self.audio_capture and self.audio_capture.is_recording:
            self.logger.info("Audio capture is ACTIVE and ready for wake word detection")
            print("*** AUDIO CAPTURE ACTIVE - LISTENING FOR WAKE WORD ***")
        else:
            self.logger.error("Audio capture is NOT active after session end!")
            print("*** ERROR: AUDIO CAPTURE NOT ACTIVE ***")
    
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
        
        # Response failed handler
        self.openai_client.on("response_failed", self._on_response_failed)
        
        # Response done handler (when OpenAI finishes creating response)
        self.openai_client.on("response.done", self._on_response_done)
    
    async def _on_audio_captured_for_wake_word(self, audio_data: bytes) -> None:
        """Handle captured audio for wake word detection"""
        # Debug audio flow
        if not hasattr(self, '_audio_flow_counter'):
            self._audio_flow_counter = 0
            print(f"DEBUG: Audio callback registered, first audio data received: {len(audio_data)} bytes", flush=True)
        
        self._audio_flow_counter += 1
        
        # ENHANCED: Check multiple conditions for muting audio during response/cooldown
        should_mute = (
            self.config.audio.mute_during_response and 
            self.config.audio.feedback_prevention and
            (self.response_active or 
             self.session_state == SessionState.RESPONDING or 
             self.session_state == SessionState.AUDIO_PLAYING or 
             self.session_state == SessionState.COOLDOWN)
        )
        
        if should_mute:
            # Skip audio processing during response playback to prevent feedback
            if hasattr(self, '_mute_debug_counter'):
                self._mute_debug_counter += 1
                if self._mute_debug_counter % 50 == 0:  # Log every 50 chunks
                    self.logger.debug(f"Audio muted: {self._mute_debug_counter} chunks skipped (state: {self.session_state.value}, response_active: {self.response_active})")
            else:
                self._mute_debug_counter = 1
                self.logger.info(f"[MUTED] Audio muted during {self.session_state.value} state (response_active: {self.response_active})")
                print(f"*** [MUTED] AUDIO MUTED DURING {self.session_state.value.upper()} STATE ***")
            return
        
        if not self.session_active:
            # Send audio to wake word detector with proper sample rate
            if self.wake_word_detector:
                # CRITICAL FIX: Audio from capture has been resampled to 24kHz
                # We must use the actual sample rate, not the device sample rate
                actual_sample_rate = 24000  # Audio capture resamples to this rate
                
                # Initialize wake word stats if needed
                if not hasattr(self, '_wake_word_stats'):
                    self._wake_word_stats = {
                        'chunks_processed': 0,
                        'total_bytes': 0,
                        'detection_attempts': 0,
                        'last_detection_time': None
                    }
                    # Log the critical fix
                    print(f"*** CRITICAL FIX APPLIED: Using actual sample rate {actual_sample_rate}Hz instead of device rate {self.config.audio.sample_rate}Hz ***")
                    self.logger.info(f"Wake word detection using corrected sample rate: {actual_sample_rate}Hz (was incorrectly using {self.config.audio.sample_rate}Hz)")
                
                self._wake_word_stats['chunks_processed'] += 1
                self._wake_word_stats['total_bytes'] += len(audio_data)
                
                # Log every 50th chunk with enhanced info
                if self._audio_flow_counter % 50 == 0:
                    duration_seconds = self._wake_word_stats['total_bytes'] / (actual_sample_rate * 2)  # PCM16 = 2 bytes per sample
                    print(f"DEBUG: Wake word listening - chunk #{self._audio_flow_counter}, "
                          f"{duration_seconds:.1f}s processed, "
                          f"{self._wake_word_stats.get('detection_attempts', 0)} detections", flush=True)
                
                # Process audio for wake word detection
                result = self.wake_word_detector.process_audio(audio_data, input_sample_rate=actual_sample_rate)
                
                # Track if wake word processing returned any result
                if result is not None and result > 0:
                    self._wake_word_stats['detection_attempts'] += 1
                    if self._wake_word_stats['detection_attempts'] % 10 == 0:
                        print(f"*** WAKE WORD ACTIVITY: {self._wake_word_stats['detection_attempts']} partial detections ***")
                
                # Debug: Log that audio is being sent to wake word detector
                if hasattr(self, '_wake_word_debug_counter'):
                    self._wake_word_debug_counter += 1
                else:
                    self._wake_word_debug_counter = 1
                    
                if self._wake_word_debug_counter % 100 == 0:  # Every 100 chunks
                    self.logger.debug(f"Sent {self._wake_word_debug_counter} audio chunks to wake word detector")
        else:
            # During active session, send audio to OpenAI (if not muted)
            await self._send_audio_to_openai(audio_data)
    
    async def _on_audio_captured_direct(self, audio_data: bytes) -> None:
        """Handle captured audio directly (development mode without wake word)"""
        # ENHANCED: Check multiple conditions for muting audio during response/cooldown
        should_mute = (
            self.config.audio.mute_during_response and 
            self.config.audio.feedback_prevention and
            (self.response_active or 
             self.session_state == SessionState.RESPONDING or 
             self.session_state == SessionState.AUDIO_PLAYING or 
             self.session_state == SessionState.COOLDOWN)
        )
        
        if should_mute:
            # Skip audio processing during response playback to prevent feedback
            if hasattr(self, '_mute_debug_counter_direct'):
                self._mute_debug_counter_direct += 1
                if self._mute_debug_counter_direct % 50 == 0:  # Log every 50 chunks
                    self.logger.debug(f"Direct audio muted: {self._mute_debug_counter_direct} chunks skipped (state: {self.session_state.value}, response_active: {self.response_active})")
            else:
                self._mute_debug_counter_direct = 1
                self.logger.info(f"[MUTED] Direct audio muted during {self.session_state.value} state (response_active: {self.response_active})")
                print(f"*** [MUTED] DIRECT AUDIO MUTED DURING {self.session_state.value.upper()} STATE ***")
            return
        
        if not self.session_active:
            # Start session on any audio (development mode)
            await self._start_session()
        
        # Send audio to OpenAI (if not muted)
        await self._send_audio_to_openai(audio_data)
    
    async def _send_audio_to_openai(self, audio_data: bytes) -> None:
        """Send audio data to OpenAI and update activity"""
        # CRITICAL CHECK: Only send audio when session is active and in appropriate state
        if not self.session_active or self.session_state == SessionState.IDLE:
            # Track blocked audio chunks for debugging
            if hasattr(self, '_blocked_audio_counter'):
                self._blocked_audio_counter += 1
            else:
                self._blocked_audio_counter = 1
                self.logger.info(f"[BLOCKED] Started blocking audio to OpenAI - session_active: {self.session_active}, state: {self.session_state.value}")
                print(f"*** [BLOCKED] AUDIO TO OPENAI - SESSION INACTIVE OR IDLE ***")
            
            if self._blocked_audio_counter % 100 == 0:  # Every 100 blocked chunks
                self.logger.debug(f"Blocked {self._blocked_audio_counter} audio chunks from OpenAI (session_active: {self.session_active}, state: {self.session_state.value})")
            return
        
        # ENHANCED SAFETY CHECK: Don't send audio during response, cooldown, or processing
        blocked_states = [
            SessionState.RESPONDING,
            SessionState.AUDIO_PLAYING,
            SessionState.COOLDOWN,
            SessionState.PROCESSING  # Block audio during processing to prevent continuous streaming
        ]
        
        # More aggressive blocking - always block during these states regardless of config
        if self.session_state in blocked_states or self.response_active:
            # Track blocked audio chunks during response states
            if hasattr(self, '_response_blocked_counter'):
                self._response_blocked_counter += 1
            else:
                self._response_blocked_counter = 1
                self.logger.warning(f"[BLOCKED] Started blocking audio during {self.session_state.value} state (response_active: {self.response_active})")
                print(f"*** [BLOCKED] AUDIO TO OPENAI DURING {self.session_state.value.upper()} STATE ***")
            
            if self._response_blocked_counter % 50 == 0:  # Every 50 blocked chunks
                self.logger.debug(f"Blocked {self._response_blocked_counter} audio chunks during response (state: {self.session_state.value}, response_active: {self.response_active})")
            return
        
        # Additional check: Block if OpenAI is in responding mode or session should be idle
        if not self.session_active:
            self.logger.debug(f"[BLOCKED] Session not active - blocking audio streaming")
            return
        
        # ADDITIONAL VALIDATION: Check OpenAI client state
        if not self.openai_client or self.openai_client.state.value != "connected":
            self.logger.warning(f"[BLOCKED] OpenAI client not connected - cannot send audio (state: {self.openai_client.state.value if self.openai_client else 'None'})")
            print("*** [BLOCKED] OPENAI CLIENT NOT CONNECTED ***")
            return
        
        # Update activity timestamp
        self.last_activity = asyncio.get_event_loop().time()
        
        # Validate audio quality before sending to OpenAI
        if not self._validate_audio_quality(audio_data):
            self.logger.debug("Audio quality validation failed - skipping OpenAI transmission")
            return
        
        # Send audio to OpenAI
        if self.openai_client:
            # Log audio levels before sending to OpenAI
            try:
                import numpy as np
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                if len(audio_array) > 0:
                    rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
                    peak = np.max(np.abs(audio_array))
                    
                    if not hasattr(self, '_audio_level_log_counter'):
                        self._audio_level_log_counter = 0
                    self._audio_level_log_counter += 1
                    
                    if self._audio_level_log_counter % 100 == 0:  # Every 100 chunks
                        self.logger.info(f"Audio to OpenAI - RMS: {rms:.1f}, Peak: {peak}, Length: {len(audio_data)} bytes")
                        print(f"*** AUDIO LEVELS TO OPENAI: RMS={rms:.1f}, Peak={peak} ***")
            except Exception as e:
                self.logger.debug(f"Could not log audio levels: {e}")
            
            await self.openai_client.send_audio(audio_data)
            # Debug: Log audio being sent to OpenAI
            if hasattr(self, '_openai_audio_counter'):
                self._openai_audio_counter += 1
            else:
                self._openai_audio_counter = 1
                self.logger.info(f"Started sending audio to OpenAI (session_active: {self.session_active}, state: {self.session_state.value})")
                print(f"*** STARTED SENDING AUDIO TO OPENAI - STATE: {self.session_state.value.upper()} ***")
                
            if self._openai_audio_counter % 50 == 0:  # Every 50 chunks
                self.logger.debug(f"Sent {self._openai_audio_counter} audio chunks to OpenAI (state: {self.session_state.value})")
        else:
            self.logger.error("No OpenAI client available to send audio!")
            print("*** ERROR: NO OPENAI CLIENT FOR AUDIO ***")
    
    def _validate_audio_quality(self, audio_data: bytes) -> bool:
        """
        Validate audio quality before sending to OpenAI
        
        Args:
            audio_data: PCM16 audio data
            
        Returns:
            True if audio quality is acceptable, False otherwise
        """
        try:
            # Convert PCM16 bytes to numpy array for analysis
            import numpy as np
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            if len(audio_array) == 0:
                return False
            
            # Convert to float for analysis
            audio_float = audio_array.astype(np.float32) / 32767.0
            
            # Calculate audio quality metrics
            rms = np.sqrt(np.mean(audio_float ** 2))
            peak = np.max(np.abs(audio_float))
            
            # Minimum quality thresholds
            min_rms = 0.0001   # Lower threshold to allow quieter audio through
            min_peak = 0.001   # Lower peak threshold as well
            max_peak = 0.95    # Maximum peak level (to detect clipping)
            
            # Check for too quiet audio
            if rms < min_rms or peak < min_peak:
                self.logger.debug(f"Audio too quiet: RMS={rms:.6f}, peak={peak:.4f}")
                return False
            
            # Check for clipping/distortion
            if peak > max_peak:
                self.logger.debug(f"Audio clipping detected: peak={peak:.4f}")
                return False
            
            # Check for reasonable dynamic range
            if rms > 0:
                dynamic_range = peak / rms
                if dynamic_range < 1.5:  # Too compressed
                    self.logger.debug(f"Audio too compressed: dynamic_range={dynamic_range:.2f}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating audio quality: {e}")
            return True  # Allow audio through if validation fails
    
    async def _on_audio_response(self, audio_data: bytes) -> None:
        """Handle audio response from OpenAI"""
        # Calculate audio duration for debugging (PCM16 at 24kHz)
        samples = len(audio_data) // 2  # 2 bytes per sample
        duration_ms = samples / 24.0  # 24kHz sample rate
        self.logger.info(f"Received audio response from OpenAI: {len(audio_data)} bytes ({samples} samples, {duration_ms:.1f}ms)")
        print(f"*** AUDIO RESPONSE RECEIVED: {len(audio_data)} bytes ({duration_ms:.1f}ms) ***")
        
        # Mark response as active and increase VAD threshold to prevent false positives
        if not self.response_active:
            self.response_active = True
            # Transition to responding state
            self._transition_to_state(SessionState.RESPONDING)
            
            
            # Increase VAD threshold during response playback
            if self.openai_client:
                await self.openai_client.update_vad_settings(threshold=0.8, silence_duration_ms=500)
                self.logger.debug("Increased VAD threshold during response playback")
            
            # Start audio response tracking
            if self.audio_playback:
                self.audio_playback.start_response()
        
        if self.audio_playback:
            self.audio_playback.play_audio(audio_data)
            self.logger.debug("Audio data sent to playback system")
            print("*** AUDIO SENT TO PLAYBACK - LISTEN FOR RESPONSE ***")
        else:
            self.logger.error("No audio playback system available!")
            print("*** ERROR: NO AUDIO PLAYBACK SYSTEM ***")
    
    async def _on_audio_response_done(self, _) -> None:
        """Handle OpenAI finishing sending audio (not actual playback completion)"""
        self.logger.info("OpenAI finished sending audio response")
        print("*** OPENAI FINISHED SENDING AUDIO - WAITING FOR PLAYBACK COMPLETION ***")
        
        # Notify audio playback that OpenAI finished sending
        if self.audio_playback:
            self.audio_playback.end_response()
        
        # Don't end session here - wait for actual audio playback completion
        # The audio completion callback will handle session ending
        
        # Transition to audio playing state
        self._transition_to_state(SessionState.AUDIO_PLAYING)
    
    def _on_audio_playback_complete(self) -> None:
        """Handle actual audio playback completion with thread safety"""
        self.logger.info("Audio playback actually completed")
        print("*** AUDIO PLAYBACK ACTUALLY COMPLETED ***")
        
        # Run async session ending in the main event loop with error handling
        if self.loop:
            try:
                future = asyncio.run_coroutine_threadsafe(self._handle_audio_completion(), self.loop)
                
                # Add timeout to prevent hanging
                try:
                    future.result(timeout=5.0)  # 5 second timeout
                except Exception as e:
                    import traceback
                    self.logger.error(f"Error in audio completion handler: {e}")
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    print(f"*** ERROR IN AUDIO COMPLETION: {e} ***")
                    # Fallback: force session end
                    self._schedule_fallback_session_end()
                    
            except Exception as e:
                self.logger.error(f"Error scheduling audio completion handler: {e}")
                # Fallback: force session end
                self._schedule_fallback_session_end()
        else:
            self.logger.error("No event loop available for audio completion")
            # Fallback: force session end
            self._schedule_fallback_session_end()
    
    async def _handle_audio_completion(self) -> None:
        """Handle audio completion in async context"""
        try:
            # Mark response as no longer active and restore normal VAD threshold
            self.response_active = False
            
            # Check if session is still active
            if not self.session_active:
                self.logger.warning("Audio completion called but session not active - ignoring")
                return
            
            # Validate we're in a valid state for audio completion
            if self.session_state not in [SessionState.AUDIO_PLAYING, SessionState.RESPONDING]:
                self.logger.warning(f"Audio completion in unexpected state: {self.session_state.value} - ignoring")
                return
            
            # Restore enhanced VAD threshold for better speech detection
            if self.openai_client:
                await self.openai_client.update_vad_settings(threshold=0.2, silence_duration_ms=800)
                self.logger.debug("Restored enhanced VAD threshold (0.2) after actual playback completion")
            
            # Check if multi-turn conversation mode is enabled
            conversation_mode = getattr(self.config.session, 'conversation_mode', 'single_turn')
            self.logger.info(f"Audio completion - conversation mode: {conversation_mode}, session_active: {self.session_active}, state: {self.session_state.value}")
            print(f"*** AUDIO COMPLETION - MODE: {conversation_mode}, ACTIVE: {self.session_active} ***")
            
            if conversation_mode == "multi_turn" and self.session_active:
                # Check if the last response contained conversation end phrases
                if self.last_user_input and self._contains_end_phrases(self.last_user_input):
                    self.logger.info("Conversation end phrase detected - ending session naturally")
                    print("*** CONVERSATION END PHRASE DETECTED - ENDING SESSION NATURALLY ***")
                    await self._end_session()
                    return
                
                # Increment conversation turn count
                self.conversation_turn_count += 1
                
                # Check if we've reached the maximum number of turns
                max_turns = getattr(self.config.session, 'multi_turn_max_turns', 10)
                if self.conversation_turn_count >= max_turns:
                    self.logger.info(f"Maximum turns ({max_turns}) reached - ending session")
                    print(f"*** MAXIMUM TURNS ({max_turns}) REACHED - ENDING SESSION ***")
                    await self._end_session()
                    return
                
                # Get timeout value safely
                multi_turn_timeout = getattr(self.config.session, 'multi_turn_timeout', 30.0)
                
                # Transition to multi-turn listening state
                self.logger.info(f"Multi-turn conversation active (turn {self.conversation_turn_count}/{max_turns})")
                print(f"*** MULTI-TURN CONVERSATION ACTIVE (TURN {self.conversation_turn_count}/{max_turns}) ***")
                print(f"*** LISTENING FOR FOLLOW-UP QUESTION (TIMEOUT: {multi_turn_timeout}s) ***")
                
                self._transition_to_state(SessionState.MULTI_TURN_LISTENING)
                
                # Set up timeout for multi-turn conversation
                self.multi_turn_timeout_task = asyncio.create_task(self._handle_multi_turn_timeout())
                self.logger.info(f"Created multi-turn timeout task (timeout: {multi_turn_timeout}s)")
                print(f"*** MULTI-TURN TIMEOUT TASK CREATED: {multi_turn_timeout}s ***")
                
                return
        
        # Original single-turn logic
        # Check if we should auto-end the session
        if (self.config.session.auto_end_after_response and 
            self.session_active and 
            self.config.session.response_cooldown_delay > 0):
            
            # Schedule session end after cooldown delay
            self.logger.info(f"Scheduling session end in {self.config.session.response_cooldown_delay} seconds")
            print(f"*** SCHEDULING SESSION END IN {self.config.session.response_cooldown_delay} SECONDS ***")
            self.response_end_task = asyncio.create_task(self._schedule_session_end())
        elif self.config.session.auto_end_after_response:
            # End session immediately if no cooldown delay
            self.logger.info("Auto-ending session after actual playback completion")
            print("*** AUTO-ENDING SESSION AFTER ACTUAL PLAYBACK COMPLETION ***")
            await self._end_session()
        
        except Exception as e:
            import traceback
            self.logger.error(f"Exception in _handle_audio_completion: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Try to end session gracefully
            if self.session_active:
                await self._end_session()
    
    def _schedule_fallback_session_end(self) -> None:
        """Schedule fallback session end from thread context"""
        if self.loop:
            try:
                self.logger.warning("Scheduling fallback session end due to audio completion failure")
                future = asyncio.run_coroutine_threadsafe(self._fallback_session_end(), self.loop)
                # Don't wait for result to avoid blocking
            except Exception as e:
                self.logger.error(f"Error scheduling fallback session end: {e}")
    
    async def _fallback_session_end(self) -> None:
        """Fallback session end when audio completion fails"""
        try:
            self.logger.warning("Executing fallback session end")
            print("*** FALLBACK SESSION END - RECOVERING FROM AUDIO COMPLETION FAILURE ***")
            
            # Force response to inactive
            self.response_active = False
            
            # Restore enhanced VAD threshold
            if self.openai_client:
                await self.openai_client.update_vad_settings(threshold=0.2, silence_duration_ms=800)
            
            # End session
            await self._end_session()
            
        except Exception as e:
            self.logger.error(f"Error in fallback session end: {e}")
            # Last resort - force session cleanup
            self.session_active = False
            self.response_active = False
            self._transition_to_state(SessionState.IDLE)
    
    async def _handle_multi_turn_timeout(self) -> None:
        """Handle timeout for multi-turn conversations"""
        try:
            self.logger.info(f"Multi-turn timeout task started - will wait {self.config.session.multi_turn_timeout}s")
            print(f"*** MULTI-TURN TIMEOUT TASK STARTED: WAITING {self.config.session.multi_turn_timeout}s ***")
            
            # Wait for multi-turn timeout
            await asyncio.sleep(self.config.session.multi_turn_timeout)
            
            # If we reach here, no follow-up question was received
            self.logger.info(f"Multi-turn timeout task completed - checking session state")
            print(f"*** MULTI-TURN TIMEOUT COMPLETED - SESSION STATE: {self.session_state.value} ***")
            
            if self.session_active and self.session_state == SessionState.MULTI_TURN_LISTENING:
                self.logger.info(f"Multi-turn conversation timeout after {self.config.session.multi_turn_timeout}s - ending session")
                print(f"*** MULTI-TURN TIMEOUT AFTER {self.config.session.multi_turn_timeout}S - ENDING SESSION ***")
                await self._end_session()
            else:
                self.logger.info(f"Multi-turn timeout completed but session not in MULTI_TURN_LISTENING state (active: {self.session_active}, state: {self.session_state.value})")
                print(f"*** MULTI-TURN TIMEOUT COMPLETED BUT SESSION STATE CHANGED: active={self.session_active}, state={self.session_state.value} ***")
        except asyncio.CancelledError:
            # Task was cancelled (user spoke again)
            self.logger.info("Multi-turn timeout task cancelled - user spoke again or session ended")
            print("*** MULTI-TURN TIMEOUT TASK CANCELLED ***")
        except Exception as e:
            self.logger.error(f"Error in multi-turn timeout handler: {e}")
            print(f"*** ERROR IN MULTI-TURN TIMEOUT HANDLER: {e} ***")
            # End session on error to prevent hanging
            if self.session_active:
                await self._end_session()
    
    def _contains_end_phrases(self, text: str) -> bool:
        """Check if text contains conversation end phrases"""
        if not text:
            return False
        
        text_lower = text.lower().strip()
        
        # Check configured end phrases
        for phrase in self.config.session.multi_turn_end_phrases:
            if phrase.lower() in text_lower:
                self.logger.info(f"End phrase detected: '{phrase}' in '{text}'")
                print(f"*** END PHRASE DETECTED: '{phrase}' in '{text}' ***")
                return True
        
        # Additional flexible matching for common end patterns
        end_patterns = [
            "nothing else",
            "that's all",
            "that'll be all",
            "that will be all",
            "i'm done",
            "we're done",
            "that's it",
            "that's everything",
            "no more",
            "finished",
            "done",
            "end session",
            "exit",
            "quit"
        ]
        
        for pattern in end_patterns:
            if pattern in text_lower:
                self.logger.info(f"End pattern detected: '{pattern}' in '{text}'")
                print(f"*** END PATTERN DETECTED: '{pattern}' in '{text}' ***")
                return True
        
        return False
    
    
    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup task to prevent hanging sessions"""
        while self.running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                if not self.running:
                    break
                    
                # Check for orphaned sessions
                current_time = asyncio.get_event_loop().time()
                
                # If we have a session that's been active too long, clean it up
                # For multi-turn sessions, use a longer timeout
                timeout_threshold = self.cleanup_interval * 2
                if self.config.session.conversation_mode == "multi_turn":
                    timeout_threshold = max(timeout_threshold, self.config.session.multi_turn_timeout * 1.5)
                
                if (self.session_active and 
                    current_time - self.last_activity > timeout_threshold):
                    
                    self.logger.warning(f"Orphaned session detected, cleaning up (inactive for {current_time - self.last_activity:.1f}s)")
                    print(f"*** ORPHANED SESSION CLEANUP AFTER {current_time - self.last_activity:.1f}S ***")
                    
                    try:
                        await self._end_session()
                    except Exception as e:
                        self.logger.error(f"Error during orphaned session cleanup: {e}")
                        # Force cleanup
                        self.session_active = False
                        self.response_active = False
                        self._transition_to_state(SessionState.IDLE)
                
                # Check for stuck multi-turn listening state
                if (self.session_state == SessionState.MULTI_TURN_LISTENING and
                    current_time - self.last_state_change > self.config.session.multi_turn_timeout * 2):
                    
                    self.logger.warning(f"Stuck multi-turn listening state detected, cleaning up (stuck for {current_time - self.last_state_change:.1f}s)")
                    print(f"*** STUCK MULTI-TURN LISTENING STATE CLEANUP AFTER {current_time - self.last_state_change:.1f}S ***")
                    
                    try:
                        await self._end_session()
                    except Exception as e:
                        self.logger.error(f"Error during stuck multi-turn cleanup: {e}")
                        # Force cleanup
                        self.session_active = False
                        self.response_active = False
                        self._transition_to_state(SessionState.IDLE)
                
                # Check for stuck audio playback
                if (self.audio_playback and 
                    self.audio_playback.is_response_active and 
                    current_time - self.last_activity > self.cleanup_interval):
                    
                    self.logger.warning("Stuck audio playback detected, forcing completion")
                    print("*** STUCK AUDIO PLAYBACK CLEANUP ***")
                    
                    try:
                        self.audio_playback._notify_completion()
                    except Exception as e:
                        self.logger.error(f"Error during audio playback cleanup: {e}")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in periodic cleanup: {e}")
                # Continue cleanup loop even on error
                await asyncio.sleep(5.0)
    
    async def _on_speech_stopped(self, event_data) -> None:
        """Handle user speech stopped"""
        # Enhanced debug logging for VAD events
        event_time = asyncio.get_event_loop().time()
        time_in_session = event_time - self.session_start_time if hasattr(self, 'session_start_time') else 0
        self.logger.info(f"User speech stopped - server VAD triggered after {time_in_session:.1f}s")
        print(f"*** USER STOPPED SPEAKING - VAD EVENT AT {time_in_session:.1f}s ***")
        
        # CRITICAL: Ignore speech events during cooldown to prevent false positives
        if self.session_state == SessionState.COOLDOWN:
            self.logger.warning("Ignoring speech_stopped event during cooldown - likely false positive from audio playback")
            print("*** IGNORING SPEECH EVENT DURING COOLDOWN - PREVENTING FALSE POSITIVE ***")
            return
        
        # Ignore speech events during response to prevent feedback
        if self.session_state == SessionState.RESPONDING:
            self.logger.warning("Ignoring speech_stopped event during response - likely audio feedback")
            print("*** IGNORING SPEECH EVENT DURING RESPONSE - PREVENTING FEEDBACK ***")
            return
        
        # Ignore speech events during audio playback to prevent feedback
        if self.session_state == SessionState.AUDIO_PLAYING:
            self.logger.warning("Ignoring speech_stopped event during audio playback - likely audio feedback")
            print("*** IGNORING SPEECH EVENT DURING AUDIO PLAYBACK - PREVENTING FEEDBACK ***")
            return
        
        # Only process speech events if we're in LISTENING or MULTI_TURN_LISTENING state
        if self.session_state not in [SessionState.LISTENING, SessionState.MULTI_TURN_LISTENING]:
            self.logger.warning(f"Ignoring speech_stopped event in {self.session_state.value} state")
            print(f"*** IGNORING SPEECH EVENT IN {self.session_state.value.upper()} STATE ***")
            return
        
        # CRITICAL: Require minimum speech duration to prevent premature VAD triggers
        if hasattr(self, 'session_start_time'):
            time_since_start = asyncio.get_event_loop().time() - self.session_start_time
            min_speech_duration = 1.5  # Require at least 1.5 seconds before processing
            
            if time_since_start < min_speech_duration:
                self.logger.warning(f"Ignoring premature speech_stopped - only {time_since_start:.1f}s since session start (min: {min_speech_duration}s)")
                print(f"*** IGNORING PREMATURE VAD TRIGGER - ONLY {time_since_start:.1f}s ELAPSED (MIN: {min_speech_duration}s) ***")
                return
        
        # Log VAD trigger timing for debugging
        if hasattr(self, 'session_start_time'):
            elapsed = asyncio.get_event_loop().time() - self.session_start_time
            print(f"*** VAD TRIGGERED AFTER {elapsed:.1f}s OF LISTENING ***")
        
        # Transition to processing state
        self._transition_to_state(SessionState.PROCESSING)
        
        # CRITICAL: Stop sending audio to OpenAI once speech is detected
        # This prevents continuous audio streaming while waiting for response
        self.logger.info("Speech detected - stopping audio transmission to OpenAI")
        print("*** STOPPING AUDIO TRANSMISSION - SPEECH DETECTED ***")
        
        # Cancel VAD timeout since speech was properly detected
        if self.vad_timeout_task and not self.vad_timeout_task.done():
            self.vad_timeout_task.cancel()
            self.vad_timeout_task = None
            self.logger.info("VAD timeout cancelled - speech properly detected")
            print("*** VAD TIMEOUT CANCELLED - SPEECH DETECTED ***")
        
        # Cancel any pending session end task since user is speaking
        if self.response_end_task and not self.response_end_task.done():
            self.response_end_task.cancel()
            self.response_end_task = None
            self.logger.debug("Cancelled pending session end - user is speaking")
        
        # Cancel multi-turn timeout task since user is speaking
        if self.multi_turn_timeout_task and not self.multi_turn_timeout_task.done():
            self.multi_turn_timeout_task.cancel()
            self.multi_turn_timeout_task = None
            self.logger.info("Cancelled multi-turn timeout - user is speaking")
            print("*** MULTI-TURN TIMEOUT TASK CANCELLED - USER IS SPEAKING ***")
        
        # Note: With server VAD enabled, OpenAI automatically commits the audio buffer
        # when speech stops. Manual commit_audio() calls cause "input_audio_buffer_commit_empty" errors
        # because the server has already processed and committed the buffer.
        self.logger.debug("Server VAD handling audio commit automatically - no manual commit needed")
        print("*** SERVER VAD WILL HANDLE RESPONSE - WAITING FOR OPENAI ***")
        
        # CRITICAL: Even with server VAD, we need to explicitly request a response
        # The server VAD only commits the buffer, it doesn't automatically generate a response
        if self.openai_client and self.openai_client.state.value == "connected":
            # Wait a small delay to ensure buffer is committed
            await asyncio.sleep(0.1)
            
            try:
                self.logger.info("Explicitly requesting response after speech_stopped")
                print("*** REQUESTING RESPONSE FROM OPENAI ***")
                await self.openai_client._send_event({"type": "response.create"})
            except Exception as e:
                self.logger.error(f"Failed to request response after speech_stopped: {e}")
                print(f"*** FAILED TO REQUEST RESPONSE: {e} ***")
    
    async def _on_openai_error(self, error_data: dict) -> None:
        """Handle OpenAI errors with recovery logic"""
        error_type = error_data.get('type', 'unknown')
        error_message = error_data.get('message', 'unknown error')
        
        self.logger.error(f"OpenAI error [{error_type}]: {error_message}")
        print(f"*** OPENAI ERROR [{error_type.upper()}]: {error_message} ***")
        
        # Handle specific error types with recovery
        if error_type == 'input_audio_buffer_commit_empty':
            self.logger.warning("Empty audio buffer - this is normal with server VAD")
            # Don't end session for this error - it's expected with server VAD
            return
        elif error_type == 'conversation_already_has_active_response':
            self.logger.warning("Conversation already has active response - ignoring duplicate request")
            # Don't end session for this error - it's expected in some race conditions
            return
        elif error_type == 'connection_error':
            self.logger.warning("Connection error - attempting reconnection")
            try:
                await self.openai_client.connect()
                return
            except Exception as e:
                self.logger.error(f"Reconnection failed: {e}")
        
        # End session on unrecoverable errors
        await self._end_session()
    
    async def _on_response_done(self, event_data: dict) -> None:
        """Handle OpenAI response completion"""
        response_data = event_data.get("response", {})
        response_id = response_data.get("id", "unknown")
        status = response_data.get("status", "unknown")
        
        self.logger.info(f"Response {response_id} completed with status: {status}")
    
    async def _on_response_failed(self, event_data: dict) -> None:
        """Handle failed OpenAI response"""
        response_id = event_data.get("response_id", "unknown")
        error_type = event_data.get("error_type", "unknown")
        error_message = event_data.get("error_message", "No error message")
        
        self.logger.error(f"Response {response_id} failed: {error_type} - {error_message}")
        print(f"*** RESPONSE FAILURE: {error_type} - {error_message} ***")
        
        # Check if we can retry based on error type
        if error_type in ["timeout", "network_error", "temporary_failure"]:
            self.logger.info("Attempting to retry response creation...")
            print("*** RETRYING RESPONSE CREATION ***")
            
            # Wait a bit before retrying
            await asyncio.sleep(0.5)
            
            # Request a new response
            if self.openai_client and self.openai_client.state.value == "connected":
                try:
                    await self.openai_client._send_event({"type": "response.create"})
                    self.logger.info("Retry response.create sent")
                except Exception as e:
                    self.logger.error(f"Failed to retry response: {e}")
                    await self._end_session()
            else:
                self.logger.error("Cannot retry - OpenAI client not connected")
                await self._end_session()
        else:
            # Non-retryable error - end session
            self.logger.error(f"Non-retryable error: {error_type}")
            await self._end_session()
    
    async def _adjust_vad_after_delay(self) -> None:
        """Adjust VAD to normal sensitivity after initial period"""
        await asyncio.sleep(2.0)  # Wait 2 seconds
        
        # Only adjust if still in listening state
        if self.session_state == SessionState.LISTENING and self.openai_client:
            await self.openai_client.update_vad_settings(threshold=0.2, silence_duration_ms=800)
            self.logger.info("Adjusted VAD to normal sensitivity after initial period")
            print("*** VAD ADJUSTED TO NORMAL SENSITIVITY ***")
    
    
    async def _vad_timeout_handler(self) -> None:
        """Handle VAD timeout - end session gracefully if no speech detected"""
        try:
            # Initial delay to allow user to start speaking after wake word
            self.logger.info("VAD timeout handler started - waiting 2s for user to begin speaking")
            print("*** WAITING FOR USER TO SPEAK (2s grace period) ***")
            await asyncio.sleep(2.0)
            
            # Now wait for actual VAD timeout (5 seconds to detect speech)
            self.logger.info("Starting VAD detection period (5s)")
            await asyncio.sleep(5.0)
            
            if self.session_active:
                self.logger.warning("VAD timeout - no speech detected after 7s total, ending session gracefully")
                print("*** VAD TIMEOUT - NO SPEECH DETECTED, ENDING SESSION ***")
                
                # End session gracefully instead of forcing a response
                # This prevents unwanted AI responses when no speech was actually detected
                await self._end_session()
                        
        except asyncio.CancelledError:
            # Task was cancelled because speech was properly detected
            self.logger.debug("VAD timeout cancelled - speech was properly detected")
        except Exception as e:
            self.logger.error(f"Error in VAD timeout handler: {e}")
            # End session on error to prevent hanging
            if self.session_active:
                await self._end_session()
    
    async def _schedule_session_end(self) -> None:
        """Schedule session end after cooldown delay"""
        try:
            # Transition to cooldown state
            self._transition_to_state(SessionState.COOLDOWN)
            
            # Wait for cooldown period
            await asyncio.sleep(self.config.session.response_cooldown_delay)
            
            # End session if still active and no response is playing
            if self.session_active and not self.response_active:
                self.logger.info(f"Auto-ending session after {self.config.session.response_cooldown_delay}s cooldown")
                print(f"*** AUTO-ENDING SESSION AFTER {self.config.session.response_cooldown_delay}S COOLDOWN ***")
                await self._end_session()
            else:
                self.logger.debug("Session end cancelled - session inactive or response active")
                
        except asyncio.CancelledError:
            # Task was cancelled (user spoke again)
            self.logger.debug("Session end task cancelled")
            # If cancelled, go back to listening state
            if self.session_active:
                self._transition_to_state(SessionState.LISTENING)
        except Exception as e:
            self.logger.error(f"Error in session end scheduler: {e}")
    
    def _on_wake_word_detected(self, model_name: str, confidence: float) -> None:
        """Handle wake word detection"""
        # ENHANCED: Add prominent visual banner
        print("\n" + "="*70)
        print("[MIC]  WAKE WORD DETECTED!  [MIC]".center(70))
        print("="*70)
        print(f"Wake Word: {model_name}".center(70))
        print(f"Confidence: {confidence:.6f}".center(70))
        print("="*70)
        print()
        
        self.logger.info(f"Wake word '{model_name}' detected with confidence {confidence:.6f}")
        
        # Increment wake word detection counter
        if not hasattr(self, '_wake_word_detection_count'):
            self._wake_word_detection_count = 0
        self._wake_word_detection_count += 1
        print(f"*** TOTAL WAKE WORD DETECTIONS THIS SESSION: {self._wake_word_detection_count} ***")
        
        # Check for wake word only mode
        wake_word_only_mode = self.config.wake_word.enabled and hasattr(self.config.wake_word, 'test_mode') and self.config.wake_word.test_mode
        
        if wake_word_only_mode:
            self.logger.info("WAKE WORD TEST MODE: Detection successful!")
            print("*** WAKE WORD TEST MODE: DETECTION SUCCESSFUL! ***")
            # Play a simple beep or confirmation sound
            if self.audio_playback:
                # Generate a simple beep tone
                import numpy as np
                sample_rate = 24000
                duration = 0.2  # 200ms beep
                freq = 800  # 800Hz tone
                t = np.linspace(0, duration, int(sample_rate * duration))
                beep = (0.3 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
                self.audio_playback.play_audio(beep.tobytes())
            return
        
        # Play confirmation beep for all wake word detections
        if self.audio_playback:
            # Generate a simple beep tone
            import numpy as np
            sample_rate = 24000
            duration = 0.15  # 150ms beep (shorter for production)
            freq = 600  # 600Hz tone (lower frequency)
            t = np.linspace(0, duration, int(sample_rate * duration))
            beep = (0.2 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
            self.audio_playback.play_audio(beep.tobytes())
            print("*** PLAYING CONFIRMATION BEEP ***")
        
        print(f"*** STARTING VOICE SESSION - SPEAK NOW ***")
        
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
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in wake word test mode (no OpenAI/HA connection)"
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
            
        # Override test mode if specified
        if args.test_mode:
            config.wake_word.test_mode = True
            # Note: Can't use logger here as it hasn't been created yet
        
        # Setup logging
        logger = setup_logging(
            level=config.system.log_level,
            log_file=config.system.log_file,
            console=not config.system.daemon
        )
        
        # Load personality
        personality = PersonalityProfile(args.persona)
        
        # Log test mode if enabled
        if args.test_mode:
            logger.info("Test mode enabled via CLI argument")
        
        logger.info("Configuration loaded successfully")
        logger.info(f"OpenAI Model: {config.openai.model}")
        logger.info(f"OpenAI Voice: {config.openai.voice}")
        logger.info(f"HA URL: {config.home_assistant.url}")
        logger.info(f"Assistant Name: {personality.backstory.name}")
        
        print("DEBUG: About to create VoiceAssistant instance", flush=True)
        # Create and start assistant
        assistant = VoiceAssistant(config, personality)
        
        print("DEBUG: About to setup signal handlers", flush=True)
        # Setup signal handlers
        setup_signal_handlers(assistant)
        
        print("DEBUG: About to start assistant", flush=True)
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