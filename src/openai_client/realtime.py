"""
OpenAI Realtime API WebSocket client
"""
import json
import base64
import asyncio
import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum

# Import websockets - required dependency
import websockets
import websockets.exceptions

# Check version and legacy support
WEBSOCKETS_VERSION = getattr(websockets, '__version__', 'unknown')
WEBSOCKETS_AVAILABLE = True
try:
    import websockets.legacy.client
    LEGACY_WEBSOCKETS_AVAILABLE = True
except ImportError:
    LEGACY_WEBSOCKETS_AVAILABLE = False

from config import OpenAIConfig, AppConfig
from utils.logger import get_logger
from utils.text_utils import sanitize_unicode_text, safe_str

# Import new migration modules
from .model_compatibility import ModelCompatibility
from .voice_manager import VoiceManager
from .performance_metrics import PerformanceMetrics

# Import native MCP support
from services.ha_client.mcp_native import NativeMCPManager


class ConnectionState(Enum):
    """WebSocket connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


@dataclass
class RealtimeEvent:
    """Represents a Realtime API event"""
    type: str
    data: Dict[str, Any]
    event_id: Optional[str] = None


class OpenAIRealtimeClient:
    """
    WebSocket client for OpenAI Realtime API
    """
    
    def __init__(self, config: OpenAIConfig, personality_prompt: str = "", text_only: bool = False, app_config: AppConfig = None):
        self.config = config
        self.app_config = app_config  # Full app config for MCP access
        self.personality_prompt = personality_prompt
        self.text_only = text_only
        self.logger = None  # Will be initialized in connect()
        
        # Initialize migration modules
        self.model_compatibility = None  # Will be initialized when logger is available
        self.voice_manager = None  # Will be initialized when logger is available
        self.performance_metrics = None  # Will be initialized when logger is available
        
        # Initialize native MCP manager if app_config provided
        self.mcp_manager = None
        if app_config:
            self.mcp_manager = NativeMCPManager(app_config)
        
        # Connection state
        self.state = ConnectionState.DISCONNECTED
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.session_id: Optional[str] = None
        self.selected_model = None  # Track which model we're actually using
        
        # Session tracking
        self.session_created_time: Optional[float] = None
        self.session_start_time: Optional[float] = None
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.function_handlers: Dict[str, Callable] = {}
        
        # Function call tracking
        self.waiting_for_function_response = False  # Track if we're waiting for response after function output
        
        # Build base session configuration
        # Get language from app config if available
        language = "en"  # Default to English
        if app_config and hasattr(app_config, 'session') and hasattr(app_config.session, 'language'):
            language = app_config.session.language
            self.configured_language = language  # Store for later use
        
        self.session_config = {
            "modalities": ["text"] if text_only else ["audio", "text"],
            "tools": [],
            "temperature": config.temperature,
            "instructions": personality_prompt,
        }
        
        # Configure audio vs text-only mode
        if not text_only:
            # Audio mode: include all audio-related settings with exact OpenAI specifications
            self.session_config["voice"] = config.voice
            self.session_config["input_audio_format"] = "pcm16"
            self.session_config["output_audio_format"] = "pcm16"
            self.session_config["turn_detection"] = {
                "type": "server_vad",
                "threshold": 0.2,  # Standard sensitivity - premature triggers handled by filtering
                "prefix_padding_ms": 500,  # Increased padding to capture speech start
                "silence_duration_ms": 800  # Longer silence before stopping to allow natural speech
            }
            self.session_config["input_audio_transcription"] = {
                "model": "whisper-1",
                "language": language  # Specify language for better transcription accuracy
            }
            
            # Enhanced configuration for better audio responses
            # Only set tool_choice if we have tools (will be updated when tools are registered)
            self.session_config["max_response_output_tokens"] = "inf"  # Allow full responses
            
            # Audio configuration will be logged in connect() when logger is available
        else:
            # Text-only mode: explicitly disable VAD to prevent audio buffer operations
            self.session_config["turn_detection"] = None
        
        # Reconnection settings
        self.reconnect_delay = 5.0
        self.max_reconnect_attempts = 10
        self.reconnect_attempts = 0
        
        # Audio buffer tracking
        self.audio_buffer = []
        self.audio_buffer_duration_ms = 0  # Track duration in milliseconds
        self.audio_chunks_sent = 0  # Track number of chunks sent
        self.last_audio_send_time = 0  # Track timing for proper streaming
        
        # Response tracking
        self.response_in_progress = False  # Track if a response is currently being created
        
    async def connect(self) -> bool:
        """
        Connect to OpenAI Realtime API
        
        Returns:
            True if connected successfully, False otherwise
        """
        # Initialize logger now that logging system is configured
        if self.logger is None:
            self.logger = get_logger("OpenAIRealtimeClient")
            
        # Initialize migration modules if not already done
        if self.model_compatibility is None:
            self.model_compatibility = ModelCompatibility(self.config, self.logger)
            self.voice_manager = VoiceManager(self.config, self.logger)
            self.performance_metrics = PerformanceMetrics(self.config, self.logger)
        
        # Validate MCP connection if native mode is enabled
        if self.mcp_manager and self.mcp_manager.enabled:
            self.logger.info("Validating native MCP connection...")
            mcp_valid = await self.mcp_manager.validate_connection()
            if not mcp_valid:
                fallback_reason = self.mcp_manager.get_fallback_reason()
                self.logger.warning(f"Native MCP validation failed: {fallback_reason}")
                if self.mcp_manager.config.home_assistant.mcp.enable_fallback:
                    self.logger.info("Falling back to bridge mode")
                    self.mcp_manager.enabled = False
                else:
                    self.logger.error("MCP fallback disabled - continuing with native mode despite validation failure")
        
        # Select model and voice using compatibility layer
        self.selected_model = self.model_compatibility.select_model()
        selected_voice = self.voice_manager.select_voice(
            self.config.voice, 
            self.selected_model,
            use_case="general"
        )
        
        # Update session config with selected voice
        if not self.text_only:
            self.session_config["voice"] = selected_voice
            
        # Log audio configuration now that logger is available
        if not self.text_only:
            self.logger.info(f"Audio session config: model={self.selected_model}, voice={selected_voice}")
            self.logger.info(f"Audio formats: input={self.session_config['input_audio_format']}, output={self.session_config['output_audio_format']}")
            self.logger.info(f"Server VAD enabled: threshold={self.session_config['turn_detection']['threshold']}, silence_duration={self.session_config['turn_detection']['silence_duration_ms']}ms")
            self.logger.info("[WARNING] Server VAD mode: Do NOT manually call commit_audio() - server will auto-commit when speech stops")
            tool_choice_info = f"tool_choice={self.session_config.get('tool_choice', 'not set')}" if 'tool_choice' in self.session_config else "tool_choice will be set when tools are registered"
            self.logger.info(f"Enhanced session config: {tool_choice_info}, max_response_output_tokens=inf for better audio responses")
            
        self.logger.debug(f"connect() called, current state: {self.state}, text_only: {self.text_only}")
        
        # Check if we actually have a connected WebSocket, not just the state
        # This handles cases where disconnect() was called but state wasn't updated properly
        if self.websocket and not self._is_websocket_closed():
            if self.state == ConnectionState.CONNECTED:
                self.logger.debug("WebSocket is already connected and open")
                return True
        
        # If we're already connecting, don't start another connection
        if self.state == ConnectionState.CONNECTING:
            self.logger.debug("Already connecting, skipping duplicate connection attempt")
            return True
            
        self.state = ConnectionState.CONNECTING
        # Reset session tracking on new connection
        self.session_created_time = None
        self.logger.info("Connecting to OpenAI Realtime API...")
        self.logger.debug("Starting connection to OpenAI Realtime API...")
        
        # Log websockets version for debugging
        if WEBSOCKETS_VERSION:
            self.logger.debug(f"Using websockets version: {WEBSOCKETS_VERSION}")
        else:
            self.logger.warning("Websockets library version not detected")
        
        # Start performance tracking
        session_id = f"session_{int(time.time() * 1000)}"
        self.performance_metrics.start_session(session_id, self.selected_model)
        self.session_start_time = time.time()
        
        try:
            # WebSocket URL with selected model parameter (required by OpenAI)
            url = f"wss://api.openai.com/v1/realtime?model={self.selected_model}"
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "OpenAI-Beta": "realtime=v1"
            }
            
            # Connect to WebSocket with version-appropriate method
            connected = False
            connection_attempts = [
                # Modern websockets (10.0+)
                {"extra_headers": headers, "max_size": None, "ping_interval": 30},
                # Older websockets with different parameter names
                {"additional_headers": headers, "max_size": None, "ping_interval": 30},
                # Very old websockets - headers as list of tuples
                {"extra_headers": list(headers.items()), "max_size": None, "ping_interval": 30},
                # Legacy format
                {"origin": None, "extensions": None, "subprotocols": None, "extra_headers": headers},
                # Basic connection without ping interval
                {"extra_headers": headers, "max_size": None},
                # Minimal parameters but with headers
                {"extra_headers": headers},
                # Final fallback - no headers at all (will fail auth but helps diagnose)
                {}
            ]
            
            for i, attempt_params in enumerate(connection_attempts):
                try:
                    self.logger.debug(f"Connection attempt {i+1}: {list(attempt_params.keys())}")
                    self.websocket = await websockets.connect(url, **attempt_params)
                    self.logger.info(f"Connected using method {i+1}: {list(attempt_params.keys())}")
                    connected = True
                    break
                except (TypeError, AttributeError, ValueError) as attempt_error:
                    self.logger.debug(f"Attempt {i+1} failed with parameter error: {type(attempt_error).__name__}: {attempt_error}")
                    continue
                except Exception as attempt_error:
                    if i == len(connection_attempts) - 1:
                        # Only log as error for the final attempt
                        self.logger.error(f"Final connection attempt failed: {type(attempt_error).__name__}: {attempt_error}")
                        self.logger.error(f"Connection details - URL: {url[:50]}..., Headers keys: {list(headers.keys())}")
                    else:
                        # Log intermediate failures as debug
                        self.logger.debug(f"Attempt {i+1} failed: {type(attempt_error).__name__}: {attempt_error}")
                    continue
            
            if not connected:
                raise ConnectionError("All WebSocket connection methods failed")
            
            # NOTE: Removed WebSocket verification check that was causing immediate disconnection
            # The check was failing due to timing or state issues
            
            # Record successful connection
            connection_latency = time.time() - self.session_start_time
            self.performance_metrics.record_connection(connection_latency)
            
            self.state = ConnectionState.CONNECTED
            self.reconnect_attempts = 0
            self.logger.info(f"Connected to OpenAI Realtime API using model: {self.selected_model}")
            
            # Start event loop
            asyncio.create_task(self._event_loop())
            
            # Send session configuration
            self.logger.debug("About to send session update...")
            await self._send_session_update()
            self.logger.debug("Session update sent successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect with model {self.selected_model}: {e}")
            self.performance_metrics.record_error(str(e))
            
            # Check if we should fallback to legacy model
            if self.model_compatibility.should_fallback(e):
                fallback_model = self.model_compatibility.get_fallback_model()
                if fallback_model:
                    self.logger.info(f"Attempting fallback to model: {fallback_model}")
                    self.selected_model = fallback_model
                    self.performance_metrics.record_fallback()
                    
                    # Update voice for fallback model
                    selected_voice = self.voice_manager.migrate_voice_preference(
                        self.config.model,
                        fallback_model,
                        self.config.voice
                    )
                    if not self.text_only:
                        self.session_config["voice"] = selected_voice
                    
                    # Retry with fallback model
                    return await self.connect()
            
            self.state = ConnectionState.FAILED
            self.performance_metrics.end_session()
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from OpenAI Realtime API"""
        if self.websocket and not self._is_websocket_closed():
            await self.websocket.close()
        
        # End performance tracking session
        if self.performance_metrics:
            session_metrics = self.performance_metrics.end_session()
            if session_metrics:
                self.logger.info(
                    f"Session ended - Duration: {session_metrics.calculate_duration():.2f}s, "
                    f"Cost: ${session_metrics.estimated_cost:.4f}"
                )
        
        self.state = ConnectionState.DISCONNECTED
        self.websocket = None
        self.session_id = None
        self.session_created_time = None
        self.logger.info("Disconnected from OpenAI Realtime API")
    
    def _is_websocket_closed(self) -> bool:
        """Check if websocket is closed, handling different websocket versions"""
        if not self.websocket:
            return True
        
        try:
            # Try the standard approach first
            return self.websocket.closed
        except AttributeError:
            # For older websocket versions, check state differently
            try:
                return self.websocket.state != 1  # OPEN state is 1
            except AttributeError:
                # Last resort: assume it's open if we can't check
                return False
    
    async def send_audio(self, audio_data: bytes) -> None:
        """
        Send audio data to OpenAI
        
        Args:
            audio_data: PCM16 audio data (24kHz, mono)
        """
        if self.state != ConnectionState.CONNECTED:
            self.logger.warning("Cannot send audio: not connected")
            return
            
        if self.text_only:
            self.logger.warning("Cannot send audio in text-only mode")
            return
            
        self.logger.info(f"SEND DEBUG: send_audio called with {len(audio_data)} bytes, text_only={self.text_only}, state={self.state}")
            
        try:
            # Log incoming audio characteristics
            import struct
            import numpy as np
            sample_count = len(audio_data) // 2
            if sample_count > 0:
                samples = struct.unpack(f'<{sample_count}h', audio_data)
                audio_array = np.array(samples, dtype=np.float32)
                incoming_rms = np.sqrt(np.mean(audio_array ** 2))
                incoming_max = np.max(np.abs(audio_array))
                self.logger.debug(f"Incoming audio: {len(audio_data)} bytes, RMS={incoming_rms:.1f}, Max={incoming_max:.0f}")
            
            # Validate audio data format
            if not self._validate_audio_format(audio_data):
                self.logger.error("Audio format validation failed - skipping audio data")
                return
            
            # Improve audio quality for better VAD detection
            processed_audio = self._improve_audio_for_vad(audio_data)
            if processed_audio is None:
                self.logger.debug("Audio chunk filtered out due to low quality")
                return
            
            # Track audio buffer duration (PCM16 at 24kHz, mono)
            # Each sample is 2 bytes (16-bit), so duration = bytes / 2 / 24000 * 1000 (ms)
            duration_ms = len(processed_audio) / 2 / 24000 * 1000
            self.audio_buffer_duration_ms += duration_ms
            self.audio_chunks_sent += 1
            
            import time
            current_time = time.time()
            if self.last_audio_send_time > 0:
                time_since_last = (current_time - self.last_audio_send_time) * 1000  # Convert to ms
                self.logger.debug(f"Time since last chunk: {time_since_last:.1f}ms")
            self.last_audio_send_time = current_time
            
            self.logger.info(f"AUDIO DEBUG: Chunk {self.audio_chunks_sent}: {len(processed_audio)} bytes, {duration_ms:.1f}ms, total buffer: {self.audio_buffer_duration_ms:.1f}ms")
            
            # Convert to base64
            audio_b64 = base64.b64encode(processed_audio).decode()
            
            # Validate base64 encoding
            if not self._validate_base64_audio(audio_b64, len(processed_audio)):
                self.logger.error("Base64 audio validation failed - skipping audio data")
                return
            
            # Test base64 roundtrip integrity
            if not self._test_base64_roundtrip(processed_audio, audio_b64):
                self.logger.error("Base64 roundtrip test failed - skipping audio data")
                return
            
            # Send audio event
            event = {
                "type": "input_audio_buffer.append",
                "audio": audio_b64
            }
            
            await self._send_event(event)
            self.logger.debug(f"Sent {duration_ms:.1f}ms of audio (total: {self.audio_buffer_duration_ms:.1f}ms)")
            
        except Exception as e:
            self.logger.error(f"Error sending audio: {e}")
    
    async def commit_audio(self) -> None:
        """Commit the audio buffer and request response"""
        if self.state != ConnectionState.CONNECTED:
            return
            
        if self.text_only:
            self.logger.warning("Cannot commit audio in text-only mode")
            return
            
        try:
            self.logger.info(f"COMMIT DEBUG: About to commit audio buffer with {self.audio_buffer_duration_ms:.1f}ms of audio ({self.audio_chunks_sent} chunks)")
            
            # Validate buffer has sufficient audio (minimum 100ms required by OpenAI)
            if self.audio_buffer_duration_ms < 100:
                self.logger.warning(f"Audio buffer too small: {self.audio_buffer_duration_ms:.1f}ms (minimum 100ms required)")
                return
            
            # Validate we actually sent chunks
            if self.audio_chunks_sent == 0:
                self.logger.error("No audio chunks were sent to buffer")
                return
                
            # Add a small delay to ensure all chunks are processed
            await asyncio.sleep(0.1)
            
            # Commit audio buffer
            self.logger.info("Sending input_audio_buffer.commit event...")
            await self._send_event({"type": "input_audio_buffer.commit"})
            
            # Small delay before requesting response
            await asyncio.sleep(0.05)
            
            # Request response
            self.logger.info("Sending response.create event...")
            await self._send_event({"type": "response.create"})
            
            self.logger.info(f"Successfully committed {self.audio_buffer_duration_ms:.1f}ms of audio ({self.audio_chunks_sent} chunks)")
            
            # Reset buffer tracking
            self.audio_buffer_duration_ms = 0
            self.audio_chunks_sent = 0
            self.last_audio_send_time = 0
            
        except Exception as e:
            self.logger.error(f"Error committing audio: {e}")
    
    async def send_text(self, text: str) -> None:
        """
        Send text message to OpenAI
        
        Args:
            text: Text message to send
        """
        if self.state != ConnectionState.CONNECTED:
            self.logger.warning("Cannot send text: not connected")
            return
            
        try:
            event = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": text
                        }
                    ]
                }
            }
            
            await self._send_event(event)
            
            # For text-only mode, we need to manually request response
            # For audio mode, let server VAD handle response generation
            if self.text_only:
                await self._send_event({"type": "response.create"})
            
        except Exception as e:
            self.logger.error(f"Error sending text: {e}")
    
    def register_function(self, name: str, handler: Callable, description: str, parameters: Dict[str, Any]) -> None:
        """
        Register a function that OpenAI can call
        
        Args:
            name: Function name
            handler: Async function to handle the call
            description: Function description
            parameters: JSON schema for parameters
        """
        # Check if native MCP is enabled - if so, skip manual function registration
        if self.mcp_manager and self.mcp_manager.should_use_native():
            self.logger.debug(f"Skipping manual function registration for '{name}' - using native MCP")
            # Still store the handler for potential fallback scenarios
            self.function_handlers[name] = handler
            return
        
        # Store handler
        self.function_handlers[name] = handler
        
        # Add to tools configuration
        tool = {
            "type": "function",
            "name": name,
            "description": description,
            "parameters": parameters
        }
        
        self.session_config["tools"].append(tool)
        
        # Set tool_choice when first tool is registered
        if len(self.session_config["tools"]) == 1 and not self.text_only:
            self.session_config["tool_choice"] = "auto"
        
        # Initialize logger if needed (lazy initialization)
        if self.logger is None:
            self.logger = get_logger("OpenAIRealtimeClient")
        self.logger.info(f"Registered function: {name}")
    
    def on(self, event_type: str, handler: Callable) -> None:
        """
        Register event handler
        
        Args:
            event_type: Type of event to handle
            handler: Function to call when event occurs
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
    
    async def _event_loop(self) -> None:
        """Main event processing loop"""
        try:
            async for message in self.websocket:
                try:
                    # Sanitize Unicode characters in the message for safe printing/logging
                    safe_message = sanitize_unicode_text(message)
                    self.logger.debug(f"WEBSOCKET RECV: {safe_message[:200]}...")  # First 200 chars
                    
                    event_data = json.loads(message)
                    event = RealtimeEvent(
                        type=event_data.get("type", "unknown"),
                        data=event_data,
                        event_id=event_data.get("event_id")
                    )
                    
                    await self._handle_event(event)
                    
                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON received: {safe_str(e)}")
                except Exception as e:
                    self.logger.error(f"Error processing event: {safe_str(e)}")
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("WebSocket connection closed")
            self.state = ConnectionState.DISCONNECTED
            await self._handle_reconnection()
            
        except Exception as e:
            self.logger.error(f"Error in event loop: {e}")
            self.state = ConnectionState.FAILED
    
    async def _handle_event(self, event: RealtimeEvent) -> None:
        """Handle incoming events from OpenAI"""
        event_type = event.type
        
        # Handle specific event types
        if event_type == "session.created":
            self.session_id = event.data.get("session", {}).get("id")
            self.session_created_time = asyncio.get_event_loop().time()
            self.logger.info(f"Session created: {self.session_id}")
            print(f"*** OPENAI SESSION CREATED (ID: {self.session_id[:8]}...) ***")
            
        elif event_type == "session.updated":
            self.logger.debug("Session configuration updated")
            
        elif event_type == "response.created":
            # Response creation started
            response_id = event.data.get("response", {}).get("id", "unknown")
            self.response_in_progress = True  # Mark that a response is in progress
            
            # Log if this is a response after function output
            if self.waiting_for_function_response:
                self.logger.info(f"[RESPONSE CREATED] OpenAI creating response after function output: {response_id}")
                print(f"*** OPENAI RESPONSE AFTER FUNCTION OUTPUT: {response_id} ***")
            else:
                self.logger.info(f"[RESPONSE CREATED] OpenAI started creating response: {response_id}")
                print(f"*** OPENAI RESPONSE CREATION STARTED: {response_id} ***")
            
        elif event_type == "response.done":
            # Response creation completed
            response_data = event.data.get("response", {})
            response_id = response_data.get("id", "unknown")
            status = response_data.get("status", "unknown")
            output_items = response_data.get("output", [])
            status_details = response_data.get("status_details", {})
            
            self.response_in_progress = False  # Mark that response is complete
            
            # Clear function response flag if set
            if self.waiting_for_function_response:
                self.waiting_for_function_response = False
                self.logger.info(f"[RESPONSE DONE] Function response completed: {response_id}, status: {status}")
                print(f"*** FUNCTION RESPONSE COMPLETED: {response_id} ***")
            
            self.logger.info(f"[RESPONSE DONE] OpenAI completed response: {response_id}, status: {status}, outputs: {len(output_items)}")
            print(f"*** OPENAI RESPONSE COMPLETED: {response_id} (status: {status}, outputs: {len(output_items)}) ***")
            
            # CRITICAL: Log full error details if response failed
            if status == "failed":
                error_type = status_details.get("type", "unknown_error")
                error_code = status_details.get("code", "")
                error_message = status_details.get("message", "No error message provided")
                
                self.logger.error(f"[RESPONSE FAILED] Response {response_id} failed:")
                self.logger.error(f"  Error Type: {error_type}")
                self.logger.error(f"  Error Code: {error_code}")
                self.logger.error(f"  Error Message: {error_message}")
                self.logger.error(f"  Full status_details: {json.dumps(status_details, indent=2)}")
                
                print(f"*** RESPONSE FAILED: {error_type} ***")
                print(f"*** ERROR: {error_message} ***")
                print(f"*** FULL ERROR DETAILS: {json.dumps(status_details)} ***")
                
                # Emit error event for handling
                await self._emit_event("response_failed", {
                    "response_id": response_id,
                    "error_type": error_type,
                    "error_message": error_message,
                    "status_details": status_details
                })
            
            # Log output item types for debugging
            for i, item in enumerate(output_items):
                item_type = item.get("type", "unknown")
                item_id = item.get("id", "unknown")
                self.logger.info(f"  Output {i+1}: type={item_type}, id={item_id}")
                print(f"*** OUTPUT {i+1}: {item_type} (id: {item_id}) ***")
            
        elif event_type == "response.audio.delta":
            # Audio response chunk
            audio_b64 = event.data.get("delta", "")
            if audio_b64:
                audio_data = base64.b64decode(audio_b64)
                self.logger.info(f"[AUDIO DELTA] Received audio response chunk: {len(audio_data)} bytes")
                print(f"*** OPENAI AUDIO DELTA: {len(audio_data)} bytes ***")
                await self._emit_event("audio_response", audio_data)
            else:
                self.logger.warning("Received empty audio delta")
                print("*** EMPTY AUDIO DELTA RECEIVED ***")
                
        elif event_type == "response.audio.done":
            # Audio response complete
            self.logger.info("[AUDIO DONE] Audio response completed")
            print("*** OPENAI AUDIO RESPONSE COMPLETE ***")
            await self._emit_event("audio_response_done", None)
            
        elif event_type == "response.text.delta":
            # Text response chunk
            text = event.data.get("delta", "")
            # Sanitize Unicode text for safe handling
            safe_text = sanitize_unicode_text(text)
            self.logger.debug(f"[TEXT] Received text response chunk: '{safe_text}'")
            await self._emit_event("text_response", safe_text)
            
        elif event_type == "response.function_call_arguments.done":
            # Function call complete
            call_data = event.data
            function_name = call_data.get("name")
            arguments_str = call_data.get("arguments", "{}")
            call_id = call_data.get("call_id")
            
            self.logger.info(f"[FUNCTION CALL] Received function call: {function_name} (call_id: {call_id})")
            print(f"*** FUNCTION CALL: {function_name} ***")
            
            try:
                arguments = json.loads(arguments_str)
                await self._handle_function_call(function_name, arguments, call_id)
            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid function arguments: {e}")
                
        elif event_type == "input_audio_buffer.speech_started":
            # User started speaking - server VAD detected speech start
            audio_start_ms = event.data.get("audio_start_ms", 0)
            self.logger.info(f"[SPEECH STARTED] Server VAD detected speech started at {audio_start_ms}ms")
            print(f"*** SERVER VAD: SPEECH STARTED (at {audio_start_ms}ms) ***")
            await self._emit_event("speech_started", event.data)
            
        elif event_type == "input_audio_buffer.speech_stopped":
            # User stopped speaking - server VAD detected end of speech
            audio_end_ms = event.data.get("audio_end_ms", 0)
            self.logger.info(f"[SPEECH STOPPED] Server VAD detected speech ended at {audio_end_ms}ms - audio buffer automatically committed")
            print(f"*** SERVER VAD: SPEECH STOPPED (at {audio_end_ms}ms) - BUFFER COMMITTED ***")
            await self._emit_event("speech_stopped", event.data)
            
        elif event_type == "conversation.item.input_audio_transcription.completed":
            # User's audio input has been transcribed
            transcript = event.data.get("transcript", "")
            item_id = event.data.get("item_id", "unknown")
            self.logger.info(f"[TRANSCRIPTION] User said: '{transcript}' (item_id: {item_id})")
            print(f"*** USER TRANSCRIPTION: '{transcript}' ***")
            
            # Emit transcription event for main.py to handle
            await self._emit_event("input_audio_transcription", {
                "transcript": transcript,
                "item_id": item_id,
                "language": event.data.get("language", "unknown")  # Include language if provided
            })
            
        elif event_type == "error":
            # Error from OpenAI
            error_data = event.data.get("error", {})
            # If error data is nested, extract it
            if not error_data and "error" not in event.data:
                # Error might be at the top level
                error_data = event.data
            
            # Create comprehensive error info
            error_info = {
                "type": error_data.get("type", error_data.get("error_type", "unknown")),
                "message": error_data.get("message", error_data.get("error_message", error_data.get("msg", str(error_data)))),
                "code": error_data.get("code", error_data.get("error_code", "")),
                "raw_data": error_data  # Include raw data for debugging
            }
            
            safe_error_info = {k: sanitize_unicode_text(str(v)) for k, v in error_info.items()}
            self.logger.debug(f"Received error event: {safe_error_info}")
            self.logger.error(f"OpenAI error - Type: {error_info['type']}, Message: {error_info['message']}, Code: {error_info['code']}")
            # Pass just the error data, not the wrapper
            await self._emit_event("error", error_data)
        
        # Handle MCP-specific events if native MCP is enabled
        elif event_type.startswith("mcp.") and self.mcp_manager:
            response = await self.mcp_manager.handle_mcp_event(event.data)
            if response:
                # Send response back to OpenAI if needed (e.g., approval response)
                await self._send_event(response)
            self.logger.debug(f"Handled MCP event: {event_type}")
        
        # Emit generic event
        await self._emit_event(event_type, event.data)
    
    async def _handle_function_call(self, function_name: str, arguments: Dict[str, Any], call_id: str) -> None:
        """Handle function call from OpenAI"""
        if function_name not in self.function_handlers:
            self.logger.error(f"Unknown function called: {function_name}")
            return
            
        try:
            # Call the function handler
            handler = self.function_handlers[function_name]
            result = await handler(arguments)
            
            # Send result back to OpenAI
            await self._send_function_result(call_id, result)
            
        except Exception as e:
            self.logger.error(f"Error calling function {function_name}: {e}")
            await self._send_function_error(call_id, str(e))
    
    async def _send_function_result(self, call_id: str, result: Any) -> None:
        """Send function call result back to OpenAI"""
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result) if not isinstance(result, str) else result
            }
        }
        
        await self._send_event(event)
        
        # Request OpenAI to generate a response with the function result
        # In server VAD mode, we need to explicitly request a response after function output
        self.logger.info("Function result sent - requesting response from OpenAI")
        self.waiting_for_function_response = True  # Mark that we're waiting for response
        await asyncio.sleep(0.1)  # Small delay to ensure function output is processed
        await self._send_event({"type": "response.create"})
        self.logger.info("Response.create sent after function output - waiting for audio response")
    
    async def _send_function_error(self, call_id: str, error: str) -> None:
        """Send function call error back to OpenAI"""
        event = {
            "type": "conversation.item.create", 
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps({"error": error})
            }
        }
        
        await self._send_event(event)
        
        # Request OpenAI to generate a response with the error
        # In server VAD mode, we need to explicitly request a response after function output
        self.logger.info("Function error sent - requesting response from OpenAI")
        self.waiting_for_function_response = True  # Mark that we're waiting for response
        await asyncio.sleep(0.1)  # Small delay to ensure function output is processed
        await self._send_event({"type": "response.create"})
        self.logger.info("Response.create sent after function error - waiting for audio response")
    
    async def _emit_event(self, event_type: str, data: Any) -> None:
        """Emit event to registered handlers"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    self.logger.error(f"Error in event handler {handler}: {e}")
    
    async def _send_event(self, event: Dict[str, Any]) -> None:
        """Send event to OpenAI"""
        if not self.websocket or self._is_websocket_closed():
            raise ConnectionError("WebSocket not connected")
            
        if event.get("type") in ["input_audio_buffer.append", "input_audio_buffer.commit"]:
            self.logger.info(f"SEND EVENT DEBUG: Sending {event['type']} event")
            
        message = json.dumps(event)
        # Sanitize Unicode for safe printing/logging
        safe_message = sanitize_unicode_text(message)
        self.logger.debug(f"WEBSOCKET SEND: {safe_message[:200]}...")  # First 200 chars
        await self.websocket.send(message)
    
    async def _send_session_update(self) -> None:
        """Send session configuration to OpenAI"""
        # Check if we should use native MCP tools
        if self.mcp_manager and self.mcp_manager.enabled:
            # Add native MCP tools to session config if not already present
            mcp_tools = self.mcp_manager.get_tool_config()
            if mcp_tools:
                # Replace traditional function tools with MCP tools
                # Keep any non-function tools that might exist
                non_function_tools = [t for t in self.session_config.get("tools", []) 
                                     if t.get("type") != "function"]
                self.session_config["tools"] = non_function_tools + mcp_tools
                self.logger.info(f"Using native MCP tools: {len(mcp_tools)} MCP server(s) configured")
        
        event = {
            "type": "session.update",
            "session": self.session_config
        }
        
        # Sanitize session config for safe logging
        safe_config = sanitize_unicode_text(json.dumps(self.session_config, indent=2))
        self.logger.info(f"SESSION CONFIG DEBUG: Sending session config: {safe_config}")
        await self._send_event(event)
        self.logger.debug("Session configuration sent")
    
    async def update_vad_settings(self, threshold: float = None, silence_duration_ms: int = None) -> None:
        """Update VAD settings dynamically"""
        if self.state != ConnectionState.CONNECTED:
            self.logger.warning("Cannot update VAD settings - not connected")
            print("*** VAD UPDATE FAILED: NOT CONNECTED ***")
            return
        
        # Update turn detection settings
        current_config = self.session_config.get("turn_detection", {})
        old_threshold = current_config.get("threshold", "unknown")
        old_silence = current_config.get("silence_duration_ms", "unknown")
        
        if threshold is not None:
            current_config["threshold"] = threshold
        
        if silence_duration_ms is not None:
            current_config["silence_duration_ms"] = silence_duration_ms
        
        # Send updated session config
        self.session_config["turn_detection"] = current_config
        
        event = {
            "type": "session.update",
            "session": {
                "turn_detection": current_config
            }
        }
        
        # Log the update attempt with detailed info
        self.logger.info(f"VAD UPDATE: threshold {old_threshold} -> {current_config.get('threshold')}, silence_duration {old_silence}ms -> {current_config.get('silence_duration_ms')}ms")
        print(f"*** VAD UPDATE: threshold {old_threshold} -> {current_config.get('threshold')}, silence_duration {old_silence}ms -> {current_config.get('silence_duration_ms')}ms ***")
        
        try:
            await self._send_event(event)
            self.logger.info(f"[SUCCESS] VAD settings updated successfully: threshold={current_config.get('threshold')}, silence_duration={current_config.get('silence_duration_ms')}ms")
            print(f"*** [SUCCESS] VAD SETTINGS UPDATED: threshold={current_config.get('threshold')}, silence_duration={current_config.get('silence_duration_ms')}ms ***")
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to update VAD settings: {e}")
            print(f"*** [ERROR] VAD UPDATE FAILED: {e} ***")
    
    async def _handle_reconnection(self) -> None:
        """Handle automatic reconnection"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            self.logger.error("Max reconnection attempts reached")
            self.state = ConnectionState.FAILED
            return
            
        self.reconnect_attempts += 1
        self.state = ConnectionState.RECONNECTING
        
        self.logger.info(f"Attempting reconnection ({self.reconnect_attempts}/{self.max_reconnect_attempts})")
        
        await asyncio.sleep(self.reconnect_delay)
        
        success = await self.connect()
        if not success:
            await self._handle_reconnection()
    
    def update_personality(self, personality_prompt: str) -> None:
        """Update personality prompt and session"""
        self.personality_prompt = personality_prompt
        self.session_config["instructions"] = personality_prompt
        
        # Send updated session if connected
        if self.state == ConnectionState.CONNECTED:
            asyncio.create_task(self._send_session_update())
    
    
    def _validate_audio_format(self, audio_data: bytes) -> bool:
        """
        Validate audio data format for OpenAI requirements
        
        Args:
            audio_data: Raw audio bytes to validate
            
        Returns:
            True if audio format is valid, False otherwise
        """
        try:
            # Check if we have audio data
            if not audio_data or len(audio_data) == 0:
                self.logger.error("Audio validation failed: No audio data")
                return False
            
            # Check if audio data length is even (each sample is 2 bytes for PCM16)
            if len(audio_data) % 2 != 0:
                self.logger.error(f"Audio validation failed: Odd number of bytes ({len(audio_data)}), PCM16 requires even number")
                return False
            
            # Check minimum audio length (at least 10ms worth of data)
            min_samples = int(24000 * 0.01)  # 10ms at 24kHz
            min_bytes = min_samples * 2  # 2 bytes per sample
            if len(audio_data) < min_bytes:
                self.logger.error(f"Audio validation failed: Too short ({len(audio_data)} bytes), minimum {min_bytes} bytes required")
                return False
            
            # Check for audio content (not all zeros)
            import struct
            sample_count = len(audio_data) // 2
            samples = struct.unpack(f'<{sample_count}h', audio_data)  # Little-endian signed 16-bit
            
            # Check for actual audio content
            max_amplitude = max(abs(sample) for sample in samples)
            if max_amplitude == 0:
                self.logger.error("Audio validation failed: All samples are zero (silence)")
                return False
            
            if max_amplitude < 100:  # Very quiet
                self.logger.warning(f"Audio validation warning: Very low amplitude ({max_amplitude})")
            
            # Log audio characteristics
            avg_amplitude = sum(abs(sample) for sample in samples) / len(samples)
            self.logger.debug(f"Audio validation: {len(audio_data)} bytes, {sample_count} samples, max_amp={max_amplitude}, avg_amp={avg_amplitude:.1f}")
            
            # Add raw audio inspection - log first few bytes and samples
            self._log_raw_audio_inspection(audio_data, samples)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Audio validation error: {e}")
            return False
    
    def _validate_base64_audio(self, audio_b64: str, expected_bytes: int) -> bool:
        """
        Validate base64 encoded audio data
        
        Args:
            audio_b64: Base64 encoded audio string
            expected_bytes: Expected number of original bytes
            
        Returns:
            True if base64 encoding is valid, False otherwise
        """
        try:
            # Check if base64 string is valid
            if not audio_b64:
                self.logger.error("Base64 validation failed: Empty string")
                return False
            
            # Test decode
            decoded = base64.b64decode(audio_b64)
            if len(decoded) != expected_bytes:
                self.logger.error(f"Base64 validation failed: Decoded length {len(decoded)} != expected {expected_bytes}")
                return False
            
            # Check base64 length is reasonable
            expected_b64_len = ((expected_bytes + 2) // 3) * 4  # Base64 encoding formula
            if abs(len(audio_b64) - expected_b64_len) > 10:  # Allow some tolerance
                self.logger.warning(f"Base64 validation warning: Length {len(audio_b64)} != expected ~{expected_b64_len}")
            
            self.logger.debug(f"Base64 validation: {len(audio_b64)} chars -> {len(decoded)} bytes")
            return True
            
        except Exception as e:
            self.logger.error(f"Base64 validation error: {e}")
            return False
    
    def _log_raw_audio_inspection(self, audio_data: bytes, samples: tuple) -> None:
        """
        Log detailed raw audio inspection for debugging
        
        Args:
            audio_data: Raw audio bytes
            samples: Unpacked audio samples
        """
        try:
            # Log first 20 bytes in hex format
            hex_bytes = ' '.join(f'{b:02x}' for b in audio_data[:20])
            self.logger.debug(f"Raw audio bytes (first 20): {hex_bytes}")
            
            # Log first 10 samples as signed 16-bit integers
            first_samples = samples[:10]
            self.logger.debug(f"First 10 samples: {list(first_samples)}")
            
            # Check byte order by examining sample values
            import struct
            
            # Test little-endian interpretation (what we should have)
            le_samples = struct.unpack('<10h', audio_data[:20])
            self.logger.debug(f"Little-endian interpretation: {list(le_samples)}")
            
            # Test big-endian interpretation (to compare)
            be_samples = struct.unpack('>10h', audio_data[:20])
            self.logger.debug(f"Big-endian interpretation: {list(be_samples)}")
            
            # Check if samples are reasonable for PCM16
            reasonable_samples = sum(1 for s in first_samples if -32768 <= s <= 32767)
            self.logger.debug(f"Samples in valid PCM16 range: {reasonable_samples}/10")
            
            # Check for obvious patterns that might indicate format issues
            if all(s == 0 for s in first_samples):
                self.logger.warning("First 10 samples are all zero")
            elif all(abs(s) < 10 for s in first_samples):
                # Only warn for extremely quiet audio (was 100, now 10)
                self.logger.debug("First 10 samples are very quiet")
            elif any(abs(s) > 30000 for s in first_samples):
                self.logger.warning("Some samples are very loud (possible clipping)")
            
            # Check for alternating pattern that might indicate channel issues
            if len(first_samples) >= 4:
                even_samples = first_samples[::2]
                odd_samples = first_samples[1::2]
                if all(s == 0 for s in even_samples) or all(s == 0 for s in odd_samples):
                    self.logger.warning("Alternating samples are zero (possible stereo/mono confusion)")
            
        except Exception as e:
            self.logger.error(f"Error in raw audio inspection: {e}")
    
    def _create_minimal_test_audio(self) -> bytes:
        """
        Create minimal test audio with exact OpenAI PCM16 specification
        
        Returns:
            PCM16 audio data as bytes (24kHz, mono, little-endian)
        """
        try:
            import struct
            import math
            
            # Create exactly 500ms of audio (well above 100ms minimum)
            sample_rate = 24000
            duration_seconds = 0.5
            frequency = 440.0  # A4 note
            
            # Generate samples
            samples = []
            for i in range(int(sample_rate * duration_seconds)):
                t = i / sample_rate
                # Generate pure sine wave
                sample_value = math.sin(2 * math.pi * frequency * t) * 0.5  # 50% amplitude
                # Convert to 16-bit signed integer
                sample_int = int(sample_value * 32767)
                # Clamp to valid range
                sample_int = max(-32768, min(32767, sample_int))
                samples.append(sample_int)
            
            # Pack as little-endian 16-bit signed integers
            audio_data = struct.pack('<' + 'h' * len(samples), *samples)
            
            self.logger.info(f"Created minimal test audio: {len(audio_data)} bytes, {len(samples)} samples, {duration_seconds}s at {sample_rate}Hz")
            
            return audio_data
            
        except Exception as e:
            self.logger.error(f"Error creating minimal test audio: {e}")
            return b''
    
    def _test_base64_roundtrip(self, original_audio: bytes, audio_b64: str) -> bool:
        """
        Test base64 encoding/decoding roundtrip integrity
        
        Args:
            original_audio: Original audio bytes
            audio_b64: Base64 encoded audio string
            
        Returns:
            True if roundtrip is successful, False otherwise
        """
        try:
            # Decode the base64 back to bytes
            decoded_audio = base64.b64decode(audio_b64)
            
            # Compare with original
            if decoded_audio != original_audio:
                self.logger.error(f"Base64 roundtrip failed: original={len(original_audio)} bytes, decoded={len(decoded_audio)} bytes")
                
                # Log first few bytes for comparison
                if len(original_audio) >= 10 and len(decoded_audio) >= 10:
                    orig_hex = ' '.join(f'{b:02x}' for b in original_audio[:10])
                    dec_hex = ' '.join(f'{b:02x}' for b in decoded_audio[:10])
                    self.logger.error(f"Original first 10 bytes: {orig_hex}")
                    self.logger.error(f"Decoded first 10 bytes:  {dec_hex}")
                
                return False
            
            # Test base64 string validity
            if not self._validate_base64_string(audio_b64):
                self.logger.error("Base64 string validation failed")
                return False
            
            self.logger.debug(f"Base64 roundtrip successful: {len(original_audio)} bytes <-> {len(audio_b64)} chars")
            return True
            
        except Exception as e:
            self.logger.error(f"Base64 roundtrip test error: {e}")
            return False
    
    def _validate_base64_string(self, b64_string: str) -> bool:
        """
        Validate base64 string format
        
        Args:
            b64_string: Base64 string to validate
            
        Returns:
            True if valid base64 string, False otherwise
        """
        try:
            # Check if string contains only valid base64 characters
            import string
            valid_chars = string.ascii_letters + string.digits + '+/='
            
            if not all(c in valid_chars for c in b64_string):
                self.logger.error("Base64 string contains invalid characters")
                return False
            
            # Check if length is multiple of 4 (base64 requirement)
            if len(b64_string) % 4 != 0:
                self.logger.error(f"Base64 string length ({len(b64_string)}) is not multiple of 4")
                return False
            
            # Check padding
            padding_count = b64_string.count('=')
            if padding_count > 2:
                self.logger.error(f"Base64 string has too much padding ({padding_count})")
                return False
            
            # Padding should only be at the end
            if '=' in b64_string[:-2]:
                self.logger.error("Base64 string has padding in wrong position")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Base64 string validation error: {e}")
            return False
    
    def _improve_audio_for_vad(self, audio_data: bytes) -> Optional[bytes]:
        """
        Improve audio quality for better VAD detection
        
        Args:
            audio_data: Raw PCM16 audio data
            
        Returns:
            Processed audio data or None if should be filtered out
        """
        try:
            import struct
            import numpy as np
            
            # Convert to numpy array for processing
            sample_count = len(audio_data) // 2
            if sample_count == 0:
                return None
                
            samples = struct.unpack(f'<{sample_count}h', audio_data)
            audio_array = np.array(samples, dtype=np.float32)
            
            # Check if audio is too quiet (likely silence or noise)
            max_amplitude = np.max(np.abs(audio_array))
            if max_amplitude < 10:  # Very low threshold - only filter complete silence
                self.logger.debug(f"Filtering out silent audio (max_amplitude: {max_amplitude})")
                return None
            
            # Calculate RMS in normalized float range for consistent processing
            audio_normalized = audio_array / 32768.0  # Normalize to [-1, 1]
            rms_normalized = np.sqrt(np.mean(audio_normalized ** 2))
            
            # Log current audio levels for debugging
            self.logger.debug(f"Audio levels - RMS (normalized): {rms_normalized:.6f}, RMS (int16): {np.sqrt(np.mean(audio_array ** 2)):.1f}, Max: {max_amplitude}")
            
            # BALANCED NORMALIZATION for OpenAI VAD detection
            # Target RMS of 0.05 in normalized range (about 1638 in int16 range)
            # Reduced from 0.1 to prevent over-amplification
            target_rms_normalized = 0.05
            
            if rms_normalized > 0:
                # Calculate required gain
                gain = target_rms_normalized / rms_normalized
                
                # Conservative gain limits to prevent distortion
                # Reduced from 10x to 3x max gain
                gain = np.clip(gain, 0.5, 3.0)
                
                # Apply gain to normalized audio
                audio_normalized *= gain
                
                # Soft limiting using tanh for smoother clipping
                # This prevents harsh distortion while still limiting peaks
                audio_normalized = np.tanh(audio_normalized * 0.9) / 0.9
                
                # Calculate actual RMS after processing
                final_rms_normalized = np.sqrt(np.mean(audio_normalized ** 2))
                
                # Convert back to int16 range
                audio_array = (audio_normalized * 32767).astype(np.float32)
                audio_array = np.clip(audio_array, -32767.0, 32767.0)
                
                # Log the transformation
                self.logger.info(f"Audio gain applied: {gain:.1f}x, RMS: {rms_normalized:.6f} -> {final_rms_normalized:.6f}")
            else:
                self.logger.warning("Audio RMS is zero, cannot normalize")
                return None
            
            # Add a high-pass filter to remove DC offset and very low frequencies
            # This can help with VAD detection
            if len(audio_array) > 100:
                # Simple DC offset removal
                audio_array -= np.mean(audio_array)
            
            # Convert back to bytes
            processed_samples = audio_array.astype(np.int16)
            processed_audio = struct.pack(f'<{len(processed_samples)}h', *processed_samples)
            
            # Final validation - ensure we have actual audio content
            final_rms = np.sqrt(np.mean(processed_samples.astype(np.float32) ** 2))
            if final_rms < 100:
                self.logger.warning(f"Processed audio still too quiet: RMS={final_rms:.1f}")
            
            return processed_audio
            
        except Exception as e:
            self.logger.error(f"Error improving audio for VAD: {e}")
            return audio_data  # Return original if processing fails