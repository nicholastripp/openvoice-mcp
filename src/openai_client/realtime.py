"""
OpenAI Realtime API WebSocket client
"""
import json
import base64
import asyncio
import websockets
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum

# Check websockets version and capabilities
WEBSOCKETS_VERSION = None
LEGACY_WEBSOCKETS_AVAILABLE = False
try:
    import websockets
    WEBSOCKETS_VERSION = websockets.__version__
    try:
        import websockets.legacy.client
        LEGACY_WEBSOCKETS_AVAILABLE = True
    except ImportError:
        pass
except ImportError:
    pass

from config import OpenAIConfig
from utils.logger import get_logger
from utils.text_utils import sanitize_unicode_text, safe_str


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
    
    def __init__(self, config: OpenAIConfig, personality_prompt: str = "", text_only: bool = False):
        self.config = config
        self.personality_prompt = personality_prompt
        self.text_only = text_only
        self.logger = get_logger("OpenAIRealtimeClient")
        
        # Connection state
        self.state = ConnectionState.DISCONNECTED
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.session_id: Optional[str] = None
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.function_handlers: Dict[str, Callable] = {}
        
        # Build base session configuration
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
                "threshold": 0.6,  # Higher threshold for more reliable detection
                "prefix_padding_ms": 300,
                "silence_duration_ms": 300  # Shorter silence duration for faster response
            }
            self.session_config["input_audio_transcription"] = {
                "model": "whisper-1"
            }
            
            # Enhanced configuration for better audio responses
            self.session_config["tool_choice"] = "auto"  # Allow OpenAI to choose when to use tools
            self.session_config["max_response_output_tokens"] = "inf"  # Allow full responses
            
            # Log audio configuration for debugging
            self.logger.info(f"Audio session config: input_format={self.session_config['input_audio_format']}, output_format={self.session_config['output_audio_format']}")
            self.logger.info(f"Server VAD enabled: threshold={self.session_config['turn_detection']['threshold']}, silence_duration={self.session_config['turn_detection']['silence_duration_ms']}ms")
            self.logger.info("⚠️  Server VAD mode: Do NOT manually call commit_audio() - server will auto-commit when speech stops")
            self.logger.info("Enhanced session config: tool_choice=auto, max_response_output_tokens=inf for better audio responses")
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
        
    async def connect(self) -> bool:
        """
        Connect to OpenAI Realtime API
        
        Returns:
            True if connected successfully, False otherwise
        """
        print(f"DEBUG: connect() called, current state: {self.state}, text_only: {self.text_only}")
        if self.state in [ConnectionState.CONNECTED, ConnectionState.CONNECTING]:
            return True
            
        self.state = ConnectionState.CONNECTING
        self.logger.info("Connecting to OpenAI Realtime API...")
        print("DEBUG: Starting connection to OpenAI Realtime API...")
        
        # Log websockets version for debugging
        if WEBSOCKETS_VERSION:
            self.logger.debug(f"Using websockets version: {WEBSOCKETS_VERSION}")
        else:
            self.logger.warning("Websockets library not properly detected")
        
        try:
            # WebSocket URL with model parameter (required by OpenAI)
            url = f"wss://api.openai.com/v1/realtime?model={self.config.model}"
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
                # Last resort - minimal parameters
                {"max_size": None}
            ]
            
            for i, attempt_params in enumerate(connection_attempts):
                try:
                    self.logger.debug(f"Connection attempt {i+1}: {list(attempt_params.keys())}")
                    self.websocket = await websockets.connect(url, **attempt_params)
                    self.logger.info(f"Connected using method {i+1}: {list(attempt_params.keys())}")
                    connected = True
                    break
                except (TypeError, AttributeError, ValueError) as attempt_error:
                    self.logger.debug(f"Attempt {i+1} failed: {attempt_error}")
                    continue
                except Exception as attempt_error:
                    self.logger.debug(f"Attempt {i+1} failed with unexpected error: {attempt_error}")
                    continue
            
            if not connected:
                raise ConnectionError("All WebSocket connection methods failed")
            
            self.state = ConnectionState.CONNECTED
            self.reconnect_attempts = 0
            self.logger.info("Connected to OpenAI Realtime API")
            
            # Start event loop
            asyncio.create_task(self._event_loop())
            
            # Send session configuration
            print("DEBUG: About to send session update...")
            await self._send_session_update()
            print("DEBUG: Session update sent successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            self.state = ConnectionState.FAILED
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from OpenAI Realtime API"""
        if self.websocket and not self._is_websocket_closed():
            await self.websocket.close()
        
        self.state = ConnectionState.DISCONNECTED
        self.websocket = None
        self.session_id = None
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
                    print(f"DEBUG WEBSOCKET RECV: {safe_message[:200]}...")  # First 200 chars
                    self.logger.info(f"WEBSOCKET RECV: {safe_message}")
                    
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
            self.logger.info(f"Session created: {self.session_id}")
            
        elif event_type == "session.updated":
            self.logger.debug("Session configuration updated")
            
        elif event_type == "response.created":
            # Response creation started
            response_id = event.data.get("response", {}).get("id", "unknown")
            self.logger.info(f"[RESPONSE CREATED] OpenAI started creating response: {response_id}")
            print(f"*** OPENAI RESPONSE CREATION STARTED: {response_id} ***")
            
        elif event_type == "response.done":
            # Response creation completed
            response_data = event.data.get("response", {})
            response_id = response_data.get("id", "unknown")
            status = response_data.get("status", "unknown")
            output_items = response_data.get("output", [])
            self.logger.info(f"[RESPONSE DONE] OpenAI completed response: {response_id}, status: {status}, outputs: {len(output_items)}")
            print(f"*** OPENAI RESPONSE COMPLETED: {response_id} (status: {status}, outputs: {len(output_items)}) ***")
            
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
            self.logger.debug(f"💬 Received text response chunk: '{safe_text}'")
            await self._emit_event("text_response", safe_text)
            
        elif event_type == "response.function_call_arguments.done":
            # Function call complete
            call_data = event.data
            function_name = call_data.get("name")
            arguments_str = call_data.get("arguments", "{}")
            call_id = call_data.get("call_id")
            
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
            
        elif event_type == "error":
            # Error from OpenAI
            error_data = event.data.get("error", {})
            safe_error_data = {k: sanitize_unicode_text(str(v)) for k, v in error_data.items()}
            print(f"DEBUG: Received error event: {safe_error_data}")
            self.logger.error(f"OpenAI error: {safe_error_data}")
            await self._emit_event("error", safe_error_data)
        
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
        
        # Request audio response after function call completion
        # This ensures OpenAI generates audio feedback for the user
        self.logger.info("Requesting audio response after function call completion")
        response_event = {
            "type": "response.create"
        }
        await self._send_event(response_event)
    
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
        
        # Request audio response after function call error
        # This ensures OpenAI generates audio feedback even for errors
        self.logger.info("Requesting audio response after function call error")
        response_event = {
            "type": "response.create"
        }
        await self._send_event(response_event)
    
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
        print(f"DEBUG WEBSOCKET SEND: {safe_message[:200]}...")  # First 200 chars
        self.logger.info(f"WEBSOCKET SEND: {safe_message}")
        await self.websocket.send(message)
    
    async def _send_session_update(self) -> None:
        """Send session configuration to OpenAI"""
        event = {
            "type": "session.update",
            "session": self.session_config
        }
        
        # Sanitize session config for safe logging
        safe_config = sanitize_unicode_text(json.dumps(self.session_config, indent=2))
        self.logger.info(f"SESSION CONFIG DEBUG: Sending session config: {safe_config}")
        await self._send_event(event)
        self.logger.debug("Session configuration sent")
    
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
            elif all(abs(s) < 100 for s in first_samples):
                self.logger.warning("First 10 samples are very quiet")
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
            
            self.logger.debug(f"Base64 roundtrip successful: {len(original_audio)} bytes ↔ {len(audio_b64)} chars")
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
            if max_amplitude < 50:  # Relaxed threshold - only filter truly silent audio
                self.logger.debug(f"Filtering out very quiet audio (max_amplitude: {max_amplitude})")
                return None
            
            # Gentle audio normalization for better VAD performance
            # Target RMS level around 2000 (more conservative)
            rms = np.sqrt(np.mean(audio_array ** 2))
            if rms > 0 and rms < 500:  # Only normalize very quiet audio
                target_rms = 2000.0
                gain = target_rms / rms
                
                # Limit gain to prevent excessive amplification
                gain = min(gain, 3.0)  # Max 3x amplification (reduced)
                
                # Apply gain
                audio_array *= gain
                
                # Clip to prevent overflow
                audio_array = np.clip(audio_array, -32767.0, 32767.0)
                
                self.logger.debug(f"Audio normalized: original_rms={rms:.1f}, gain={gain:.2f}, new_rms={np.sqrt(np.mean(audio_array ** 2)):.1f}")
            
            # Convert back to bytes
            processed_samples = audio_array.astype(np.int16)
            processed_audio = struct.pack(f'<{len(processed_samples)}h', *processed_samples)
            
            return processed_audio
            
        except Exception as e:
            self.logger.error(f"Error improving audio for VAD: {e}")
            return audio_data  # Return original if processing fails