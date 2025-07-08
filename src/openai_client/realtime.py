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
            # Audio mode: include all audio-related settings
            self.session_config["voice"] = config.voice
            self.session_config["input_audio_format"] = "pcm16"
            self.session_config["output_audio_format"] = "pcm16"
            self.session_config["turn_detection"] = {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 200
            }
            self.session_config["input_audio_transcription"] = {
                "model": "whisper-1"
            }
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
            # Track audio buffer duration (PCM16 at 24kHz, mono)
            # Each sample is 2 bytes (16-bit), so duration = bytes / 2 / 24000 * 1000 (ms)
            duration_ms = len(audio_data) / 2 / 24000 * 1000
            self.audio_buffer_duration_ms += duration_ms
            
            self.logger.info(f"AUDIO DEBUG: Received {len(audio_data)} bytes, calculated {duration_ms:.1f}ms, total buffer: {self.audio_buffer_duration_ms:.1f}ms")
            
            # Convert to base64
            audio_b64 = base64.b64encode(audio_data).decode()
            
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
            self.logger.info(f"COMMIT DEBUG: About to commit audio buffer with {self.audio_buffer_duration_ms:.1f}ms of audio")
            
            # Validate buffer has sufficient audio (minimum 100ms required by OpenAI)
            if self.audio_buffer_duration_ms < 100:
                self.logger.warning(f"Audio buffer too small: {self.audio_buffer_duration_ms:.1f}ms (minimum 100ms required)")
                return
            
            # Commit audio buffer
            await self._send_event({"type": "input_audio_buffer.commit"})
            
            # Request response
            await self._send_event({"type": "response.create"})
            
            self.logger.debug(f"Committed {self.audio_buffer_duration_ms:.1f}ms of audio")
            
            # Reset buffer duration tracking
            self.audio_buffer_duration_ms = 0
            
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
            
        elif event_type == "response.audio.delta":
            # Audio response chunk
            audio_b64 = event.data.get("delta", "")
            if audio_b64:
                audio_data = base64.b64decode(audio_b64)
                await self._emit_event("audio_response", audio_data)
                
        elif event_type == "response.audio.done":
            # Audio response complete
            await self._emit_event("audio_response_done", None)
            
        elif event_type == "response.text.delta":
            # Text response chunk
            text = event.data.get("delta", "")
            # Sanitize Unicode text for safe handling
            safe_text = sanitize_unicode_text(text)
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
                
        elif event_type == "input_audio_buffer.speech_stopped":
            # User stopped speaking
            await self._emit_event("speech_stopped", None)
            
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