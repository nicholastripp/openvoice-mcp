"""Model Context Protocol (MCP) client for Home Assistant integration."""

import asyncio
import json
import logging
import ssl
import time
from typing import Any, Dict, List, Optional, Callable, AsyncGenerator, Tuple
from urllib.parse import urljoin
import aiohttp
import uuid

logger = logging.getLogger(__name__)


class MCPError(Exception):
    """Base exception for MCP-related errors."""
    pass


class MCPConnectionError(MCPError):
    """Raised when connection to MCP server fails."""
    pass


class MCPProtocolError(MCPError):
    """Raised when protocol-level errors occur."""
    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(f"MCP Error {code}: {message}")
        self.code = code
        self.message = message
        self.data = data


class MCPClient:
    """Client for Home Assistant's Model Context Protocol server."""
    
    def __init__(
        self,
        base_url: str,
        access_token: str,
        sse_endpoint: str = "/mcp_server/sse",
        connection_timeout: int = 30,
        reconnect_attempts: int = 3,
        reconnect_delay: float = 1.0,
        ssl_verify: bool = True,
        ssl_ca_bundle: Optional[str] = None
    ):
        """Initialize MCP client.
        
        Args:
            base_url: Home Assistant base URL
            access_token: Long-lived access token
            sse_endpoint: SSE endpoint path (default: /mcp_server/sse)
            connection_timeout: Connection timeout in seconds
            reconnect_attempts: Number of reconnection attempts
            reconnect_delay: Initial delay between reconnection attempts
            ssl_verify: Whether to verify SSL certificates (default: True)
            ssl_ca_bundle: Path to custom CA bundle file (optional)
        """
        self.base_url = base_url.rstrip('/')
        self.access_token = access_token
        self.sse_endpoint = sse_endpoint
        self.connection_timeout = connection_timeout
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.ssl_verify = ssl_verify
        self.ssl_ca_bundle = ssl_ca_bundle
        
        self.sse_url = urljoin(self.base_url, sse_endpoint.lstrip('/'))
        self._session: Optional[aiohttp.ClientSession] = None
        self._ssl_context: Optional[ssl.SSLContext] = None
        self._sse_response = None  # The SSE response object
        self._response_futures: Dict[str, asyncio.Future] = {}
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._tools: List[Dict[str, Any]] = []
        self._capabilities: Dict[str, Any] = {}
        self._connected = False
        self._initialized = False  # Track if protocol is initialized
        self._reconnect_task = None
        self._message_task = None
        self._message_endpoint: Optional[str] = None  # Session-specific message endpoint
        
    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context with custom settings."""
        if self.ssl_verify:
            context = ssl.create_default_context()
            if self.ssl_ca_bundle:
                context.load_verify_locations(self.ssl_ca_bundle)
            logger.debug("SSL verification enabled")
        else:
            # Create unverified context for testing
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            logger.warning("SSL verification DISABLED - this is insecure!")
        
        # Log SSL details
        logger.debug(f"SSL Protocol: {context.protocol}")
        return context
    
    async def connect(self) -> None:
        """Establish connection to MCP server."""
        if self._connected:
            logger.debug("Already connected to MCP server")
            return
            
        logger.info(f"Connecting to MCP server at {self.sse_url}")
        logger.info(f"SSL verification: {'enabled' if self.ssl_verify else 'DISABLED'}")
        
        try:
            await self._establish_connection()
            # Set connected to True after SSE connection is established
            self._connected = True
            
            # Only initialize protocol if not already initialized
            if not self._initialized:
                await self._initialize_protocol()
                self._initialized = True
            else:
                logger.debug("Skipping protocol initialization - already initialized")
                
            logger.info("Successfully connected to MCP server")
        except Exception as e:
            # Reset connected state on failure
            self._connected = False
            logger.error(f"Failed to connect to MCP server: {e}")
            raise MCPConnectionError(f"Connection failed: {str(e)}")
    
    async def _establish_connection(self) -> None:
        """Establish SSE connection with retry logic."""
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
        
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=self.connection_timeout)
            
            # Create SSL context
            if self.base_url.startswith('https'):
                self._ssl_context = self._create_ssl_context()
                connector = aiohttp.TCPConnector(ssl=self._ssl_context)
            else:
                connector = aiohttp.TCPConnector()
                
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={'Connection': 'keep-alive'}
            )
        
        last_error = None
        for attempt in range(self.reconnect_attempts):
            try:
                logger.debug(f"Connection attempt {attempt + 1}/{self.reconnect_attempts}")
                
                # First, test basic connectivity
                async with self._session.get(self.base_url) as resp:
                    if resp.status != 200:
                        raise MCPConnectionError(f"Home Assistant returned status {resp.status}")
                    logger.debug("Basic connectivity test passed")
                
                # First check if endpoint returns SSE
                logger.debug(f"Checking SSE endpoint: {self.sse_url}")
                async with self._session.get(self.sse_url, headers=headers) as resp:
                    content_type = resp.headers.get('Content-Type', '')
                    logger.debug(f"SSE endpoint returned Content-Type: {content_type}")
                    
                    if resp.status != 200:
                        body = await resp.text()
                        raise MCPConnectionError(
                            f"SSE endpoint returned {resp.status}: {body[:200]}"
                        )
                    
                    if 'text/event-stream' not in content_type and 'text/plain' not in content_type:
                        # Read a bit of the response to see what it is
                        chunk = await resp.content.read(500)
                        text = chunk.decode('utf-8', errors='replace')
                        
                        if text.strip().startswith('<!DOCTYPE') or text.strip().startswith('<html'):
                            raise MCPConnectionError(
                                "SSE endpoint returned HTML instead of event stream. "
                                "MCP server integration may not be properly configured."
                            )
                        else:
                            logger.warning(f"Unexpected content type: {content_type}")
                            logger.warning(f"Response preview: {text[:100]}")
                
                # Now establish SSE connection with manual streaming
                logger.debug(f"Establishing SSE connection to {self.sse_url}")
                # Use no total timeout for SSE streaming
                self._sse_response = await self._session.get(
                    self.sse_url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=None, connect=self.connection_timeout)
                )
                
                if self._sse_response.status != 200:
                    body = await self._sse_response.text()
                    raise MCPConnectionError(
                        f"SSE endpoint returned {self._sse_response.status}: {body[:200]}"
                    )
                
                # Start message processing task
                self._message_task = asyncio.create_task(self._process_messages())
                
                # Wait a bit longer to ensure connection is established
                await asyncio.sleep(0.5)
                
                # Check if task is still running
                if self._message_task.done():
                    error = self._message_task.exception()
                    if error:
                        raise MCPConnectionError(f"Message processing failed immediately: {error}")
                
                # Wait for endpoint event
                logger.debug("Waiting for endpoint event...")
                max_wait = 5.0  # Maximum 5 seconds to wait for endpoint
                start_time = asyncio.get_event_loop().time()
                
                while not self._message_endpoint:
                    if asyncio.get_event_loop().time() - start_time > max_wait:
                        raise MCPConnectionError("Timeout waiting for message endpoint from MCP server")
                    await asyncio.sleep(0.1)
                
                logger.debug(f"Got message endpoint: {self._message_endpoint}")
                logger.debug("SSE connection established successfully")
                return
                
            except ssl.SSLError as e:
                last_error = e
                logger.error(f"SSL Error on attempt {attempt + 1}: {e}")
                logger.error("This might be a certificate verification issue")
                
            except aiohttp.ClientConnectorSSLError as e:
                last_error = e
                logger.error(f"SSL Connection Error on attempt {attempt + 1}: {e}")
                logger.error("Try disabling SSL verification for testing")
                
            except Exception as e:
                last_error = e
                logger.error(f"Connection attempt {attempt + 1} failed: {type(e).__name__}: {e}")
                
                # Cancel message task if it's running
                if self._message_task and not self._message_task.done():
                    self._message_task.cancel()
                    try:
                        await self._message_task
                    except asyncio.CancelledError:
                        pass
                
                if attempt < self.reconnect_attempts - 1:
                    delay = self.reconnect_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    
        raise MCPConnectionError(f"Failed after {self.reconnect_attempts} attempts: {last_error}")
    
    async def _parse_sse_stream(self) -> AsyncGenerator[Tuple[str, str], None]:
        """Parse SSE stream manually.
        
        Yields:
            Tuples of (event_type, event_data)
        """
        buffer = ""
        event_type = "message"
        event_data = ""
        
        try:
            # Check if response is closed before reading
            if self._sse_response.closed:
                logger.debug("SSE response is closed")
                return
                
            async for chunk in self._sse_response.content:
                # Decode chunk and add to buffer
                buffer += chunk.decode('utf-8', errors='replace')
                
                # Process complete lines
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip('\r')
                    
                    if not line:
                        # Empty line means end of event
                        if event_data:
                            yield (event_type, event_data.strip())
                            event_type = "message"
                            event_data = ""
                    elif line.startswith('event:'):
                        event_type = line[6:].strip()
                    elif line.startswith('data:'):
                        # Append data (SSE allows multiple data lines)
                        if event_data:
                            event_data += "\n"
                        event_data += line[5:].strip()
                    elif line.startswith(':'):
                        # Comment line, ignore
                        pass
                    else:
                        # Non-standard line, log it
                        logger.debug(f"Non-standard SSE line: {line}")
                        
        except aiohttp.ClientPayloadError as e:
            # Connection closed by server
            if self._connected:
                # Only log as error if we're still supposed to be connected
                logger.warning(f"SSE stream closed unexpectedly: {e}")
            else:
                logger.debug(f"SSE stream closed during shutdown: {e}")
            # Yield any remaining data
            if event_data:
                logger.debug(f"Yielding incomplete event on stream close: type={event_type}, data={event_data[:200]}...")
                yield (event_type, event_data.strip())
            # Also check if there's data in the buffer
            if buffer.strip():
                logger.warning(f"Unprocessed data in buffer on stream close: {buffer[:200]}...")
        except Exception as e:
            logger.error(f"Error in SSE parser: {type(e).__name__}: {e}")
            raise
    
    async def _process_messages(self) -> None:
        """Process incoming SSE messages."""
        try:
            logger.debug("Starting message processing loop")
            if not self._sse_response:
                raise MCPError("SSE response not initialized")
            
            # Process SSE events using our manual parser
            async for event_type, event_data in self._parse_sse_stream():
                logger.debug(f"SSE event - type: {event_type}, data: {event_data[:100] if event_data else 'None'}")
                
                if event_type == 'endpoint':
                    # Handle the endpoint event that provides the message URL
                    self._message_endpoint = event_data
                    logger.info(f"Received message endpoint: {self._message_endpoint}")
                    # The endpoint event means we're connected
                    continue
                    
                elif event_type == 'message':
                    logger.debug(f"Received message event: {event_data[:100]}...")
                    # Check if this looks like a complete JSON message
                    if event_data.strip().endswith('}'):
                        await self._handle_message(event_data)
                    else:
                        logger.warning(f"Received incomplete message: {event_data[:200]}...")
                elif event_type == 'error':
                    logger.error(f"SSE error event: {event_data}")
                elif event_type == 'ping':
                    logger.debug("Received ping from server")
                else:
                    logger.debug(f"Received unknown event type: {event_type}")
            
            # Stream ended 
            if self._connected:
                logger.warning("SSE stream ended while still connected - will reconnect")
                # Trigger reconnection
                self._connected = False
                if self._reconnect_task is None or self._reconnect_task.done():
                    self._reconnect_task = asyncio.create_task(self._reconnect())
            else:
                logger.debug("SSE stream ended during shutdown")
                
        except asyncio.CancelledError:
            logger.debug("Message processing cancelled")
            raise
        except Exception as e:
            logger.error(f"Error processing messages: {type(e).__name__}: {e}")
            if hasattr(e, '__traceback__'):
                import traceback
                logger.debug("Traceback: " + ''.join(traceback.format_tb(e.__traceback__)))
            
            if self._connected:
                # Only reconnect if we're supposed to be connected
                self._connected = False
                if self._reconnect_task is None or self._reconnect_task.done():
                    self._reconnect_task = asyncio.create_task(self._reconnect())
    
    async def _handle_message(self, data: str) -> None:
        """Handle incoming JSON-RPC message."""
        try:
            message = json.loads(data)
            
            # Check if it's a response to a request
            if 'id' in message and message['id'] in self._response_futures:
                logger.debug(f"Processing response for request {message['id']}")
                future = self._response_futures.pop(message['id'])
                if 'error' in message:
                    error = message['error']
                    logger.error(f"Received error response: {error}")
                    future.set_exception(MCPProtocolError(
                        error.get('code', -1),
                        error.get('message', 'Unknown error'),
                        error.get('data')
                    ))
                else:
                    logger.debug(f"Received successful response for {message['id']}")
                    future.set_result(message.get('result'))
            
            # Handle notifications (no id field)
            elif 'method' in message and 'id' not in message:
                await self._handle_notification(message['method'], message.get('params', {}))
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {e}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _handle_notification(self, method: str, params: Dict[str, Any]) -> None:
        """Handle server notifications."""
        logger.debug(f"Received notification: {method}")
        
        # Call registered handlers
        if method in self._event_handlers:
            for handler in self._event_handlers[method]:
                try:
                    await handler(params)
                except Exception as e:
                    logger.error(f"Error in notification handler for {method}: {e}")
    
    async def _send_request(self, method: str, params: Dict[str, Any] = None) -> Any:
        """Send JSON-RPC request and wait for response."""
        if not self._connected:
            raise MCPConnectionError("Not connected to MCP server")
        
        # Wait for message endpoint to be available
        if not self._message_endpoint:
            # Give it a moment for the endpoint event to arrive
            await asyncio.sleep(0.5)
            if not self._message_endpoint:
                raise MCPConnectionError("No message endpoint received from MCP server")
        
        request_id = str(uuid.uuid4())
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": request_id
        }
        
        # Create future for response
        future = asyncio.Future()
        self._response_futures[request_id] = future
        
        # Build the full message endpoint URL
        message_url = urljoin(self.base_url, self._message_endpoint.lstrip('/'))
        
        try:
            # Send request via POST to the message endpoint (not SSE endpoint)
            logger.debug(f"Sending request: {method} with id {request_id} to {message_url}")
            async with self._session.post(
                message_url,
                json=request,
                headers={
                    "Authorization": f"Bearer {self.access_token}",
                    "Content-Type": "application/json"
                }
            ) as response:
                if response.status != 200:
                    body = await response.text()
                    logger.error(f"Request failed: status={response.status}, body={body[:200]}")
                    raise MCPConnectionError(f"Request failed with status {response.status}: {body[:200]}")
                else:
                    logger.debug(f"POST request sent successfully, status: {response.status}")
            
            # Wait for response with timeout
            logger.debug(f"Waiting for response to request {request_id}")
            result = await asyncio.wait_for(future, timeout=30.0)
            logger.debug(f"Received response for request {request_id}")
            return result
            
        except asyncio.TimeoutError:
            self._response_futures.pop(request_id, None)
            raise MCPError(f"Request timeout for method {method}")
        except Exception as e:
            self._response_futures.pop(request_id, None)
            raise
    
    async def _initialize_protocol(self) -> None:
        """Initialize MCP protocol handshake."""
        logger.debug("Initializing MCP protocol")
        
        # Send initialize request
        result = await self._send_request("initialize", {
            "protocolVersion": "0.1.0",
            "capabilities": {
                "tools": {}
            },
            "clientInfo": {
                "name": "openvoice-mcp",
                "version": "2.0.0"
            }
        })
        
        self._capabilities = result.get('capabilities', {})
        logger.debug(f"Server capabilities: {self._capabilities}")
        
        # List available tools
        await self._list_tools()
    
    async def _list_tools(self) -> None:
        """Retrieve available tools from server."""
        logger.debug("Listing available tools")
        
        try:
            result = await self._send_request("tools/list", {})
            self._tools = result.get('tools', [])
            logger.info(f"Available tools: {[tool['name'] for tool in self._tools]}")
        except Exception as e:
            logger.warning(f"Failed to list tools: {e}")
            logger.warning("Continuing without tools - this may be a Home Assistant MCP limitation")
            self._tools = []
            # Don't re-raise - we can operate without tools
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server.
        
        Args:
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        if not self._tools:
            raise MCPError("No tools available - tools discovery may have failed")
            
        logger.debug(f"Calling tool: {name}")
        
        result = await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments
        })
        
        return result
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools."""
        return self._tools.copy()
    
    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """Get specific tool by name."""
        for tool in self._tools:
            if tool['name'] == name:
                return tool
        return None
    
    async def _reconnect(self) -> None:
        """Attempt to reconnect to server."""
        logger.info("Attempting to reconnect to MCP server")
        
        self._connected = False
        
        # Clean up existing connection
        if self._message_task and not self._message_task.done():
            self._message_task.cancel()
            try:
                await self._message_task
            except asyncio.CancelledError:
                pass
        
        if self._sse_response:
            self._sse_response.close()
            self._sse_response = None
        
        # Wait a bit before reconnecting
        await asyncio.sleep(1.0)
        
        try:
            # Reconnect - this will skip initialization if already done
            await self.connect()
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            # Schedule another reconnection attempt
            await asyncio.sleep(5.0)
            if not self._connected and (self._reconnect_task is None or self._reconnect_task.done()):
                self._reconnect_task = asyncio.create_task(self._reconnect())
    
    async def disconnect(self) -> None:
        """Disconnect from MCP server."""
        logger.info("Disconnecting from MCP server")
        
        self._connected = False
        
        # Cancel tasks
        if self._message_task and not self._message_task.done():
            self._message_task.cancel()
            try:
                await self._message_task
            except asyncio.CancelledError:
                pass
            
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
        
        # Close connections
        if self._sse_response:
            try:
                self._sse_response.close()
            except Exception as e:
                logger.debug(f"Error closing SSE response: {e}")
            self._sse_response = None
            
        if self._session:
            try:
                await self._session.close()
            except Exception as e:
                logger.debug(f"Error closing session: {e}")
            self._session = None
        
        # Clear state
        self._response_futures.clear()
        self._tools.clear()
        self._capabilities.clear()
        self._message_endpoint = None
        self._initialized = False
    
    def on_event(self, event: str, handler: Callable) -> None:
        """Register event handler for notifications.
        
        Args:
            event: Event name
            handler: Async function to handle event
        """
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)
    
    def off_event(self, event: str, handler: Callable) -> None:
        """Unregister event handler.
        
        Args:
            event: Event name
            handler: Handler to remove
        """
        if event in self._event_handlers:
            self._event_handlers[event].remove(handler)
            if not self._event_handlers[event]:
                del self._event_handlers[event]
    
    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected