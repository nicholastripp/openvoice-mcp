"""Model Context Protocol (MCP) client for Home Assistant integration."""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Callable
from urllib.parse import urljoin
import aiohttp
from aiohttp_sse_client import client as sse_client
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
        reconnect_delay: float = 1.0
    ):
        """Initialize MCP client.
        
        Args:
            base_url: Home Assistant base URL
            access_token: Long-lived access token
            sse_endpoint: SSE endpoint path (default: /mcp_server/sse)
            connection_timeout: Connection timeout in seconds
            reconnect_attempts: Number of reconnection attempts
            reconnect_delay: Initial delay between reconnection attempts
        """
        self.base_url = base_url.rstrip('/')
        self.access_token = access_token
        self.sse_endpoint = sse_endpoint
        self.connection_timeout = connection_timeout
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        
        self.sse_url = urljoin(self.base_url, sse_endpoint.lstrip('/'))
        self._session: Optional[aiohttp.ClientSession] = None
        self._sse_client = None
        self._response_futures: Dict[str, asyncio.Future] = {}
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._tools: List[Dict[str, Any]] = []
        self._capabilities: Dict[str, Any] = {}
        self._connected = False
        self._reconnect_task = None
        self._message_task = None
        
    async def connect(self) -> None:
        """Establish connection to MCP server."""
        if self._connected:
            logger.debug("Already connected to MCP server")
            return
            
        logger.info(f"Connecting to MCP server at {self.sse_url}")
        
        try:
            await self._establish_connection()
            await self._initialize_protocol()
            self._connected = True
            logger.info("Successfully connected to MCP server")
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise MCPConnectionError(f"Connection failed: {str(e)}")
    
    async def _establish_connection(self) -> None:
        """Establish SSE connection with retry logic."""
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache"
        }
        
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=self.connection_timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        
        last_error = None
        for attempt in range(self.reconnect_attempts):
            try:
                self._sse_client = sse_client.EventSource(
                    self.sse_url,
                    session=self._session,
                    headers=headers
                )
                
                # Start message processing task
                self._message_task = asyncio.create_task(self._process_messages())
                
                # Give it a moment to establish
                await asyncio.sleep(0.1)
                return
                
            except Exception as e:
                last_error = e
                if attempt < self.reconnect_attempts - 1:
                    delay = self.reconnect_delay * (2 ** attempt)
                    logger.warning(f"Connection attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                    
        raise MCPConnectionError(f"Failed after {self.reconnect_attempts} attempts: {last_error}")
    
    async def _process_messages(self) -> None:
        """Process incoming SSE messages."""
        try:
            async for event in self._sse_client:
                if event.type == 'message':
                    await self._handle_message(event.data)
                elif event.type == 'error':
                    logger.error(f"SSE error event: {event.data}")
                elif event.type == 'ping':
                    logger.debug("Received ping from server")
        except Exception as e:
            logger.error(f"Error processing messages: {e}")
            self._connected = False
            # Trigger reconnection if needed
            if self._reconnect_task is None or self._reconnect_task.done():
                self._reconnect_task = asyncio.create_task(self._reconnect())
    
    async def _handle_message(self, data: str) -> None:
        """Handle incoming JSON-RPC message."""
        try:
            message = json.loads(data)
            
            # Check if it's a response to a request
            if 'id' in message and message['id'] in self._response_futures:
                future = self._response_futures.pop(message['id'])
                if 'error' in message:
                    error = message['error']
                    future.set_exception(MCPProtocolError(
                        error.get('code', -1),
                        error.get('message', 'Unknown error'),
                        error.get('data')
                    ))
                else:
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
        
        try:
            # Send request via POST to the same endpoint
            async with self._session.post(
                self.sse_url,
                json=request,
                headers={"Authorization": f"Bearer {self.access_token}"}
            ) as response:
                if response.status != 200:
                    raise MCPConnectionError(f"Request failed with status {response.status}")
            
            # Wait for response with timeout
            result = await asyncio.wait_for(future, timeout=30.0)
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
                "name": "ha-realtime-assist",
                "version": "1.0.0"
            }
        })
        
        self._capabilities = result.get('capabilities', {})
        logger.debug(f"Server capabilities: {self._capabilities}")
        
        # List available tools
        await self._list_tools()
    
    async def _list_tools(self) -> None:
        """Retrieve available tools from server."""
        logger.debug("Listing available tools")
        
        result = await self._send_request("tools/list", {})
        self._tools = result.get('tools', [])
        
        logger.info(f"Available tools: {[tool['name'] for tool in self._tools]}")
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server.
        
        Args:
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
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
        
        if self._sse_client:
            await self._sse_client.close()
        
        try:
            await self.connect()
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from MCP server."""
        logger.info("Disconnecting from MCP server")
        
        self._connected = False
        
        # Cancel tasks
        if self._message_task and not self._message_task.done():
            self._message_task.cancel()
            
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
        
        # Close connections
        if self._sse_client:
            await self._sse_client.close()
            
        if self._session:
            await self._session.close()
            self._session = None
        
        # Clear state
        self._response_futures.clear()
        self._tools.clear()
        self._capabilities.clear()
    
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