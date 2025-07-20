"""Model Context Protocol (MCP) client for Home Assistant using official SDK."""

import asyncio
import logging
from typing import Any, Dict, List, Optional
import httpx
import time

from mcp.client.sse import sse_client
from mcp import ClientSession
from mcp.types import Tool

logger = logging.getLogger(__name__)


class MCPError(Exception):
    """Base exception for MCP-related errors."""
    pass


class MCPConnectionError(MCPError):
    """Raised when connection to MCP server fails."""
    pass


class BearerTokenAuth(httpx.Auth):
    """Custom auth handler for Home Assistant bearer tokens."""
    
    def __init__(self, token: str):
        self.token = token
    
    def auth_flow(self, request):
        request.headers["Authorization"] = f"Bearer {self.token}"
        yield request


class MCPClient:
    """Client for Home Assistant's Model Context Protocol server using official SDK."""
    
    def __init__(
        self,
        base_url: str,
        access_token: str,
        sse_endpoint: str = "/mcp_server/sse",
        connection_timeout: int = 30,
        sse_read_timeout: int = 300,
        ssl_verify: bool = True,
        max_reconnect_attempts: int = 5,
        reconnect_base_delay: float = 1.0,
        reconnect_max_delay: float = 60.0
    ):
        """Initialize MCP client.
        
        Args:
            base_url: Home Assistant base URL
            access_token: Long-lived access token
            sse_endpoint: SSE endpoint path (default: /mcp_server/sse)
            connection_timeout: Connection timeout in seconds
            sse_read_timeout: SSE read timeout in seconds
            ssl_verify: Whether to verify SSL certificates
            max_reconnect_attempts: Maximum number of reconnection attempts
            reconnect_base_delay: Base delay between reconnection attempts
            reconnect_max_delay: Maximum delay between reconnection attempts
        """
        self.base_url = base_url.rstrip('/')
        self.access_token = access_token
        self.sse_endpoint = sse_endpoint
        self.connection_timeout = connection_timeout
        self.sse_read_timeout = sse_read_timeout
        self.ssl_verify = ssl_verify
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_base_delay = reconnect_base_delay
        self.reconnect_max_delay = reconnect_max_delay
        
        self.sse_url = f"{self.base_url}{self.sse_endpoint}"
        
        # Context managers for proper lifecycle management
        self._streams_context = None
        self._session_context = None
        self._session: Optional[ClientSession] = None
        
        # State tracking
        self._tools: List[Tool] = []
        self._capabilities: Dict[str, Any] = {}
        self._connected = False
        self._initialized = False
        self._shutting_down = False
        self._reconnect_task = None
        self._last_successful_connection = None
    
    def _httpx_client_factory(self, headers=None, auth=None, timeout=None):
        """Create httpx client with SSL settings and provided parameters."""
        return httpx.AsyncClient(
            verify=self.ssl_verify,
            headers=headers,
            auth=auth,
            timeout=timeout
        )
    
    async def connect(self) -> None:
        """Establish connection and initialize MCP protocol with retry logic."""
        if self._initialized:
            logger.debug("Already initialized")
            return
        
        attempt = 0
        last_error = None
        
        while attempt < self.max_reconnect_attempts and not self._shutting_down:
            try:
                await self._connect_once()
                self._last_successful_connection = time.time()
                return
            except Exception as e:
                last_error = e
                attempt += 1
                
                if attempt >= self.max_reconnect_attempts:
                    logger.error(f"Failed to connect after {self.max_reconnect_attempts} attempts")
                    raise MCPConnectionError(f"Failed to connect to MCP server after {self.max_reconnect_attempts} attempts: {last_error}")
                
                # Calculate delay with exponential backoff
                delay = min(
                    self.reconnect_base_delay * (2 ** (attempt - 1)),
                    self.reconnect_max_delay
                )
                
                logger.warning(f"Connection attempt {attempt} failed: {e}. Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
    
    async def _connect_once(self) -> None:
        """Single connection attempt."""
        logger.info(f"Connecting to MCP server at {self.sse_url}")
        logger.info(f"SSL verification: {'enabled' if self.ssl_verify else 'DISABLED'}")
        
        try:
            # Create and enter SSE context with error handling
            self._streams_context = sse_client(
                url=self.sse_url,
                auth=BearerTokenAuth(self.access_token),
                headers={
                    "Accept": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                },
                timeout=float(self.connection_timeout),
                sse_read_timeout=float(self.sse_read_timeout),
                httpx_client_factory=self._httpx_client_factory
            )
            
            try:
                streams = await self._streams_context.__aenter__()
                logger.debug("SSE streams created")
            except httpx.RemoteProtocolError as e:
                logger.error(f"SSE connection failed with protocol error: {e}")
                raise MCPConnectionError(f"SSE connection failed: {str(e)}")
            except httpx.NetworkError as e:
                logger.error(f"SSE connection failed with network error: {e}")
                raise MCPConnectionError(f"Network error connecting to SSE: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error creating SSE streams: {e}")
                raise
            
            # Create and enter session context with error handling
            try:
                self._session_context = ClientSession(*streams)
                self._session = await self._session_context.__aenter__()
                logger.debug("MCP session created")
            except Exception as e:
                logger.error(f"Failed to create MCP session: {e}")
                # Clean up streams context if session creation fails
                await self._cleanup_connection()
                raise MCPConnectionError(f"Failed to create MCP session: {str(e)}")
            
            # Initialize protocol
            await self._initialize_protocol(self._session)
            
            # List available tools (may fail on some HA versions)
            await self._list_tools(self._session)
            
            self._connected = True
            self._initialized = True
            logger.info("Successfully connected to MCP server")
                
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            # Clean up on failure
            await self._cleanup_connection()
            raise
    
    async def _cleanup_connection(self) -> None:
        """Clean up connection contexts."""
        # Exit session context first
        if self._session_context:
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"Error cleaning up session context: {e}")
            finally:
                self._session_context = None
        
        # Then exit streams context with special handling for shutdown
        if self._streams_context:
            try:
                await self._streams_context.__aexit__(None, None, None)
            except RuntimeError as e:
                # Common error during shutdown when context is closed from different task
                if "different task" in str(e) or "Cannot enter" in str(e):
                    logger.debug("SSE context cleanup during shutdown - this is expected")
                else:
                    logger.warning(f"Unexpected runtime error during SSE cleanup: {e}")
            except GeneratorExit:
                # This is expected during shutdown
                logger.debug("SSE generator exit during cleanup - this is expected")
            except Exception as e:
                # Log unexpected errors
                logger.warning(f"Unexpected error cleaning up streams context: {type(e).__name__}: {e}")
            finally:
                self._streams_context = None
        
        # Reset state
        self._session = None
        self._connected = False
        self._initialized = False
        self._shutting_down = False
    
    async def _initialize_protocol(self, session: ClientSession) -> None:
        """Initialize MCP protocol handshake."""
        logger.debug("Initializing MCP protocol")
        
        try:
            result = await session.initialize()
            
            if result.capabilities:
                self._capabilities = {
                    'experimental': result.capabilities.experimental or {},
                    'tools': getattr(result.capabilities, 'tools', {}),
                    'prompts': getattr(result.capabilities, 'prompts', {}),
                    'resources': getattr(result.capabilities, 'resources', {})
                }
            else:
                self._capabilities = {}
                
            logger.debug(f"Protocol version: {result.protocolVersion}")
            logger.debug(f"Server capabilities: {self._capabilities}")
            
        except httpx.RemoteProtocolError as e:
            logger.error(f"Protocol error during initialization: {e}")
            raise MCPError(f"Protocol initialization failed with protocol error: {str(e)}")
        except httpx.NetworkError as e:
            logger.error(f"Network error during initialization: {e}")
            raise MCPError(f"Protocol initialization failed with network error: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to initialize protocol: {e}")
            raise MCPError(f"Protocol initialization failed: {str(e)}")
    
    async def _list_tools(self, session: ClientSession) -> None:
        """Retrieve available tools from server."""
        logger.debug("Listing available tools")
        
        try:
            result = await session.list_tools()
            
            if result and result.tools:
                self._tools = result.tools
                logger.info(f"Available tools: {[tool.name for tool in self._tools]}")
            else:
                self._tools = []
                logger.warning("No tools found")
                
        except Exception as e:
            logger.warning(f"Failed to list tools: {e}")
            logger.warning("Continuing without tools - this may be a Home Assistant MCP limitation")
            self._tools = []
            # Don't re-raise - we can operate without tools
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server with automatic reconnection.
        
        Args:
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        if not self._initialized or not self._session:
            # Try to reconnect if not connected
            logger.info("Not connected, attempting to reconnect...")
            await self.connect()
            
        if not self._tools:
            logger.warning("No tools available - tools discovery may have failed")
            
        logger.debug(f"Calling tool: {name}")
        
        try:
            # Use the persistent session directly
            result = await self._session.call_tool(name, arguments)
            
            if hasattr(result, 'content'):
                return result.content
            else:
                return result
                
        except (httpx.RemoteProtocolError, httpx.NetworkError, ConnectionError) as e:
            logger.warning(f"Connection error during tool call: {e}")
            
            # Try to reconnect and retry once
            if not self._shutting_down:
                logger.info("Attempting to reconnect and retry tool call...")
                await self._handle_connection_error()
                
                # Retry the tool call once after reconnection
                if self._initialized and self._session:
                    try:
                        result = await self._session.call_tool(name, arguments)
                        if hasattr(result, 'content'):
                            return result.content
                        else:
                            return result
                    except Exception as retry_error:
                        logger.error(f"Tool call failed after reconnection: {retry_error}")
                        raise MCPError(f"Failed to call tool '{name}' after reconnection: {str(retry_error)}")
            
            raise MCPError(f"Failed to call tool '{name}': {str(e)}")
            
        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            raise MCPError(f"Failed to call tool '{name}': {str(e)}")
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools as dictionaries."""
        # Convert Tool objects to dictionaries for compatibility
        return [
            {
                'name': tool.name,
                'description': tool.description,
                'inputSchema': getattr(tool, 'inputSchema', {})
            }
            for tool in self._tools
        ]
    
    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """Get specific tool by name."""
        for tool in self._tools:
            if tool.name == name:
                return {
                    'name': tool.name,
                    'description': tool.description,
                    'inputSchema': getattr(tool, 'inputSchema', {})
                }
        return None
    
    async def disconnect(self) -> None:
        """Disconnect from MCP server."""
        if not self._initialized:
            logger.debug("Already disconnected")
            return
            
        logger.info("Disconnecting from MCP server")
        self._shutting_down = True
        
        # Clear data first
        self._tools.clear()
        self._capabilities.clear()
        
        # Perform cleanup
        await self._cleanup_connection()
        logger.info("MCP server disconnected")
    
    async def _handle_connection_error(self) -> None:
        """Handle connection errors by attempting to reconnect."""
        if self._reconnect_task and not self._reconnect_task.done():
            # Reconnection already in progress
            logger.debug("Reconnection already in progress")
            await self._reconnect_task
            return
        
        # Start reconnection task
        self._reconnect_task = asyncio.create_task(self._reconnect())
        await self._reconnect_task
    
    async def _reconnect(self) -> None:
        """Reconnect to the MCP server."""
        logger.info("Starting reconnection process...")
        
        # Clean up existing connection
        await self._cleanup_connection()
        
        # Wait a bit before reconnecting
        await asyncio.sleep(0.5)
        
        # Attempt to reconnect
        try:
            await self.connect()
            logger.info("Reconnection successful")
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            raise
    
    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected and self._initialized and self._session is not None
        
    @property
    def capabilities(self) -> Dict[str, Any]:
        """Get server capabilities."""
        return self._capabilities.copy()