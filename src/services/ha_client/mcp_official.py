"""Model Context Protocol (MCP) client for Home Assistant using official SDK."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable
from contextlib import asynccontextmanager
import httpx

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
        ssl_verify: bool = True
    ):
        """Initialize MCP client.
        
        Args:
            base_url: Home Assistant base URL
            access_token: Long-lived access token
            sse_endpoint: SSE endpoint path (default: /mcp_server/sse)
            connection_timeout: Connection timeout in seconds
            sse_read_timeout: SSE read timeout in seconds
            ssl_verify: Whether to verify SSL certificates
        """
        self.base_url = base_url.rstrip('/')
        self.access_token = access_token
        self.sse_endpoint = sse_endpoint
        self.connection_timeout = connection_timeout
        self.sse_read_timeout = sse_read_timeout
        self.ssl_verify = ssl_verify
        
        self.sse_url = f"{self.base_url}{self.sse_endpoint}"
        self._session: Optional[ClientSession] = None
        self._tools: List[Tool] = []
        self._capabilities: Dict[str, Any] = {}
        self._connected = False
        self._initialized = False
        
    @asynccontextmanager
    async def _create_session(self):
        """Create an MCP session with proper error handling."""
        try:
            # Create httpx client factory that respects SSL settings
            def httpx_client_factory():
                return httpx.AsyncClient(verify=self.ssl_verify)
            
            # Create SSE client with Home Assistant auth
            async with sse_client(
                url=self.sse_url,
                auth=BearerTokenAuth(self.access_token),
                headers={
                    "Accept": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                },
                timeout=float(self.connection_timeout),
                sse_read_timeout=float(self.sse_read_timeout),
                httpx_client_factory=httpx_client_factory
            ) as (read_stream, write_stream):
                logger.debug("SSE client connected")
                
                # Create MCP session
                async with ClientSession(read_stream, write_stream) as session:
                    self._session = session
                    self._connected = True
                    yield session
                    
        except Exception as e:
            logger.error(f"Failed to create MCP session: {e}")
            raise MCPConnectionError(f"Connection failed: {str(e)}")
        finally:
            self._session = None
            self._connected = False
    
    async def connect(self) -> None:
        """Establish connection and initialize MCP protocol."""
        if self._initialized:
            logger.debug("Already initialized")
            return
            
        logger.info(f"Connecting to MCP server at {self.sse_url}")
        logger.info(f"SSL verification: {'enabled' if self.ssl_verify else 'DISABLED'}")
        
        try:
            async with self._create_session() as session:
                # Initialize protocol
                await self._initialize_protocol(session)
                
                # List available tools (may fail on some HA versions)
                await self._list_tools(session)
                
                self._initialized = True
                logger.info("Successfully connected to MCP server")
                
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise
    
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
        """Call a tool on the MCP server.
        
        Args:
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        if not self._initialized:
            raise MCPError("Client not initialized - call connect() first")
            
        if not self._tools:
            raise MCPError("No tools available - tools discovery may have failed")
            
        logger.debug(f"Calling tool: {name}")
        
        async with self._create_session() as session:
            try:
                # Re-initialize if needed (for new session)
                if not session._initialized:
                    await self._initialize_protocol(session)
                
                # Call the tool
                result = await session.call_tool(name, arguments)
                
                if hasattr(result, 'content'):
                    return result.content
                else:
                    return result
                    
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
        logger.info("Disconnecting from MCP server")
        self._initialized = False
        self._tools.clear()
        self._capabilities.clear()
        # Session cleanup is handled by context manager
    
    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected and self._initialized
        
    @property
    def capabilities(self) -> Dict[str, Any]:
        """Get server capabilities."""
        return self._capabilities.copy()