"""
Client-side MCP Manager for local/stdio MCP servers

Handles connections to local MCP servers that cannot be accessed by OpenAI's
native MCP support (e.g., stdio-based servers running as subprocesses).
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from config import MCPServerConfig

logger = logging.getLogger(__name__)


class ClientMCPError(Exception):
    """Base exception for client-side MCP errors"""
    pass


@dataclass
class MCPServerConnection:
    """Represents a connection to a single MCP server"""
    name: str
    config: MCPServerConfig
    session: Optional[ClientSession] = None
    context: Optional[Any] = None  # Context manager for connection lifecycle
    tools: List[Dict[str, Any]] = None
    connected: bool = False

    def __post_init__(self):
        if self.tools is None:
            self.tools = []


class ClientMCPManager:
    """
    Manages client-side connections to local MCP servers

    This manager handles stdio-based MCP servers that run as local subprocesses.
    It discovers tools from these servers and provides a unified interface for
    calling them.
    """

    def __init__(self, server_configs: Dict[str, MCPServerConfig]):
        """
        Initialize client-side MCP manager

        Args:
            server_configs: Dictionary of server name -> MCPServerConfig
                           Only servers with mode="client" will be managed
        """
        self.logger = logging.getLogger("ClientMCPManager")

        # Filter for client-mode servers only
        self.server_configs = {
            name: config
            for name, config in server_configs.items()
            if config.mode == "client" and config.enabled
        }

        # Track connections to each server
        self.connections: Dict[str, MCPServerConnection] = {}

        # Map tool names to server names for routing
        self.tool_to_server: Dict[str, str] = {}

        self.logger.info(f"Initialized ClientMCPManager with {len(self.server_configs)} server(s)")

    async def connect_all(self) -> None:
        """
        Connect to all configured client-side MCP servers

        Establishes connections in parallel for efficiency.
        """
        if not self.server_configs:
            self.logger.info("No client-side MCP servers configured")
            return

        self.logger.info(f"Connecting to {len(self.server_configs)} client-side MCP server(s)...")

        # Connect to all servers in parallel
        tasks = [
            self._connect_server(name, config)
            for name, config in self.server_configs.items()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log results
        successful = sum(1 for r in results if r is True)
        failed = len(results) - successful

        if successful > 0:
            self.logger.info(f"Successfully connected to {successful}/{len(results)} MCP server(s)")
        if failed > 0:
            self.logger.warning(f"Failed to connect to {failed}/{len(results)} MCP server(s)")

    async def _connect_server(self, name: str, config: MCPServerConfig) -> bool:
        """
        Connect to a single MCP server

        Args:
            name: Server name
            config: Server configuration

        Returns:
            True if connection successful, False otherwise
        """
        self.logger.info(f"Connecting to MCP server '{name}' (transport: {config.transport})")

        try:
            connection = MCPServerConnection(name=name, config=config)

            # Currently only stdio transport is supported
            if config.transport != "stdio":
                self.logger.error(f"Unsupported transport '{config.transport}' for server '{name}'")
                return False

            # Validate stdio configuration
            if not config.command:
                self.logger.error(f"No command specified for stdio server '{name}'")
                return False

            # Create stdio server parameters
            server_params = StdioServerParameters(
                command=config.command,
                args=config.args or [],
                env=config.env or {}
            )

            # Create stdio client context
            self.logger.debug(f"Starting stdio server for '{name}': {config.command} {config.args or []}")

            # Enter context and create session
            connection.context = stdio_client(server_params)
            read_stream, write_stream = await asyncio.wait_for(
                connection.context.__aenter__(),
                timeout=config.timeout
            )

            # Create and initialize session
            session_context = ClientSession(read_stream, write_stream)
            connection.session = await session_context.__aenter__()

            # Initialize MCP protocol
            init_result = await asyncio.wait_for(
                connection.session.initialize(),
                timeout=config.timeout
            )
            self.logger.debug(f"Server '{name}' initialized: protocol version {init_result.protocolVersion}")

            # List available tools
            tools_result = await asyncio.wait_for(
                connection.session.list_tools(),
                timeout=config.timeout
            )

            if tools_result and tools_result.tools:
                connection.tools = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": getattr(tool, "inputSchema", {})
                    }
                    for tool in tools_result.tools
                ]

                # Map tools to server for routing
                for tool in connection.tools:
                    tool_name = tool["name"]
                    self.tool_to_server[tool_name] = name
                    self.logger.debug(f"Registered tool '{tool_name}' from server '{name}'")

                self.logger.info(f"Server '{name}' provides {len(connection.tools)} tool(s): {[t['name'] for t in connection.tools]}")
            else:
                self.logger.warning(f"Server '{name}' has no tools")

            connection.connected = True
            self.connections[name] = connection

            return True

        except asyncio.TimeoutError:
            self.logger.error(f"Timeout connecting to server '{name}' (timeout: {config.timeout}s)")
            return False

        except Exception as e:
            self.logger.error(f"Failed to connect to server '{name}': {type(e).__name__}: {e}")
            return False

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool on the appropriate MCP server

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool execution result

        Raises:
            ClientMCPError: If tool not found or call fails
        """
        # Find which server has this tool
        server_name = self.tool_to_server.get(tool_name)

        if not server_name:
            raise ClientMCPError(f"Tool '{tool_name}' not found in any connected MCP server")

        connection = self.connections.get(server_name)

        if not connection or not connection.connected or not connection.session:
            raise ClientMCPError(f"Server '{server_name}' not connected")

        self.logger.info(f"Calling tool '{tool_name}' on server '{server_name}'")
        self.logger.debug(f"Tool arguments: {arguments}")

        try:
            result = await connection.session.call_tool(tool_name, arguments)

            # Extract content from result
            if hasattr(result, "content"):
                return result.content
            else:
                return result

        except Exception as e:
            self.logger.error(f"Tool call failed for '{tool_name}' on server '{server_name}': {e}")
            raise ClientMCPError(f"Failed to call tool '{tool_name}': {str(e)}")

    def get_all_tools(self) -> List[Dict[str, Any]]:
        """
        Get all tools from all connected servers

        Returns:
            List of tool definitions with server information
        """
        all_tools = []

        for server_name, connection in self.connections.items():
            if connection.connected:
                for tool in connection.tools:
                    # Add server information to tool
                    tool_with_server = tool.copy()
                    tool_with_server["_server"] = server_name
                    all_tools.append(tool_with_server)

        return all_tools

    def get_tools_for_openai(self) -> List[Dict[str, Any]]:
        """
        Get tool definitions formatted for OpenAI function calling

        Returns:
            List of OpenAI function definitions
        """
        openai_functions = []

        for tool in self.get_all_tools():
            openai_function = {
                "type": "function",
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("inputSchema", {"type": "object", "properties": {}})
            }
            openai_functions.append(openai_function)

        return openai_functions

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers"""
        if not self.connections:
            return

        self.logger.info(f"Disconnecting from {len(self.connections)} MCP server(s)...")

        # Disconnect from all servers in parallel
        tasks = [
            self._disconnect_server(name, connection)
            for name, connection in self.connections.items()
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

        self.connections.clear()
        self.tool_to_server.clear()

        self.logger.info("All MCP servers disconnected")

    async def _disconnect_server(self, name: str, connection: MCPServerConnection) -> None:
        """
        Disconnect from a single MCP server

        Args:
            name: Server name
            connection: Server connection to disconnect
        """
        try:
            if connection.session:
                # Exit session context
                await connection.session.__aexit__(None, None, None)

            if connection.context:
                # Exit stdio context
                await connection.context.__aexit__(None, None, None)

            self.logger.info(f"Disconnected from server '{name}'")

        except Exception as e:
            self.logger.warning(f"Error disconnecting from server '{name}': {e}")

        finally:
            connection.connected = False
            connection.session = None
            connection.context = None

    def is_connected(self) -> bool:
        """Check if any servers are connected"""
        return any(conn.connected for conn in self.connections.values())

    def get_connected_servers(self) -> List[str]:
        """Get list of connected server names"""
        return [name for name, conn in self.connections.items() if conn.connected]
