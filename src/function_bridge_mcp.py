"""
Function calling bridge between OpenAI and Home Assistant using MCP
"""
import json
import asyncio
from typing import Dict, Any, Optional, List

from services.ha_client.mcp_official import MCPClient
from services.ha_client.mcp_native import NativeMCPManager
from utils.logger import get_logger
from config import AppConfig


class MCPFunctionBridge:
    """
    Bridge that handles OpenAI function calls and routes them to Home Assistant via MCP
    Supports both native MCP mode and custom bridge mode with automatic fallback
    """
    
    def __init__(self, mcp_client: MCPClient, app_config: AppConfig = None):
        self.mcp_client = mcp_client
        self.app_config = app_config
        self._logger = None  # Lazy initialization
        self._tools_cache: List[Dict[str, Any]] = []
        self._primary_tool_name: Optional[str] = None
        
        # Initialize native MCP manager if config provided
        self.native_mcp = None
        if app_config:
            self.native_mcp = NativeMCPManager(app_config)
        
        # Track which mode we're using
        self.mode = "bridge"  # "native" or "bridge"
        
    @property
    def logger(self):
        """Lazy logger initialization"""
        if self._logger is None:
            self._logger = get_logger("MCPFunctionBridge")
        return self._logger
    
    async def initialize(self) -> None:
        """Initialize the bridge by discovering available MCP tools"""
        # Check if we should use native MCP mode
        if self.native_mcp and self.native_mcp.enabled:
            self.logger.info("Checking native MCP availability...")
            if await self.native_mcp.validate_connection():
                self.mode = "native"
                self.logger.info("Using native MCP mode for Home Assistant integration")
                return
            else:
                fallback_reason = self.native_mcp.get_fallback_reason()
                self.logger.warning(f"Native MCP not available: {fallback_reason}")
                if self.native_mcp.config.home_assistant.mcp.enable_fallback:
                    self.logger.info("Falling back to bridge mode")
                    self.mode = "bridge"
                else:
                    raise RuntimeError(f"Native MCP validation failed and fallback disabled: {fallback_reason}")
        
        # Use bridge mode
        self.mode = "bridge"
        if not self.mcp_client.is_connected:
            raise RuntimeError("MCP client must be connected before initializing function bridge")
        
        # Get available tools from MCP
        self._tools_cache = self.mcp_client.get_tools()
        self.logger.info(f"Discovered {len(self._tools_cache)} MCP tools")
        
        # Find the primary control tool (usually for natural language commands)
        for tool in self._tools_cache:
            if any(keyword in tool['name'].lower() for keyword in ['control', 'command', 'process']):
                self._primary_tool_name = tool['name']
                self.logger.info(f"Using '{self._primary_tool_name}' as primary control tool")
                break
        
        if not self._primary_tool_name and self._tools_cache:
            # Fallback to first tool if no control tool found
            self._primary_tool_name = self._tools_cache[0]['name']
            self.logger.warning(f"No control tool found, using '{self._primary_tool_name}' as fallback")
    
    def get_function_definitions(self) -> List[Dict[str, Any]]:
        """
        Get OpenAI function definitions based on discovered MCP tools
        
        Returns:
            List of function definitions for OpenAI (empty if using native mode)
        """
        # In native mode, OpenAI handles tool discovery directly
        if self.mode == "native":
            self.logger.debug("Native MCP mode - OpenAI will discover tools directly")
            return []  # No manual function definitions needed
        
        # Bridge mode - return discovered tools
        if not self._tools_cache:
            self.logger.warning("No MCP tools discovered, returning default function")
            return self._get_default_function_definitions()
        
        # Convert MCP tools to OpenAI function format
        functions = []
        
        for tool in self._tools_cache:
            # Map MCP tool to OpenAI function schema
            function_def = {
                "type": "function",
                "name": self._sanitize_function_name(tool['name']),
                "description": tool.get('description', f"MCP tool: {tool['name']}"),
            }
            
            # Convert MCP inputSchema to OpenAI parameters
            if 'inputSchema' in tool:
                function_def["parameters"] = tool['inputSchema']
            else:
                # Default parameters if none specified
                function_def["parameters"] = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            
            functions.append(function_def)
        
        # If we have many tools, also add a general control function
        if len(functions) > 5:
            functions.insert(0, self._get_control_function_definition())
        
        return functions
    
    def _get_default_function_definitions(self) -> List[Dict[str, Any]]:
        """Get default function definition when no MCP tools are discovered"""
        return [
            {
                "type": "function",
                "name": "control_home_assistant",
                "description": (
                    "Control Home Assistant devices, check device states, or answer questions about the home. "
                    "This function can handle any smart home command including turning devices on/off, "
                    "setting brightness/temperature, checking status, and querying information."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": (
                                "The natural language command to send to Home Assistant. "
                                "Examples: 'turn on the living room lights', 'set thermostat to 72 degrees', "
                                "'what's the temperature in the bedroom', 'are any lights on in the kitchen'"
                            )
                        }
                    },
                    "required": ["command"]
                }
            }
        ]
    
    def _get_control_function_definition(self) -> Dict[str, Any]:
        """Get a general control function for natural language commands"""
        return {
            "type": "function",
            "name": "control_home_assistant",
            "description": (
                "General purpose function to control Home Assistant using natural language. "
                "Use this for complex commands that don't fit specific tool patterns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Natural language command for Home Assistant"
                    }
                },
                "required": ["command"]
            }
        }
    
    def _sanitize_function_name(self, name: str) -> str:
        """Sanitize MCP tool name to be a valid OpenAI function name"""
        # Replace invalid characters with underscores
        sanitized = name.replace("-", "_").replace(" ", "_").replace(".", "_")
        # Ensure it starts with a letter
        if sanitized and not sanitized[0].isalpha():
            sanitized = f"mcp_{sanitized}"
        return sanitized
    
    async def handle_function_call(self, function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle function calls from OpenAI
        
        Args:
            function_name: Name of the function being called
            arguments: Function arguments
            
        Returns:
            Function result to send back to OpenAI
        """
        # Native mode shouldn't receive manual function calls
        if self.mode == "native":
            self.logger.warning(f"Received function call '{function_name}' in native mode - this shouldn't happen")
            # Try to handle it anyway for robustness
        
        self.logger.debug(f"Function call: {function_name} with args: {arguments}")
        
        try:
            # Handle general control function
            if function_name == "control_home_assistant":
                return await self._handle_control_command(arguments)
            
            # Find corresponding MCP tool
            mcp_tool_name = self._find_mcp_tool_name(function_name)
            if not mcp_tool_name:
                self.logger.error(f"No MCP tool found for function: {function_name}")
                return {
                    "success": False,
                    "error": f"Unknown function: {function_name}",
                    "message": "I don't know how to handle that request."
                }
            
            # Call MCP tool
            result = await self.mcp_client.call_tool(mcp_tool_name, arguments)
            
            # Convert MCP result to function result
            return self._convert_mcp_result(result, function_name, arguments)
            
        except Exception as e:
            self.logger.error(f"Error handling function call '{function_name}': {e}")
            return {
                "success": False,
                "error": "function_call_error",
                "message": f"I encountered an error while processing your request: {str(e)}"
            }
    
    async def _handle_control_command(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general control command using primary MCP tool"""
        command = arguments.get("command", "")
        if not command:
            return {
                "success": False,
                "error": "missing_command",
                "message": "No command provided"
            }
        
        if not self._primary_tool_name:
            return {
                "success": False,
                "error": "no_primary_tool",
                "message": "No primary control tool available in Home Assistant"
            }
        
        try:
            # Call primary tool with command
            result = await self.mcp_client.call_tool(
                self._primary_tool_name,
                {"command": command}  # Adapt to expected format
            )
            
            return self._convert_mcp_result(result, "control_home_assistant", arguments)
            
        except Exception as e:
            self.logger.error(f"Error processing control command '{command}': {e}")
            return {
                "success": False,
                "error": "mcp_error",
                "message": f"I encountered an error while trying to {command.lower()}. Please try again."
            }
    
    def _find_mcp_tool_name(self, function_name: str) -> Optional[str]:
        """Find original MCP tool name from sanitized function name"""
        # Direct match first
        for tool in self._tools_cache:
            if self._sanitize_function_name(tool['name']) == function_name:
                return tool['name']
        
        # Try case-insensitive match
        function_lower = function_name.lower()
        for tool in self._tools_cache:
            if self._sanitize_function_name(tool['name']).lower() == function_lower:
                return tool['name']
        
        return None
    
    def _convert_mcp_result(self, mcp_result: Any, function_name: str, 
                           original_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert MCP tool result to OpenAI function result
        
        Args:
            mcp_result: Result from MCP tool call
            function_name: Name of the function that was called
            original_args: Original arguments passed to function
            
        Returns:
            Function result for OpenAI
        """
        # Handle direct list results (from mcp_client.call_tool which returns result.content)
        if isinstance(mcp_result, list):
            # Extract text content from list of TextContent objects
            text_content = []
            serializable_result = []
            
            for item in mcp_result:
                if hasattr(item, 'text'):
                    # TextContent object
                    text_content.append(item.text)
                    serializable_result.append({
                        'type': getattr(item, 'type', 'text'),
                        'text': item.text
                    })
                elif isinstance(item, dict) and 'text' in item:
                    # Already a dictionary
                    text_content.append(item['text'])
                    serializable_result.append(item)
                else:
                    # Unknown format - convert to string
                    text = str(item)
                    text_content.append(text)
                    serializable_result.append({'type': 'text', 'text': text})
            
            message = ' '.join(text_content) if text_content else "Action completed"
            
            return {
                "success": True,
                "function": function_name,
                "message": message,
                "raw_result": serializable_result
            }
        
        # MCP results typically have a content array
        elif isinstance(mcp_result, dict) and 'content' in mcp_result:
            content_items = mcp_result['content']
            
            # Extract text content
            text_content = []
            for item in content_items:
                if item.get('type') == 'text':
                    text_content.append(item.get('text', ''))
            
            message = ' '.join(text_content) if text_content else "Action completed"
            
            # Convert raw_result to serializable format to avoid TextContent serialization errors
            serializable_result = []
            if isinstance(content_items, list):
                for item in content_items:
                    if isinstance(item, dict):
                        serializable_result.append(item)
                    elif hasattr(item, '__dict__'):
                        # Convert TextContent object to dict
                        serializable_result.append({
                            'type': getattr(item, 'type', 'text'),
                            'text': getattr(item, 'text', str(item))
                        })
                    else:
                        serializable_result.append(str(item))
            
            return {
                "success": True,
                "function": function_name,
                "message": message,
                "raw_result": serializable_result
            }
        
        # Handle string results
        elif isinstance(mcp_result, str):
            return {
                "success": True,
                "function": function_name,
                "message": mcp_result
            }
        
        # Handle other result types
        else:
            return {
                "success": True,
                "function": function_name,
                "message": "Action completed successfully",
                "result": mcp_result
            }
    
    def get_tools_summary(self) -> Dict[str, Any]:
        """
        Get summary of available tools
        
        Returns:
            Summary of discovered MCP tools
        """
        return {
            "tool_count": len(self._tools_cache),
            "tools": [
                {
                    "name": tool['name'],
                    "function_name": self._sanitize_function_name(tool['name']),
                    "description": tool.get('description', 'No description')
                }
                for tool in self._tools_cache
            ],
            "primary_tool": self._primary_tool_name
        }
    
    def is_native_mode(self) -> bool:
        """Check if currently using native MCP mode"""
        return self.mode == "native"
    
    def get_mode_info(self) -> Dict[str, Any]:
        """Get information about current MCP mode"""
        info = {
            "mode": self.mode,
            "native_enabled": self.native_mcp and self.native_mcp.enabled if self.native_mcp else False,
        }
        
        if self.native_mcp:
            info["native_metrics"] = self.native_mcp.get_metrics()
        
        return info