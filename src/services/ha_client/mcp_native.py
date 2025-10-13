"""
Native MCP Integration Client for Home Assistant
Implements direct MCP support through OpenAI's native capabilities
"""

import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import aiohttp
from config import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class MCPMetrics:
    """Metrics for MCP performance tracking"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_latency_ms: float = 0
    last_error: Optional[str] = None
    last_check: float = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_calls == 0:
            return 100.0
        return (self.successful_calls / self.total_calls) * 100
    
    @property
    def average_latency(self) -> float:
        """Calculate average latency in milliseconds"""
        if self.successful_calls == 0:
            return 0
        return self.total_latency_ms / self.successful_calls


class NativeMCPManager:
    """
    Manages native MCP connection through OpenAI's built-in support
    Replaces custom bridge implementation with direct integration
    """
    
    def __init__(self, config: AppConfig):
        """
        Initialize Native MCP Manager

        Args:
            config: Application configuration
        """
        self.config = config

        # Check if Home Assistant is configured
        if not config.home_assistant or not config.home_assistant.url:
            # Home Assistant not configured - disable native MCP
            self.ha_url = None
            self.ha_token = None
            self.enabled = False
            self.approval_mode = 'never'
            self.approval_timeout = 5000
            self.mcp_endpoint = '/mcp_server/sse'
        else:
            self.ha_url = config.home_assistant.url
            self.ha_token = config.home_assistant.token

            # Native MCP configuration
            self.enabled = getattr(config.home_assistant.mcp, 'native_mode', False)
            self.approval_mode = getattr(config.home_assistant.mcp, 'approval_mode', 'never')
            self.approval_timeout = getattr(config.home_assistant.mcp, 'approval_timeout', 5000)
            self.mcp_endpoint = getattr(config.home_assistant.mcp, 'endpoint', '/mcp_server/sse')
        
        # Performance tracking
        self.metrics = MCPMetrics()
        
        # Connection state
        self._validated = False
        self._validation_error = None
        
        logger.info(f"Native MCP Manager initialized - Mode: {'ENABLED' if self.enabled else 'DISABLED'}")
    
    def get_tool_config(self) -> List[Dict[str, Any]]:
        """
        Generate MCP tool configuration for OpenAI session
        
        Returns:
            List of MCP tool configurations (empty if disabled)
        """
        if not self.enabled:
            logger.debug("Native MCP disabled, returning empty tool config")
            return []
        
        # Build MCP configuration for OpenAI
        mcp_config = {
            "type": "mcp",
            "server_label": "home_assistant",
            "server_url": self._build_mcp_url(),
            "authorization": f"Bearer {self.ha_token}",
            "require_approval": self.approval_mode,
            "timeout": 30000,  # 30 seconds timeout
            "retry_policy": {
                "max_attempts": 3,
                "backoff_multiplier": 2,
                "initial_delay": 1000  # 1 second
            },
            "capabilities": {
                "supports_context": True,
                "supports_tools": True,
                "supports_streaming": False,
                "supports_approval": True
            }
        }
        
        # Add approval timeout if approval is enabled
        if self.approval_mode != "never":
            mcp_config["approval_timeout"] = self.approval_timeout
        
        logger.info(f"Generated native MCP tool config for {mcp_config['server_url']}")
        return [mcp_config]
    
    def _build_mcp_url(self) -> str:
        """
        Build complete MCP server URL
        
        Returns:
            Full URL to MCP endpoint
        """
        # Remove trailing slash from base URL if present
        base_url = self.ha_url.rstrip('/')
        # Ensure endpoint starts with /
        endpoint = self.mcp_endpoint if self.mcp_endpoint.startswith('/') else f"/{self.mcp_endpoint}"
        
        full_url = f"{base_url}{endpoint}"
        logger.debug(f"Built MCP URL: {full_url}")
        return full_url
    
    async def validate_connection(self) -> bool:
        """
        Test MCP server availability and configuration
        
        Returns:
            True if connection is valid, False otherwise
        """
        if not self.enabled:
            logger.debug("Native MCP disabled, skipping validation")
            return True
        
        mcp_url = self._build_mcp_url()
        logger.info(f"Validating MCP connection to {mcp_url}")
        
        try:
            start_time = time.time()
            
            # Test the MCP endpoint with a simple request
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.ha_token}",
                    "Accept": "text/event-stream"
                }
                
                # Try to connect to the SSE endpoint
                async with session.get(mcp_url, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        # Successfully connected
                        latency_ms = (time.time() - start_time) * 1000
                        self.metrics.total_latency_ms += latency_ms
                        self.metrics.successful_calls += 1
                        self.metrics.total_calls += 1
                        self.metrics.last_check = time.time()
                        
                        self._validated = True
                        self._validation_error = None
                        
                        logger.info(f"MCP connection validated successfully (latency: {latency_ms:.2f}ms)")
                        return True
                    else:
                        error_msg = f"MCP validation failed with status {response.status}"
                        logger.error(error_msg)
                        self._validation_error = error_msg
                        self.metrics.failed_calls += 1
                        self.metrics.total_calls += 1
                        self.metrics.last_error = error_msg
                        return False
                        
        except aiohttp.ClientTimeout:
            error_msg = "MCP connection timeout"
            logger.error(error_msg)
            self._validation_error = error_msg
            self.metrics.failed_calls += 1
            self.metrics.total_calls += 1
            self.metrics.last_error = error_msg
            return False
            
        except Exception as e:
            error_msg = f"MCP validation error: {str(e)}"
            logger.error(error_msg)
            self._validation_error = error_msg
            self.metrics.failed_calls += 1
            self.metrics.total_calls += 1
            self.metrics.last_error = error_msg
            return False
    
    def handle_approval_request(self, request: Dict[str, Any]) -> bool:
        """
        Handle tool approval request from OpenAI
        
        Args:
            request: Approval request details from OpenAI
            
        Returns:
            True to approve, False to deny
        """
        if self.approval_mode == "never":
            # Auto-approve all requests
            logger.debug("Auto-approving MCP tool request (approval_mode: never)")
            return True
        
        tool_name = request.get("tool", {}).get("name", "unknown")
        tool_params = request.get("tool", {}).get("parameters", {})
        
        logger.info(f"MCP approval request for tool: {tool_name}")
        logger.debug(f"Tool parameters: {tool_params}")
        
        if self.approval_mode == "always":
            # In production, this would integrate with a UI component
            # For now, we'll auto-approve with logging
            logger.warning(f"Approval required for {tool_name} - auto-approving (configure UI for manual approval)")
            return True
        
        elif self.approval_mode == "on_error":
            # Check if this is a retry after an error
            if request.get("retry_after_error", False):
                logger.info(f"Approval requested after error for {tool_name}")
                # Could implement logic to check error type and decide
                return True
            else:
                # Not an error case, auto-approve
                return True
        
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for native MCP
        
        Returns:
            Dictionary of performance metrics
        """
        return {
            "enabled": self.enabled,
            "validated": self._validated,
            "validation_error": self._validation_error,
            "total_calls": self.metrics.total_calls,
            "successful_calls": self.metrics.successful_calls,
            "failed_calls": self.metrics.failed_calls,
            "success_rate": f"{self.metrics.success_rate:.2f}%",
            "average_latency_ms": f"{self.metrics.average_latency:.2f}",
            "last_error": self.metrics.last_error,
            "last_check": self.metrics.last_check,
            "approval_mode": self.approval_mode
        }
    
    def should_use_native(self) -> bool:
        """
        Determine if native MCP should be used
        
        Returns:
            True if native mode is enabled and validated
        """
        if not self.enabled:
            return False
        
        # If we haven't validated yet, try to validate
        if not self._validated and self._validation_error is None:
            # This would normally be async, but for config check we return current state
            logger.debug("Native MCP not yet validated, defaulting to bridge mode")
            return False
        
        return self._validated
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics = MCPMetrics()
        logger.info("Native MCP metrics reset")
    
    async def handle_mcp_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handle MCP-related events from OpenAI
        
        Args:
            event: Event data from OpenAI
            
        Returns:
            Response data if needed, None otherwise
        """
        event_type = event.get("type", "")
        
        if event_type == "mcp.approval_request":
            # Handle approval request
            approved = self.handle_approval_request(event)
            return {
                "type": "mcp.approval_response",
                "approved": approved,
                "request_id": event.get("request_id")
            }
        
        elif event_type == "mcp.tool_error":
            # Log tool errors for debugging
            tool_name = event.get("tool", {}).get("name", "unknown")
            error = event.get("error", "Unknown error")
            logger.error(f"MCP tool error for {tool_name}: {error}")
            
            self.metrics.failed_calls += 1
            self.metrics.total_calls += 1
            self.metrics.last_error = error
            
        elif event_type == "mcp.tool_success":
            # Track successful tool calls
            tool_name = event.get("tool", {}).get("name", "unknown")
            latency_ms = event.get("latency_ms", 0)
            
            self.metrics.successful_calls += 1
            self.metrics.total_calls += 1
            self.metrics.total_latency_ms += latency_ms
            
            logger.debug(f"MCP tool {tool_name} succeeded (latency: {latency_ms}ms)")
        
        return None
    
    def get_fallback_reason(self) -> Optional[str]:
        """
        Get reason for fallback to bridge mode
        
        Returns:
            Reason string if fallback is needed, None otherwise
        """
        if not self.enabled:
            return "Native MCP mode disabled in configuration"
        
        if not self._validated:
            if self._validation_error:
                return f"MCP validation failed: {self._validation_error}"
            else:
                return "MCP connection not yet validated"
        
        # Check if error rate is too high
        if self.metrics.total_calls > 10 and self.metrics.success_rate < 80:
            return f"High error rate detected ({100 - self.metrics.success_rate:.1f}% failures)"
        
        return None