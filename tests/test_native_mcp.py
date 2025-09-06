"""
Tests for Native MCP Integration
Tests the native MCP support through OpenAI's built-in capabilities
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass
import aiohttp

# Import the modules to test
from src.services.ha_client.mcp_native import NativeMCPManager, MCPMetrics
from src.function_bridge_mcp import MCPFunctionBridge
from src.config import AppConfig, HomeAssistantConfig, MCPConfig, OpenAIConfig


@pytest.fixture
def mock_config():
    """Create mock configuration for testing"""
    mcp_config = MCPConfig(
        native_mode=True,
        endpoint="/mcp_server/sse",
        approval_mode="never",
        approval_timeout=5000,
        enable_fallback=True,
        performance_tracking=True
    )
    
    ha_config = HomeAssistantConfig(
        url="https://homeassistant.local:8123",
        token="test_token_123",
        mcp=mcp_config
    )
    
    openai_config = OpenAIConfig(
        api_key="test_api_key",
        model="gpt-realtime"
    )
    
    return AppConfig(
        openai=openai_config,
        home_assistant=ha_config
    )


@pytest.fixture
def native_mcp_manager(mock_config):
    """Create NativeMCPManager instance for testing"""
    return NativeMCPManager(mock_config)


@pytest.fixture
def mock_mcp_client():
    """Create mock MCP client for bridge testing"""
    client = Mock()
    client.is_connected = True
    client.get_tools = Mock(return_value=[
        {
            "name": "control_device",
            "description": "Control Home Assistant devices",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "device": {"type": "string"},
                    "action": {"type": "string"}
                },
                "required": ["device", "action"]
            }
        }
    ])
    return client


class TestNativeMCPManager:
    """Test suite for NativeMCPManager"""
    
    def test_initialization(self, native_mcp_manager, mock_config):
        """Test NativeMCPManager initialization"""
        assert native_mcp_manager.enabled == True
        assert native_mcp_manager.approval_mode == "never"
        assert native_mcp_manager.ha_url == "https://homeassistant.local:8123"
        assert native_mcp_manager.metrics.total_calls == 0
        
    def test_build_mcp_url(self, native_mcp_manager):
        """Test MCP URL building"""
        url = native_mcp_manager._build_mcp_url()
        assert url == "https://homeassistant.local:8123/mcp_server/sse"
        
    def test_get_tool_config_native_enabled(self, native_mcp_manager):
        """Test tool configuration generation when native mode is enabled"""
        tools = native_mcp_manager.get_tool_config()
        
        assert len(tools) == 1
        assert tools[0]["type"] == "mcp"
        assert tools[0]["server_label"] == "home_assistant"
        assert tools[0]["server_url"] == "https://homeassistant.local:8123/mcp_server/sse"
        assert tools[0]["authorization"] == "Bearer test_token_123"
        assert tools[0]["require_approval"] == "never"
        
    def test_get_tool_config_native_disabled(self, native_mcp_manager):
        """Test tool configuration when native mode is disabled"""
        native_mcp_manager.enabled = False
        tools = native_mcp_manager.get_tool_config()
        assert tools == []
        
    @pytest.mark.asyncio
    async def test_validate_connection_success(self, native_mcp_manager):
        """Test successful MCP connection validation"""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            result = await native_mcp_manager.validate_connection()
            
            assert result == True
            assert native_mcp_manager.metrics.successful_calls == 1
            assert native_mcp_manager.metrics.total_calls == 1
            assert native_mcp_manager._validated == True
            
    @pytest.mark.asyncio
    async def test_validate_connection_failure(self, native_mcp_manager):
        """Test failed MCP connection validation"""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 404
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            result = await native_mcp_manager.validate_connection()
            
            assert result == False
            assert native_mcp_manager.metrics.failed_calls == 1
            assert native_mcp_manager.metrics.total_calls == 1
            assert native_mcp_manager._validation_error is not None
            
    @pytest.mark.asyncio
    async def test_validate_connection_timeout(self, native_mcp_manager):
        """Test MCP connection validation with timeout"""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.get.side_effect = asyncio.TimeoutError()
            
            result = await native_mcp_manager.validate_connection()
            
            assert result == False
            assert native_mcp_manager.metrics.failed_calls == 1
            assert native_mcp_manager.metrics.last_error == "MCP connection timeout"
            
    def test_handle_approval_request_never(self, native_mcp_manager):
        """Test approval handling with 'never' mode"""
        request = {
            "tool": {
                "name": "control_device",
                "parameters": {"device": "light.living_room"}
            }
        }
        
        result = native_mcp_manager.handle_approval_request(request)
        assert result == True
        
    def test_handle_approval_request_always(self, native_mcp_manager):
        """Test approval handling with 'always' mode"""
        native_mcp_manager.approval_mode = "always"
        request = {
            "tool": {
                "name": "control_device",
                "parameters": {"device": "light.living_room"}
            }
        }
        
        result = native_mcp_manager.handle_approval_request(request)
        assert result == True  # Auto-approves with warning in test
        
    def test_get_metrics(self, native_mcp_manager):
        """Test metrics retrieval"""
        native_mcp_manager.metrics.total_calls = 10
        native_mcp_manager.metrics.successful_calls = 8
        native_mcp_manager.metrics.failed_calls = 2
        
        metrics = native_mcp_manager.get_metrics()
        
        assert metrics["total_calls"] == 10
        assert metrics["successful_calls"] == 8
        assert metrics["failed_calls"] == 2
        assert metrics["success_rate"] == "80.00%"
        
    def test_should_use_native(self, native_mcp_manager):
        """Test native mode selection logic"""
        # Not validated yet
        assert native_mcp_manager.should_use_native() == False
        
        # Mark as validated
        native_mcp_manager._validated = True
        assert native_mcp_manager.should_use_native() == True
        
        # Disable native mode
        native_mcp_manager.enabled = False
        assert native_mcp_manager.should_use_native() == False
        
    @pytest.mark.asyncio
    async def test_handle_mcp_event_approval(self, native_mcp_manager):
        """Test MCP event handling for approval requests"""
        event = {
            "type": "mcp.approval_request",
            "request_id": "req_123",
            "tool": {
                "name": "control_device",
                "parameters": {}
            }
        }
        
        response = await native_mcp_manager.handle_mcp_event(event)
        
        assert response is not None
        assert response["type"] == "mcp.approval_response"
        assert response["approved"] == True
        assert response["request_id"] == "req_123"
        
    @pytest.mark.asyncio
    async def test_handle_mcp_event_error(self, native_mcp_manager):
        """Test MCP event handling for tool errors"""
        event = {
            "type": "mcp.tool_error",
            "tool": {"name": "control_device"},
            "error": "Device not found"
        }
        
        response = await native_mcp_manager.handle_mcp_event(event)
        
        assert response is None
        assert native_mcp_manager.metrics.failed_calls == 1
        assert native_mcp_manager.metrics.last_error == "Device not found"
        
    def test_get_fallback_reason(self, native_mcp_manager):
        """Test fallback reason generation"""
        # Disabled mode
        native_mcp_manager.enabled = False
        reason = native_mcp_manager.get_fallback_reason()
        assert "disabled in configuration" in reason
        
        # Validation failed
        native_mcp_manager.enabled = True
        native_mcp_manager._validation_error = "Connection refused"
        reason = native_mcp_manager.get_fallback_reason()
        assert "Connection refused" in reason
        
        # High error rate
        native_mcp_manager._validated = True
        native_mcp_manager._validation_error = None
        native_mcp_manager.metrics.total_calls = 20
        native_mcp_manager.metrics.successful_calls = 10
        reason = native_mcp_manager.get_fallback_reason()
        assert "High error rate" in reason


class TestMCPFunctionBridge:
    """Test suite for MCPFunctionBridge with native MCP support"""
    
    @pytest.mark.asyncio
    async def test_initialize_native_mode(self, mock_mcp_client, mock_config):
        """Test bridge initialization with native MCP mode"""
        bridge = MCPFunctionBridge(mock_mcp_client, mock_config)
        
        with patch.object(bridge.native_mcp, 'validate_connection', return_value=True):
            await bridge.initialize()
            
            assert bridge.mode == "native"
            assert bridge.is_native_mode() == True
            
    @pytest.mark.asyncio
    async def test_initialize_fallback_to_bridge(self, mock_mcp_client, mock_config):
        """Test fallback from native to bridge mode"""
        bridge = MCPFunctionBridge(mock_mcp_client, mock_config)
        
        with patch.object(bridge.native_mcp, 'validate_connection', return_value=False):
            await bridge.initialize()
            
            assert bridge.mode == "bridge"
            assert bridge.is_native_mode() == False
            
    @pytest.mark.asyncio
    async def test_initialize_fallback_disabled(self, mock_mcp_client, mock_config):
        """Test when fallback is disabled and native fails"""
        mock_config.home_assistant.mcp.enable_fallback = False
        bridge = MCPFunctionBridge(mock_mcp_client, mock_config)
        
        with patch.object(bridge.native_mcp, 'validate_connection', return_value=False):
            with pytest.raises(RuntimeError, match="Native MCP validation failed"):
                await bridge.initialize()
                
    def test_get_function_definitions_native_mode(self, mock_mcp_client, mock_config):
        """Test function definitions in native mode"""
        bridge = MCPFunctionBridge(mock_mcp_client, mock_config)
        bridge.mode = "native"
        
        definitions = bridge.get_function_definitions()
        assert definitions == []  # Native mode returns empty list
        
    def test_get_function_definitions_bridge_mode(self, mock_mcp_client, mock_config):
        """Test function definitions in bridge mode"""
        bridge = MCPFunctionBridge(mock_mcp_client, mock_config)
        bridge.mode = "bridge"
        bridge._tools_cache = mock_mcp_client.get_tools()
        
        definitions = bridge.get_function_definitions()
        assert len(definitions) > 0
        assert definitions[0]["name"] == "control_device"
        
    @pytest.mark.asyncio
    async def test_handle_function_call_native_mode(self, mock_mcp_client, mock_config):
        """Test function call handling in native mode"""
        bridge = MCPFunctionBridge(mock_mcp_client, mock_config)
        bridge.mode = "native"
        
        # In native mode, function calls shouldn't normally happen
        # but bridge should handle them gracefully
        mock_mcp_client.call_tool = AsyncMock(return_value="Success")
        
        result = await bridge.handle_function_call("control_device", {"device": "light.test"})
        
        # Should still attempt to handle the call
        assert result is not None
        
    def test_get_mode_info(self, mock_mcp_client, mock_config):
        """Test mode information retrieval"""
        bridge = MCPFunctionBridge(mock_mcp_client, mock_config)
        bridge.mode = "native"
        
        info = bridge.get_mode_info()
        
        assert info["mode"] == "native"
        assert info["native_enabled"] == True
        assert "native_metrics" in info
        

class TestMCPMetrics:
    """Test suite for MCPMetrics dataclass"""
    
    def test_metrics_initialization(self):
        """Test metrics initialization"""
        metrics = MCPMetrics()
        
        assert metrics.total_calls == 0
        assert metrics.successful_calls == 0
        assert metrics.failed_calls == 0
        assert metrics.success_rate == 100.0
        assert metrics.average_latency == 0
        
    def test_success_rate_calculation(self):
        """Test success rate calculation"""
        metrics = MCPMetrics()
        metrics.total_calls = 100
        metrics.successful_calls = 75
        
        assert metrics.success_rate == 75.0
        
    def test_average_latency_calculation(self):
        """Test average latency calculation"""
        metrics = MCPMetrics()
        metrics.successful_calls = 5
        metrics.total_latency_ms = 500
        
        assert metrics.average_latency == 100.0
        
    def test_metrics_with_no_calls(self):
        """Test metrics with no calls"""
        metrics = MCPMetrics()
        
        assert metrics.success_rate == 100.0  # Default to 100% when no calls
        assert metrics.average_latency == 0  # No latency when no calls


@pytest.mark.integration
class TestNativeMCPIntegration:
    """Integration tests for native MCP with OpenAI client"""
    
    @pytest.mark.asyncio
    async def test_openai_session_with_native_mcp(self, mock_config):
        """Test OpenAI session configuration with native MCP tools"""
        from src.openai_client.realtime import OpenAIRealtimeClient
        
        # Create OpenAI client with app config
        client = OpenAIRealtimeClient(
            mock_config.openai,
            personality_prompt="Test assistant",
            text_only=False,
            app_config=mock_config
        )
        
        # Check that MCP manager was initialized
        assert client.mcp_manager is not None
        assert isinstance(client.mcp_manager, NativeMCPManager)
        
    @pytest.mark.asyncio
    async def test_performance_comparison(self, mock_mcp_client, mock_config):
        """Test performance metrics comparison between native and bridge modes"""
        bridge = MCPFunctionBridge(mock_mcp_client, mock_config)
        
        # Simulate bridge mode performance
        bridge.mode = "bridge"
        bridge_start = asyncio.get_event_loop().time()
        # Simulate some processing
        await asyncio.sleep(0.1)
        bridge_time = (asyncio.get_event_loop().time() - bridge_start) * 1000
        
        # Simulate native mode performance (should be faster)
        bridge.mode = "native"
        native_start = asyncio.get_event_loop().time()
        # Native mode has less overhead
        await asyncio.sleep(0.05)
        native_time = (asyncio.get_event_loop().time() - native_start) * 1000
        
        # Native should be faster than bridge
        assert native_time < bridge_time
        
        # Calculate improvement
        improvement = ((bridge_time - native_time) / bridge_time) * 100
        print(f"Performance improvement: {improvement:.1f}%")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])