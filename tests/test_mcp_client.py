"""Tests for MCP client implementation."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from aiohttp import ClientSession
from aiohttp_sse_client import client as sse_client

from src.services.ha_client.mcp import MCPClient, MCPError, MCPConnectionError, MCPProtocolError


@pytest.fixture
async def mcp_client():
    """Create MCP client instance."""
    client = MCPClient(
        base_url="http://localhost:8123",
        access_token="test_token",
        connection_timeout=5,
        reconnect_attempts=2
    )
    yield client
    if client.is_connected:
        await client.disconnect()


@pytest.mark.asyncio
async def test_client_initialization(mcp_client):
    """Test MCP client initialization."""
    assert mcp_client.base_url == "http://localhost:8123"
    assert mcp_client.access_token == "test_token"
    assert mcp_client.sse_url == "http://localhost:8123/mcp_server/sse"
    assert not mcp_client.is_connected


@pytest.mark.asyncio
async def test_successful_connection():
    """Test successful connection to MCP server."""
    with patch('aiohttp_sse_client.client.EventSource') as mock_sse:
        # Mock SSE client
        mock_event_source = AsyncMock()
        mock_sse.return_value = mock_event_source
        
        # Mock the async iterator for messages
        async def mock_messages():
            # Simulate initialize response
            yield MagicMock(type='message', data=json.dumps({
                "jsonrpc": "2.0",
                "result": {
                    "protocolVersion": "0.1.0",
                    "capabilities": {"tools": {}}
                },
                "id": "test-id"
            }))
            # Simulate tools/list response
            yield MagicMock(type='message', data=json.dumps({
                "jsonrpc": "2.0",
                "result": {
                    "tools": [
                        {
                            "name": "control_device",
                            "description": "Control a device",
                            "inputSchema": {"type": "object"}
                        }
                    ]
                },
                "id": "test-id-2"
            }))
            # Keep the connection alive
            while True:
                await asyncio.sleep(1)
        
        mock_event_source.__aiter__.return_value = mock_messages()
        
        # Mock session and POST requests
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            # Mock POST responses for requests
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_session.post.return_value.__aenter__.return_value = mock_response
            
            # Create client and set up request tracking
            client = MCPClient("http://localhost:8123", "test_token")
            
            # Mock _send_request to return expected responses
            original_send = client._send_request
            call_count = 0
            
            async def mock_send_request(method, params=None):
                nonlocal call_count
                call_count += 1
                
                if method == "initialize":
                    client._response_futures["test-id"] = asyncio.Future()
                    client._response_futures["test-id"].set_result({
                        "protocolVersion": "0.1.0",
                        "capabilities": {"tools": {}}
                    })
                    return await client._response_futures["test-id"]
                    
                elif method == "tools/list":
                    client._response_futures["test-id-2"] = asyncio.Future()
                    client._response_futures["test-id-2"].set_result({
                        "tools": [{
                            "name": "control_device",
                            "description": "Control a device",
                            "inputSchema": {"type": "object"}
                        }]
                    })
                    return await client._response_futures["test-id-2"]
                    
                return await original_send(method, params)
            
            client._send_request = mock_send_request
            
            # Connect
            await client.connect()
            
            # Verify connection
            assert client.is_connected
            assert len(client.get_tools()) == 1
            assert client.get_tool("control_device") is not None
            
            # Cleanup
            await client.disconnect()


@pytest.mark.asyncio
async def test_connection_failure(mcp_client):
    """Test connection failure handling."""
    with patch('aiohttp_sse_client.client.EventSource') as mock_sse:
        mock_sse.side_effect = Exception("Connection failed")
        
        with pytest.raises(MCPConnectionError):
            await mcp_client.connect()
        
        assert not mcp_client.is_connected


@pytest.mark.asyncio
async def test_protocol_error_handling():
    """Test handling of protocol errors."""
    client = MCPClient("http://localhost:8123", "test_token")
    
    # Simulate error response
    error_data = {
        "jsonrpc": "2.0",
        "error": {
            "code": -32601,
            "message": "Method not found"
        },
        "id": "test-id"
    }
    
    # Create a future and simulate error
    future = asyncio.Future()
    client._response_futures["test-id"] = future
    
    await client._handle_message(json.dumps(error_data))
    
    with pytest.raises(MCPProtocolError) as exc_info:
        await future
    
    assert exc_info.value.code == -32601
    assert "Method not found" in str(exc_info.value)


@pytest.mark.asyncio
async def test_tool_invocation():
    """Test calling a tool through MCP."""
    client = MCPClient("http://localhost:8123", "test_token")
    client._connected = True
    
    # Mock _send_request
    async def mock_send_request(method, params=None):
        if method == "tools/call":
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Turned on the lights"
                    }
                ]
            }
        return {}
    
    client._send_request = mock_send_request
    
    # Call tool
    result = await client.call_tool("control_device", {"command": "turn on lights"})
    
    assert result["content"][0]["text"] == "Turned on the lights"


@pytest.mark.asyncio
async def test_event_handlers():
    """Test event handler registration and notification handling."""
    client = MCPClient("http://localhost:8123", "test_token")
    
    # Track handler calls
    handler_called = False
    handler_params = None
    
    async def test_handler(params):
        nonlocal handler_called, handler_params
        handler_called = True
        handler_params = params
    
    # Register handler
    client.on_event("test_event", test_handler)
    
    # Simulate notification
    await client._handle_notification("test_event", {"data": "test"})
    
    assert handler_called
    assert handler_params == {"data": "test"}
    
    # Unregister handler
    client.off_event("test_event", test_handler)
    assert "test_event" not in client._event_handlers


@pytest.mark.asyncio
async def test_reconnection_logic():
    """Test automatic reconnection on connection loss."""
    client = MCPClient("http://localhost:8123", "test_token", reconnect_attempts=2)
    
    # Mock connection methods
    connect_attempts = 0
    
    async def mock_establish_connection():
        nonlocal connect_attempts
        connect_attempts += 1
        if connect_attempts < 2:
            raise Exception("Connection failed")
        # Success on second attempt
        client._sse_client = AsyncMock()
        client._message_task = asyncio.create_task(asyncio.sleep(0))
    
    client._establish_connection = mock_establish_connection
    client._initialize_protocol = AsyncMock()
    
    # Should succeed after retry
    await client.connect()
    
    assert connect_attempts == 2
    assert client.is_connected


@pytest.mark.asyncio
async def test_disconnect():
    """Test proper cleanup on disconnect."""
    client = MCPClient("http://localhost:8123", "test_token")
    
    # Mock connection components
    client._connected = True
    client._session = AsyncMock()
    client._sse_client = AsyncMock()
    client._message_task = asyncMock()
    client._message_task.done.return_value = False
    
    # Add some state
    client._response_futures["test"] = asyncio.Future()
    client._tools = [{"name": "test_tool"}]
    
    # Disconnect
    await client.disconnect()
    
    # Verify cleanup
    assert not client.is_connected
    assert len(client._response_futures) == 0
    assert len(client._tools) == 0
    assert client._session.close.called