# Native MCP Integration Setup Guide

## Overview

Native MCP (Model Context Protocol) integration allows OpenAI's Realtime API to directly communicate with Home Assistant's MCP server, eliminating the need for a custom bridge implementation. This results in:

- **Reduced Latency**: >20% improvement by removing translation layer
- **Simplified Architecture**: Direct OpenAI-to-Home Assistant communication
- **Better Reliability**: Fewer points of failure
- **Automatic Tool Discovery**: OpenAI discovers available tools dynamically

## Prerequisites

### Home Assistant Requirements
- Home Assistant version 2025.2 or later
- MCP server integration installed and configured
- Long-lived access token with appropriate permissions

### OpenAI Requirements
- OpenAI API key with Realtime API access
- Model that supports native MCP (`gpt-realtime` or later)

## Configuration

### 1. Enable Native MCP Mode

Edit your `config/config.yaml` file and update the MCP section:

```yaml
home_assistant:
  url: "https://your-homeassistant.local:8123"
  token: ${HA_TOKEN}
  
  mcp:
    # Enable native MCP support
    native_mode: true
    
    # MCP server endpoint (usually /mcp_server/sse)
    endpoint: "/mcp_server/sse"
    
    # Tool approval mode
    approval_mode: "never"  # Options: "never", "always", "on_error"
    
    # Fallback to bridge mode if native fails
    enable_fallback: true
    
    # Performance tracking
    performance_tracking: true
```

### 2. Configuration Options Explained

#### `native_mode`
- **Type**: Boolean
- **Default**: `false`
- **Description**: Enables native MCP integration through OpenAI

#### `endpoint`
- **Type**: String
- **Default**: `/mcp_server/sse`
- **Description**: MCP server endpoint path on your Home Assistant instance

#### `approval_mode`
- **Type**: String
- **Options**: `"never"`, `"always"`, `"on_error"`
- **Default**: `"never"`
- **Description**: Controls when tool calls require approval
  - `"never"`: All tool calls are automatically approved
  - `"always"`: Every tool call requires manual approval
  - `"on_error"`: Only failed tool calls require approval

#### `approval_timeout`
- **Type**: Integer
- **Default**: `5000`
- **Description**: Milliseconds to wait for approval (when approval_mode != "never")

#### `enable_fallback`
- **Type**: Boolean
- **Default**: `true`
- **Description**: Automatically fallback to bridge mode if native MCP connection fails

#### `performance_tracking`
- **Type**: Boolean
- **Default**: `true`
- **Description**: Track performance metrics for native MCP calls

#### `cache_tool_definitions`
- **Type**: Boolean
- **Default**: `true`
- **Description**: Cache discovered tools for improved performance

#### `tool_timeout`
- **Type**: Integer
- **Default**: `30000`
- **Description**: Tool execution timeout in milliseconds

## Migration from Bridge Mode

### Step 1: Test Native Mode
Before fully migrating, test native mode alongside your existing setup:

```yaml
mcp:
  native_mode: true
  enable_fallback: true  # Ensures automatic fallback if issues arise
```

### Step 2: Monitor Performance
Check the logs for performance metrics:

```bash
grep "MCP Mode" logs/assistant.log
grep "Native MCP" logs/assistant.log
```

### Step 3: Verify Tool Discovery
Native mode should automatically discover all Home Assistant tools:

```bash
# Check discovered tools in logs
grep "Using native MCP tools" logs/assistant.log
```

### Step 4: Disable Fallback (Optional)
Once confident in native mode stability:

```yaml
mcp:
  native_mode: true
  enable_fallback: false  # Disable fallback for pure native mode
```

## Troubleshooting

### Connection Validation Failed

If you see "Native MCP validation failed" in logs:

1. **Verify Home Assistant URL**: Ensure the URL is accessible from the assistant
   ```bash
   curl -H "Authorization: Bearer YOUR_TOKEN" https://your-ha.local:8123/mcp_server/sse
   ```

2. **Check SSL Certificates**: For self-signed certificates, set:
   ```yaml
   mcp:
     ssl_verify: false
   ```

3. **Verify MCP Server**: Ensure the MCP integration is installed in Home Assistant

### High Error Rate

If native mode shows high error rates:

1. **Check Logs**: Look for specific error patterns
   ```bash
   grep "MCP tool error" logs/assistant.log
   ```

2. **Enable Fallback**: Temporarily enable fallback while debugging
   ```yaml
   enable_fallback: true
   ```

3. **Adjust Timeouts**: Increase timeouts for slower networks
   ```yaml
   tool_timeout: 60000  # 60 seconds
   ```

### Tools Not Discovered

If OpenAI doesn't discover Home Assistant tools:

1. **Verify MCP Endpoint**: Ensure the endpoint is correct
2. **Check Authentication**: Verify the token has necessary permissions
3. **Test Manual Discovery**: Temporarily disable native mode to test bridge mode

## Performance Monitoring

### View Metrics

The native MCP manager tracks performance metrics:

```python
# Metrics available in logs
- Total calls
- Successful calls
- Failed calls
- Success rate (%)
- Average latency (ms)
```

### Compare Modes

To compare native vs bridge performance:

1. Run with bridge mode for a period
2. Switch to native mode
3. Compare metrics in logs:

```bash
# Bridge mode metrics
grep "bridge mode" logs/assistant.log | grep latency

# Native mode metrics
grep "native mode" logs/assistant.log | grep latency
```

## Security Considerations

### Approval Workflows

For sensitive environments, enable approval mode:

```yaml
mcp:
  approval_mode: "always"  # Require approval for all tool calls
  approval_timeout: 10000  # 10 seconds to approve
```

### Token Security

- Use environment variables for tokens: `${HA_TOKEN}`
- Rotate tokens regularly
- Use tokens with minimal required permissions

### SSL/TLS

For production environments:

```yaml
mcp:
  ssl_verify: true
  ssl_ca_bundle: "/path/to/ca-bundle.crt"  # Custom CA if needed
```

## Advanced Configuration

### Custom Approval Logic

For advanced users, approval logic can be customized by modifying `mcp_native.py`:

```python
def handle_approval_request(self, request: Dict[str, Any]) -> bool:
    tool_name = request.get("tool", {}).get("name")
    
    # Custom logic based on tool type
    if "delete" in tool_name.lower() or "remove" in tool_name.lower():
        # Always require approval for destructive operations
        return False  # Deny by default
    
    return True  # Approve other operations
```

### Performance Tuning

Optimize for your network conditions:

```yaml
mcp:
  # Fast local network
  tool_timeout: 5000      # 5 seconds
  connection_timeout: 10  # 10 seconds
  
  # OR slow/remote network
  tool_timeout: 60000     # 60 seconds
  connection_timeout: 60  # 60 seconds
```

## Rollback Procedure

If you need to rollback to bridge mode:

1. **Disable Native Mode**:
   ```yaml
   mcp:
     native_mode: false
   ```

2. **Restart the Assistant**:
   ```bash
   systemctl restart ha-realtime-assist
   ```

3. **Verify Bridge Mode**:
   ```bash
   grep "Mode: bridge" logs/assistant.log
   ```

## Support

For issues or questions:

1. Check logs: `logs/assistant.log`
2. Review this documentation
3. Check Home Assistant MCP server status
4. Verify network connectivity between assistant and Home Assistant

## Appendix: Architecture Comparison

### Bridge Mode (Traditional)
```
User → OpenAI → function_bridge_mcp.py → MCP Client → Home Assistant
         ↑                                    ↓
         └────── Function Results ←───────────┘
```

### Native Mode (New)
```
User → OpenAI (with native MCP) → Home Assistant
         ↑                              ↓
         └──── Direct MCP Response ←────┘
```

The native mode eliminates the intermediate translation layer, resulting in faster response times and simpler architecture.