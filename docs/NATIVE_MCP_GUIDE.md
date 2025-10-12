# Hybrid MCP Architecture Guide

**OpenVoice MCP v2.0.0 - Hybrid Native/Client-Side MCP Support**

## Overview

OpenVoice MCP v2 implements a **hybrid MCP architecture** that combines OpenAI's native MCP support with client-side MCP handling. This provides the best of both worlds:

- **Native Mode**: OpenAI manages remote MCP server connections directly (low latency, scalable)
- **Client Mode**: Local MCP servers run as subprocesses (supports stdio transport, local tools)

This architecture was introduced in October 2025 following OpenAI's August 2025 Realtime API update that added native MCP protocol support.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   OpenVoice MCP Application                  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐         ┌────────────────────────┐   │
│  │ Native MCP Mode  │         │ Client MCP Mode        │   │
│  │ (Remote Servers) │         │ (Local Servers)        │   │
│  └────────┬─────────┘         └──────────┬─────────────┘   │
│           │                               │                  │
│           │ Session Config                │ Tool Registration│
│           │ w/ MCP Tools                  │ as Functions     │
│           │                               │                  │
│  ┌────────▼───────────────────────────────▼─────────────┐   │
│  │         OpenAI Realtime API Client                   │   │
│  │         (realtime.py)                                │   │
│  └────────┬───────────────────────────────┬─────────────┘   │
│           │                               │                  │
└───────────┼───────────────────────────────┼─────────────────┘
            │                               │
            │ WebSocket                     │ stdio/subprocess
            │                               │
    ┌───────▼─────────┐           ┌─────────▼──────────┐
    │  OpenAI Servers │           │  Local MCP Servers │
    │  (Native MCP)   │           │  (ClientMCPManager)│
    └────────┬────────┘           └─────────┬──────────┘
             │                               │
             │ HTTP/SSE                      │ subprocess
             │                               │
    ┌────────▼────────┐           ┌─────────▼──────────┐
    │  Remote MCP     │           │  Local MCP         │
    │  Servers        │           │  Servers           │
    │  (e.g., HA)     │           │  (e.g., filesystem)│
    └─────────────────┘           └────────────────────┘
```

## When to Use Each Mode

### Native Mode (Recommended for Remote Servers)
**Use when:**
- MCP server is accessible via HTTP/HTTPS
- Server supports SSE or Streamable HTTP transport
- You want lowest latency (OpenAI manages connection)
- Server is on remote infrastructure (Home Assistant, cloud APIs)

**Benefits:**
- Lower latency (direct connection from OpenAI)
- Scalable (OpenAI manages connection pool)
- Simplified error handling
- Built-in approval workflows

**Example Servers:**
- Home Assistant MCP Server (remote HA instance)
- Cloud-based API services
- Remote database connections

### Client Mode (Required for Local Servers)
**Use when:**
- MCP server uses stdio transport (subprocess)
- Server must run locally (filesystem access, local resources)
- Server requires local environment/credentials
- You need more control over server lifecycle

**Benefits:**
- Supports stdio transport
- Access to local resources
- Full control over server execution
- Custom environment variables

**Example Servers:**
- MCP Filesystem Server (local file access)
- Local database tools
- System administration tools
- Development/debugging servers

## Configuration

### Basic Configuration Structure

```yaml
mcp_servers:
  # Native mode server (remote)
  home_assistant:
    mode: native
    enabled: true
    server_url: https://homeassistant.local/mcp_server/sse
    authorization: Bearer your_ha_token_here
    description: Home Assistant integration
    require_approval: never  # or "always", or dict with tool-specific rules
    allowed_tools:  # Optional: filter which tools to expose
      - get_state
      - call_service
    priority: 100  # Lower = higher priority

  # Client mode server (local)
  filesystem:
    mode: client
    enabled: true
    transport: stdio
    command: uvx
    args:
      - mcp-server-filesystem
      - /Users/username/Documents
    env:
      MCP_DEBUG: "1"
    timeout: 30
    priority: 200
```

### Configuration Fields

#### Common Fields (All Servers)
- **`name`** (string, auto-set): Server identifier
- **`mode`** (string, required): `"native"` or `"client"`
- **`enabled`** (boolean): Enable/disable server (default: `true`)
- **`priority`** (integer): Tool routing priority (default: `100`, lower = higher priority)

#### Native Mode Fields
- **`server_url`** (string, required): MCP server endpoint URL
- **`authorization`** (string, optional): Authorization header value (e.g., `"Bearer token"`)
- **`description`** (string, optional): Human-readable description
- **`require_approval`** (string or dict, default: `"always"`):
  - `"always"`: Always require approval for all tools
  - `"never"`: Auto-approve all tools
  - `{"tool_name": "never", "other_tool": "always"}`: Tool-specific rules
- **`allowed_tools`** (array, optional): Whitelist of tool names to expose

#### Client Mode Fields
- **`transport`** (string, required): Transport type (currently only `"stdio"` supported)
- **`command`** (string, required for stdio): Command to execute
- **`args`** (array, optional): Command arguments
- **`env`** (dict, optional): Environment variables
- **`timeout`** (integer, default: `30`): Connection timeout in seconds

## Example Configurations

### Example 1: Native-Only (Remote Home Assistant)

```yaml
openai:
  api_key: ${OPENAI_API_KEY}
  model: gpt-realtime
  voice: coral

home_assistant:
  url: https://homeassistant.local
  token: ${HA_TOKEN}

mcp_servers:
  home_assistant:
    mode: native
    enabled: true
    server_url: https://homeassistant.local/mcp_server/sse
    authorization: Bearer ${HA_TOKEN}
    description: Control smart home devices
    require_approval: never
    allowed_tools:
      - get_state
      - call_service
      - list_entities
```

### Example 2: Client-Only (Local Filesystem)

```yaml
openai:
  api_key: ${OPENAI_API_KEY}
  model: gpt-realtime
  voice: coral

mcp_servers:
  filesystem:
    mode: client
    enabled: true
    transport: stdio
    command: uvx
    args:
      - mcp-server-filesystem
      - /Users/username/Documents
    env:
      MCP_DEBUG: "1"
```

### Example 3: Hybrid (Both Native and Client)

```yaml
openai:
  api_key: ${OPENAI_API_KEY}
  model: gpt-realtime
  voice: coral

home_assistant:
  url: https://homeassistant.local
  token: ${HA_TOKEN}

mcp_servers:
  # Remote HA server (native)
  home_assistant:
    mode: native
    enabled: true
    server_url: https://homeassistant.local/mcp_server/sse
    authorization: Bearer ${HA_TOKEN}
    description: Smart home control
    require_approval: never
    priority: 100  # Higher priority

  # Local filesystem (client)
  filesystem:
    mode: client
    enabled: true
    transport: stdio
    command: uvx
    args:
      - mcp-server-filesystem
      - /Users/username/Documents
    priority: 200  # Lower priority

  # Local git tools (client)
  git:
    mode: client
    enabled: true
    transport: stdio
    command: uvx
    args:
      - mcp-server-git
      - --repository
      - /Users/username/projects
    priority: 200
```

## Approval Policies

Native MCP servers support approval workflows for security. Configure approval policies per server:

### Never Require Approval (Auto-approve all tools)
```yaml
mcp_servers:
  home_assistant:
    mode: native
    require_approval: never
```

### Always Require Approval (All tools need approval)
```yaml
mcp_servers:
  home_assistant:
    mode: native
    require_approval: always
    # Note: Currently auto-approved with TODO warning
    # Future: Will prompt user for approval
```

### Tool-Specific Approval Rules
```yaml
mcp_servers:
  home_assistant:
    mode: native
    require_approval:
      # Dangerous tools require approval
      call_service: always
      restart_service: always
      # Safe read-only tools auto-approved
      get_state: never
      list_entities: never
```

## Implementation Details

### Native MCP Flow

1. **Configuration**: MCP server defined in `config.yaml` with `mode: native`
2. **Session Config**: `_send_session_update()` builds OpenAI native MCP tool config
3. **OpenAI Handles**:
   - Connection to remote MCP server
   - Tool discovery
   - Tool execution
   - Error handling
4. **Approval Events**: If required, OpenAI sends `mcp.approval_request` events
5. **Our Handler**: `_handle_mcp_approval_request()` applies policy and responds

### Client MCP Flow

1. **Configuration**: MCP server defined with `mode: client` and `transport: stdio`
2. **Manager Init**: `ClientMCPManager` created with client-mode servers
3. **Connection**: During OpenAI connect, `client_mcp_manager.connect_all()`:
   - Spawns subprocess for each stdio server
   - Creates MCP client session
   - Initializes protocol handshake
   - Discovers available tools
4. **Registration**: Tools registered as OpenAI functions with closure handlers
5. **Execution**: When OpenAI calls function:
   - Closure captures tool name
   - Routes to `client_mcp_manager.call_tool()`
   - Executes on appropriate local server
   - Returns result to OpenAI
6. **Cleanup**: On disconnect, all subprocess servers are terminated

### Code References

**Configuration**:
- `src/config.py:MCPServerConfig` - Server configuration dataclass
- `src/config.py:AppConfig.mcp_servers` - Multi-server config field

**Native MCP**:
- `src/openai_client/realtime.py:_send_session_update()` (lines 970-1042) - Builds native MCP config
- `src/openai_client/realtime.py:_handle_mcp_approval_request()` (lines 772-846) - Approval handler

**Client MCP**:
- `src/services/ha_client/mcp_client_manager.py` - Client-side manager
- `src/openai_client/realtime.py:__init__()` (lines 79-89) - Manager initialization
- `src/openai_client/realtime.py:connect()` (lines 308-341) - Client server connection and registration
- `src/openai_client/realtime.py:disconnect()` (lines 383-390) - Client server cleanup

## Migration from v0.1.0

### Breaking Changes
- Configuration structure changed (see examples above)
- Native MCP now used by default for remote servers
- Legacy single-server `home_assistant.mcp` config deprecated

### Migration Steps

1. **Update configuration structure**:
   ```yaml
   # OLD (v0.1.0)
   home_assistant:
     mcp:
       enabled: true
       # ...

   # NEW (v2.0.0)
   mcp_servers:
     home_assistant:
       mode: native
       enabled: true
       # ...
   ```

2. **Test with existing setup** (backward compatible for now)

3. **Add new servers** using hybrid configuration

4. **Review approval policies** for security

## Troubleshooting

### Native MCP Issues

**Problem**: Tools not appearing in session
- Check `server_url` is correct and accessible
- Verify `authorization` header if required
- Check server logs for connection errors
- Ensure server supports OpenAI's native MCP format

**Problem**: Approval requests failing
- Check `require_approval` policy configuration
- Review logs for approval event handling
- Ensure tool names match in `allowed_tools`

### Client MCP Issues

**Problem**: Server won't start (stdio)
- Verify `command` is in PATH or use absolute path
- Check `args` are correct for the server
- Review subprocess logs for errors
- Ensure `timeout` is sufficient for startup

**Problem**: Tools not discovered
- Check server implements MCP protocol correctly
- Review server's `list_tools` response
- Enable debug logging: `env: {MCP_DEBUG: "1"}`

**Problem**: Tool execution fails
- Verify server has required permissions
- Check environment variables in `env`
- Review tool arguments match server schema

## Best Practices

1. **Use Native for Remote**: Always use native mode for remote/cloud MCP servers
2. **Secure Tokens**: Use environment variables for tokens/credentials
3. **Test Locally First**: Test client mode locally before deploying
4. **Set Priorities**: Use priority to control tool preference when names conflict
5. **Limit Tools**: Use `allowed_tools` to reduce surface area
6. **Approval for Write Operations**: Require approval for destructive tools
7. **Monitor Logs**: Enable debug logging during development
8. **Graceful Degradation**: Use `enabled: false` to disable without removing config

## Future Enhancements

- [ ] User approval UI for native MCP approval requests
- [ ] Support for HTTP transport in client mode
- [ ] Automatic failover from native to client mode
- [ ] Tool usage analytics and monitoring
- [ ] Configuration validation and schema checking
- [ ] Dynamic server enable/disable at runtime

## Related Documentation

- [OpenAI Realtime API - MCP Support](https://platform.openai.com/docs/guides/realtime-mcp)
- [Model Context Protocol Specification](https://spec.modelcontextprotocol.io)
- [MCP SDK Documentation](https://github.com/modelcontextprotocol/python-sdk)
- [OpenVoice MCP README](../README.md)
- [CHANGELOG](../CHANGELOG.md)

---

**Last Updated**: October 12, 2025
**Version**: 2.0.0
**Authors**: OpenVoice MCP Development Team
