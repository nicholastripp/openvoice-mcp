# OpenVoice MCP Configuration Examples

This directory contains example configuration files for OpenVoice MCP v2.0.0's hybrid MCP architecture.

## Quick Start

1. **Choose a configuration example** based on your needs
2. **Copy to your config directory**: `cp examples/config/[example].yaml config/config.yaml`
3. **Edit environment variables** in `.env` file
4. **Customize** the configuration for your setup
5. **Run the assistant**: `python src/main.py`

## Available Examples

### 🔗 native-only.yaml

**Best for**: Remote servers only (e.g., Home Assistant on network)

**Includes**:
- Home Assistant MCP server (native mode)
- Example cloud API configuration (commented)
- Lowest latency configuration

**Use when**:
- All your MCP servers are accessible via HTTP/HTTPS
- You want OpenAI to manage connections
- You prioritize lowest latency

### 💻 client-only.yaml

**Best for**: Local tools only (filesystem, git, etc.)

**Includes**:
- Filesystem server for local file access
- Git server for repository operations
- Example memory/notes server (commented)
- Custom server template (commented)

**Use when**:
- You only need local tool access
- No remote MCP servers required
- You want full control over server execution

### ⚡ hybrid.yaml (Recommended)

**Best for**: Combining remote and local servers

**Includes**:
- Home Assistant (native mode)
- Filesystem access (client mode)
- Git operations (client mode)
- Example cloud API (commented)
- Priority-based tool routing
- Comprehensive configuration guide

**Use when**:
- You want both remote and local MCP servers
- You need maximum flexibility
- You want to leverage best of both worlds

## Configuration Guide

### Environment Variables

Create a `.env` file in the project root with:

```bash
# Required
OPENAI_API_KEY=sk-...
PICOVOICE_ACCESS_KEY=...

# For Home Assistant
HA_URL=https://homeassistant.local
HA_TOKEN=eyJ0eXAiOiJKV1...

# For cloud APIs (optional)
CLOUD_API_TOKEN=...
```

### Configuration Modes

#### Native Mode
- **Transport**: HTTP/SSE
- **Managed by**: OpenAI
- **Latency**: <100ms
- **Best for**: Remote servers

```yaml
server_name:
  mode: native
  server_url: https://example.com/mcp/sse
  authorization: Bearer ${TOKEN}
```

#### Client Mode
- **Transport**: stdio (subprocess)
- **Managed by**: OpenVoice MCP
- **Latency**: <200ms
- **Best for**: Local servers

```yaml
server_name:
  mode: client
  transport: stdio
  command: uvx
  args:
    - mcp-server-name
    - --arg1
    - value1
```

### Priority System

Lower number = higher priority:

- **100**: Critical services (Home Assistant)
- **200**: Frequently used (filesystem)
- **300+**: Less frequently used

When multiple servers provide similar tools, the lowest priority number wins.

### Approval Policies

Control which tool calls require user approval:

```yaml
# Auto-approve everything
require_approval: never

# Require approval for everything
require_approval: always

# Tool-specific rules
require_approval:
  read_state: never    # Auto-approve reads
  call_service: always # Require approval for actions
```

## Server Installation

### Filesystem Server
```bash
# Install via uvx (recommended)
uvx mcp-server-filesystem

# Or via npm
npx -y @modelcontextprotocol/server-filesystem
```

### Git Server
```bash
uvx mcp-server-git
# or
npx -y @modelcontextprotocol/server-git
```

### Other Servers
Visit [MCP Servers Directory](https://github.com/modelcontextprotocol/servers) for more options.

## Testing Your Configuration

1. **Validate syntax**:
   ```bash
   python -c "import yaml; yaml.safe_load(open('config/config.yaml'))"
   ```

2. **Test with verbose logging**:
   ```bash
   python src/main.py --verbose
   ```

3. **Check MCP connections**:
   Look for these log messages:
   - `[NATIVE MCP] Connected to server 'server_name'`
   - `[CLIENT MCP] Connected to server 'server_name'`

4. **Test specific servers**:
   Enable one server at a time and verify functionality.

## Troubleshooting

### Native Mode Issues

**Problem**: Server not connecting
- Check `server_url` is correct and accessible
- Verify `authorization` token is valid
- Check firewall/network settings
- Test URL directly: `curl -H "Authorization: Bearer TOKEN" URL`

**Problem**: Tools not appearing
- Check `allowed_tools` filter (if set)
- Verify server is MCP-compatible
- Check server logs for errors

### Client Mode Issues

**Problem**: Server won't start
- Verify `command` is in PATH: `which uvx`
- Check `args` are correct for the server
- Ensure required permissions
- Check `timeout` is sufficient

**Problem**: Tools not discovered
- Enable debug logging: `env: {MCP_DEBUG: "1"}`
- Check server implements MCP protocol
- Verify server's `list_tools` response
- Check subprocess logs in system log file

### General Issues

**Problem**: Conflicting tool names
- Use `priority` to choose preferred server
- Use `allowed_tools` to filter tools
- Disable less important servers

**Problem**: High latency
- Native servers should be <100ms
- Client servers should be <200ms
- Check network for native servers
- Check subprocess startup time for client

## Advanced Configuration

### Custom stdio Server

```yaml
custom:
  mode: client
  transport: stdio
  command: python
  args:
    - /path/to/your/server.py
  env:
    CUSTOM_VAR: value
    ANOTHER_VAR: value2
  timeout: 60
  priority: 500
```

### Multiple Filesystem Paths

```yaml
filesystem:
  mode: client
  transport: stdio
  command: uvx
  args:
    - mcp-server-filesystem
    - /path/one
    - /path/two
    - /path/three
```

### Tool Filtering

```yaml
home_assistant:
  mode: native
  server_url: https://ha.local/mcp/sse
  allowed_tools:
    - get_state      # Only expose these 3 tools
    - call_service
    - list_entities
  # All other tools from this server are hidden
```

## Security Best Practices

1. **Use environment variables** for all sensitive data
2. **Set strict approval policies** for write operations
3. **Limit filesystem access** to specific directories
4. **Review allowed_tools** for each server
5. **Use verify_ssl: true** for production
6. **Regularly update** MCP server packages
7. **Monitor logs** for suspicious activity

## Additional Resources

- [Hybrid MCP Architecture Guide](../../docs/NATIVE_MCP_GUIDE.md)
- [OpenAI Realtime API Docs](https://platform.openai.com/docs/guides/realtime)
- [Model Context Protocol Spec](https://spec.modelcontextprotocol.io)
- [MCP Servers Directory](https://github.com/modelcontextprotocol/servers)

## Need Help?

- Check [Troubleshooting Guide](../../docs/TROUBLESHOOTING.md)
- Review [NATIVE_MCP_GUIDE.md](../../docs/NATIVE_MCP_GUIDE.md)
- See [CHANGELOG.md](../../CHANGELOG.md) for latest changes
- Open an issue on GitHub

---

**Version**: 2.0.0
**Last Updated**: October 12, 2025
