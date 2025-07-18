# Migration Guide: 0.5.x to 1.0.0

This guide helps you upgrade from version 0.5.x to 1.0.0 of the Home Assistant Realtime Voice Assistant.

## ⚠️ Breaking Changes

Version 1.0.0 introduces a complete replacement of the Home Assistant integration layer:

- **Old**: Used Home Assistant Conversation API
- **New**: Uses Model Context Protocol (MCP)
- **Impact**: No backward compatibility - requires Home Assistant 2025.2+

## Prerequisites

Before upgrading, ensure you have:

1. **Home Assistant 2025.2 or later** installed
2. **MCP Server integration** available in your HA instance
3. **Backup** of your current configuration

## Migration Steps

### Step 1: Update Home Assistant

First, ensure your Home Assistant is running version 2025.2 or later:

```bash
# Check current version
ha core info

# Update if needed
ha core update
```

### Step 2: Install MCP Server Integration

1. Open Home Assistant web interface
2. Navigate to **Settings** → **Devices & Services**
3. Click **"+ Add Integration"**
4. Search for **"Model Context Protocol Server"**
5. Install and configure:
   - **Important**: Enable **"Control Home Assistant"** option
   - This allows the assistant to control your devices

### Step 3: Generate New Access Token

The MCP integration may require a new access token:

1. In Home Assistant, click your username (bottom left)
2. Go to **Security** tab
3. Under **Long-Lived Access Tokens**, click **"Create Token"**
4. Name it: `MCP Voice Assistant`
5. Copy and save the token immediately

### Step 4: Update Your Installation

```bash
# Navigate to your installation directory
cd ~/ha-realtime-assist

# Backup your current configuration
cp .env .env.backup
cp config/config.yaml config/config.yaml.backup

# Pull the latest code
git pull origin main

# Update dependencies
source venv/bin/activate
pip install -r requirements.txt
```

### Step 5: Update Configuration

#### Update .env file

Replace your old token with the new one:

```bash
# Edit .env file
nano .env

# Update the token
HA_TOKEN=your_new_mcp_token_here
```

#### Update config.yaml (if customized)

The default configuration should work, but if you've customized `config.yaml`, add the MCP section:

```yaml
home_assistant:
  url: ${HA_URL}
  token: ${HA_TOKEN}
  language: "en"
  timeout: 10
  
  # Add this section for MCP
  mcp:
    sse_endpoint: "/mcp_server/sse"
    auth_method: "token"
    connection_timeout: 30
    reconnect_attempts: 3
```

### Step 6: Test the Connection

Before running the full assistant, test the MCP connection:

```bash
# Activate virtual environment
source venv/bin/activate

# Test connection
python examples/test_ha_connection.py
```

You should see:
- Successful connection to Home Assistant
- MCP tools discovered
- No errors about missing endpoints

### Step 7: Run the Assistant

```bash
python src/main.py
```

## Troubleshooting

### "404 Not Found" Error

If you see:
```
Client error '404 Not Found' for url 'http://homeassistant.local:8123/mcp_server/sse'
```

**Solution**: MCP Server integration is not installed. Go back to Step 2.

### "401 Unauthorized" Error

**Solution**: Your token is invalid. Generate a new token (Step 3) and update your .env file.

### Connection Timeouts

**Solutions**:
1. Verify Home Assistant is accessible
2. Check if using HTTPS requires SSL verification settings
3. Increase timeout in config.yaml if needed

### No Tools Discovered

**Solution**: In MCP Server integration settings, ensure "Control Home Assistant" is enabled.

## What's Different?

### Better Device Control
- Direct tool-based control instead of natural language processing
- More reliable and predictable device operations
- Comprehensive device state awareness via GetLiveContext

### Improved Performance
- Faster response times for device control
- Better error messages when operations fail
- More efficient communication protocol

### Enhanced Features
- Real-time device state queries
- Support for all Home Assistant services
- Better handling of complex operations

## Need Help?

If you encounter issues:

1. Check the [Troubleshooting Guide](docs/TROUBLESHOOTING.md)
2. Review the [MCP Testing Guide](docs/MCP_TESTING_GUIDE.md)
3. Open an issue on GitHub with:
   - Your Home Assistant version
   - Error messages
   - Steps to reproduce

## Rolling Back

If you need to rollback to 0.5.x:

```bash
# Restore your backups
cp .env.backup .env
cp config/config.yaml.backup config/config.yaml

# Checkout the old version
git checkout v0.5.0-beta

# Reinstall old dependencies
pip install -r requirements.txt
```

Note: Version 0.5.x will not receive further updates. We recommend completing the migration to 1.0.0 for continued support.

---

# Migration Guide: 1.0.0 to 1.1.0

This guide helps you upgrade from version 1.0.0 to 1.1.0 of the Home Assistant Realtime Voice Assistant.

## ✅ Non-Breaking Update

Version 1.1.0 is a feature release that adds the web UI without breaking existing functionality. Your current installation will continue to work exactly as before.

## What's New in 1.1.0

- **Complete Web UI** with setup wizard and real-time monitoring
- **Security features** including HTTPS and authentication
- **Visual configuration** editors for all settings
- **Enhanced installation** process with optional web UI setup
- **Natural multi-turn conversations** with VAD-based silence detection
  - Conversations end naturally after 8 seconds of silence
  - Replaced arbitrary 30s timeout with intelligent silence detection
  - 5-minute safety timeout for extended conversations

## Upgrade Steps

### Step 1: Update Your Installation

```bash
# Navigate to your installation directory
cd ~/ha-realtime-assist

# Pull the latest code
git pull origin main

# Update dependencies
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Enable Web UI (Optional)

The web UI is disabled by default. To enable it:

#### Option A: Run the installer for security setup
```bash
./install.sh
```

When prompted:
1. Choose to enable web UI with authentication
2. Set a username (default: admin)
3. Enter a secure password

#### Option B: Manual configuration
1. Edit `config/config.yaml`:
   ```yaml
   web_ui:
     enabled: true
     host: "0.0.0.0"  # Or "127.0.0.1" for localhost only
     port: 8443
   ```

2. If you want authentication (recommended):
   ```bash
   # Generate a password hash
   python -c "import bcrypt; print(bcrypt.hashpw(b'your-password', bcrypt.gensalt(12)).decode())"
   
   # Add to .env file
   echo "WEB_UI_PASSWORD_HASH=your-hash-here" >> .env
   ```

### Step 3: Access the Web UI

Start the assistant with web UI:
```bash
python src/main.py --web
```

Access at `https://localhost:8443` (accept the self-signed certificate warning).

## New Features Usage

### Web UI Features
- **Setup Wizard**: Automatically shown on first run without .env
- **Configuration Editor**: Edit all settings via `https://localhost:8443/config/yaml`
- **Personality Editor**: Customize assistant at `https://localhost:8443/persona`
- **Status Dashboard**: Monitor in real-time at `https://localhost:8443/status`
- **Audio Testing**: Configure devices at `https://localhost:8443/config/audio`

### Security Configuration
- **HTTPS**: Enabled by default with self-signed certificates
- **Authentication**: Basic auth with bcrypt password hashing
- **Session Management**: Configurable timeout (default 1 hour)

### Using Custom Certificates
```yaml
web_ui:
  tls:
    cert_file: "/path/to/cert.pem"
    key_file: "/path/to/key.pem"
```

### Multi-turn Conversation Improvements
The multi-turn conversation behavior has been enhanced:
```yaml
session:
  multi_turn_timeout: 300.0       # Increased from 30s to 5 minutes (safety only)
  extended_silence_threshold: 8.0 # New: Natural conversation end after silence
```

## No Action Required

If you don't want to use the web UI, no action is required. The assistant will continue to work exactly as it did in v1.0.0.