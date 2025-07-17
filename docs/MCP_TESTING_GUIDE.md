# MCP Integration Testing Guide

This guide provides step-by-step instructions for testing the Model Context Protocol (MCP) integration with Home Assistant Realtime Voice Assistant on a Raspberry Pi.

## ⚠️ Important Requirements

- **Home Assistant 2025.2 or later** (REQUIRED - MCP was introduced in this version)
- **MCP Server Integration** must be installed and enabled in Home Assistant
- **New access token** may be required for MCP authentication

## 📋 Prerequisites

Before starting, ensure you have:
- [ ] Raspberry Pi with working audio (microphone and speaker)
- [ ] Home Assistant 2025.2+ installed and accessible
- [ ] SSH access to your Raspberry Pi
- [ ] OpenAI API key
- [ ] Picovoice access key (for wake word)
- [ ] Git installed on the Raspberry Pi

## 🏠 Phase 1: Home Assistant Setup

### Step 1: Verify Home Assistant Version

```bash
# SSH into your Home Assistant instance or use Terminal addon
ha core info
```

Look for version 2025.2 or higher. If you need to update:

```bash
ha core update
```

### Step 2: Install MCP Server Integration

1. Open Home Assistant web interface
2. Navigate to **Settings** → **Devices & Services**
3. Click the **"+ Add Integration"** button
4. Search for **"Model Context Protocol Server"**
5. Click on it to install
6. **IMPORTANT**: During setup, enable **"Control Home Assistant"** option
7. Click Submit/Finish

### Step 3: Create a New Access Token

1. Click your username in the bottom left corner
2. Go to the **Security** tab
3. Scroll down to **"Long-Lived Access Tokens"**
4. Click **"Create Token"**
5. Name it: `MCP Voice Assistant`
6. Copy the token immediately (you won't see it again!)
7. Save it securely - you'll need it shortly

## 🔧 Phase 2: Raspberry Pi Setup

### Step 4: Clone the MCP Branch

SSH into your Raspberry Pi and run:

```bash
# Navigate to your preferred directory
cd ~

# Clone the repository (or if already cloned, just cd into it)
git clone https://github.com/nicholastripp/ha-realtime-assist.git
cd ha-realtime-assist

# Fetch latest changes and checkout MCP branch
git fetch origin
git checkout feature/mcp-integration

# Verify you're on the correct branch
git branch
# Should show: * feature/mcp-integration
```

### Step 5: Set Up Python Environment

```bash
# Create virtual environment (if not already created)
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
pip install -r requirements.txt
```

### Step 6: Configure Environment Variables

Create or update your `.env` file:

```bash
# Create .env file
cat > .env << 'EOF'
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Home Assistant Configuration
HA_URL=http://homeassistant.local:8123
HA_TOKEN=your_new_mcp_token_from_step3

# Picovoice Configuration
PICOVOICE_ACCESS_KEY=your_picovoice_key_here
EOF

# Set proper permissions
chmod 600 .env
```

### Step 7: Verify MCP Configuration

Check that your `config/config.yaml` includes the MCP section:

```bash
# Copy example if needed
cp config/config.yaml.example config/config.yaml

# Edit configuration
nano config/config.yaml
```

Ensure the `home_assistant` section includes MCP settings:

```yaml
home_assistant:
  url: ${HA_URL}
  token: ${HA_TOKEN}
  language: "en"
  timeout: 10
  
  # Model Context Protocol (MCP) configuration
  mcp:
    sse_endpoint: "/mcp_server/sse"       # Default endpoint
    auth_method: "token"                  # Use token authentication
    connection_timeout: 30                # Connection timeout
    reconnect_attempts: 3                 # Retry attempts
```

## 🧪 Phase 3: Connection Testing

### Step 8: Test MCP Connection

Create and run a connection test:

```bash
# Create test script
cat > test_mcp.py << 'EOF'
import asyncio
import sys
sys.path.insert(0, 'src')

from services.ha_client.mcp import MCPClient
from config import load_config

async def test_mcp_connection():
    print("Loading configuration...")
    config = load_config("config/config.yaml")
    
    print(f"Connecting to Home Assistant at: {config.home_assistant.url}")
    print(f"Using MCP endpoint: {config.home_assistant.mcp.sse_endpoint}")
    
    client = MCPClient(
        base_url=config.home_assistant.url,
        access_token=config.home_assistant.token,
        sse_endpoint=config.home_assistant.mcp.sse_endpoint,
        connection_timeout=config.home_assistant.mcp.connection_timeout,
        reconnect_attempts=config.home_assistant.mcp.reconnect_attempts
    )
    
    try:
        print("\nAttempting connection...")
        await client.connect()
        print("✅ Successfully connected to MCP server!")
        
        tools = client.get_tools()
        print(f"\n✅ Discovered {len(tools)} tools from Home Assistant:")
        for tool in tools:
            print(f"   - {tool['name']}: {tool.get('description', 'No description')}")
        
        # Test a simple tool call if available
        if tools:
            print(f"\n🔧 Testing tool invocation...")
            # This would depend on what tools HA exposes
            
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Connection failed: {type(e).__name__}: {e}")
        print("\nTroubleshooting:")
        print("1. Check Home Assistant is running and accessible")
        print("2. Verify MCP Server integration is installed")
        print("3. Confirm your access token is correct")
        print("4. Check the URL in your .env file")
        
    finally:
        await client.disconnect()
        print("\nTest complete.")

# Run the test
asyncio.run(test_mcp_connection())
EOF

# Execute test
python test_mcp.py
```

## 🎤 Phase 4: Voice Assistant Testing

### Step 9: Test Audio Devices

First, verify your audio devices:

```bash
# List audio devices
python -m src.main --list-devices

# Test wake word detection only
python -m src.main --test
```

### Step 10: Run the Voice Assistant

```bash
# Run with normal logging
python -m src.main

# Or with verbose logging for debugging
python -m src.main --verbose
```

### Step 11: Test Voice Commands

1. Say the wake word: **"Picovoice"** (or your configured wake word)
2. Wait for the confirmation sound
3. Try these test commands:
   - "Turn on the living room lights"
   - "What's the temperature in the bedroom?"
   - "Turn off all lights"
   - "Set the thermostat to 72 degrees"
   - "Are any lights on?"

## 🔍 Phase 5: Troubleshooting

### Common Issues and Solutions

#### 1. "404 Not Found" Error
```
Client error '404 Not Found' for url 'http://localhost:8123/mcp_server/sse'
```
**Solution**: MCP Server integration not installed. Go back to Step 2.

#### 2. "401 Unauthorized" Error
```
Client error '401 Unauthorized' for url 'http://localhost:8123/mcp_server/sse'
```
**Solution**: Invalid token. Create a new token (Step 3) and update .env file.

#### 3. Connection Timeout
**Solutions**:
- Check if Home Assistant is accessible: `curl http://homeassistant.local:8123`
- Verify the URL in your .env file
- Check network connectivity
- Increase `connection_timeout` in config.yaml

#### 4. No Tools Discovered
**Solution**: In MCP Server integration settings, ensure "Control Home Assistant" is enabled.

#### 5. Wake Word Not Detected
**Solutions**:
- Increase `audio_gain` in config.yaml (try 1.5 or 2.0)
- Check microphone with: `arecord -d 5 test.wav && aplay test.wav`
- Verify Picovoice access key is valid

### Debug Commands

```bash
# Check Home Assistant logs
ssh homeassistant
docker logs homeassistant | grep mcp_server

# Test raw SSE connection
curl -N \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Accept: text/event-stream" \
  http://homeassistant.local:8123/mcp_server/sse

# Monitor system resources
htop  # While running the assistant
```

## 📊 Phase 6: Performance Testing

### Measure Response Times

```bash
# Create performance test
cat > test_performance.py << 'EOF'
import time
import asyncio
from test_mcp import test_mcp_connection

start = time.time()
asyncio.run(test_mcp_connection())
end = time.time()

print(f"\nTotal connection and discovery time: {end - start:.2f} seconds")
EOF

python test_performance.py
```

### Expected Performance Metrics
- MCP connection: < 2 seconds
- Tool discovery: < 1 second
- Voice command execution: < 3 seconds total
- Wake word detection: < 500ms

## ✅ Testing Checklist

Complete all items for successful validation:

- [ ] Home Assistant 2025.2+ verified
- [ ] MCP Server integration installed
- [ ] New access token created
- [ ] MCP connection test passes
- [ ] Tools discovered from Home Assistant
- [ ] Wake word triggers session
- [ ] Voice commands execute successfully
- [ ] Audio responses play correctly
- [ ] Multi-turn conversations work (if enabled)
- [ ] Logs show no errors
- [ ] Performance is acceptable

## 📝 Reporting Issues

If you encounter problems:

1. Run with verbose logging: `python -m src.main --verbose`
2. Save the logs: `python -m src.main --verbose 2>&1 | tee test_log.txt`
3. Note:
   - Exact error messages
   - What command triggered the error
   - Your Home Assistant version
   - Your configuration (without tokens!)

## 🎉 Success!

If all tests pass, the MCP integration is working correctly! The voice assistant is now using Home Assistant's Model Context Protocol for all smart home control.

### Next Steps
- Test with your specific devices and automations
- Try multi-turn conversations
- Test edge cases (device offline, ambiguous commands)
- Monitor for stability over extended periods

---

**Note**: This is a breaking change from the previous Conversation API implementation. There is no backward compatibility, so ensure you're ready to fully migrate to MCP before updating.