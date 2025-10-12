# OpenVoice MCP - Hybrid Multi-Server Voice Assistant

![Version](https://img.shields.io/badge/version-2.0.0--beta-orange)
![Status](https://img.shields.io/badge/status-testing-yellow)
![Python](https://img.shields.io/badge/python-3.9+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A next-generation voice assistant with **hybrid native/client-side MCP support**, enabling both remote cloud services and local tool execution through OpenAI's Realtime API.

**🚀 What's New in v2.0.0**: Hybrid MCP architecture combining OpenAI's native MCP support (August 2025) with client-side management for local servers. Based on ha-realtime-assist v1.2.0.

## Overview

This project creates a dedicated voice interface for Home Assistant that runs on a Raspberry Pi. Unlike traditional voice assistants that use a sequential "speak-wait-respond" pattern, this assistant enables natural, real-time conversations with <600ms response latency using OpenAI's production Realtime API.

## Key Features

### 🌟 v2.0.0 Hybrid MCP Architecture
- **🔗 Native MCP Mode**: OpenAI directly manages remote MCP servers (lowest latency)
- **💻 Client MCP Mode**: Local stdio servers for filesystem, git, and local tools
- **⚡ Multi-Server Support**: Connect to multiple MCP servers simultaneously
- **🛡️ Approval Workflows**: Policy-based tool execution approval for security
- **🔄 Automatic Routing**: Intelligent tool routing based on priority and availability

### 🎙️ Voice & Conversation
- **Natural Conversations**: Real-time bidirectional audio streaming with multi-turn support
- **Low Latency**: <600ms voice-to-voice response time
- **10+ Voices**: Choose from 10 OpenAI voices including Cedar, Marin, Verse
- **Multi-Language**: End phrases in 6 languages (EN, DE, ES, FR, IT, NL)

### 🏠 Home Assistant Integration
- **Native MCP**: Direct OpenAI connection to Home Assistant MCP server
- **Full Control**: Complete device and service control through MCP protocol
- **Real-time State**: Live entity state updates and monitoring

### 🔧 Local Tools & Servers
- **Filesystem Access**: Browse and manage files through MCP filesystem server
- **Git Integration**: Repository operations via MCP git server
- **Custom Servers**: Add any stdio-based MCP server
- **Subprocess Management**: Automatic lifecycle handling for local servers

### 🎛️ Audio & Wake Words
- **Porcupine Wake Words**: >95% accurate detection with built-in + custom keywords
- **Automatic Gain Control**: AGC prevents clipping and maintains optimal levels
- **Audio Diagnostics**: Professional-grade optimization tools

### 🌐 Interface & Security
- **Web UI**: Complete web interface with setup wizard and monitoring
- **Enterprise Security**: CSRF protection, rate limiting, secure authentication
- **Cost Effective**: 20% lower API costs with production models

## How It Works

1. **Wake Word Detection**: Local detection using Picovoice Porcupine
2. **Audio Streaming**: Captures voice and streams to OpenAI Realtime API
3. **Smart Control**: OpenAI understands intent and calls HA functions
4. **Natural Response**: Speaks back with natural, conversational responses
5. **Multi-turn Conversations**: Continue talking without repeating wake word
6. **Smart End Phrases**: Say "stop" or "that's all" to end conversation immediately

### Multi-Turn Conversations

The assistant supports natural multi-turn conversations where you can continue speaking without repeating the wake word. The conversation continues until:
- You say an end phrase like "stop", "that's all", or "goodbye"
- 8.5 seconds of silence is detected
- The conversation timeout is reached (configurable, default 5 minutes)

**Supported End Phrases**: Smart detection in 6 languages - English ("stop", "goodbye", "that's all"), German ("stopp", "ende"), Spanish, French, Italian, and Dutch. Single-word "stop" ends conversation without triggering device actions.

## Requirements

### Software Requirements
- Home Assistant 2025.2 or later (required for MCP support)
- MCP Server integration installed and enabled in Home Assistant

### Hardware Requirements

#### Minimum Setup
- Raspberry Pi 3B+ or newer
- USB microphone
- Speaker (3.5mm jack or USB)
- 8GB+ SD card

#### Recommended Setup
- Raspberry Pi 4 (2GB+ RAM)
- USB conference speakerphone (e.g., Jabra Speak 410)
- -OR- ReSpeaker 2-Mic HAT
- Ethernet connection

## Quick Start

```bash
# Clone the repository
git clone https://github.com/nicholastripp/openvoice-mcp
cd openvoice-mcp

# Install dependencies and set up configuration
./install.sh

# Edit your API keys and settings
nano .env
# Add:
#   OPENAI_API_KEY=sk-...
#   HA_URL=http://homeassistant.local:8123
#   HA_TOKEN=your_home_assistant_token
#   PICOVOICE_ACCESS_KEY=your_picovoice_key

# (Optional) Customize additional settings
nano config/config.yaml  # Audio settings, wake word, etc.
nano config/persona.ini  # Assistant personality

# Activate the virtual environment (required for all Python commands)
source venv/bin/activate

# Start the assistant
python src/main.py

# Or start with web UI for easy configuration (opens at https://localhost:8443)
python src/main.py --web
```

**Note**: Always activate the virtual environment (`source venv/bin/activate`) before running any Python commands. You'll see `(venv)` in your terminal prompt when it's active.

## Configuration

The project includes example configuration files (`.example` suffix) that are automatically copied to working files during installation. Always edit the copies, not the example files.

### Required Settings in .env
All user-specific settings are managed in the `.env` file:
- **OPENAI_API_KEY**: Get from [OpenAI Platform](https://platform.openai.com)
- **HA_URL**: Your Home Assistant instance URL
- **HA_TOKEN**: Long-lived access token (generate in HA Profile settings)
- **PICOVOICE_ACCESS_KEY**: Get from [Picovoice Console](https://console.picovoice.ai)

### Example .env
```bash
OPENAI_API_KEY=sk-...
HA_URL=http://homeassistant.local:8123
HA_TOKEN=eyJ0eXAiOiJKV1...
PICOVOICE_ACCESS_KEY=your_key_here
```

### config.yaml Settings
The `config.yaml` file contains application settings that typically don't need to be changed:
```yaml
openai:
  api_key: ${OPENAI_API_KEY}  # Uses value from .env
  model: "gpt-realtime"      # Production API (20% cheaper)
  voice: "nova"               # 10 voices available

home_assistant:
  url: ${HA_URL}               # Uses value from .env
  token: ${HA_TOKEN}           # Uses value from .env

wake_word:
  enabled: true
  model: "picovoice"
  sensitivity: 1.0
```

### MCP Server Configuration (v2.0.0)

OpenVoice MCP v2 supports **hybrid MCP architecture** with multiple servers. Configure servers in the `mcp_servers` section:

```yaml
mcp_servers:
  # Native mode - OpenAI manages remote servers (recommended for remote)
  home_assistant:
    mode: native
    enabled: true
    server_url: https://homeassistant.local/mcp_server/sse
    authorization: Bearer ${HA_TOKEN}
    description: Smart home control
    require_approval: never  # "always", "never", or tool-specific dict
    priority: 100

  # Client mode - Local stdio servers (required for local tools)
  filesystem:
    mode: client
    enabled: true
    transport: stdio
    command: uvx
    args:
      - mcp-server-filesystem
      - /Users/username/Documents
    priority: 200

  # Add more servers as needed
  git:
    mode: client
    enabled: false
    transport: stdio
    command: uvx
    args:
      - mcp-server-git
      - --repository
      - /path/to/repo
```

**Configuration Modes**:
- **Native Mode**: OpenAI connects directly to remote MCP servers (lowest latency)
- **Client Mode**: Local stdio servers managed as subprocesses (for filesystem, git, etc.)

See the [Hybrid MCP Architecture Guide](docs/NATIVE_MCP_GUIDE.md) for detailed configuration examples and best practices.

### Wake Word Setup

The assistant uses Picovoice Porcupine for accurate wake word detection. Built-in keywords include:
- "picovoice" (default)
- "alexa", "computer", "jarvis"
- Custom wake words via .ppn files from Picovoice Console
- And more!

See the [Wake Word Setup Guide](docs/WAKE_WORD_SETUP.md) for detailed configuration.

### Audio Configuration

The assistant now includes **Automatic Gain Control (AGC)** to handle varying microphone levels:

```yaml
audio:
  agc_enabled: true  # Enable automatic volume adjustment
  input_volume: 1.0  # Manual gain (0.1-5.0, <1.0 reduces volume)
```

See the [Audio Setup Guide](docs/AUDIO_SETUP.md) for optimal configuration.

## Latest Features

### v2.0.0 (October 2025)
- 🔗 **Hybrid MCP Architecture** - Native OpenAI MCP + client-side stdio server support
- ⚡ **Multi-Server Support** - Connect to multiple MCP servers simultaneously
- 💻 **Local Tool Integration** - Filesystem, git, and custom stdio-based servers
- 🛡️ **Approval Workflows** - Policy-based tool execution security
- 🔄 **Intelligent Tool Routing** - Priority-based tool selection and routing

### v1.2.0 Base Features (Inherited from ha-realtime-assist)
- 🎯 **Audio Pipeline Diagnostics** - Professional tools for analyzing and optimizing audio quality
- 💰 **20% Cost Reduction** - Migration to OpenAI production API with better pricing
- 🎤 **Enhanced Voice Options** - 10 voices including Cedar, Marin, Verse, and Juniper
- 🌍 **Multi-Language End Phrases** - Smart detection in 6 languages prevents false positives
- 📊 **Wake Word Accuracy >95%** - Optimized audio pipeline improves detection

See [CHANGELOG.md](CHANGELOG.md) for detailed release notes.

## Logging & Output Control

The assistant now features a clean, minimal console output by default:

```bash
# Normal operation (clean output)
python src/main.py

# Verbose mode (includes debug information)
python src/main.py --verbose
# or
python src/main.py -v

# Quiet mode (errors only)
python src/main.py --quiet
# or
python src/main.py -q

# Custom log level
python src/main.py --log-level DEBUG
```

**Console Output Examples:**
```
✓ OpenVoice MCP v2.0.0
✓ Connected to Home Assistant 2024.1.0
✓ Listening for wake word 'picovoice'

► Wake word detected: picovoice
● Listening...
● Processing...
● Responding...
✓ Ready
```

**Logging Configuration** (in `config.yaml`):
```yaml
system:
  log_level: "INFO"         # File logging level
  console_log_level: "INFO"  # Console output level
  log_to_file: true          # Enable file logging
  log_max_size_mb: 10        # Max log file size
  log_backup_count: 3        # Number of rotated files
```

### Testing & Validation

Run these tests to verify your setup:

```bash
# First, activate the virtual environment
source venv/bin/activate

# Test wake word detection
python examples/test_wake_word.py --interactive

# Test audio devices
python examples/test_audio_devices.py

# NEW: Run audio diagnostics and optimization
python tools/audio_pipeline_diagnostic.py
python tools/gain_optimization_wizard.py

# Test Home Assistant connection
python examples/test_ha_connection.py

# Test OpenAI connection
python examples/test_openai_connection.py

# Run the full assistant
python src/main.py
```

**Convenient Test Runner**: Use the provided script to run tests easily:
```bash
# Run all tests
./run_tests.sh

# Run specific tests
./run_tests.sh audio
./run_tests.sh ha
./run_tests.sh openai
./run_tests.sh wake     # Interactive wake word test
./run_tests.sh help     # Show all options
```

## Documentation

- [Installation Guide](docs/INSTALLATION.md) - Step-by-step setup instructions
- [Usage Guide](docs/USAGE.md) - How to use your assistant
- [Configuration Guide](docs/CONFIGURATION.md) - All configuration options
- [Hybrid MCP Architecture Guide](docs/NATIVE_MCP_GUIDE.md) - **NEW v2.0.0**: Native + client-side MCP setup
- [Web UI Guide](docs/WEB_UI_GUIDE.md) - Web interface for easy configuration
- [Audio Setup](docs/AUDIO_SETUP.md) - Microphone and speaker configuration
- [Wake Word Setup](docs/WAKE_WORD_SETUP.md) - Wake word configuration
- [OpenAI Migration](docs/OPENAI_MIGRATION.md) - Guide for v1.2.0 API changes
- [Audio Diagnostics](tools/README_AUDIO_DIAGNOSTIC.md) - Audio optimization tools
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](./CONTRIBUTING.md) first.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Performance

Based on ha-realtime-assist v1.2.0 with v2.0.0 enhancements:
- **Response Latency**: <600ms voice-to-voice (native MCP even lower)
- **Wake Word Accuracy**: >95%
- **API Costs**: 20% reduction from production model
- **Transcription Accuracy**: >98%
- **Multi-language Support**: 6 languages
- **MCP Tool Latency**: Native mode <100ms, client mode <200ms

## Acknowledgments

- **Built on** [ha-realtime-assist v1.2.0](https://github.com/nicholastripp/ha-realtime-assist) - The foundation voice assistant platform
- Inspired by the [Billy B-Assistant](https://github.com/nickschaub/billy-b-assistant) project
- Built with [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime)
- Powered by [Model Context Protocol](https://modelcontextprotocol.io)
- Integrates with [Home Assistant](https://www.home-assistant.io)