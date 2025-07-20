# Home Assistant Realtime Voice Assistant

![Version](https://img.shields.io/badge/version-1.1.3-blue)
![Status](https://img.shields.io/badge/status-stable-green)
![Python](https://img.shields.io/badge/python-3.9+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A standalone Raspberry Pi voice assistant that provides natural, low-latency conversations for Home Assistant control using OpenAI's Realtime API.

## 🚀 Project Status: v1.1.3 - Critical Fix for OpenAI Session Timeout!

Fixed the 30-minute session timeout issue that prevented the assistant from working after extended idle periods. OpenAI connections are now established on-demand for better reliability and resource efficiency.

## Overview

This project creates a dedicated voice interface for Home Assistant that runs on a Raspberry Pi. Unlike traditional voice assistants that use a sequential "speak-wait-respond" pattern, this assistant enables natural, real-time conversations with <800ms response latency.

## Key Features

- 🎙️ **Natural Conversations**: Real-time bidirectional audio streaming with multi-turn support
- ⚡ **Low Latency**: <800ms voice-to-voice response time  
- 🏠 **Full HA Control**: Uses Home Assistant's Model Context Protocol (MCP)
- 👂 **Porcupine Wake Words**: Accurate detection with built-in keywords + custom wake word support
- 🔊 **Automatic Gain Control**: AGC prevents clipping and maintains optimal audio levels
- 🌍 **Multi-Language**: Configurable language support
- 🎭 **Personality**: Customizable assistant personality
- 🌐 **Web UI**: Complete web interface with setup wizard and real-time monitoring
- 🔒 **Enterprise Security**: CSRF protection, rate limiting, security headers, and secure authentication
- 🚀 **Easy Setup**: Simple configuration and installation with guided wizard
- 🔄 **Web-based Management**: Configure and restart without CLI access

## How It Works

1. **Wake Word Detection**: Local detection using Picovoice Porcupine
2. **Audio Streaming**: Captures voice and streams to OpenAI Realtime API
3. **Smart Control**: OpenAI understands intent and calls HA functions
4. **Natural Response**: Speaks back with natural, conversational responses
5. **Multi-turn Conversations**: Continue talking without repeating wake word

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
git clone https://github.com/nicholastripp/ha-realtime-assist
cd ha-realtime-assist

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
  voice: "nova"

home_assistant:
  url: ${HA_URL}               # Uses value from .env
  token: ${HA_TOKEN}           # Uses value from .env

wake_word:
  enabled: true
  model: "picovoice"
  sensitivity: 1.0
```

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

## What's New in v1.1.0

- 🌐 **Complete Web UI** - Full-featured web interface with setup wizard
- 🔒 **Security Features** - HTTPS encryption and authentication built-in
- 🎨 **Visual Configuration** - Edit all settings through the browser
- 📊 **Real-time Dashboard** - Monitor status, logs, and statistics
- 🎙️ **Audio Testing** - Test and configure audio devices via web
- 🎭 **Personality Editor** - Customize assistant personality with visual sliders
- 🔧 **Natural Multi-turn** - VAD-based silence detection for natural conversation endings

See [CHANGELOG.md](CHANGELOG.md) for complete details.

### Previous Release (v1.0.0)
- Model Context Protocol (MCP) integration for direct Home Assistant control
- Real-time device state awareness with GetLiveContext
- Automatic tool discovery and SSL certificate support

### ⚠️ Breaking Changes from 0.x

**This is a major release with breaking changes:**
- Requires Home Assistant 2025.2+ (for MCP support)
- MCP Server integration must be installed and enabled
- No backward compatibility with Conversation API
- New access token may be required

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
✓ Home Assistant Voice Assistant v0.5.0-beta
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
- [Web UI Guide](docs/WEB_UI_GUIDE.md) - Web interface for easy configuration
- [Audio Setup](docs/AUDIO_SETUP.md) - Microphone and speaker configuration
- [Wake Word Setup](docs/WAKE_WORD_SETUP.md) - Wake word configuration
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](./CONTRIBUTING.md) first.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by the [Billy B-Assistant](https://github.com/nickschaub/billy-b-assistant) project
- Built with [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime)
- Integrates with [Home Assistant](https://www.home-assistant.io)