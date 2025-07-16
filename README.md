# Home Assistant Realtime Voice Assistant

![Version](https://img.shields.io/badge/version-0.4.0--beta-blue)
![Status](https://img.shields.io/badge/status-beta-yellow)
![Python](https://img.shields.io/badge/python-3.9+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A standalone Raspberry Pi voice assistant that provides natural, low-latency conversations for Home Assistant control using OpenAI's Realtime API.

## 🚀 Project Status: Beta Release!

The assistant is now in beta with all core features implemented and tested. Ready for real-world use with ongoing improvements.

## Overview

This project creates a dedicated voice interface for Home Assistant that runs on a Raspberry Pi. Unlike traditional voice assistants that use a sequential "speak-wait-respond" pattern, this assistant enables natural, real-time conversations with <800ms response latency.

## Key Features

- 🎙️ **Natural Conversations**: Real-time bidirectional audio streaming with multi-turn support
- ⚡ **Low Latency**: <800ms voice-to-voice response time  
- 🏠 **Full HA Control**: Uses Home Assistant's Conversation API
- 👂 **Porcupine Wake Words**: Accurate detection with built-in keywords
- 🔊 **Automatic Gain Control**: AGC prevents clipping and maintains optimal audio levels
- 🌍 **Multi-Language**: Configurable language support
- 🎭 **Personality**: Customizable assistant personality
- 🚀 **Easy Setup**: Simple configuration and installation

## How It Works

1. **Wake Word Detection**: Local detection using Picovoice Porcupine
2. **Audio Streaming**: Captures voice and streams to OpenAI Realtime API
3. **Smart Control**: OpenAI understands intent and calls HA functions
4. **Natural Response**: Speaks back with natural, conversational responses
5. **Multi-turn Conversations**: Continue talking without repeating wake word

## Hardware Requirements

### Minimum Setup
- Raspberry Pi 3B+ or newer
- USB microphone
- Speaker (3.5mm jack or USB)
- 8GB+ SD card

### Recommended Setup
- Raspberry Pi 4 (2GB+ RAM)
- USB conference speakerphone (e.g., Jabra Speak 410)
- -OR- ReSpeaker 2-Mic HAT
- Ethernet connection

## Quick Start

```bash
# Clone the repository
git clone https://github.com/nicholastripp/ha-realtime-assist
cd ha-realtime-assist

# Install dependencies and create virtual environment
./install.sh

# Setup configuration files (copies examples to working files)
./setup_config.sh

# Edit your API keys
nano .env

# Edit Home Assistant URL and other settings
nano config/config.yaml

# (Optional) Customize assistant personality
nano config/persona.ini

# Activate the virtual environment (required for all Python commands)
source venv/bin/activate

# Start the assistant
python src/main.py
```

**Note**: Always activate the virtual environment (`source venv/bin/activate`) before running any Python commands. You'll see `(venv)` in your terminal prompt when it's active.

## Configuration

The project includes example configuration files (`.example` suffix) that are copied to working files by `setup_config.sh`. Always edit the copies, not the example files.

### Required Settings
- **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com)
- **Home Assistant URL**: Your HA instance address
- **HA Long-Lived Token**: Generate in HA Profile settings

### Example config.yaml
```yaml
openai:
  api_key: ${OPENAI_API_KEY}
  voice: "nova"

home_assistant:
  url: "http://homeassistant.local:8123"
  token: ${HA_TOKEN}

wake_word:
  enabled: true
  model: "picovoice"
  sensitivity: 1.0
```

### Wake Word Setup

The assistant uses Picovoice Porcupine for accurate wake word detection. Built-in keywords include:
- "picovoice" (default)
- "alexa", "computer", "jarvis"
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

## What's New in v0.4.0-beta

- 🎤 **Custom Wake Words** - Create your own wake words like "Jarvis" with Picovoice Console
- 🛡️ **Robust Error Handling** - Graceful failures with helpful troubleshooting for connection issues
- 🔄 **Connection Retry Logic** - Automatic retry with exponential backoff for transient failures
- 📁 **Better Setup Process** - Automated configuration file setup including persona.ini
- 🔧 **Improved Diagnostics** - New connection testing tools and detailed error messages

See [CHANGELOG.md](CHANGELOG.md) for complete details.

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
python src/main.py --log-level DEBUG
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
- [Audio Setup](docs/AUDIO_SETUP.md) - Microphone and speaker configuration
- [Wake Word Setup](docs/WAKE_WORD_SETUP.md) - Wake word configuration
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](./CONTRIBUTING.md) first.

## License

TBD

## Acknowledgments

- Inspired by the [Billy B-Assistant](https://github.com/nickschaub/billy-b-assistant) project
- Built with [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime)
- Integrates with [Home Assistant](https://www.home-assistant.io)