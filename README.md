# Home Assistant Realtime Voice Assistant

A standalone Raspberry Pi voice assistant that provides natural, low-latency conversations for Home Assistant control using OpenAI's Realtime API.

## 🚀 Project Status: Implementation Complete!

All core components have been implemented and validated. The system is ready for real-world testing on Raspberry Pi hardware.

## Overview

This project creates a dedicated voice interface for Home Assistant that runs on a Raspberry Pi. Unlike traditional voice assistants that use a sequential "speak-wait-respond" pattern, this assistant enables natural, real-time conversations with <800ms response latency.

## Key Features

- 🎙️ **Natural Conversations**: Real-time bidirectional audio streaming
- ⚡ **Low Latency**: <800ms voice-to-voice response time  
- 🏠 **Full HA Control**: Uses Home Assistant's Conversation API
- 👂 **Local Wake Words**: OpenWakeWord with multiple model support
- 🌍 **Multi-Language**: Configurable language support
- 🎭 **Personality**: Customizable assistant personality
- 🔊 **Local Audio**: All audio processing happens on the Pi
- 🚀 **Easy Setup**: Simple configuration and installation

## How It Works

1. **Wake Word Detection**: Local detection using OpenWakeWord (hey_jarvis, alexa, etc.)
2. **Audio Streaming**: Captures voice and streams to OpenAI Realtime API
3. **Smart Control**: OpenAI understands intent and calls HA functions
4. **Natural Response**: Speaks back with natural, conversational responses

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
git clone https://github.com/yourusername/ha-realtime-voice-assistant
cd ha-realtime-voice-assistant

# Install dependencies
./install.sh

# Setup configuration files
./setup_config.sh

# Edit your settings (API keys and configuration)
nano .env
nano config/config.yaml

# Start the assistant (use virtual environment)
./venv/bin/python src/main.py
```

## Configuration

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
  model: "hey_jarvis"
  auto_download: true
  sensitivity: 0.5
```

### Wake Word Setup

The assistant uses OpenWakeWord for local wake word detection. Models are automatically downloaded on first run.

**Troubleshooting wake word issues:**

1. **Manual model download:**
   ```bash
   python download_wake_word_models.py --download-all
   ```

2. **Test wake word detection:**
   ```bash
   ./venv/bin/python examples/test_wake_word.py --interactive
   ```

3. **Check available models:**
   ```bash
   python download_wake_word_models.py --list
   ```

4. **Test model loading:**
   ```bash
   python download_wake_word_models.py --test
   ```

## Implementation Status

✅ **Core Implementation Complete** - All major components are implemented and ready for testing.

### Completed Components
- [x] **OpenAI Integration** - Full WebSocket client with event handling and function calling
- [x] **Audio System** - Capture/playback with resampling and device management
- [x] **Wake Word Detection** - OpenWakeWord integration with multiple models
- [x] **HA Integration** - REST and Conversation API clients
- [x] **Personality System** - Configurable traits and custom prompts
- [x] **Configuration** - YAML-based config with environment variables
- [x] **Test Scripts** - Comprehensive testing utilities

### Available Wake Words
- `hey_jarvis` - Default wake word
- `alexa` - Amazon Alexa compatible
- `hey_mycroft` - Mycroft compatible
- `hey_rhasspy` - Rhasspy compatible
- `ok_nabu` - Nabu Casa compatible

**Note**: Wake word models are automatically downloaded on first run. If you encounter issues, run:
```bash
python download_wake_word_models.py --download-all
```

### Testing & Validation

**Important**: All test scripts must be run using the virtual environment:

```bash
# Test wake word detection
./venv/bin/python examples/test_wake_word.py --interactive

# Test audio devices
./venv/bin/python examples/test_audio_devices.py

# Test Home Assistant connection
./venv/bin/python examples/test_ha_connection.py

# Test OpenAI connection
./venv/bin/python examples/test_openai_connection.py

# Run the full assistant
./venv/bin/python src/main.py --log-level DEBUG
```

**Alternative**: Activate the virtual environment first:
```bash
source venv/bin/activate
python3 examples/test_audio_devices.py
python3 src/main.py
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

- [CLAUDE.md](./CLAUDE.md) - Comprehensive technical documentation
- [Installation Guide](./docs/INSTALL.md) - Detailed setup instructions
- [Configuration Guide](./docs/CONFIG.md) - All configuration options
- [Hardware Guide](./docs/HARDWARE.md) - Hardware recommendations

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](./CONTRIBUTING.md) first.

## License

TBD

## Acknowledgments

- Inspired by the [Billy B-Assistant](https://github.com/nickschaub/billy-b-assistant) project
- Built with [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime)
- Integrates with [Home Assistant](https://www.home-assistant.io)