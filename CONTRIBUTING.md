# Contributing to HA Realtime Voice Assistant

Thank you for your interest in contributing! This project aims to create a natural voice interface for Home Assistant using OpenAI's Realtime API.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ha-realtime-voice-assistant
   cd ha-realtime-voice-assistant
   ```

2. **Install development dependencies**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements-dev.txt
   ```

3. **Set up configuration**
   ```bash
   cp config/config.yaml.example config/config.yaml
   cp config/persona.ini.example config/persona.ini
   cp .env.example .env
   # Edit these files with your settings
   ```

## Project Structure

- `src/` - Main source code
  - `audio/` - Audio capture and playback
  - `ha_client/` - Home Assistant API integration
  - `openai_client/` - OpenAI Realtime API client
  - `wake_word/` - Wake word detection
  - `utils/` - Utility functions
- `config/` - Configuration files
- `tests/` - Test files
- `docs/` - Documentation
- `systemd/` - Service files

## Code Style

We use Black for code formatting and flake8 for linting:

```bash
black src/
flake8 src/
```

## Testing

Run tests with pytest:

```bash
pytest tests/
```

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Areas for Contribution

- **Core Implementation**: OpenAI client, audio pipeline, HA integration
- **Wake Word Detection**: Integration with various wake word systems
- **Documentation**: Setup guides, troubleshooting, examples
- **Testing**: Unit tests, integration tests, hardware testing
- **Hardware Support**: Support for different Pi models and audio hardware

## Questions?

Feel free to open an issue for discussion before working on major features.