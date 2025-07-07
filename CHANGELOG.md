# Changelog

All notable changes to the Home Assistant Realtime Voice Assistant project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-12-07

### Added
- Initial implementation of all core components
- OpenAI Realtime API WebSocket client with full event handling
- Home Assistant REST API client for entity state queries
- Home Assistant Conversation API client for natural language processing
- Function calling bridge connecting OpenAI to Home Assistant
- Audio capture system with device selection and real-time monitoring
- Audio playback system with queue management and volume control
- Real-time audio resampling between device rates and OpenAI's 24kHz requirement
- OpenWakeWord integration for local wake word detection
- Support for multiple wake word models (hey_jarvis, alexa, hey_mycroft, etc.)
- Two-stage audio pipeline design (wake word → OpenAI session)
- Configurable personality system based on Billy B-Assistant patterns
- YAML-based configuration with environment variable support
- Comprehensive test scripts for all major components
- Full async/await implementation for optimal performance
- Automatic reconnection and error recovery
- Session management with timeouts and activity tracking
- Multi-language support for both OpenAI and Home Assistant

### Technical Details
- **Architecture**: Modular component design with clear separation of concerns
- **Audio Format**: PCM16 at 24kHz for OpenAI, with automatic resampling
- **Wake Word**: OpenWakeWord at 16kHz with configurable sensitivity
- **Latency Target**: <800ms voice-to-voice response time
- **Python Version**: 3.9+ required
- **Platform**: Optimized for Raspberry Pi deployment

### Configuration
- Main configuration via `config/config.yaml`
- Personality configuration via `config/persona.ini`
- Environment variables for sensitive data (API keys, tokens)
- Support for multiple audio devices and custom wake words

### Testing
- All modules pass Python syntax validation
- Test scripts provided for:
  - Audio device enumeration and testing
  - Home Assistant connection validation
  - Wake word detection with multiple modes
  - Component integration testing

### Known Limitations
- Initial version focused on core functionality
- No GUI or web interface yet
- Limited to pre-trained wake word models
- No built-in usage tracking or analytics

### Next Steps
- Real-world testing on Raspberry Pi hardware
- Performance optimization based on testing results
- Additional wake word model support
- Enhanced error handling and recovery
- Usage analytics and cost tracking