# Changelog

All notable changes to the Home Assistant Realtime Voice Assistant project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.4] - 2025-07-20

### Fixed
- **Multiple Audio Timeout Issues** - Fixed various hardcoded timeouts that were cutting off long AI responses:
  - AudioPlayback now uses configurable `session.max_duration` instead of hardcoded 30 seconds
  - "Stuck audio" detection in main.py now uses `session.max_duration` instead of hardcoded 45 seconds
  - Periodic cleanup "stuck audio" check now uses `session.max_duration` instead of 30-second cleanup interval
  - Session state duration check now uses `session.max_duration` for audio states (RESPONDING, AUDIO_PLAYING)
  - All audio timeout mechanisms now respect the same configurable value (default 300 seconds)
- **WebSocket Reconnection Rate Limiting** - Fixed aggressive reconnection loop after session expiry:
  - Implemented exponential backoff for reconnection attempts (3s → 6s → 12s... up to 120s)
  - Added proper authentication check before accepting WebSocket connections
  - Dashboard shows clear "Session Expired" message instead of endless reconnection attempts
- **Web UI Issues**:
  - Fixed shutdown hang caused by unclosed WebSocket connections
  - Fixed OpenAI status showing "Disconnected" instead of "Ready (On-Demand)"
  - Fixed version display to dynamically show current version instead of hardcoded v1.1.0
  - Fixed module import error when running main.py directly

### Changed
- **Configuration Clarification** - `session.max_duration` now clearly documented as "Maximum audio response duration"
- **Default Session Timeout** - Web UI session timeout increased from 1 hour to 7 days for better usability

## [1.1.3] - 2025-07-20

### Fixed
- **OpenAI 30-minute session timeout** - Complete fix for "Your session hit the maximum duration of 30 minutes" error:
  - OpenAI connection now established on-demand (after wake word) instead of at startup
  - WebSocket properly disconnects after each voice session
  - Fixed reconnection logic to check actual WebSocket status, not just connection state
  - No more idle connections consuming resources or hitting timeout limits

### Changed
- **Connection Management** - OpenAI WebSocket connections are now created per-session for better resource efficiency

## [1.1.2] - 2025-07-19

### Fixed
- **MCP Test Script** - Rewrote test_ha_connection.py to use new MCP architecture instead of deprecated Conversation API
- **Audio Test Page** - Fixed poor text contrast (light gray on white) with proper CSS styling
- **--skip-ha-check Flag** - Multiple fixes for proper operation:
  - Flag now works when Home Assistant is available (not just on connection failure)
  - OpenAI connection no longer fails when no tools are available
  - Audio output now works correctly (was silent despite wake word and recording working)
- **MCP SSE Shutdown** - Fixed scary traceback on Ctrl+C with improved signal handling and graceful cleanup
- **README License** - Updated license section from "TBD" to proper MIT license reference

### Added
- **Wake Word Instructions** - Added helpful text with link to Picovoice Console for creating custom wake words

### Changed
- **OpenAI Initialization** - Restructured to ensure audio event handlers are always set up when not in wake-word-only mode

## [1.1.1] - 2025-07-18

### Security
- **CSRF Protection** - Double-submit cookie pattern for all state-changing operations
- **Rate Limiting** - Sliding window rate limiting for authentication (5/min), API (100/min), and config (20/min) endpoints
- **Security Headers** - Comprehensive OWASP-recommended headers including CSP, HSTS, X-Frame-Options
- **File Permissions** - Automatic security warnings for insecure config directory permissions

### Added
- **Web UI Enhancements**
  - Custom wake word upload functionality (.ppn files)
  - Secure application restart via web UI (Apply & Restart button)
  - Conversation mode setting (single-turn/multi-turn) in OpenAI settings
  - Health check endpoint for monitoring restart completion
- **Configuration Improvements**
  - YAML comment preservation system (temporarily disabled due to edge cases)
  - Better config file consistency when saving via web UI

### Changed
- Wake word dropdown now shows only valid Porcupine built-in keywords
- Custom wake words display user-friendly names in UI
- Terminal output uses ASCII characters instead of Unicode for better compatibility
- Web UI branding consistently shows "HA Realtime Voice Assistant"
- Extended silence threshold adjusted to 8.5 seconds in example config

### Fixed
- YAML config corruption when saving via web UI (section boundaries preserved)
- Custom wake word crash (now preserves .ppn extension in config)
- Wake word upload directory (now correctly uses config/wake_words/)
- UI element ordering (wake word selection after sensitivity settings)
- Removed invalid wake word options (jarvis, hey google, hey siri, ok google)
- Unicode character display issues in terminal output
- Octal notation display (0o755 → 755) for file permissions

### Removed
- Invalid wake word options that aren't built into Porcupine

## [1.1.0] - 2025-07-17

### Added
- **Complete Web UI** - Full-featured web interface for configuration and monitoring
  - Setup wizard for first-time configuration
  - Environment variable editor with secure handling
  - YAML configuration editor
  - Personality editor with visual sliders and preview
  - Real-time status dashboard with WebSocket updates
  - Audio device testing and configuration
  - Log viewer with syntax highlighting
- **Security Features**
  - HTTPS/TLS support with automatic self-signed certificate generation
  - Basic authentication with bcrypt password hashing
  - Secure session management with configurable timeout
  - Password configuration during installation
- **Enhanced Installation**
  - Interactive web UI security setup in install.sh
  - Automatic bcrypt password hashing
  - Environment variable storage for password hash

### Changed
- **Password Storage** - Migrated from SHA256 to bcrypt for improved security
- **Web UI Port** - Now respects config file port setting when using --web flag
- **Installation Process** - Added optional web UI security configuration prompts
- **Multi-turn Conversations** - Natural conversation endings using VAD-based silence detection
  - Replaced fixed 30s timeout with 8s extended silence threshold
  - Multi-turn timeout increased to 5 minutes (safety fallback only)
  - Conversations end naturally when both parties stop talking

### Fixed
- Web UI port configuration now correctly uses config.yaml settings
- WebUIConfig creation error with nested dataclasses resolved
- Multi-turn conversation response completion issues
- Certificate generation for web UI HTTPS support
- WebSocket real-time status updates (proper wss:// protocol for HTTPS)
- Config file preservation when saving through web UI (deep merge)
- Web UI connection status display showing actual states
- Installer now updates password_hash field in existing config files

### Security
- Passwords are now hashed using bcrypt (12 rounds) instead of SHA256
- Password hashes stored in .env file instead of config.yaml
- HTTPS enabled by default for web UI with strong cipher suites
- Authentication required by default when web UI listens on all interfaces

## [1.0.0] - 2025-07-16

### BREAKING CHANGES
- Replaced Home Assistant Conversation API with Model Context Protocol (MCP)
- Requires Home Assistant 2025.2 or later
- Requires MCP Server integration to be installed and enabled
- No backward compatibility with previous versions

### Added
- **Model Context Protocol (MCP) integration** - Direct tool-based control of Home Assistant
- **GetLiveContext tool support** - Comprehensive device state awareness
- **Automatic tool discovery** - Dynamically discovers and maps all available MCP tools
- **SSL certificate verification options** - Support for self-signed certificates
- **Enhanced multi-turn conversation** - Fixed timeout issues for smoother conversations
- **Improved error handling** - Better user feedback for connection and execution errors

### Fixed
- Multi-turn conversation timeouts no longer cut off audio responses
- JSON serialization errors with MCP TextContent objects
- Unicode encoding issues on Raspberry Pi
- Audio responses after function calls now play to completion
- GetLiveContext response logging moved to DEBUG level

### Changed
- Complete rewrite of Home Assistant integration layer using MCP
- Improved device state caching and management
- Better error handling with specific MCP error messages
- Enhanced logging for debugging MCP connections
- Reduced console output for cleaner user experience

### Removed
- Home Assistant Conversation API support (replaced by MCP)
- Legacy REST API client code
- Over 70 development test scripts (cleaned up for release)
- Backward compatibility with pre-MCP versions

## [0.5.0-beta] - 2025-07-15

### Added
- **Improved Logging System** - Separate console and file log levels for better user experience
- **Console Output Control** - New CLI flags: `--verbose` for debug output, `--quiet` for errors only
- **Smart Console Formatting** - Clean, minimal output with contextual prefixes (✓, ●, ►, ✗, ⚠)
- **Log Rotation** - Configurable log file size limits and automatic rotation
- **Memory Bank Documentation** - Comprehensive project state tracking in `.claude/memory/`

### Changed
- **Cleaner Default Output** - Replaced verbose debug prints with concise status messages
- **Logging Configuration** - Added `console_log_level`, `log_to_file`, `log_max_size_mb`, `log_backup_count` options
- **Debug Information** - All technical details now go to log file, shown on console only with --verbose

### Fixed
- **Invalid API Call** - Removed call to non-existent `/api/conversation/agents` endpoint

## [0.4.0-beta] - 2025-07-14

### Added
- **Custom Wake Word Support** - Use custom Picovoice wake words (.ppn files)
- **Connection Diagnostics** - New `test_connection()` method with detailed error analysis
- **Retry Logic** - Automatic retry with exponential backoff for transient failures
- **Connection Test Script** - `examples/test_connection_error_handling.py` for debugging
- **Skip HA Check Flag** - `--skip-ha-check` to run without Home Assistant connection
- **Wake Words Directory** - `config/wake_words/` for storing custom wake word files

### Changed
- **Enhanced Error Messages** - User-friendly error messages with troubleshooting steps
- **Improved Setup Process** - `setup_config.sh` now handles persona.ini copying
- **Better URL Validation** - Validates Home Assistant URL format before connection
- **Updated Documentation** - Clearer quickstart instructions and custom wake word guide

### Fixed
- **Setup Instructions** - README now correctly documents copying example files
- **Wake Word Documentation** - Corrected list of built-in Porcupine keywords
- **Error Handling** - Graceful failures instead of crashes on connection errors

## [0.3.0] - 2025-07-13

### Changed
- **BREAKING**: Removed OpenWakeWord support - Porcupine is now the only wake word engine
- Simplified wake word configuration by removing engine selection
- Reduced dependencies by ~100MB (removed TensorFlow Lite, ONNX runtime)

### Removed
- OpenWakeWord engine and all related code
- Wake word model download functionality
- Engine selection from configuration

## [0.2.0-beta] - 2025-07-12

### Added
- **Multi-turn conversation support** - Natural back-and-forth conversations without re-triggering wake word
- **Automatic Gain Control (AGC)** - Automatically adjusts input volume to prevent clipping and maintain optimal levels
- **Porcupine wake word engine** - Support for Picovoice Porcupine with built-in wake words
- **Audio diagnostics tools** - Comprehensive audio testing and debugging utilities
- **Response ID tracking** - Ensures proper audio playback for multiple responses
- **High-pass filter configuration** - Required for Porcupine, configurable cutoff frequency
- **Volume attenuation support** - Input volume can now be set below 1.0 for loud microphones
- **Language enforcement** - Explicit language instructions to prevent responses in wrong languages
- Wake word audio gain configuration (fixed and dynamic modes)
- Configurable gain values (1.0-5.0 range)
- Natural conversation ending with phrase detection
- Comprehensive audio tuning guide (docs/audio_tuning_guide.md)
- Soft limiting for audio to prevent harsh distortion
- Clipping detection and logging throughout audio pipeline

### Changed
- **Project structure** - Reorganized into clean directory structure:
  - `examples/` - User-facing test utilities
  - `tests/` - All test scripts and audio samples
  - `docs/` - Consolidated documentation with archives
- **Default audio settings** to prevent distortion:
  - Wake word audio_gain default: 3.5 → 1.0
  - Input volume supports values < 1.0 for attenuation
  - Consistent audio normalization using 32767
- **Development Status** - Updated from Alpha to Beta

### Fixed
- **Audio completion hang** - Fixed infinite loop in audio completion callbacks causing "Audio underrun #2" errors
- **Multi-turn response initialization** - Fixed start_response() not being called for follow-up responses
- **Wake word detection issues**:
  - Invalid wake word configurations (e.g., "hey_jarvis" not being built-in)
  - Multi-word wake words with spaces breaking Porcupine
  - Porcupine instance cleanup preventing reinitialization
  - Audio gain causing distortion and preventing wake word detection
- **Home Assistant device exposure** - Removed artificial limits on exposed devices
- **Audio clipping with high gain** - Fixed integer overflow issues with gain multiplication
- Wake word model stuck at low confidence values
- Audio buffer flushing mechanism for model reset
- False positive wake word detections
- Configuration synchronization between config.yaml and config.yaml.example
- OpenAI receiving distorted audio leading to misunderstanding
- Inconsistent normalization values causing DC bias

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