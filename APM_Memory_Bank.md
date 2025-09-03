# APM Memory Bank: HA Realtime Voice Assistant

## Project Overview

### Basic Information
- **Project Name:** Home Assistant Realtime Voice Assistant
- **Project Type:** Voice Interface / Smart Home Control
- **Primary Language:** Python 3.9+
- **Target Platform:** Raspberry Pi (3B+ or newer)
- **Current Version:** v1.1.5
- **Repository:** https://github.com/nicholastripp/ha-realtime-assist
- **License:** MIT

### Project Purpose
A standalone Raspberry Pi voice assistant providing natural, low-latency conversations for Home Assistant control using OpenAI's Realtime API. Unlike traditional sequential voice assistants, this enables real-time bidirectional audio streaming with <800ms response latency.

### Development Status
- **Overall Completion:** ~65%
- **Core Features:** COMPLETE
- **Production Ready:** YES (with known issues)
- **Active Development:** YES
- **Last Update:** 2025-07-20

## Technical Architecture

### Technology Stack

#### Core Technologies
- **Language:** Python 3.9+ with asyncio
- **Audio Processing:** sounddevice, numpy, scipy
- **Wake Word:** Picovoice Porcupine v3.0
- **Speech API:** OpenAI Realtime API (WebSocket)
- **Home Assistant:** Model Context Protocol (MCP)
- **Web Framework:** aiohttp with Jinja2

#### Key Dependencies
```
websockets>=10.0,<12.0  # OpenAI WebSocket
sounddevice>=0.4.6      # Audio I/O
numpy>=1.24.0,<2.0      # Audio processing
scipy>=1.11.0           # Resampling
mcp>=1.0.0              # Home Assistant integration
pvporcupine>=3.0.0      # Wake word detection
aiohttp>=3.9.0          # Web server & HA client
bcrypt>=4.2.0           # Password hashing
```

### System Architecture

#### Component Structure
```
src/
├── main.py                 # Main application entry
├── config.py               # Configuration management
├── personality.py          # Assistant personality
├── function_bridge_mcp.py  # OpenAI ↔ HA bridge
├── audio/
│   ├── capture.py          # Microphone input (48kHz → 24kHz)
│   ├── playback.py         # Speaker output (24kHz → 48kHz)
│   └── agc.py              # Automatic Gain Control
├── openai_client/
│   └── realtime.py         # WebSocket client
├── services/ha_client/
│   └── mcp_official.py     # MCP integration
├── wake_word/
│   └── porcupine_detector.py
└── web/                    # Web UI (port 8443)
    ├── app.py
    ├── auth.py
    └── routes/
```

#### Audio Processing Pipeline
1. **Input Capture** (48kHz) → sounddevice
2. **Volume Adjustment** → input_volume multiplier
3. **AGC Processing** → Dynamic gain control
4. **Resampling** → scipy.signal.resample to 24kHz
5. **Format Conversion** → Float32 to PCM16
6. **Wake Word Detection** → Porcupine at 16kHz
7. **OpenAI Streaming** → WebSocket at 24kHz

### Design Patterns
- **Event-Driven Architecture:** Async/await throughout
- **Pipeline Pattern:** Audio processing stages
- **Bridge Pattern:** OpenAI ↔ Home Assistant
- **Observer Pattern:** WebSocket event handling
- **Singleton:** Configuration management

## Current Implementation Status

### ✅ Completed Features

#### Core Voice Pipeline
- Real-time bidirectional audio streaming
- Wake word detection with Porcupine
- OpenAI Realtime API integration
- Multi-turn conversation support
- VAD-based conversation endings

#### Home Assistant Integration
- Model Context Protocol (MCP) support
- Dynamic tool discovery
- GetLiveContext for device states
- Function calling bridge
- SSL certificate support

#### Audio System
- Automatic Gain Control (AGC)
- Multi-device support
- Resampling (48kHz ↔ 24kHz)
- Feedback prevention
- Buffer management

#### Web Interface
- Complete dashboard (https://localhost:8443)
- Real-time status monitoring
- Configuration editor
- Personality customization
- Audio device testing
- Log viewer

#### Security
- HTTPS with self-signed certs
- bcrypt authentication
- CSRF protection
- Rate limiting
- Security headers

### 🔧 Known Issues

#### Critical
1. **Audio Distortion:** Multiple gain stages causing clipping
2. **SSE Connection Drops:** MCP connection instability
3. **Session Timeouts:** OpenAI 30-minute limit (partially fixed)

#### Important
1. **Audio Underrun:** Buffer management issues
2. **Wake Word Sensitivity:** Poor detection at normal volume
3. **Transcription Errors:** Due to distorted audio input

#### Minor
1. **Memory Usage:** Grows over time on Pi 3B+
2. **UI Responsiveness:** Slow on older Pi models
3. **Log Rotation:** Not working properly

## Configuration & Settings

### Environment Variables (.env)
```bash
OPENAI_API_KEY=sk-...              # OpenAI API key
HA_URL=http://homeassistant.local:8123  # HA instance
HA_TOKEN=eyJ0eXAi...              # HA long-lived token
PICOVOICE_ACCESS_KEY=...          # Porcupine key
WEB_UI_USERNAME=admin              # Web UI auth
WEB_UI_PASSWORD_HASH=$2b$12$...   # bcrypt hash
```

### Key Configuration (config.yaml)
```yaml
openai:
  model: "gpt-4o-realtime-preview"  # Needs update to gpt-realtime
  voice: "alloy"                    # 8 voices available
  
audio:
  sample_rate: 48000                # Device rate
  input_volume: 1.0                 # Gain multiplier
  agc_enabled: true                 # Auto gain control
  
wake_word:
  model: "picovoice"                # Built-in keyword
  sensitivity: 1.0                  # Detection threshold
  audio_gain: 1.0                   # Additional gain

session:
  conversation_mode: "multi_turn"   # or "single_turn"
  max_duration: 300                 # Audio timeout (seconds)
```

## Development Workflow

### Git Branch Strategy
- **main:** Stable releases
- **develop:** Active development
- **feature/*:** New features
- **fix/*:** Bug fixes

### Testing Approach
```bash
# Test suite locations
examples/
├── test_audio_devices.py     # Audio hardware test
├── test_ha_connection.py     # MCP/HA validation
├── test_openai_connection.py # API connectivity
├── test_wake_word.py         # Wake word detection
└── test_web_ui.py            # Web interface test
```

### Build & Deployment
```bash
# Installation
./install.sh              # Sets up environment

# Development
source venv/bin/activate  # Activate venv
python src/main.py       # Run assistant
python src/main.py --web # With web UI

# Systemd service
systemd/ha-voice-assistant.service
```

## Performance Metrics

### Current Performance
- **Response Latency:** ~800ms voice-to-voice
- **Wake Word Accuracy:** ~85% (degraded by audio issues)
- **Transcription Accuracy:** ~95% (when audio clean)
- **Memory Usage:** 250-400MB
- **CPU Usage:** 15-25% on Pi 4

### Target Performance
- **Response Latency:** <600ms
- **Wake Word Accuracy:** >95%
- **Transcription Accuracy:** >98%
- **Memory Usage:** <300MB
- **CPU Usage:** <20%

## API Integration Details

### OpenAI Realtime API
- **Protocol:** WebSocket
- **Audio Format:** PCM16 @ 24kHz
- **Model:** gpt-4o-realtime-preview (updating to gpt-realtime)
- **Pricing:** $40/1M input, $80/1M output tokens

### Home Assistant MCP
- **Endpoint:** /mcp_server/sse
- **Protocol:** Server-Sent Events
- **Tools:** Dynamic discovery
- **Context:** GetLiveContext for device states

## Lessons Learned

### What Works Well
1. Porcupine wake word detection (when audio clean)
2. WebSocket streaming for low latency
3. MCP for Home Assistant integration
4. Web UI for configuration

### What Needs Improvement
1. Audio pipeline complexity (7 transformation stages)
2. Gain stage management
3. Connection reliability
4. Error recovery

### Critical Insights
1. Audio quality is foundational - fix before features
2. Multiple gain stages compound distortion
3. Resampling quality matters for speech
4. AGC can mask underlying issues

## Future Considerations

### Immediate Priorities (Phase 1-2)
1. Fix audio pipeline distortion
2. Migrate to gpt-realtime model
3. Implement native MCP support
4. Add image input capability

### Medium-term Goals (Phase 3-4)
1. Stabilize connections
2. Add custom wake words
3. Implement conversation history
4. Multi-user support

### Long-term Vision (Phase 5+)
1. WebRTC for lower latency
2. SIP for phone integration
3. Local LLM fallback
4. Edge ML optimization

## Resource Links

### Documentation
- [README.md](./README.md) - Project overview
- [CHANGELOG.md](./CHANGELOG.md) - Version history
- [docs/](./docs/) - Detailed guides

### External Resources
- [OpenAI Realtime Docs](https://platform.openai.com/docs/guides/realtime)
- [Picovoice Console](https://console.picovoice.ai/)
- [Home Assistant MCP](https://www.home-assistant.io/integrations/mcp/)

### Community
- GitHub Issues: Bug reports and features
- Discord: Real-time support
- Forum: Long-form discussions

## Agent Instructions

### For Audio DSP Agent
- Focus on `src/audio/` directory
- Test with various microphones
- Measure signal quality at each stage
- Document optimal settings

### For API Migration Agent
- Update `src/openai_client/realtime.py`
- Test new model features
- Maintain backward compatibility
- Document breaking changes

### For Testing Agent
- Create comprehensive test matrices
- Automate audio quality tests
- Validate across Pi models
- Generate metrics reports

### For Integration Agent
- Focus on `src/services/ha_client/`
- Test MCP native support
- Improve connection resilience
- Document error patterns

## Task Completion Log

### Phase 1, Task 1.1: Audio Pipeline Analysis Tool Development
**Status**: COMPLETED  
**Agent**: Audio DSP Implementation Agent  
**Date**: 2025-09-03  
**Duration**: ~1 hour  

**Deliverables Created**:
1. `tools/audio_pipeline_diagnostic.py` - Main diagnostic tool (600+ lines)
2. `tools/audio_analysis/metrics.py` - Comprehensive audio metrics calculator
3. `tools/audio_analysis/visualization.py` - Visualization suite for audio analysis
4. `tools/audio_analysis/stage_capture.py` - Pipeline stage capture hooks
5. `tools/README_AUDIO_DIAGNOSTIC.md` - Complete documentation

**Key Achievements**:
- Implemented capture for all 7 pipeline stages
- Created 12+ audio quality metrics (RMS, THD, SNR, clipping, etc.)
- Built real-time monitoring with visual feedback
- Developed automated report generation (JSON/CSV)
- Added visualization suite (waveforms, spectrograms, gain cascade)

**Technical Findings**:
1. **Multiple Gain Stages**: Identified cumulative gain up to 25x possible
2. **PCM16 Conversion Issue**: Using 32767 multiplier may cause asymmetry
3. **Resampling Quality**: scipy.signal.resample lacks anti-aliasing config
4. **Test Signal Analysis**: THD ~37% in synthetic test (expected for multi-tone)

**Recommendations Implemented**:
- Stage-by-stage capture with isolated transformation analysis
- Real-time clipping detection and visualization
- Frequency response analysis per stage
- Automated gain optimization recommendations

**Next Steps**:
- Hook into actual audio pipeline (currently using simulation)
- Test with real microphone input
- Create device-specific profiles based on findings
- Integrate findings into main application configuration

## Revision History

| Date | Version | Changes |
|------|---------|---------|
| 2025-09-03 | 1.0 | Initial Memory Bank creation |
| 2025-09-03 | 1.1 | Added OpenAI API update details |
| 2025-09-03 | 1.2 | Completed Phase 1, Task 1.1 - Audio Pipeline Diagnostic Tool |

---

*This Memory Bank is the single source of truth for the HA Realtime Voice Assistant project. All agents should reference and update this document throughout development.*