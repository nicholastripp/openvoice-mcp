**Condensed CLAUDE.md (Project-Critical Info Only)**

---

# PROJECT SUMMARY  
**Goal**: Standalone Raspberry Pi voice assistant for Home Assistant using OpenAI’s Realtime API (low-latency, natural conversation, NO HACS, independent from HA updates).  
- Audio handled locally; <800ms latency target  
- Local wake word detection (OpenWakeWord)  
- REST/Conversation API to communicate with HA  
- Configurable personality/multi-language support  
- RasPi hardware: audio always local, wake word processed locally

## Repository  
- [GitHub](https://github.com/nicholastripp/ha-realtime-assist) | MIT license | Public | v0.1.0  
- **Current Status (July 2025)**: Core functionality working, recent audio quality improvements  
- Major technical issues (buffer, unicode) resolved; ongoing audio processing optimization

---

# SYSTEM ARCHITECTURE

- **App**: Standalone modular Pi app (not HACS); runs independently  
- **Wake Word**: OpenWakeWord, not HA  
- **Control Flow**: Audio input (Mic) → Wake Word → Audio to OpenAI Realtime API → Intent detection → If controlling HA, OpenAI function call bridges to HA Conversation API → HA action → Voice feedback  
- **Voice**: <800ms voice-to-voice latency, OpenAI’s streaming Realtime API (PCM16, 24kHz, mono)  
- **NEVER use Emoji**: ASCII only, avoid Unicode crashes on Pi (latin-1 terminals)

---

# TECHNICAL IMPLEMENTATION

**Key Components**  
- **WebSocket client**: OpenAI Realtime API  
- **Audio**: sounddevice (I/O), numpy/scipy (resampling), PCM16 at 24kHz  
- **Wake Word**: OpenWakeWord (16kHz) with configurable models  
- **HA client**: REST + Conversation API  
- **Config**: YAML (main), INI (persona), dotenv for secrets  
- **Service**: systemd (autostart)  
- **Testing**: Must run/validate on Raspberry Pi, not just macOS

**Architecture Diagram**  
```text
Pi Voice Assistant ──[WebSocket]──► OpenAI Realtime API ──► HA API (REST/Conversation)
  | WakeWord | Audio I/O | Function Call Bridge |   | Device control/feedback |
```

**Functions & Patterns**  
- Modular design: Audio capture (wake & convo), OpenAI session, HA API client  
- Function calls: OpenAI does not directly control HA–the app bridges via function tool definitions and conversation API  
- Server VAD: Use OpenAI's server VAD; *do not* call `input_audio_buffer.commit` if using VAD—fixes empty buffer errors  
- Asynchronous flow throughout (async/await)

**Wake Word Pipeline**  
1. Stage 1: OpenWakeWord always listening (~16kHz, low CPU)  
2. Stage 2: On detection, switch to OpenAI streaming (24kHz, bidirectional, <800ms V2V)

**Audio Format**  
- PCM16, 24kHz, mono, little-endian, base64  
- Chunks: 50ms/1200 samples  
- Min buffer: 100ms/2400 samples  
- Only server VAD handles commit; never call commit if VAD is on

---

# ENVIRONMENT & TESTING

## Key Environment Differences  
- **Development:** macOS, up-to-date libraries, UTF-8 terminal  
- **Production:** Raspberry Pi OS (ARM), system Python (possible encoding issues); default terminal often latin-1  
- **Audio device enumeration**: cross-platform differences

**Critical Requirements:**  
- All features must be tested on Pi HW, not just emulated  
- No emoji usage in comments or output  
- Always sanitize Unicode/ASCII for output/logs  
- Validate audio library compatibility across platforms  

---

# DEPENDENCIES (EXCERPT OF REQUIREMENTS)  
- websockets>=12.0, sounddevice>=0.4.6, numpy>=1.24.0, scipy>=1.11.0, aiohttp>=3.9.0  
- openwakeword>=0.6.0, onnxruntime>=1.16.0  
- pyyaml>=6.0, configparser>=5.3.0, python-dotenv>=1.0.0

---

# TROUBLESHOOTING

## Critical Issues and Fixes

### Buffer Issues  
- **Symptom:** "input_audio_buffer_commit_empty"  
- **Cause:** Manual commits with server VAD enabled  
- **Fix:** Remove all manual `input_audio_buffer.commit` calls; rely on VAD  
- **Validation:** Test with correct audio format (PCM16, 24kHz, min 100ms); use test scripts

### Wake Word Detection Regressions (July 2025)
- **Symptom:** "'WakeWordDetector' object has no attribute '_reset_model'"  
- **Cause:** Method name mismatch in session cleanup code  
- **Fix:** Use `reset_audio_buffers()` instead of `_reset_model()`  
- **Status:** Fixed in commit a48a930

### Audio Quality Issues
- **Symptom:** Low audio levels, garbled OpenAI responses, VAD timeout  
- **Cause:** Insufficient audio gain, wrong device selection, RMS normalization issues  
- **Fixes Applied:**
  - RMS-based audio normalization (target 0.1 RMS level)
  - Intelligent gain control (1.0x to 20.0x range)
  - Audio quality validation before OpenAI transmission
  - Session state validation to prevent excessive streaming costs
- **Current Status:** Audio processing improved but may need further tuning

### Configuration Issues
- **Wake Word Model Mismatch:** Config showing "alexa" but system using "hey_jarvis_v0.1"  
- **Audio Device Selection:** Direct USB device (index 1) vs system default  
- **Sensitivity Threshold:** Ongoing optimization between 0.0001-0.0005 range

---

# SAMPLE CONFIGURATION

**Current Working `config.yaml` (July 2025)**  
```yaml
openai:
  api_key: ${OPENAI_API_KEY}
  voice: "alloy"                    # Current: alloy (was nova)
  model: "gpt-4o-realtime-preview"
  temperature: 0.8
  language: "en"
home_assistant:
  url: "https://friday.earth616.cc"  # Production URL
  token: ${HA_TOKEN}
  language: "en"
  timeout: 10
audio:
  input_device: "default"           # Reverted from device index 1
  output_device: "default"
  sample_rate: 48000
  channels: 1
  chunk_size: 1200                 # 50ms at 24kHz
  input_volume: 2.0                # Increased for VAD detection
  output_volume: 1.0
  feedback_prevention: true        # Prevent cost issues
wake_word:
  enabled: true
  model: "hey_jarvis"              # Fixed: was "alexa", now matches system
  sensitivity: 0.004               # Balance between false positives and detection
  timeout: 5.0
  vad_enabled: false               # Disabled (was causing issues)
  cooldown: 2.0
  audio_gain: 3.5                  # Audio amplification (1.0-5.0)
  audio_gain_mode: "fixed"         # "fixed" or "dynamic"
session:
  conversation_mode: "multi_turn"  # Working multi-turn support
  multi_turn_timeout: 30.0
  multi_turn_max_turns: 10
  timeout: 30
  auto_end_silence: 3.0
  max_duration: 300
system:
  log_level: "INFO"
  daemon: false
```

---

# MULTI-TURN CONVERSATION SUPPORT

**Feature Overview:**  
- Support for natural conversation flow without requiring wake words for follow-up questions
- Configurable conversation modes: single-turn (default) or multi-turn
- Automatic session management with timeout and turn limits
- Natural conversation ending with phrase detection

**Configuration Options:**
```yaml
session:
  conversation_mode: "multi_turn"      # Enable multi-turn conversations
  multi_turn_timeout: 30.0            # Seconds to wait for follow-up questions
  multi_turn_max_turns: 10            # Maximum conversation turns per session
  multi_turn_end_phrases:             # Phrases to end conversation
    - "goodbye"
    - "stop"
    - "that's all" 
    - "thank you"
    - "bye"
```

**Usage Example:**
1. User: "Hey Jarvis" (wake word)
2. Assistant: "How can I help you?"
3. User: "What's the weather today?" (no wake word needed)
4. Assistant: "It's 72 degrees and sunny"
5. User: "What about tomorrow?" (no wake word needed)
6. Assistant: "Tomorrow will be 68 degrees with light rain"
7. User: "Thank you" (conversation ends automatically)

**Technical Implementation:**
- New session state: `MULTI_TURN_LISTENING`
- Leverages OpenAI's stateful WebSocket connection for conversation context
- VAD remains active after responses for follow-up detection
- Comprehensive error recovery and timeout handling
- Session watchdog prevents stuck conversations

**Benefits:**
- Natural conversation flow similar to commercial voice assistants
- Reduced wake word fatigue for multi-step interactions
- Better user experience for complex queries requiring clarification
- Maintains existing single-turn compatibility

---

# NEXT STEPS / ENHANCEMENTS

**Deployment Checklist:**  
- Install on Pi, configure HA/OpenAI creds, test wake word, verify audio  
- Measure end-to-end latency  
- Test failure/recovery/multi-lang scenarios

**Planned Features:**  
- LED feedback, usage analytics, advanced error handling, easier install, multi-room

---

# CURRENT PROJECT STATE (November 2024)

## Latest Updates
1. **Wake Word Gain System**: Configurable audio gain (default 3.5x, range 1.0-5.0)
2. **Audio Buffer Flushing**: Implemented to prevent model stuck states
3. **Project Organization**: Test scripts moved to `tools/`, docs organized in `docs/`
4. **Multi-turn Conversations**: Fully working with configurable timeout and phrases

## Configuration Updates
- **Wake Word Gain**: Now configurable in `config.yaml`:
  ```yaml
  wake_word:
    audio_gain: 3.5              # Audio amplification (1.0-5.0)
    audio_gain_mode: "fixed"     # "fixed" or "dynamic"
  ```
- **Sensitivity Tuning**: Balance between false positives and detection reliability
- **Model Selection**: Ensure wake word model matches configuration

## Known Issues & Solutions
1. **Low Wake Word Confidence**:
   - Increase `audio_gain` to 4.0 or 4.5 if needed
   - Check microphone input levels in system settings
   - Consider bounded dynamic gain mode for variable conditions

2. **False Positives**:
   - Reduce gain or increase sensitivity threshold
   - Current sweet spot: gain 3.5x with sensitivity 0.004-0.006

3. **Model Stuck States**:
   - Audio flush mechanism implemented
   - Automatic model reset on detection
   - Variance-based stuck detection

## Project Structure
```
ha-realtime-assist/
├── src/                     # Main application code
├── config/                  # Configuration files
├── tools/                   # Test scripts and utilities
├── docs/                    # Project documentation
│   └── troubleshooting/     # Specific issue guides
├── reference-documentation/ # External references
└── examples/               # Clean example scripts
```

## Testing Notes
- **Platform**: All testing must be done on Raspberry Pi
- **Audio Diagnostics**: Use `tools/test_audio_diagnostics.py`
- **Gain Testing**: Use `tools/test_gain_settings.py`
- **Wake Word Testing**: Check RMS levels and confidence values in logs

---

# DEVELOPMENT RULES (SUMMARY)

- Prioritize: Simplicity, clarity, maintainability, code quality, consistency  
- Always write and run tests cross-platform  
- Comprehensive debug logs; record lessons learned  
- *Never expose secrets*; always sanitize input/output  
- Keep code modular, under 200-300 lines per file, small focused modules  
- Follow version control and branching best practice

---