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
- Production-ready (Jan 2025); all components tested on real Pi HW  
- Major technical issues (buffer, unicode) resolved

---

# SYSTEM ARCHITECTURE

- **App**: Standalone modular Pi app (not HACS); runs independently  
- **Wake Word**: OpenWakeWord, not HA  
- **Control Flow**: Audio input (Mic) → Wake Word → Audio to OpenAI Realtime API → Intent detection → If controlling HA, OpenAI function call bridges to HA Conversation API → HA action → Voice feedback  
- **Voice**: <800ms voice-to-voice latency, OpenAI’s streaming Realtime API (PCM16, 24kHz, mono)  
- **No Emoji**: ASCII only, avoid Unicode crashes on Pi (latin-1 terminals)

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

## Buffer Issues  
- **Symptom:** "input_audio_buffer_commit_empty"  
- **Cause:** Manual commits with server VAD enabled  
- **Fix:** Remove all manual `input_audio_buffer.commit` calls; rely on VAD  
- **Validation:** Test with correct audio format (PCM16, 24kHz, min 100ms); use test scripts

---

# SAMPLE CONFIGURATION

**Sample `config.yaml`**  
```yaml
openai:
  api_key: ${OPENAI_API_KEY}
  voice: "nova"
  temperature: 0.8
  language: "en"
home_assistant:
  url: "http://homeassistant.local:8123"
  token: ${HA_TOKEN}
  language: "en"
audio:
  input_device: "default"
  output_device: "default"
wake_word:
  enabled: true
  model: "hey_jarvis"
  sensitivity: 0.5
  timeout: 5.0
system:
  log_level: "INFO"
  session_timeout: 30
```

---

# NEXT STEPS / ENHANCEMENTS

**Deployment Checklist:**  
- Install on Pi, configure HA/OpenAI creds, test wake word, verify audio  
- Measure end-to-end latency  
- Test failure/recovery/multi-lang scenarios

**Planned Features:**  
- LED feedback, usage analytics, advanced error handling, easier install, multi-room

---

# DEVELOPMENT RULES (SUMMARY)

- Prioritize: Simplicity, clarity, maintainability, code quality, consistency  
- Always write and run tests cross-platform  
- Comprehensive debug logs; record lessons learned  
- *Never expose secrets*; always sanitize input/output  
- Keep code modular, under 200-300 lines per file, small focused modules  
- Follow version control and branching best practice

---