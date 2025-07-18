# Configuration Guide

This guide covers all configuration options for the Home Assistant Realtime Voice Assistant.

## Configuration Files

The assistant uses three main configuration files:

1. **`.env`** - Environment variables for sensitive data (API keys, tokens)
2. **`config/config.yaml`** - Main configuration file
3. **`config/persona.ini`** - Assistant personality configuration

## Environment Variables (.env)

Create a `.env` file in the project root with your API credentials:

```bash
# OpenAI API Key
OPENAI_API_KEY=sk-your-openai-api-key-here

# Home Assistant Long-Lived Access Token
HA_TOKEN=your-home-assistant-token-here

# Web UI Password Hash (set by installer, uses bcrypt)
WEB_UI_PASSWORD_HASH=
```

## Main Configuration (config/config.yaml)

### OpenAI Settings

```yaml
openai:
  api_key: ${OPENAI_API_KEY}        # From environment variable
  voice: "alloy"                    # Voice options: alloy, ash, ballad, coral, echo, sage, shimmer, verse
  model: "gpt-4o-realtime-preview"  # OpenAI Realtime model
  temperature: 0.8                  # Response creativity (0.0-1.0)
  language: "en"                    # Language code (en, es, fr, de, etc.)
```

### Home Assistant Settings

```yaml
home_assistant:
  url: "http://homeassistant.local:8123"  # Your HA instance URL
  token: ${HA_TOKEN}                      # Long-lived access token
  language: "en"                          # Must match OpenAI language
  timeout: 10                             # API timeout in seconds
```

### Audio Configuration

```yaml
audio:
  input_device: "default"           # Audio input device (name or index)
  output_device: "default"          # Speaker device (name or index)
  sample_rate: 48000               # Device sample rate (48000, 44100, etc.)
  channels: 1                      # Audio channels (1=mono, 2=stereo)
  chunk_size: 1200                 # Audio chunk size (50ms at 24kHz)
  input_volume: 1.0                # Input volume multiplier (0.1-5.0, <1.0 reduces)
  output_volume: 2.0               # Output volume multiplier
  feedback_prevention: true        # Prevent audio loops
  feedback_threshold: 0.1          # Feedback detection threshold
  mute_during_response: true       # Mute mic during assistant response
  
  # Automatic Gain Control (AGC)
  agc_enabled: false              # Enable automatic volume adjustment
  agc_target_rms: 0.3            # Target audio level (0.0-1.0, 0.3 = 30% of max)
  agc_max_gain: 3.0              # Maximum gain multiplier
  agc_min_gain: 0.1              # Minimum gain multiplier
  agc_attack_time: 0.5           # Seconds to reduce gain when clipping
  agc_release_time: 2.0          # Seconds to increase gain when quiet
  agc_clipping_threshold: 0.05   # Maximum acceptable clipping ratio (5%)
```

### Wake Word Configuration

```yaml
wake_word:
  enabled: true                   # Enable wake word detection
  model: "picovoice"             # Porcupine built-in keyword
  sensitivity: 1.0               # Detection sensitivity (0.0-1.0)
  timeout: 5.0                   # Session timeout after wake word
  vad_enabled: false             # Voice activity detection
  cooldown: 2.0                  # Seconds between detections
  audio_gain: 1.0                # Audio amplification (1.0-5.0)
  audio_gain_mode: "fixed"       # Gain mode: "fixed" or "dynamic"
  
  # Porcupine settings
  porcupine_access_key: ${PICOVOICE_ACCESS_KEY}  # From environment variable
  highpass_filter_enabled: true   # Required for Porcupine
  highpass_filter_cutoff: 80.0    # Hz - removes low frequency noise
```

Available Porcupine built-in keywords:
- `picovoice` - Default wake word
- `alexa` - Amazon Alexa compatible
- `computer` - Star Trek style
- `jarvis` - Iron Man inspired
- `terminator` - Sci-fi themed
- Plus: americano, blueberry, bumblebee, grapefruit, grasshopper, hey google, hey siri, ok google, porcupine

### Session Configuration

```yaml
session:
  timeout: 30                     # Session timeout in seconds
  auto_end_silence: 3.0          # End session after silence
  max_duration: 300              # Maximum session duration
  interrupt_threshold: 0.5       # Voice activity to interrupt AI
  
  # Multi-turn conversation settings
  conversation_mode: "multi_turn"  # "single_turn" or "multi_turn"
  multi_turn_timeout: 300.0       # Safety fallback timeout (5 minutes)
  extended_silence_threshold: 8.0 # Natural conversation end after extended silence
  multi_turn_max_turns: 10        # Max conversation turns
  multi_turn_end_phrases:         # Phrases to end conversation
    - "goodbye"
    - "stop"
    - "that's all"
    - "thank you"
    - "bye"
```

**v1.1.0 Multi-turn Improvements**:
- Natural conversation endings based on VAD-detected silence
- Extended silence threshold (default 8s) replaces the fixed 30s timeout
- Multi-turn timeout increased to 5 minutes (safety fallback only)
- Conversations end naturally when both parties stop talking

### System Configuration

```yaml
system:
  log_level: "INFO"              # DEBUG, INFO, WARNING, ERROR, CRITICAL
  log_file: "logs/assistant.log" # Log file path
  led_gpio: null                 # GPIO pin for status LED (optional)
  daemon: false                  # Run as daemon process
```

### Web UI Configuration

Enable and configure the optional web interface with security features:

```yaml
web_ui:
  enabled: false                # Enable web UI on startup
  host: "0.0.0.0"              # Listen address (0.0.0.0 for all interfaces, 127.0.0.1 for localhost only)
  port: 8443                   # Web UI port (8443 for HTTPS by default)
  
  # Authentication settings
  auth:
    enabled: true              # Require login (highly recommended)
    username: "admin"          # Default username
    password_hash: ${WEB_UI_PASSWORD_HASH}  # From .env (set by installer)
    session_timeout: 3600      # Session timeout in seconds
  
  # TLS/HTTPS settings
  tls:
    enabled: true              # Use HTTPS (highly recommended)
    cert_file: ""              # Path to certificate (empty = self-signed)
    key_file: ""               # Path to private key (empty = self-signed)
```

**Security Features**:
- **Authentication**: Basic auth with bcrypt hashed passwords (12 rounds)
- **HTTPS**: Encrypted connections with TLS 1.2+
- **Self-signed certificates**: Generated automatically using OpenSSL
- **Session management**: Configurable timeouts with secure cookies

**Setup**: The installer will prompt you to configure web UI security during installation:
1. Choose a username (default: admin)
2. Set a secure password (min 8 characters recommended)
3. The installer automatically:
   - Hashes your password with bcrypt (12 rounds)
   - Stores the hash in .env file
   - Generates a self-signed certificate
   - Updates config.yaml with username

**Certificate Management**:
- Self-signed certificates are stored in `config/certs/`
- Valid for 365 days from creation
- To use custom certificates, place them in `config/certs/` and update the config:
  ```yaml
  tls:
    cert_file: "config/certs/your-cert.pem"
    key_file: "config/certs/your-key.pem"
  ```

## Personality Configuration (config/persona.ini)

Customize the assistant's personality and behavior:

```ini
[personality]
style = friendly
curiosity = helpful and informative
humor = light and clever
verbosity = concise
formality = casual

[messages]
greeting = How can I help you today?
confirmation = Got it!
error = I'm having trouble with that right now.
thinking = Let me think about that...

[instructions]
# Custom instructions for the assistant
# Add specific behaviors or knowledge here
```

## Audio Device Selection

To find available audio devices:

```bash
# Activate virtual environment first
source venv/bin/activate

# Then list devices
python tools/test_audio_devices.py
```

Then update your config with the device name or index:

```yaml
audio:
  input_device: "USB Audio Device"  # Use device name
  output_device: 2                   # Or use device index
```

## Wake Word Sensitivity Tuning

The wake word sensitivity determines how easily the assistant responds:

- **Lower values (0.001-0.003)**: More sensitive, may have false positives
- **Default (0.004)**: Balanced detection
- **Higher values (0.005-0.010)**: Less sensitive, may miss quiet commands

To test and tune:

```bash
# Activate virtual environment first
source venv/bin/activate

# Then test wake word
python tools/test_wake_word.py --interactive
```

## Audio Gain and AGC

### Manual Gain Control

Adjust input volume for your microphone:

```yaml
audio:
  input_volume: 1.0              # 0.1-5.0 (<1.0 reduces, >1.0 amplifies)
```

### Automatic Gain Control (AGC)

Enable AGC for automatic volume adjustment:

```yaml
audio:
  agc_enabled: true              # Automatically adjusts volume
  agc_target_rms: 0.3           # Target level (30% of maximum)
```

AGC is recommended when:
- Multiple people use the assistant
- Microphone distance varies
- Background noise levels change

## Multi-Language Support

Both OpenAI and Home Assistant must use the same language:

```yaml
openai:
  language: "es"    # Spanish

home_assistant:
  language: "es"    # Must match OpenAI
```

Supported languages include: en, es, fr, de, it, pt, nl, pl, and more.

## Troubleshooting Configuration Issues

1. **API Key Issues**: Ensure your `.env` file is in the project root
2. **Audio Device Errors**: Run the audio test script to find correct device names
3. **Wake Word Not Detecting**: Increase sensitivity or audio gain
4. **False Wake Word Triggers**: Decrease sensitivity
5. **Connection Errors**: Verify URLs and tokens are correct

## Example Configurations

### USB Speakerphone Setup

```yaml
audio:
  input_device: "Jabra Speak 410 USB"
  output_device: "Jabra Speak 410 USB"
  sample_rate: 48000
  input_volume: 1.0
  output_volume: 1.5
  agc_enabled: true  # Recommended for conference speakers
```

### Raspberry Pi with ReSpeaker HAT

```yaml
audio:
  input_device: "seeed-2mic-voicecard"
  output_device: "seeed-2mic-voicecard"
  sample_rate: 48000
  
system:
  led_gpio: 12  # GPIO pin for LED feedback
```

### High Sensitivity Wake Word

```yaml
wake_word:
  model: "computer"
  sensitivity: 1.0      # Maximum sensitivity
  audio_gain: 1.5      # Slight boost
  cooldown: 3.0        # Prevent rapid triggers
```

## Security Best Practices

1. Never commit `.env` files to version control
2. Use strong, unique API keys
3. Restrict Home Assistant token permissions
4. Keep configuration files readable only by the service user
5. Regularly rotate API keys and tokens