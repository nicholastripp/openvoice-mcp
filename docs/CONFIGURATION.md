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
  input_volume: 5.0                # Input volume multiplier
  output_volume: 2.0               # Output volume multiplier
  feedback_prevention: true        # Prevent audio loops
```

### Wake Word Configuration

```yaml
wake_word:
  enabled: true                   # Enable wake word detection
  model: "hey_jarvis"            # Wake word model
  sensitivity: 0.004             # Detection sensitivity (0.0-1.0)
  timeout: 5.0                   # Session timeout after wake word
  vad_enabled: false             # Voice activity detection
  cooldown: 2.0                  # Seconds between detections
  audio_gain: 3.5                # Audio amplification (1.0-5.0)
  audio_gain_mode: "fixed"       # Gain mode: "fixed" or "dynamic"
```

Available wake word models:
- `hey_jarvis` - Default wake word
- `alexa` - Amazon Alexa compatible
- `hey_mycroft` - Mycroft compatible
- `hey_rhasspy` - Rhasspy compatible
- `ok_nabu` - Nabu Casa compatible

### Session Configuration

```yaml
session:
  timeout: 30                     # Session timeout in seconds
  auto_end_silence: 3.0          # End session after silence
  max_duration: 300              # Maximum session duration
  interrupt_threshold: 0.5       # Voice activity to interrupt AI
  
  # Multi-turn conversation settings
  conversation_mode: "multi_turn"  # "single_turn" or "multi_turn"
  multi_turn_timeout: 30.0        # Wait time for follow-ups
  multi_turn_max_turns: 10        # Max conversation turns
  multi_turn_end_phrases:         # Phrases to end conversation
    - "goodbye"
    - "stop"
    - "that's all"
    - "thank you"
    - "bye"
```

### System Configuration

```yaml
system:
  log_level: "INFO"              # DEBUG, INFO, WARNING, ERROR, CRITICAL
  log_file: "logs/assistant.log" # Log file path
  led_gpio: null                 # GPIO pin for status LED (optional)
  daemon: false                  # Run as daemon process
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
./venv/bin/python tools/test_audio_devices.py
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
./venv/bin/python tools/test_wake_word.py --interactive
```

## Audio Gain Configuration

Adjust audio gain to improve wake word detection:

```yaml
wake_word:
  audio_gain: 3.5                # Amplification factor (1.0-5.0)
  audio_gain_mode: "fixed"       # "fixed" or "dynamic"
```

- **Fixed mode**: Constant amplification (recommended)
- **Dynamic mode**: Automatic adjustment based on audio levels

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
  input_volume: 3.0
  output_volume: 1.5
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
  sensitivity: 0.002
  audio_gain: 4.0
  cooldown: 3.0  # Prevent rapid triggers
```

## Security Best Practices

1. Never commit `.env` files to version control
2. Use strong, unique API keys
3. Restrict Home Assistant token permissions
4. Keep configuration files readable only by the service user
5. Regularly rotate API keys and tokens