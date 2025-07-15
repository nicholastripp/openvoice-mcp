# Usage Guide

Learn how to use your Home Assistant Realtime Voice Assistant effectively.

## Basic Operation

### Starting the Assistant

1. **Normal Mode**:
   ```bash
   # Activate virtual environment first
   source venv/bin/activate
   
   # Then run the assistant
   python src/main.py
   ```

2. **Debug Mode** (for troubleshooting):
   ```bash
   # With virtual environment activated
   python src/main.py --log-level DEBUG
   ```

3. **As a System Service** (if configured):
   ```bash
   sudo systemctl start ha-voice-assistant
   ```

### Using the Assistant

1. **Wake the Assistant**
   - Say the wake word (default: "Picovoice")
   - You'll hear a confirmation sound
   - The assistant is now listening

2. **Give Commands**
   - Speak naturally, for example:
     - "Turn on the living room lights"
     - "What's the temperature in the bedroom?"
     - "Set the thermostat to 72 degrees"
     - "Turn off all lights"

3. **Multi-turn Conversations**
   - After the assistant responds, you can continue the conversation
   - No need to say the wake word again
   - The conversation continues for 30 seconds (configurable)
   - Say "goodbye", "stop", or "that's all" to end early

## Voice Commands

### Device Control
- **Lights**: "Turn on/off the [room] lights"
- **Switches**: "Turn on/off the [device name]"
- **Dimmers**: "Set the [light] to 50 percent"
- **Colors**: "Change the [light] to blue"

### Climate Control
- **Temperature**: "Set the thermostat to [temperature]"
- **Mode**: "Set the AC to cool/heat/auto"
- **Query**: "What's the temperature setting?"

### Queries
- **State**: "Is the [device] on?"
- **Temperature**: "What's the temperature in the [room]?"
- **Sensors**: "What's the [sensor name] reading?"

### Scenes and Scripts
- **Scenes**: "Activate [scene name]"
- **Scripts**: "Run the [script name]"

## Configuration Options

### Conversation Modes

1. **Single-turn Mode** (default):
   ```yaml
   session:
     conversation_mode: "single_turn"
   ```
   - Say wake word for each command
   - Session ends after response

2. **Multi-turn Mode**:
   ```yaml
   session:
     conversation_mode: "multi_turn"
     multi_turn_timeout: 30.0
   ```
   - Continue conversation without wake word
   - Configurable timeout

### Audio Settings

1. **Volume Control**:
   ```yaml
   audio:
     input_volume: 1.0    # Microphone gain (0.1-5.0)
     output_volume: 2.0   # Speaker volume
   ```

2. **Automatic Gain Control** (recommended for varying conditions):
   ```yaml
   audio:
     agc_enabled: true
     agc_target_rms: 0.3
   ```

### Wake Word Options

Available built-in wake words:
- `picovoice` (default)
- `alexa`
- `computer`
- `terminator`
- `americano`
- `blueberry`
- `bumblebee`
- `grapefruit`
- `grasshopper`
- `porcupine`

Change in `config.yaml`:
```yaml
wake_word:
  model: "computer"  # or any from the list above
```

## Tips for Best Results

### Speaking Tips
1. **Clear Speech**: Speak clearly and at a normal pace
2. **Distance**: Stay within 6 feet of the microphone
3. **Background Noise**: Minimize background noise when possible
4. **Natural Language**: Use natural phrases, not robotic commands

### Microphone Placement
1. **Central Location**: Place in a central area of the room
2. **Away from Speakers**: Avoid placing near TVs or speakers
3. **Elevated Position**: Place at chest height or higher
4. **Clear Path**: Ensure no obstacles between you and the mic

### Troubleshooting Commands

If the assistant doesn't understand:
1. **Be Specific**: Use exact device names from Home Assistant
2. **Simple Commands**: Start with simple on/off commands
3. **Check Exposure**: Ensure devices are exposed to the assistant in HA
4. **Review Logs**: Check logs for what the assistant heard

## Advanced Features

### Custom Wake Words

For Porcupine engine:
1. Create custom wake word at [console.picovoice.ai](https://console.picovoice.ai)
2. Download the `.ppn` file
3. Place in `config/wake_words/`
4. Update configuration to use the file path

### Personality Customization

Edit `config/persona.ini` to customize:
- Assistant name
- Response style
- Personality traits
- Custom instructions

### Home Assistant Exposure

Control what the assistant can access:
1. In Home Assistant, go to Settings → Voice Assistants
2. Configure exposed entities
3. Group devices logically
4. Use areas for room-based control

## Monitoring and Logs

### View Logs
```bash
# Real-time logs
tail -f logs/assistant.log

# Today's logs
cat logs/assistant.log | grep "$(date +%Y-%m-%d)"

# Error logs only
grep ERROR logs/assistant.log
```

### Performance Monitoring
- Wake word detections: Look for "Wake word detected" in logs
- Response times: Check timestamps between detection and response
- Audio levels: Monitor "Audio level" entries for gain issues

## Stopping the Assistant

1. **If running in terminal**: Press `Ctrl+C`
2. **If running as service**: `sudo systemctl stop ha-voice-assistant`
3. **Emergency stop**: `pkill -f main.py`

## Getting Help

- Review the [Troubleshooting Guide](TROUBLESHOOTING.md)
- Check [Audio Setup](AUDIO_SETUP.md) for audio issues
- See [Wake Word Setup](WAKE_WORD_SETUP.md) for detection problems
- Submit issues on [GitHub](https://github.com/nicholastripp/ha-realtime-assist/issues)