# Troubleshooting Guide

This guide helps you resolve common issues with the Home Assistant Realtime Voice Assistant.

## Quick Diagnostics

Run these commands first to identify issues:

```bash
# Test all components
./run_tests.sh all

# Check specific components
./run_tests.sh audio    # Audio devices
./run_tests.sh ha       # Home Assistant connection
./run_tests.sh openai   # OpenAI connection
./run_tests.sh wake     # Wake word detection
```

## Common Issues and Solutions

### Installation Issues

#### "No module named 'sounddevice'" or similar import errors

**Cause**: Not using the virtual environment

**Solution**:
```bash
# Always activate the virtual environment first
source venv/bin/activate

# Then run normally
python src/main.py
```

#### "Permission denied" errors

**Cause**: User not in audio group or file permissions

**Solution**:
```bash
# Add user to audio group
sudo usermod -a -G audio $USER
# Logout and login again

# Fix file permissions
chmod +x install.sh run_tests.sh setup_config.sh
```

### Audio Issues

#### No audio input detected

**Diagnosis**:
```bash
# Activate virtual environment first
source venv/bin/activate

# Then test audio
python examples/test_audio_devices.py
```

**Solutions**:
1. Check device is connected: `ls /dev/snd/`
2. Verify in system: `arecord -l`
3. Update config with correct device name
4. Try default device: `input_device: "default"`

#### Audio clipping/distortion

**Symptoms**: 
- Wake word only works with soft speech
- Crackling or distorted audio
- Log shows: "AUDIO CLIPPING: X% of samples clipped"

**Solution**:
```yaml
audio:
  input_volume: 0.5  # Reduce below 1.0
  # Or enable AGC
  agc_enabled: true
```

#### Echo or feedback

**Solution**:
```yaml
audio:
  feedback_prevention: true
  mute_during_response: true
  output_volume: 1.0  # Reduce if needed
```

### Wake Word Issues

#### Wake word not detecting

**Diagnosis**:
```bash
# Activate virtual environment first
source venv/bin/activate

# Then test wake word
python examples/test_wake_word.py --interactive
```

**Solutions in order**:

1. **Check audio input**: Ensure microphone works (see Audio Issues)

2. **Verify engine settings**:
   ```yaml
   wake_word:
     enabled: true
     engine: "porcupine"  # or "openwakeword"
   ```

3. **For Porcupine - Enable high-pass filter**:
   ```yaml
   wake_word:
     highpass_filter_enabled: true  # REQUIRED
     highpass_filter_cutoff: 80.0
   ```

4. **Adjust sensitivity**:
   ```yaml
   wake_word:
     sensitivity: 1.0  # Increase for Porcupine
     # or
     sensitivity: 0.7  # Increase for OpenWakeWord
   ```

5. **Increase audio gain**:
   ```yaml
   wake_word:
     audio_gain: 1.5  # Amplify for wake word
   ```

#### False wake word triggers

**Solution**:
```yaml
wake_word:
  sensitivity: 0.3  # Reduce sensitivity
  cooldown: 3.0     # Increase cooldown period
```

#### "Invalid wake word" error

**Cause**: Using non-existent wake word model

**Solution**: Use built-in keywords:
- Porcupine: `picovoice`, `alexa`, `computer`, `terminator`
- OpenWakeWord: `hey_jarvis`, `alexa`, `hey_mycroft`

### OpenAI Connection Issues

#### "Connection to OpenAI Realtime API failed"

**Diagnosis**:
```bash
# Activate virtual environment first
source venv/bin/activate

# Then test connection
python examples/test_openai_connection.py
```

**Solutions**:
1. Verify API key in `.env` file
2. Check internet connection
3. Ensure Realtime API access is enabled in OpenAI account
4. Verify billing is set up in OpenAI

#### Assistant responds in wrong language

**Solution**:
```yaml
openai:
  language: "en"  # Set explicitly
```

#### Poor speech recognition

**Solutions**:
1. Check audio quality (no clipping)
2. Reduce background noise
3. Speak clearly and naturally
4. Check microphone distance (< 6 feet)

### Home Assistant Issues

#### "Failed to connect to Home Assistant"

**Diagnosis**:
```bash
# Activate virtual environment first
source venv/bin/activate

# Then test connection
python examples/test_ha_connection.py
```

**Solutions**:
1. Verify URL is accessible: `curl http://your-ha-url:8123/api/`
2. Check long-lived token is valid
3. Ensure HA is not using HTTPS if URL uses HTTP
4. Check network connectivity

#### "No devices found" or limited device access

**Solution**:
1. In Home Assistant: Settings → Voice Assistants → Expose Entities
2. Select entities to expose
3. Restart the voice assistant
4. Check logs for exposed devices list

#### Commands not working

**Diagnosis**: Check what assistant heard
```bash
grep "User said" logs/assistant.log
```

**Solutions**:
1. Use exact entity names from Home Assistant
2. Check entity is exposed to assist pipeline
3. Try simple commands first: "Turn on [exact entity name]"

### Performance Issues

#### High CPU usage

**Monitor**:
```bash
top -p $(pgrep -f main.py)
```

**Solutions**:
1. Switch to less CPU-intensive wake word
2. Increase chunk size:
   ```yaml
   audio:
     chunk_size: 2400  # Larger chunks = less CPU
   ```
3. Disable unnecessary features:
   ```yaml
   wake_word:
     vad_enabled: false
     speex_noise_suppression: false
   ```

#### Slow response times

**Solutions**:
1. Use wired Ethernet instead of WiFi
2. Check internet speed
3. Reduce audio quality if needed:
   ```yaml
   audio:
     sample_rate: 16000  # Lower sample rate
   ```

### Session Issues

#### Session ends immediately

**Solution**:
```yaml
session:
  auto_end_after_response: false
  multi_turn_timeout: 30.0
```

#### Can't continue conversation

**Solution**:
```yaml
session:
  conversation_mode: "multi_turn"
  multi_turn_max_turns: 10
```

## Debug Mode

Enable comprehensive debugging:

```bash
# Activate virtual environment first
source venv/bin/activate

# Run with debug logging
python src/main.py --log-level DEBUG

# Watch logs in real-time
tail -f logs/assistant.log

# Filter for specific issues
grep -i error logs/assistant.log
grep -i "wake word" logs/assistant.log
grep -i clipping logs/assistant.log
```

## Log Analysis

### Understanding log entries

```
2025-01-15 10:23:45 INFO Wake word detected: picovoice (confidence: 0.95)
                    ^    ^                    ^          ^
                    Level Event               Model      Score

2025-01-15 10:23:46 WARNING Audio clipping detected: 1250 samples (5.2%)
                    ^        ^                        ^
                    Level    Issue                    Details
```

### Key log patterns to watch

- `ERROR` - Critical issues requiring attention
- `WARNING` - Non-critical issues (clipping, connection retry)
- `Wake word detected` - Successful activation
- `User said` - What was transcribed
- `Calling function` - Home Assistant commands
- `Audio clipping` - Input volume too high

## Getting Further Help

### Before asking for help

1. Run all diagnostic tests
2. Enable debug logging
3. Collect relevant log entries
4. Note your configuration (OS, Pi model, audio devices)

### Where to get help

1. **GitHub Issues**: [Report bugs](https://github.com/nicholastripp/ha-realtime-assist/issues)
2. **Logs**: Include relevant portions from `logs/assistant.log`
3. **Configuration**: Share your `config.yaml` (remove sensitive data)
4. **System Info**: Include output of:
   ```bash
   uname -a
   python --version
   pip list | grep -E "(openai|sounddevice|websockets)"
   ```

## Emergency Recovery

If the assistant is completely broken:

```bash
# Stop any running instances
pkill -f main.py

# Reset to clean state
git stash  # Save your changes
git pull   # Get latest code
./install.sh  # Reinstall dependencies

# Restore your config
git stash pop
```

## Prevention Tips

1. **Test changes**: Always test configuration changes
2. **Backup config**: Keep working configurations
3. **Monitor logs**: Watch for warnings before they become errors
4. **Update carefully**: Test updates in a safe environment first
5. **Document what works**: Note your working settings