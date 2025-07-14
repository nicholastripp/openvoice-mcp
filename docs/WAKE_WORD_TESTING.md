# Wake Word Configuration Testing Guide

## Summary of Changes

We've fixed the wake word configuration issue where changing to "hey_jarvis" didn't work. The key changes:

1. **Added comprehensive wake word mappings** including all Porcupine built-in keywords
2. **Improved error handling** to show clear error messages for invalid wake words
3. **Updated documentation** to list all valid wake words
4. **Created verification script** to test wake word configurations

## Valid Porcupine Wake Words

### Voice Assistants
- `alexa`
- `hey_google`
- `ok_google`
- `hey_siri`

### Picovoice Brand
- `picovoice`
- `hey_picovoice` (alias)
- `ok_picovoice` (alias)

### Fun Wake Words
- `americano`
- `blueberry`
- `bumblebee`
- `grapefruit`
- `grasshopper`
- `porcupine`
- `terminator`

### Other
- `computer`
- `hey_barista`
- `pico_clock`

**Important:** `jarvis` and `hey_jarvis` are NOT built-in wake words!

## Testing in Your Environment

### 1. Test Wake Word Verification Script

```bash
cd /path/to/ha-realtime-assist
python3 scripts/verify_wake_words.py
```

This will:
- List all available wake words
- Test common configurations
- Check your current config.yaml settings

### 2. Test Invalid Wake Word Error Handling

Try setting an invalid wake word in config.yaml:

```yaml
wake_word:
  engine: "porcupine"
  model: "hey_jarvis"  # This should now show a clear error
```

Run the assistant and you should see:
```
Invalid wake word: 'hey_jarvis'
Available options:
  Config names: alexa, americano, blueberry, bumblebee, computer, grapefruit, grasshopper, hey_barista, hey_google, hey_picovoice, hey_siri, ok_google, ok_picovoice, pico_clock, picovoice, porcupine, terminator
  Direct keywords: alexa, americano, blueberry, bumblebee, computer, grapefruit, grasshopper, hey barista, hey google, hey siri, ok google, pico clock, picovoice, porcupine, terminator
  For custom wake words, visit https://console.picovoice.ai/
```

### 3. Test Different Wake Words

Try these valid wake words in config.yaml:

```yaml
wake_word:
  model: "computer"      # Star Trek style!
```

```yaml
wake_word:
  model: "hey_google"    # Google Assistant style
```

```yaml
wake_word:
  model: "terminator"    # Fun wake word
```

### 4. Monitor Logs

When starting the assistant, you should see:
```
Using wake word 'picovoice' (mapped from config 'picovoice')
```

Or for direct matches:
```
Using wake word 'computer' (direct match)
```

## Custom Wake Words (like Jarvis)

To use "jarvis" or "hey_jarvis":

1. Visit https://console.picovoice.ai/
2. Click "Create Wake Word"
3. Enter "hey jarvis" as your phrase
4. Select your platform (e.g., Raspberry Pi)
5. Download the .ppn model file
6. Use the custom model path in your config (this feature needs additional implementation)

## Troubleshooting

- If you see "Invalid wake word" error, check the list above
- The wake word in config.yaml must exactly match one from the list
- Wake words are case-sensitive in the config
- Some wake words (like 'hey_google') map to multi-word phrases ('hey google')