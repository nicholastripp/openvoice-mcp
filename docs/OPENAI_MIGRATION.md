# OpenAI Realtime API Migration Guide

## Overview

This guide covers the migration from the preview models (`gpt-4o-realtime-preview` and `gpt-4o-mini-realtime-preview`) to the new production model (`gpt-realtime`). The migration brings significant improvements in performance, cost, and capabilities while maintaining full backward compatibility.

## What's New in gpt-realtime

### Performance Improvements

| Metric | Preview Model | Production Model | Improvement |
|--------|--------------|------------------|-------------|
| Big Bench Audio | 65.6% | 82.8% | **+26%** |
| Instruction Following | 20.6% | 30.5% | **+48%** |
| Function Calling | 49.7% | 66.5% | **+34%** |
| Cost (per 1M tokens) | $40/$80 | $32/$64 | **-20%** |

### New Features

1. **New Voices**: Cedar (masculine, rich, authoritative) and Marin (feminine, crisp, articulate)
2. **Native MCP Support**: Built-in Model Context Protocol server integration
3. **Image Input**: Support for visual inputs (coming soon)
4. **Async Functions**: Enhanced asynchronous function calling
5. **Better Instruction Following**: 48% improvement in following complex instructions

## Migration Options

### 1. Automatic Migration (Recommended)

The default configuration uses automatic migration with fallback support:

```yaml
openai:
  model_selection: "auto"  # Default setting
```

This will:
- Start with the new `gpt-realtime` model
- Automatically fallback to `gpt-4o-realtime-preview` if connection fails
- Select compatible voices based on the active model

### 2. Force New Model

To use only the new production model without fallback:

```yaml
openai:
  model_selection: "new"
  model: "gpt-realtime"
  voice: "cedar"  # Or "marin" for new voices
```

### 3. Stay on Legacy Model

To continue using the preview model:

```yaml
openai:
  model_selection: "legacy"
```

## Configuration Changes

### Basic Configuration

Update your `config/config.yaml`:

```yaml
openai:
  api_key: ${OPENAI_API_KEY}
  voice: "alloy"                    # Works with all models
  model: "gpt-realtime"            # New production model
  legacy_model: "gpt-4o-realtime-preview"  # Fallback option
  model_selection: "auto"           # auto, new, or legacy
  voice_fallback: "alloy"           # Fallback if voice unavailable
  auto_select_voice: true           # Auto-select compatible voice
  temperature: 0.8
  language: "en"
```

### Voice Selection

Available voices by model:

- **All models**: alloy, ash, ballad, coral, echo, sage, shimmer, verse
- **gpt-realtime only**: cedar, marin

Example configurations:

```yaml
# Use new voice with fallback
openai:
  voice: "cedar"           # New voice
  voice_fallback: "echo"   # Fallback for legacy models
  auto_select_voice: true

# Force specific voice
openai:
  voice: "marin"
  auto_select_voice: false  # Will error if unavailable
```

## Migration Steps

### Step 1: Backup Current Configuration

```bash
cp config/config.yaml config/config.yaml.backup
```

### Step 2: Update Configuration

Edit `config/config.yaml` to add new fields:

```yaml
openai:
  # ... existing fields ...
  model: "gpt-realtime"
  legacy_model: "gpt-4o-realtime-preview"
  model_selection: "auto"
  voice_fallback: "alloy"
  auto_select_voice: true
```

### Step 3: Test Connection

Run the assistant and verify successful connection:

```bash
python src/main.py
```

You should see:
```
Connected to OpenAI Realtime API using model: gpt-realtime
Audio session config: model=gpt-realtime, voice=alloy
```

### Step 4: Try New Voices (Optional)

Test the new voices exclusive to gpt-realtime:

```yaml
openai:
  voice: "cedar"  # Rich, authoritative masculine voice
  # OR
  voice: "marin"  # Crisp, articulate feminine voice
```

## Rollback Procedure

If you encounter issues, you can quickly rollback:

### Option 1: Force Legacy Model

```yaml
openai:
  model_selection: "legacy"
```

### Option 2: Full Rollback

```bash
# Restore backup configuration
cp config/config.yaml.backup config/config.yaml

# Restart the assistant
python src/main.py
```

## Troubleshooting

### Voice Not Available

**Error**: `Voice 'cedar' not available for model 'gpt-4o-realtime-preview'`

**Solution**: Cedar and Marin are exclusive to gpt-realtime. Either:
- Use `model_selection: "new"` to force new model
- Set `auto_select_voice: true` for automatic fallback
- Choose a universal voice like "alloy"

### Connection Failed

**Error**: `Failed to connect with model gpt-realtime`

**Solution**: The system will automatically fallback if `model_selection: "auto"`. Otherwise:
- Check your API key has access to the new model
- Verify your account has been upgraded
- Try `model_selection: "legacy"` temporarily

### Increased Latency

If you experience higher latency with the new model:

1. Check metrics: The new model should have ~20% lower latency
2. Verify your region/endpoint
3. Monitor with performance metrics:
   ```python
   # Metrics are automatically tracked in metrics/ directory
   ```

## Performance Monitoring

The migration includes automatic performance tracking. Monitor improvements:

```python
# View metrics in metrics/ directory
ls metrics/session_*.json

# Key metrics to watch:
- average_response_latency  # Should decrease by ~20%
- estimated_cost            # Should decrease by ~20%
- function_success_rate     # Should increase by ~34%
```

## Cost Savings

The new model provides 20% cost reduction:

| Usage | Preview Model | Production Model | Monthly Savings |
|-------|--------------|------------------|-----------------|
| Light (100K tokens/day) | $12/month | $9.60/month | $2.40 |
| Medium (500K tokens/day) | $60/month | $48/month | $12 |
| Heavy (2M tokens/day) | $240/month | $192/month | $48 |

## API Changes

### Session Configuration

The new model supports additional session parameters:

```python
# Available with gpt-realtime
session_config = {
    "model": "gpt-realtime",
    "mcp_servers": [...]      # Native MCP support
    "image_input": True,       # Coming soon
    "async_functions": True,   # Enhanced function calling
}
```

### Function Calling

Improved function calling accuracy (66.5% vs 49.7%):

```python
# Functions work the same but with better accuracy
# No code changes required
```

## Best Practices

1. **Start with Auto Mode**: Let the system handle model selection
2. **Test New Voices**: Cedar and Marin offer improved clarity
3. **Monitor Costs**: Track savings with performance metrics
4. **Gradual Migration**: Test in development before production
5. **Keep Fallback**: Maintain legacy model configuration

## Support

For issues or questions:

1. Check logs: `logs/assistant.log`
2. Review metrics: `metrics/session_*.json`
3. Verify configuration: `config/config.yaml`
4. Test connection: Run with `model_selection: "legacy"` to isolate issues

## Changelog

### Version 2.0.0 - OpenAI Model Migration
- Added support for gpt-realtime production model
- Implemented automatic fallback to legacy models
- Added Cedar and Marin voice support
- Integrated performance metrics tracking
- Added model compatibility layer
- Implemented voice migration system
- 20% cost reduction with new model
- 34% improvement in function calling accuracy
- 48% improvement in instruction following