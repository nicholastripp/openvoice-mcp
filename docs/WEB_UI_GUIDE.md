# Web UI Guide

The Home Assistant Realtime Voice Assistant includes an optional web interface for easy configuration and monitoring.

## Overview

The web UI provides:
- **Setup Wizard** - First-time configuration with guided setup
- **Configuration Editors** - Modify all settings through the browser
- **Personality Editor** - Customize your assistant's personality
- **Audio Testing** - Test and configure audio devices
- **Status Dashboard** - Real-time monitoring and statistics
- **Log Viewer** - View system logs with syntax highlighting

## Starting the Web UI

```bash
# Start the assistant with web UI enabled
python src/main.py --web

# Use a custom port (default is 8080)
python src/main.py --web --web-port 8090
```

## Features

### 1. Setup Wizard (First Run)

On first run (when no `.env` file exists), the web UI automatically shows a setup wizard:

- Welcome page with feature overview
- API key configuration form
- Connection testing for each service
- Automatic configuration file creation

Access: `http://localhost:8080/setup`

### 2. Environment Variables

Securely manage your API keys and tokens:

- Masked display of sensitive values
- Update individual keys without exposing others
- Automatic `.env` file updates

Access: `http://localhost:8080/config/env`

### 3. Configuration Settings

Edit all `config.yaml` settings through an intuitive interface:

- **OpenAI Settings**: Voice selection, temperature, language
- **Audio Settings**: Volume controls, AGC, device selection
- **Wake Word Settings**: Sensitivity, confirmation beep
- **Session Settings**: Timeouts, multi-turn configuration

Access: `http://localhost:8080/config/yaml`

### 4. Personality Editor

Customize your assistant's personality with:

- **Identity**: Name, role, and personality description
- **Trait Sliders**: 10 personality traits (0-100 scale)
  - Helpfulness, Humor, Formality, Patience, Verbosity
  - Warmth, Curiosity, Confidence, Optimism, Respectfulness
- **Backstory**: Origin, purpose, and specialties
- **Advanced**: Custom instructions and response style
- **Preview**: See sample responses before saving

Access: `http://localhost:8080/persona`

### 5. Audio Device Testing

Test and configure audio devices:

- List all available input/output devices
- Test individual devices
- Monitor real-time audio levels
- Visual level meter with dB display

Access: `http://localhost:8080/config/audio`

### 6. Status Dashboard

Monitor your assistant in real-time:

- Current state indicator (idle, listening, processing, etc.)
- Connection status for all services
- Audio level visualization
- Activity log with recent commands
- Statistics: uptime, wake detections, commands, response times
- WebSocket updates for live data

Access: `http://localhost:8080/status`

### 7. Log Viewer

View and monitor system logs:

- Last 100 lines of logs
- Syntax highlighting by log level
- Auto-refresh option (5-second interval)
- Clear display button

Access: `http://localhost:8080/status/logs`

## Navigation

The web UI includes a consistent navigation bar with links to all major sections:
- Status
- Environment
- Configuration  
- Personality
- Audio
- Logs

## Security

- By default, the web UI only listens on localhost (127.0.0.1)
- API keys are masked in the interface
- No authentication required for local access
- For network access, consider using a reverse proxy with authentication

## Troubleshooting

### Web UI Won't Start

1. Ensure you have installed the dependencies:
   ```bash
   pip install aiohttp-jinja2 jinja2
   ```

2. Check if port 8080 is already in use:
   ```bash
   lsof -i :8080  # On macOS/Linux
   ```

3. Try a different port:
   ```bash
   python src/main.py --web --web-port 8090
   ```

### Cannot Access Web UI

1. Ensure the assistant started successfully
2. Check the console output for the web UI URL
3. Try accessing `http://localhost:8080` (not https)
4. Check firewall settings if accessing from another device

### Changes Not Taking Effect

Some configuration changes require a restart:
- Audio device changes
- Wake word settings
- OpenAI model/voice changes

The UI will indicate when a restart is required.

## Development

To test the web UI independently:

```bash
python examples/test_web_ui.py
```

This starts only the web server without the voice assistant components.