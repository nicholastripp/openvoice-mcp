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

### Method 1: Configuration File (Recommended)

Add to your `config/config.yaml`:

```yaml
web_ui:
  enabled: true         # Enable web UI on startup
  host: "0.0.0.0"      # Listen on all interfaces for remote access
  port: 8443           # Default HTTPS port (8080 for HTTP if TLS disabled)
  auth:
    enabled: true       # Authentication enabled by default
  tls:
    enabled: true       # HTTPS enabled by default
```

Then start normally:
```bash
python src/main.py
```

### Method 2: Command Line

```bash
# Start the assistant with web UI enabled
python src/main.py --web

# Use a custom port (default is 8443 for HTTPS)
python src/main.py --web --web-port 8090
```

Note: CLI flags override config file settings for the port, but the host address is always taken from the config file for security reasons.

## Features

### 1. Setup Wizard (First Run)

On first run (when no `.env` file exists), the web UI automatically shows a setup wizard:

- Welcome page with feature overview
- API key configuration form
- Connection testing for each service
- Automatic configuration file creation

Access: `https://localhost:8443/setup` (accept the self-signed certificate warning)

### 2. Environment Variables

Securely manage your API keys and tokens:

- Masked display of sensitive values
- Update individual keys without exposing others
- Automatic `.env` file updates

Access: `https://localhost:8443/config/env`

### 3. Configuration Settings

Edit all `config.yaml` settings through an intuitive interface:

- **OpenAI Settings**: Voice selection, temperature, language
- **Audio Settings**: Volume controls, AGC, device selection
- **Wake Word Settings**: Sensitivity, confirmation beep
- **Session Settings**: Timeouts, multi-turn configuration

Access: `https://localhost:8443/config/yaml`

### 4. Personality Editor

Customize your assistant's personality with:

- **Identity**: Name, role, and personality description
- **Trait Sliders**: 10 personality traits (0-100 scale)
  - Helpfulness, Humor, Formality, Patience, Verbosity
  - Warmth, Curiosity, Confidence, Optimism, Respectfulness
- **Backstory**: Origin, purpose, and specialties
- **Advanced**: Custom instructions and response style
- **Preview**: See sample responses before saving

Access: `https://localhost:8443/persona`

### 5. Audio Device Testing

Test and configure audio devices:

- List all available input/output devices
- Test individual devices
- Monitor real-time audio levels
- Visual level meter with dB display

Access: `https://localhost:8443/config/audio`

### 6. Status Dashboard

Monitor your assistant in real-time:

- Current state indicator (idle, listening, processing, etc.)
- Connection status for all services
- Audio level visualization
- Activity log with recent commands
- Statistics: uptime, wake detections, commands, response times
- WebSocket updates for live data

Access: `https://localhost:8443/status`

### 7. Log Viewer

View and monitor system logs:

- Last 100 lines of logs
- Syntax highlighting by log level
- Auto-refresh option (5-second interval)
- Clear display button

Access: `https://localhost:8443/status/logs`

## Navigation

The web UI includes a consistent navigation bar with links to all major sections:
- Status
- Environment
- Configuration  
- Personality
- Audio
- Logs

## Security

The web UI now includes comprehensive security features:

### Authentication
- **Basic Authentication** required by default
- Username and password configured during installation
- Secure session management with configurable timeout
- Password stored as SHA256 hash (never plaintext)

### HTTPS/TLS
- **HTTPS enabled by default** using self-signed certificates
- Automatic certificate generation on first run
- Support for custom certificates
- Strong cipher suites and TLS 1.2+ only

### Configuration
```yaml
web_ui:
  enabled: true
  host: "0.0.0.0"      # Listen address
  port: 8443           # HTTPS port
  auth:
    enabled: true      # Require authentication
    username: "admin"
    password_hash: ""  # Set by installer
    session_timeout: 3600
  tls:
    enabled: true      # Use HTTPS
    cert_file: ""      # Custom cert (optional)
    key_file: ""       # Custom key (optional)
```

### Security Best Practices
1. **Always use authentication** when host is "0.0.0.0"
2. **Accept the self-signed certificate** on first access
3. **Use strong passwords** during installation (8+ characters)
4. **Consider custom certificates** for production use
5. **Monitor access logs** for unauthorized attempts
6. **Change default username** from "admin" if desired
7. **Set appropriate session timeout** based on your security needs

### Custom Certificates
To use your own SSL certificate:
1. Place your certificate and key files in a secure location
2. Update config.yaml:
   ```yaml
   tls:
     cert_file: "/path/to/your/cert.pem"
     key_file: "/path/to/your/key.pem"
   ```
3. Restart the assistant

### SSH Tunneling (Alternative)
For maximum security, use localhost with SSH tunnel:
```bash
# On your computer
ssh -L 8443:localhost:8443 pi@your-pi

# Then access https://localhost:8443
```

## Troubleshooting

### Web UI Won't Start

1. Ensure you have installed the dependencies:
   ```bash
   pip install aiohttp-jinja2 jinja2
   ```

2. Check if port 8443 is already in use:
   ```bash
   lsof -i :8443  # On macOS/Linux
   ```

3. Try a different port:
   ```bash
   python src/main.py --web --web-port 8090
   ```

### Cannot Access Web UI

1. Ensure the assistant started successfully
2. Check the console output for the web UI URL
3. Accept the self-signed certificate warning in your browser
4. Try accessing `https://localhost:8443` (HTTPS is default)
5. If authentication is enabled, use the credentials you set during installation
6. Check firewall settings if accessing from another device

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