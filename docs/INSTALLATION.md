# Installation Guide

This guide will walk you through setting up the Home Assistant Realtime Voice Assistant on your Raspberry Pi or other Linux system.

## Prerequisites

### Hardware Requirements
- **Minimum**: Raspberry Pi 3B+ with 1GB RAM
- **Recommended**: Raspberry Pi 4 with 2GB+ RAM
- USB microphone or USB conference speakerphone
- Speaker (3.5mm jack or USB)
- 8GB+ SD card
- Stable internet connection (Ethernet recommended)

### Software Requirements
- Raspberry Pi OS (Bullseye or newer) or Ubuntu 20.04+
- Python 3.9 or newer
- Git
- Home Assistant 2025.2+ (required for MCP support)

### Required Accounts
1. **OpenAI Account** with API access
   - Sign up at [platform.openai.com](https://platform.openai.com)
   - Create an API key with Realtime API access
   - Note: Realtime API requires payment setup

2. **Home Assistant Instance**
   - Running Home Assistant 2025.2 or later
   - MCP Server integration installed and enabled
   - Long-lived access token created
   - "Control Home Assistant" option enabled in MCP settings

3. **Picovoice Account** (Optional, for Porcupine wake words)
   - Sign up at [console.picovoice.ai](https://console.picovoice.ai)
   - Get a free access key

## Step-by-Step Installation

### 1. System Preparation

Update your system:
```bash
sudo apt update && sudo apt upgrade -y
```

Install required system packages:
```bash
sudo apt install -y python3-pip python3-venv git portaudio19-dev python3-pyaudio
```

For Speex noise suppression (optional):
```bash
sudo apt install -y libspeexdsp-dev
```

### 2. Clone the Repository

```bash
git clone https://github.com/nicholastripp/ha-realtime-assist
cd ha-realtime-assist
```

### 3. Run the Installation Script

The installation script will set up the Python virtual environment and install all dependencies:

```bash
./install.sh
```

This script will:
- Create a Python virtual environment
- Install all required Python packages (including web UI dependencies)
- Set up the basic directory structure
- Optionally configure web UI security (authentication and HTTPS)

#### Web UI Security Setup (Optional)

During installation, you'll be asked about web UI configuration:

1. **Enable Web UI?** - Choose Yes to set up the web interface
2. **Enable Authentication?** - Strongly recommended if web UI will be network-accessible
3. **Username** - Default is "admin" (you can change this)
4. **Password** - Enter a strong password (8+ characters recommended)

The installer will:
- Generate a secure bcrypt password hash
- Store the hash in your `.env` file as `WEB_UI_PASSWORD_HASH`
- Configure HTTPS with self-signed certificates
- Update your `config.yaml` with web UI settings

### 4. Configure the Assistant

#### Set up configuration files:
```bash
./setup_config.sh
```

#### Edit the environment file with your API keys:
```bash
nano .env
```

Add your credentials:
```bash
OPENAI_API_KEY=your-openai-api-key-here
HA_TOKEN=your-home-assistant-long-lived-token-here
PICOVOICE_ACCESS_KEY=your-picovoice-key-here  # Optional
```

#### Edit the main configuration:
```bash
nano config/config.yaml
```

Key settings to configure:
- `home_assistant.url`: Your Home Assistant URL (e.g., `http://192.168.1.100:8123`)
- `wake_word.model`: Choose your wake word (default: `picovoice`)
- `audio.input_device`: Set your microphone (use `default` or run audio test to find device name)
- `audio.output_device`: Set your speaker (use `default` or run audio test to find device name)

### 5. Test Your Setup

#### Test audio devices:
```bash
# Activate virtual environment first
source venv/bin/activate

# Then test audio
python examples/test_audio_devices.py
```

#### Test Home Assistant connection:
```bash
# With virtual environment activated
python examples/test_ha_connection.py
```

#### Test OpenAI connection:
```bash
# With virtual environment activated
python examples/test_openai_connection.py
```

#### Test wake word detection:
```bash
# With virtual environment activated
python examples/test_wake_word.py --interactive
```

### 6. Run the Assistant

```bash
# Activate virtual environment first
source venv/bin/activate

# Then run the assistant
python src/main.py
```

Or with debug logging:
```bash
# With virtual environment activated
python src/main.py --log-level DEBUG
```

Or with the web UI enabled:
```bash
# With virtual environment activated
python src/main.py --web
# Access the web UI at https://localhost:8443
```

## Web UI Access

### Starting with Web UI

If you enabled the web UI during installation:

```bash
# Activate virtual environment
source venv/bin/activate

# Start with web UI enabled
python src/main.py --web

# Or with a custom port
python src/main.py --web --web-port 8090
```

### Accessing the Web Interface

1. **Local Access**: Navigate to `https://localhost:8443`
2. **Remote Access**: Use `https://your-pi-ip:8443`
3. **Certificate Warning**: Accept the self-signed certificate
   - This is normal on first access
   - The certificate is generated automatically for security
4. **Authentication**: Log in with the credentials you set during installation

### Web UI Features

Once logged in, you can:
- **Status Dashboard** - Real-time monitoring with WebSocket updates
- **Environment Variables** - Securely edit API keys and tokens
- **Configuration Editor** - Modify all settings through the browser
- **Personality Editor** - Customize assistant behavior with visual sliders
- **Audio Testing** - Test and configure audio devices
- **Log Viewer** - View system logs with syntax highlighting

### Security Considerations

- **HTTPS is enabled by default** to protect your credentials
- **Authentication is required** when accessing from the network
- **Session timeout** is configurable (default: 1 hour)
- **Password hashes** use bcrypt with 12 rounds

For maximum security when accessing remotely, consider:
1. Using a VPN to your home network
2. Setting up custom SSL certificates
3. Using SSH tunneling:
   ```bash
   ssh -L 8443:localhost:8443 pi@your-pi
   # Then access https://localhost:8443
   ```

## Setting up as a System Service (Optional)

To run the assistant automatically on boot:

1. Edit the systemd service file:
```bash
sudo nano systemd/ha-voice-assistant.service
```

2. Update the paths and user in the service file

3. Install and enable the service:
```bash
sudo cp systemd/ha-voice-assistant.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ha-voice-assistant
sudo systemctl start ha-voice-assistant
```

4. Check service status:
```bash
sudo systemctl status ha-voice-assistant
```

## Troubleshooting Installation

### Common Issues

1. **"No module named 'sounddevice'" error**
   - Ensure you're using the virtual environment: `source venv/bin/activate`

2. **Audio device not found**
   - Activate venv: `source venv/bin/activate`
   - Run `python examples/test_audio_devices.py` to list devices
   - Update `config.yaml` with the correct device name or index

3. **Wake word not working**
   - Ensure Picovoice access key is set in `.env`
   - Check that `highpass_filter_enabled: true` in config

4. **Permission denied errors**
   - Add your user to the audio group: `sudo usermod -a -G audio $USER`
   - Logout and login again

### Getting Help

- Check the [Troubleshooting Guide](TROUBLESHOOTING.md)
- Review logs in `logs/assistant.log`
- Submit issues at [GitHub Issues](https://github.com/nicholastripp/ha-realtime-assist/issues)

## Next Steps

- Read the [Usage Guide](USAGE.md) to learn how to use the assistant
- Configure [Audio Settings](AUDIO_SETUP.md) for optimal performance
- Customize [Wake Words](WAKE_WORD_SETUP.md) for your preferences