#!/bin/bash

# Installation script for Home Assistant Realtime Voice Assistant
# For Raspberry Pi and other Linux systems

set -e

echo "========================================"
echo "HA Realtime Voice Assistant Installer"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on supported system
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo -e "${RED}Error: This installer is designed for Linux systems${NC}"
    echo "For other systems, please install manually using pip"
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo -e "${RED}Error: Python 3.9 or higher is required${NC}"
    echo "Current version: $python_version"
    exit 1
fi

echo -e "${GREEN}✓ Python $python_version detected${NC}"

# Install system dependencies
echo -e "${YELLOW}Installing system dependencies...${NC}"

# Detect package manager
if command -v apt &> /dev/null; then
    sudo apt update
    sudo apt install -y python3-pip python3-venv portaudio19-dev git
elif command -v yum &> /dev/null; then
    sudo yum install -y python3-pip python3-venv portaudio-devel git
elif command -v pacman &> /dev/null; then
    sudo pacman -S python-pip python-virtualenv portaudio git
else
    echo -e "${YELLOW}Warning: Could not detect package manager${NC}"
    echo "Please ensure the following are installed:"
    echo "  - python3-pip"
    echo "  - python3-venv" 
    echo "  - portaudio development headers"
    echo "  - git"
fi

# Create virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install -r requirements.txt

echo -e "${GREEN}✓ Dependencies installed${NC}"

# Create configuration files
echo -e "${YELLOW}Setting up configuration...${NC}"

if [ ! -f "config/config.yaml" ]; then
    cp config/config.yaml.example config/config.yaml
    echo -e "${GREEN}✓ Created config/config.yaml${NC}"
    echo -e "${YELLOW}  Please edit this file with your settings${NC}"
else
    echo -e "${YELLOW}  config.yaml already exists${NC}"
fi

if [ ! -f "config/persona.ini" ]; then
    cp config/persona.ini.example config/persona.ini
    echo -e "${GREEN}✓ Created config/persona.ini${NC}"
else
    echo -e "${YELLOW}  persona.ini already exists${NC}"
fi

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${GREEN}✓ Created .env file${NC}"
    echo -e "${YELLOW}  Please edit this file with your API keys${NC}"
else
    echo -e "${YELLOW}  .env already exists${NC}"
fi

# Create logs directory
mkdir -p logs

# Auto-configure wake word models
echo -e "${YELLOW}Configuring wake word models...${NC}"
./venv/bin/python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('src')))

try:
    import openwakeword
    from openwakeword import Model as WakeWordModel
    import yaml
    
    # Check available models
    try:
        model = WakeWordModel()
        available_models = list(model.models.keys())
        
        if available_models:
            # Prefer certain models in order
            preferred_models = ['alexa', 'hey_mycroft', 'ok_nabu', 'hey_rhasspy']
            selected_model = None
            
            for preferred in preferred_models:
                if preferred in available_models:
                    selected_model = preferred
                    break
            
            if not selected_model:
                selected_model = available_models[0]
            
            print(f'Available models: {available_models}')
            print(f'Selected model: {selected_model}')
            
            # Update config file if it exists
            config_file = Path('config/config.yaml')
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                if 'wake_word' not in config:
                    config['wake_word'] = {}
                config['wake_word']['model'] = selected_model
                
                with open(config_file, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                
                print(f'Updated config to use wake word model: {selected_model}')
            else:
                print('Config file not found, will update during setup')
        else:
            print('No wake word models available, attempting to download...')
            
            # Try to download common models
            try:
                import urllib.request
                import os
                
                models_to_download = [
                    ('alexa', 'https://github.com/dscripka/openWakeWord/raw/main/openwakeword/resources/models/alexa_v0.1.tflite'),
                    ('hey_mycroft', 'https://github.com/dscripka/openWakeWord/raw/main/openwakeword/resources/models/hey_mycroft_v0.1.tflite')
                ]
                
                models_dir = Path(openwakeword.__file__).parent / 'resources' / 'models'
                models_dir.mkdir(parents=True, exist_ok=True)
                
                downloaded_models = []
                for model_name, url in models_to_download:
                    model_file = models_dir / f'{model_name}_v0.1.tflite'
                    if not model_file.exists():
                        try:
                            print(f'Downloading {model_name} model...')
                            urllib.request.urlretrieve(url, model_file)
                            downloaded_models.append(model_name)
                            print(f'Downloaded {model_name} model successfully')
                        except Exception as e:
                            print(f'Failed to download {model_name}: {e}')
                
                if downloaded_models:
                    print(f'Downloaded models: {downloaded_models}')
                    # Re-check available models
                    model = WakeWordModel()
                    available_models = list(model.models.keys())
                    if available_models:
                        selected_model = downloaded_models[0]  # Use first downloaded model
                        
                        config_file = Path('config/config.yaml')
                        if config_file.exists():
                            with open(config_file, 'r') as f:
                                config = yaml.safe_load(f)
                            
                            if 'wake_word' not in config:
                                config['wake_word'] = {}
                            config['wake_word']['model'] = selected_model
                            
                            with open(config_file, 'w') as f:
                                yaml.dump(config, f, default_flow_style=False)
                            
                            print(f'Updated config to use downloaded model: {selected_model}')
                else:
                    print('No models could be downloaded, disabling wake word detection')
                    
                    # Update config to disable wake word
                    config_file = Path('config/config.yaml')
                    if config_file.exists():
                        with open(config_file, 'r') as f:
                            config = yaml.safe_load(f)
                        
                        if 'wake_word' not in config:
                            config['wake_word'] = {}
                        config['wake_word']['enabled'] = False
                        
                        with open(config_file, 'w') as f:
                            yaml.dump(config, f, default_flow_style=False)
                        
                        print('Disabled wake word detection in config')
                        
            except Exception as e:
                print(f'Model download failed: {e}')
                print('Disabling wake word detection')
                
                # Update config to disable wake word
                config_file = Path('config/config.yaml')
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    if 'wake_word' not in config:
                        config['wake_word'] = {}
                    config['wake_word']['enabled'] = False
                    
                    with open(config_file, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False)
                    
                    print('Disabled wake word detection in config')
    except Exception as e:
        print(f'Could not configure wake word models: {e}')
        print('Wake word setup can be done manually later')
        
except ImportError:
    print('OpenWakeWord not available for configuration')
" 2>/dev/null || echo -e "${YELLOW}  Wake word configuration will be done during setup${NC}"

echo -e "${GREEN}✓ Wake word models configured${NC}"

# Test basic functionality
echo -e "${YELLOW}Testing installation...${NC}"
echo -e "${YELLOW}Running test: ./venv/bin/python src/main.py --help${NC}"
if ./venv/bin/python src/main.py --help; then
    echo -e "${GREEN}✓ Installation test passed${NC}"
else
    echo -e "${RED}✗ Installation test failed${NC}"
    echo -e "${RED}Error details shown above${NC}"
    exit 1
fi

echo ""
echo "========================================"
echo -e "${GREEN}Installation completed successfully!${NC}"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Edit configuration files:"
echo "   - config/config.yaml"
echo "   - .env"
echo ""
echo "2. Add your API keys:"
echo "   - OpenAI API key in .env"
echo "   - Home Assistant token in .env"
echo ""
echo "3. Test the installation:"
echo "   source venv/bin/activate"
echo "   python3 src/main.py --help"
echo ""
echo "4. Run the assistant:"
echo "   python3 src/main.py"
echo ""
echo "For help, see README.md or visit:"
echo "https://github.com/yourusername/ha-realtime-voice-assistant"
echo ""