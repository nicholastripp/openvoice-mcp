#!/bin/bash

# Setup script for HA Realtime Voice Assistant
# Creates initial configuration files and guides user through setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "================================================"
echo "HA Realtime Voice Assistant Configuration Setup"
echo "================================================"

# Check if we're in the right directory
if [ ! -f "install.sh" ] || [ ! -d "src" ]; then
    echo -e "${RED}Error: This script must be run from the project root directory${NC}"
    exit 1
fi

# Create configuration files
echo -e "${YELLOW}Setting up configuration files...${NC}"

# Copy config files if they don't exist
if [ ! -f "config/config.yaml" ]; then
    if [ -f "config/config.yaml.example" ]; then
        cp config/config.yaml.example config/config.yaml
        echo -e "${GREEN}✓ Created config/config.yaml${NC}"
    else
        echo -e "${RED}✗ config/config.yaml.example not found${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}  config/config.yaml already exists${NC}"
fi

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}✓ Created .env${NC}"
    else
        echo -e "${RED}✗ .env.example not found${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}  .env already exists${NC}"
fi

echo ""

# Check and configure wake word models if virtual environment exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}Checking wake word models...${NC}"
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
            
            print(f'Available wake word models: {available_models}')
            print(f'Recommended model: {selected_model}')
            
            # Update config file if it exists
            config_file = Path('config/config.yaml')
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                current_model = config.get('wake_word', {}).get('model', 'unknown')
                if current_model not in available_models:
                    if 'wake_word' not in config:
                        config['wake_word'] = {}
                    config['wake_word']['model'] = selected_model
                    
                    with open(config_file, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False)
                    
                    print(f'Updated config to use available model: {selected_model}')
                else:
                    print(f'Current model \"{current_model}\" is available, no change needed')
        else:
            print('No wake word models available')
            print('Wake word detection will be disabled')
            
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
        print(f'Could not check wake word models: {e}')
        print('Wake word configuration should be checked manually')
        
except ImportError:
    print('OpenWakeWord not available - wake word setup will need manual configuration')
" 2>/dev/null || echo -e "${YELLOW}  Wake word models will be configured when dependencies are installed${NC}"
    
    echo -e "${GREEN}✓ Wake word models checked${NC}"
else
    echo -e "${YELLOW}Virtual environment not found - wake word models will be configured during installation${NC}"
fi

echo ""
echo -e "${BLUE}Configuration files created successfully!${NC}"
echo ""

# Provide setup guidance
echo "================================================"
echo "Next Steps:"
echo "================================================"
echo ""
echo -e "${YELLOW}1. Configure your API keys:${NC}"
echo "   Edit the .env file and add your keys:"
echo "   nano .env"
echo ""
echo "   Required keys:"
echo "   - OPENAI_API_KEY=sk-..."
echo "   - HA_TOKEN=your_home_assistant_token"
echo ""

echo -e "${YELLOW}2. Configure your settings:${NC}"
echo "   Edit the config.yaml file:"
echo "   nano config/config.yaml"
echo ""
echo "   Update:"
echo "   - Home Assistant URL"
echo "   - Audio device settings"
echo "   - Wake word preferences"
echo ""

echo -e "${YELLOW}3. Test your setup:${NC}"
echo "   Run individual tests to verify everything works:"
echo "   ./run_tests.sh audio       # Test audio devices"
echo "   ./run_tests.sh ha          # Test HA connection"
echo "   ./run_tests.sh openai      # Test OpenAI connection"
echo ""

echo -e "${YELLOW}4. Run the voice assistant:${NC}"
echo "   ./venv/bin/python src/main.py"
echo ""

echo "================================================"
echo -e "${GREEN}Setup complete!${NC}"
echo "================================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo -e "${YELLOW}Note: Virtual environment not found.${NC}"
    echo "Run './install.sh' first to install dependencies."
fi