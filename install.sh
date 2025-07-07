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

# Test basic functionality
echo -e "${YELLOW}Testing installation...${NC}"
if python3 src/main.py --help &> /dev/null; then
    echo -e "${GREEN}✓ Installation test passed${NC}"
else
    echo -e "${RED}✗ Installation test failed${NC}"
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