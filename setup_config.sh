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