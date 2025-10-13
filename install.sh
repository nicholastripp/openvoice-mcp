#!/bin/bash

# Installation script for OpenVoice MCP
# For Raspberry Pi and other Linux systems

set -e

echo "========================================"
echo "OpenVoice MCP Installer"
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
echo ""
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
        # Set secure permissions on .env file
        chmod 600 .env
        echo -e "${GREEN}✓ Set secure permissions on .env file${NC}"
    else
        echo -e "${RED}✗ .env.example not found${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}  .env already exists${NC}"
    # Check and fix permissions if needed
    current_perms=$(stat -c "%a" .env 2>/dev/null || stat -f "%OLp" .env 2>/dev/null)
    if [ "$current_perms" != "600" ]; then
        chmod 600 .env
        echo -e "${GREEN}✓ Fixed permissions on .env file (was $current_perms, now 600)${NC}"
    fi
fi

if [ ! -f "config/persona.ini" ]; then
    if [ -f "config/persona.ini.example" ]; then
        cp config/persona.ini.example config/persona.ini
        echo -e "${GREEN}✓ Created config/persona.ini${NC}"
    else
        echo -e "${RED}✗ config/persona.ini.example not found${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}  config/persona.ini already exists${NC}"
fi

# Create logs directory
mkdir -p logs
echo -e "${GREEN}✓ Created logs directory${NC}"

# Configure Web UI security (optional)
echo ""
echo -e "${YELLOW}Web UI Security Configuration (Optional)${NC}"
echo "The web UI can be accessed remotely. Would you like to set up authentication?"
read -p "Enable web UI with authentication? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Prompt for username
    read -p "Enter web UI username (default: admin): " web_username
    web_username=${web_username:-admin}
    
    # Prompt for password
    while true; do
        read -s -p "Enter web UI password: " web_password
        echo
        read -s -p "Confirm password: " web_password_confirm
        echo
        
        if [ "$web_password" = "$web_password_confirm" ] && [ -n "$web_password" ]; then
            break
        else
            echo -e "${RED}Passwords don't match or are empty. Please try again.${NC}"
        fi
    done
    
    # Generate password hash using bcrypt
    web_password_hash=$(./venv/bin/python -c "
import bcrypt
password = '$web_password'
salt = bcrypt.gensalt(rounds=12)
hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
print(hashed.decode('utf-8'))
    ")
    
    # Update config.yaml for username and enable web UI
    echo -e "${YELLOW}Updating config/config.yaml with web UI settings...${NC}"
    
    # Enable web UI
    sed -i.bak "s/^web_ui:$/web_ui:\n  enabled: true/" config/config.yaml 2>/dev/null || \
    sed -i '' "s/^web_ui:$/web_ui:\n  enabled: true/" config/config.yaml 2>/dev/null || \
    echo "Note: Please manually enable web_ui in config.yaml"
    
    # Update username in config
    sed -i.bak "s/username: \"admin\"/username: \"$web_username\"/" config/config.yaml 2>/dev/null || \
    sed -i '' "s/username: \"admin\"/username: \"$web_username\"/" config/config.yaml 2>/dev/null
    
    # Update password_hash to use environment variable
    sed -i.bak 's/password_hash: ""/password_hash: ${WEB_UI_PASSWORD_HASH}/' config/config.yaml 2>/dev/null || \
    sed -i '' 's/password_hash: ""/password_hash: ${WEB_UI_PASSWORD_HASH}/' config/config.yaml 2>/dev/null
    
    # Add password hash to .env file
    echo -e "${YELLOW}Saving password hash to .env file...${NC}"
    if grep -q "^WEB_UI_PASSWORD_HASH=" .env 2>/dev/null; then
        # Update existing entry
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s|^WEB_UI_PASSWORD_HASH=.*|WEB_UI_PASSWORD_HASH=$web_password_hash|" .env
        else
            sed -i "s|^WEB_UI_PASSWORD_HASH=.*|WEB_UI_PASSWORD_HASH=$web_password_hash|" .env
        fi
    else
        # Add new entry
        echo "" >> .env
        echo "# Web UI Authentication (set by installer)" >> .env
        echo "WEB_UI_PASSWORD_HASH=$web_password_hash" >> .env
    fi
    
    # Ensure .env has secure permissions after modification
    chmod 600 .env
    echo -e "${GREEN}✓ Ensured secure permissions on .env file${NC}"
    
    echo -e "${GREEN}✓ Web UI authentication configured${NC}"
    echo -e "${GREEN}  Username: $web_username${NC}"
    echo -e "${GREEN}  Access at: https://<your-ip>:8443${NC}"
    echo -e "${YELLOW}  Note: You'll see a certificate warning on first access (self-signed cert)${NC}"
fi

# Test basic functionality
echo ""
echo -e "${YELLOW}Testing installation...${NC}"
if ./venv/bin/python src/main.py --help > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Installation test passed${NC}"
else
    echo -e "${RED}✗ Installation test failed${NC}"
    echo -e "${RED}Please check the error messages above${NC}"
    exit 1
fi

echo ""
echo "================================================"
echo -e "${GREEN}Installation completed successfully!${NC}"
echo "================================================"
echo ""
echo -e "${BLUE}Configuration files created successfully!${NC}"
echo ""
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
echo "   - HA_URL=http://homeassistant.local:8123"
echo "   - HA_TOKEN=your_home_assistant_token"
echo "   - PICOVOICE_ACCESS_KEY=your_picovoice_key"
echo ""

echo -e "${YELLOW}2. (Optional) Customize your assistant:${NC}"
echo "   Edit the configuration files:"
echo "   nano config/config.yaml      # Audio settings, wake word, etc."
echo "   nano config/persona.ini      # Assistant personality"
echo ""

echo -e "${YELLOW}3. Run the voice assistant:${NC}"
echo "   source venv/bin/activate"
echo "   python src/main.py"
echo ""

echo "================================================"
echo -e "${GREEN}Setup complete!${NC}"
echo "================================================"
echo ""
echo "For detailed documentation, see README.md"
echo "GitHub: https://github.com/nicholastripp/openvoice-mcp"
echo ""