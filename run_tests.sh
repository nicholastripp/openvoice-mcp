#!/bin/bash

# Convenient test runner script for HA Realtime Voice Assistant
# This script automatically uses the virtual environment - no need to activate it first

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "HA Realtime Voice Assistant Test Runner"
echo "========================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}Error: Virtual environment not found${NC}"
    echo "Please run ./install.sh first"
    exit 1
fi

# Function to run a test
run_test() {
    local test_name="$1"
    local test_file="$2"
    local test_args="${3:-}"
    
    echo -e "${YELLOW}Running $test_name...${NC}"
    if ./venv/bin/python "$test_file" $test_args; then
        echo -e "${GREEN}✓ $test_name passed${NC}"
    else
        echo -e "${RED}✗ $test_name failed${NC}"
        return 1
    fi
    echo ""
}

# Parse command line arguments
case "${1:-all}" in
    "audio")
        run_test "Audio Devices Test" "examples/test_audio_devices.py"
        ;;
    "ha")
        run_test "Home Assistant Connection Test" "examples/test_ha_connection.py"
        ;;
    "openai")
        run_test "OpenAI Connection Test" "examples/test_openai_connection.py"
        ;;
    "wake")
        run_test "Wake Word Test" "examples/test_wake_word.py" "--interactive"
        ;;
    "integration")
        run_test "Full Integration Test" "examples/test_full_integration.py"
        ;;
    "all")
        echo "Running all tests (except interactive wake word test)..."
        echo ""
        run_test "Audio Devices Test" "examples/test_audio_devices.py"
        run_test "Home Assistant Connection Test" "examples/test_ha_connection.py"
        run_test "OpenAI Connection Test" "examples/test_openai_connection.py"
        run_test "Full Integration Test" "examples/test_full_integration.py"
        echo -e "${YELLOW}Note: Run './run_tests.sh wake' separately for interactive wake word testing${NC}"
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [test_name]"
        echo ""
        echo "Available tests:"
        echo "  audio       - Test audio device detection"
        echo "  ha          - Test Home Assistant connection"
        echo "  openai      - Test OpenAI Realtime API connection"
        echo "  wake        - Test wake word detection (interactive)"
        echo "  integration - Run full integration test"
        echo "  all         - Run all tests (default, except wake word)"
        echo "  help        - Show this help message"
        exit 0
        ;;
    *)
        echo -e "${RED}Error: Unknown test '$1'${NC}"
        echo "Use '$0 help' to see available tests"
        exit 1
        ;;
esac

echo -e "${GREEN}Test run completed!${NC}"