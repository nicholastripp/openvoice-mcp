# Understanding Test Scripts in This Project

This guide clarifies the different types of test scripts in the project and how to run them correctly.

## Types of Test Scripts

### 1. Standalone Example Scripts (`examples/` directory)

These are meant to be run directly by users to test functionality:

- **Location**: `examples/` directory
- **Purpose**: User-facing tests and demonstrations
- **How to run**: 
  ```bash
  python examples/test_mcp_connection.py
  # or
  python3 examples/test_mcp_connection.py
  ```

**Example scripts include:**
- `test_mcp_connection.py` - Tests MCP connection to Home Assistant
- `test_ha_connection.py` - Tests basic Home Assistant connectivity
- `test_audio_devices.py` - Lists and tests audio devices

### 2. Unit Tests (`tests/` directory)

These are pytest-based unit tests for developers:

- **Location**: `tests/` directory  
- **Purpose**: Automated testing during development
- **Framework**: pytest
- **How to run**:
  ```bash
  # Run all tests
  pytest
  
  # Run specific test file
  pytest tests/test_mcp_client.py
  
  # Run with verbose output
  pytest -v
  ```

**DO NOT** run these files directly with `python` or `./`

## Common Mistakes and Solutions

### Mistake 1: Running Python files as shell scripts
```bash
# WRONG - This tries to execute as shell script
./test_mcp.py

# Error you'll see:
# ./test_mcp.py: line 1: import: command not found
```

**Solution**: Use Python interpreter
```bash
# CORRECT
python test_mcp.py
```

### Mistake 2: Running pytest files directly
```bash
# WRONG - Unit tests aren't standalone scripts
python tests/test_mcp_client.py

# WRONG - Same issue
./tests/test_mcp_client.py
```

**Solution**: Use pytest
```bash
# CORRECT
pytest tests/test_mcp_client.py
```

### Mistake 3: Wrong directory
```bash
# WRONG - Running from wrong directory
cd tests
python test_mcp.py  # Can't find imports
```

**Solution**: Run from project root
```bash
# CORRECT - From project root
cd ~/ha-realtime-assist
python examples/test_mcp_connection.py
```

## Quick Reference

| Script Type | Location | Run Command | Purpose |
|------------|----------|-------------|---------|
| Example scripts | `examples/` | `python examples/script.py` | User testing |
| Unit tests | `tests/` | `pytest tests/test_file.py` | Development |
| Main app | `src/` | `python -m src.main` | Run assistant |

## Making Scripts Executable (Optional)

If you want to run scripts with `./`, they need:

1. Shebang line at the top:
   ```python
   #!/usr/bin/env python3
   ```

2. Execute permission:
   ```bash
   chmod +x script.py
   ```

3. Then you can run:
   ```bash
   ./script.py
   ```

But using `python script.py` always works and is recommended.

## For MCP Testing Specifically

To test MCP connection:
```bash
# From project root directory
cd ~/ha-realtime-assist

# Make sure you're in virtual environment
source venv/bin/activate

# Run the test
python examples/test_mcp_connection.py
```

This will verify your MCP setup is working before running the full voice assistant.