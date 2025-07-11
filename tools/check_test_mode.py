#!/usr/bin/env python3
"""
Check if test_mode is being parsed correctly
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import load_config

# Load config
config = load_config("config/config.yaml")

print(f"Wake word enabled: {config.wake_word.enabled}")
print(f"Wake word test_mode: {config.wake_word.test_mode}")
print(f"Type of test_mode: {type(config.wake_word.test_mode)}")

# Check the condition
wake_word_only_mode = config.wake_word.enabled and hasattr(config.wake_word, 'test_mode') and config.wake_word.test_mode
print(f"\nCondition result: {wake_word_only_mode}")
print(f"Has test_mode attr: {hasattr(config.wake_word, 'test_mode')}")