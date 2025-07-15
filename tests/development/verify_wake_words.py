#!/usr/bin/env python3
"""
Verify and list available Porcupine wake words
This script helps diagnose wake word configuration issues
"""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from wake_word.porcupine_detector import PorcupineDetector
from config import WakeWordConfig

def list_available_wake_words():
    """List all available wake words from the mapping"""
    print("Available Porcupine Wake Words")
    print("=" * 50)
    print("\nYou can use any of these values for the 'model' field in config.yaml:\n")
    
    # Group by category
    categories = {
        "Voice Assistants": ['alexa', 'hey_google', 'ok_google', 'hey_siri'],
        "Picovoice": ['picovoice', 'hey_picovoice', 'ok_picovoice'],
        "Fun Wake Words": ['americano', 'blueberry', 'bumblebee', 'grapefruit', 
                          'grasshopper', 'porcupine', 'terminator'],
        "Other": ['computer', 'hey_barista', 'pico_clock']
    }
    
    for category, words in categories.items():
        print(f"{category}:")
        for word in words:
            # Check if it's in the mapping
            if word in PorcupineDetector.KEYWORD_MAPPING:
                mapped_to = PorcupineDetector.KEYWORD_MAPPING[word]
                if word != mapped_to:
                    print(f"  - {word} (maps to '{mapped_to}')")
                else:
                    print(f"  - {word}")
            else:
                print(f"  - {word} [WARNING: Not in mapping!]")
        print()
    
    print("\nNOTE: 'jarvis' and 'hey_jarvis' are NOT built-in wake words!")
    print("To use custom wake words, create them at https://console.picovoice.ai/")
    print()

def test_wake_word_config(model_name):
    """Test if a wake word configuration is valid"""
    print(f"\nTesting wake word: '{model_name}'")
    print("-" * 40)
    
    # Create test config
    config = WakeWordConfig()
    config.engine = "porcupine"
    config.model = model_name
    config.sensitivity = 0.5
    
    try:
        # Try to create detector (won't actually initialize Porcupine)
        detector = PorcupineDetector(config)
        print(f"✓ Valid wake word: '{model_name}'")
        print(f"  Will use Porcupine keyword: '{detector.keywords[0]}'")
        return True
    except ValueError as e:
        print(f"✗ Invalid wake word: '{model_name}'")
        print(f"  Error: {str(e).split('\\n')[0]}")
        return False
    except Exception as e:
        print(f"✗ Error testing wake word: {e}")
        return False

def main():
    """Main function"""
    # List available wake words
    list_available_wake_words()
    
    # Test some common configurations
    print("\nTesting Common Configurations")
    print("=" * 50)
    
    test_cases = [
        "picovoice",      # Valid
        "alexa",          # Valid
        "hey_google",     # Valid
        "jarvis",         # Invalid - common mistake
        "hey_jarvis",     # Invalid - common mistake
        "computer",       # Valid
    ]
    
    valid_count = 0
    for test_case in test_cases:
        if test_wake_word_config(test_case):
            valid_count += 1
    
    print(f"\nSummary: {valid_count}/{len(test_cases)} wake words are valid")
    
    # Check current config
    print("\n\nChecking Current Configuration")
    print("=" * 50)
    
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            wake_config = config_data.get('wake_word', {})
            engine = wake_config.get('engine', 'unknown')
            model = wake_config.get('model', 'unknown')
            
            print(f"Current engine: {engine}")
            print(f"Current model: {model}")
            
            if engine == 'porcupine':
                test_wake_word_config(model)
            else:
                print(f"Not using Porcupine engine (using {engine})")
                
        except Exception as e:
            print(f"Error reading config: {e}")
    else:
        print("Config file not found")

if __name__ == "__main__":
    main()