#!/usr/bin/env python3
"""
Check available wake word models and help with setup
"""
import sys
import os
from pathlib import Path

# Simple Unicode test without surrogates
def test_unicode_support():
    """Test if Unicode is supported safely"""
    try:
        # Test with simple Unicode characters (no surrogates)
        test_char = "✅"  # Direct Unicode instead of escape
        print(test_char, end='', file=sys.stderr)
        print('\r', end='', file=sys.stderr)  # Clear test
        return True
    except (UnicodeEncodeError, UnicodeDecodeError):
        return False

USE_UNICODE = test_unicode_support()

# Define symbols based on Unicode support (using safe characters)
if USE_UNICODE:
    OK = "✅"
    ERROR = "❌"
    TIP = "💡"
    CONFIG = "🔧"
    INFO = "🔍"
else:
    OK = "[OK]"
    ERROR = "[ERROR]"
    TIP = "[TIP]"
    CONFIG = "[CONFIG]"
    INFO = "[INFO]"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_available_models():
    """Check what wake word models are available"""
    try:
        import openwakeword
        from openwakeword import Model as WakeWordModel
        
        print("OpenWakeWord is installed!")
        print()
        
        # Try to create a model to see what's available
        try:
            model = WakeWordModel()
            available_models = list(model.models.keys())
            
            if available_models:
                print(f"{OK} Available pre-installed models:")
                for model_name in sorted(available_models):
                    print(f"  - {model_name}")
                print()
                print("You can use any of these models in your config/config.yaml:")
                print("wake_word:")
                print(f"  model: \"{available_models[0]}\"  # Example")
            else:
                print(f"{ERROR} No pre-installed models found")
                
        except Exception as e:
            print(f"{ERROR} Error checking models: {e}")
            
        # Check for custom models directory
        try:
            import pkg_resources
            oww_path = pkg_resources.resource_filename('openwakeword', 'resources/models/')
            models_dir = Path(oww_path)
            
            print(f"Models directory: {models_dir}")
            
            if models_dir.exists():
                model_files = list(models_dir.glob("*.tflite"))
                print(f"Found {len(model_files)} .tflite model files:")
                for model_file in sorted(model_files):
                    print(f"  - {model_file.name}")
            else:
                print("Models directory does not exist")
                
        except Exception as e:
            print(f"Error checking models directory: {e}")
            
    except ImportError:
        print(f"{ERROR} OpenWakeWord is not installed")
        print("Install with: pip install openwakeword")
        return
        
    print()
    print(f"{TIP} Wake Word Setup Tips:")
    print("1. If no models are available, try using a different wake word model")
    print("2. You can download additional models from:")
    print("   https://github.com/dscripka/openWakeWord/tree/main/openwakeword/resources")
    print("3. Or try using 'alexa' or 'hey_mycroft' which might be pre-installed")

def suggest_config_change():
    """Suggest a config change to use an available model"""
    try:
        import openwakeword
        from openwakeword import Model as WakeWordModel
        
        model = WakeWordModel()
        available_models = list(model.models.keys())
        
        if available_models:
            recommended = available_models[0]
            print()
            print(f"{CONFIG} Suggested config change:")
            print(f"Edit config/config.yaml and change the wake word model to:")
            print()
            print("wake_word:")
            print(f"  model: \"{recommended}\"")
            print()
            
    except:
        pass

if __name__ == "__main__":
    print(f"{INFO} Checking Wake Word Models...")
    print("=" * 50)
    check_available_models()
    suggest_config_change()