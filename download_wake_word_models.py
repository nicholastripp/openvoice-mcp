#!/usr/bin/env python3
"""
Standalone script to download and manage OpenWakeWord models
"""
import sys
import argparse
from pathlib import Path

# Add src to path so we can import utilities
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.logger import setup_logging, get_logger


def download_all_models():
    """Download all available OpenWakeWord models"""
    logger = get_logger("ModelDownloader")
    
    try:
        import openwakeword
        from openwakeword import utils
        
        logger.info("Downloading OpenWakeWord models...")
        logger.info("This may take several minutes depending on your internet connection...")
        
        # Download all available models
        utils.download_models()
        
        logger.info("[OK] All models downloaded successfully!")
        
        # List downloaded models
        list_models()
        
        return True
        
    except ImportError as e:
        logger.error(f"[ERROR] OpenWakeWord not installed: {e}")
        logger.error("Install with: pip install openwakeword>=0.6.0")
        return False
    except Exception as e:
        logger.error(f"[ERROR] Model download failed: {e}")
        logger.error("")
        logger.error("🔧 Troubleshooting steps:")
        logger.error("1. Check your internet connection")
        logger.error("2. Check if you have write permissions in the package directory")
        logger.error("3. Try running with different arguments")
        logger.error("")
        return False


def download_specific_models(model_names):
    """Download specific OpenWakeWord models"""
    logger = get_logger("ModelDownloader")
    
    try:
        import openwakeword
        from openwakeword import utils
        
        logger.info(f"Downloading specific models: {model_names}")
        
        # Download specific models
        utils.download_models(model_names=model_names)
        
        logger.info("[OK] Specific models downloaded successfully!")
        
        # List downloaded models
        list_models()
        
        return True
        
    except ImportError as e:
        logger.error(f"[ERROR] OpenWakeWord not installed: {e}")
        logger.error("Install with: pip install openwakeword>=0.6.0")
        return False
    except Exception as e:
        logger.error(f"[ERROR] Model download failed: {e}")
        logger.error("")
        logger.error("Available models: alexa, hey_jarvis, hey_mycroft, hey_rhasspy, ok_nabu")
        logger.error("")
        return False


def list_models():
    """List available OpenWakeWord models"""
    logger = get_logger("ModelLister")
    
    try:
        import openwakeword
        from openwakeword import Model as WakeWordModel
        from pathlib import Path
        
        # Check for model files in the package directory
        models_dir = Path(openwakeword.__file__).parent / "resources" / "models"
        
        if models_dir.exists():
            model_files = list(models_dir.glob("*.tflite"))
            if model_files:
                logger.info("[FILES] Available model files:")
                for file in sorted(model_files):
                    logger.info(f"  - {file.name}")
            else:
                logger.warning("No model files found in models directory")
        else:
            logger.warning(f"Models directory does not exist: {models_dir}")
        
        # Try to load a test model to see what's available
        try:
            test_model = WakeWordModel()
            if test_model.models:
                logger.info("[MODELS] Loaded models:")
                for model_name in sorted(test_model.models.keys()):
                    logger.info(f"  - {model_name}")
            else:
                logger.warning("No models could be loaded")
        except Exception as e:
            logger.debug(f"Could not load test model: {e}")
        
        return True
        
    except ImportError as e:
        logger.error(f"[ERROR] OpenWakeWord not installed: {e}")
        return False
    except Exception as e:
        logger.error(f"[ERROR] Error listing models: {e}")
        return False


def test_models():
    """Test OpenWakeWord model loading"""
    logger = get_logger("ModelTester")
    
    try:
        import openwakeword
        from openwakeword import Model as WakeWordModel
        import numpy as np
        
        logger.info("Testing OpenWakeWord model loading...")
        
        # Test model loading with common models
        test_models = ['alexa_v0.1', 'hey_jarvis_v0.1', 'hey_mycroft_v0.1']
        
        for model_name in test_models:
            try:
                logger.info(f"Testing model: {model_name}")
                test_model = WakeWordModel(wakeword_models=[model_name])
                
                # Test with dummy audio
                dummy_audio = np.zeros(1280, dtype=np.float32)
                predictions = test_model.predict(dummy_audio)
                
                logger.info(f"[OK] {model_name} loaded successfully")
                logger.info(f"   Available models: {list(test_model.models.keys())}")
                
            except Exception as e:
                logger.warning(f"[WARNING] {model_name} failed to load: {e}")
        
        return True
        
    except ImportError as e:
        logger.error(f"[ERROR] OpenWakeWord not installed: {e}")
        return False
    except Exception as e:
        logger.error(f"[ERROR] Model testing failed: {e}")
        return False


def cleanup_models():
    """Clean up downloaded models"""
    logger = get_logger("ModelCleaner")
    
    try:
        import openwakeword
        from pathlib import Path
        
        models_dir = Path(openwakeword.__file__).parent / "resources" / "models"
        
        if models_dir.exists():
            model_files = list(models_dir.glob("*.tflite"))
            if model_files:
                logger.info(f"Found {len(model_files)} model files to clean up")
                for file in model_files:
                    file.unlink()
                    logger.info(f"Removed: {file.name}")
                logger.info("[OK] Model cleanup completed")
            else:
                logger.info("No model files found to clean up")
        else:
            logger.info("Models directory does not exist - nothing to clean up")
        
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Model cleanup failed: {e}")
        return False


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Download and manage OpenWakeWord models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_wake_word_models.py --download-all
  python download_wake_word_models.py --download alexa hey_jarvis
  python download_wake_word_models.py --list
  python download_wake_word_models.py --test
  python download_wake_word_models.py --cleanup
        """
    )
    
    parser.add_argument("--download-all", action="store_true", 
                       help="Download all available models")
    parser.add_argument("--download", nargs="+", metavar="MODEL",
                       help="Download specific models (e.g., alexa hey_jarvis)")
    parser.add_argument("--list", action="store_true",
                       help="List available models")
    parser.add_argument("--test", action="store_true",
                       help="Test model loading")
    parser.add_argument("--cleanup", action="store_true",
                       help="Remove all downloaded models")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Set logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, console=True)
    logger = get_logger("ModelManager")
    
    # If no arguments provided, download all models (default behavior)
    if not any([args.download_all, args.download, args.list, args.test, args.cleanup]):
        logger.info("No arguments provided, downloading all models...")
        success = download_all_models()
    else:
        success = True
        
        # Execute requested operations
        if args.download_all:
            success &= download_all_models()
        
        if args.download:
            success &= download_specific_models(args.download)
        
        if args.list:
            success &= list_models()
        
        if args.test:
            success &= test_models()
        
        if args.cleanup:
            success &= cleanup_models()
    
    # Exit with appropriate code
    if success:
        logger.info("[OK] All operations completed successfully")
        sys.exit(0)
    else:
        logger.error("[ERROR] Some operations failed")
        sys.exit(1)


if __name__ == "__main__":
    main()