#!/usr/bin/env python3
"""
Download Wake Word Models for OpenWakeWord

This script downloads the necessary wake word models for OpenWakeWord.
Run this script if you encounter errors about missing wake word models.
"""

import sys
import os
from pathlib import Path
import logging

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def download_models():
    """Download wake word models"""
    logger = logging.getLogger("WakeWordDownloader")
    
    try:
        import openwakeword
        from openwakeword.utils import download_models
        
        logger.info("🔄 Downloading OpenWakeWord models...")
        logger.info("This may take a few minutes on first run...")
        
        # Download all available models
        download_models()
        
        logger.info("✅ Wake word models downloaded successfully!")
        
        # List available models
        logger.info("\n📋 Available wake word models:")
        
        # Try to list models
        try:
            from openwakeword.model import Model
            
            # Get the models directory
            models_dir = Path(openwakeword.__file__).parent / "resources" / "models"
            
            if models_dir.exists():
                model_files = list(models_dir.glob("*.tflite"))
                if model_files:
                    for model_file in sorted(model_files):
                        model_name = model_file.stem
                        logger.info(f"  - {model_name}")
                else:
                    logger.warning("No model files found in models directory")
            else:
                logger.warning("Models directory not found")
                
        except Exception as e:
            logger.warning(f"Could not list models: {e}")
            
        logger.info("\n💡 Common wake word models:")
        logger.info("  - alexa")
        logger.info("  - hey_mycroft")
        logger.info("  - hey_jarvis")
        logger.info("  - ok_nabu")
        logger.info("  - hey_rhasspy")
        
        logger.info("\n🔧 You can now run the wake word test:")
        logger.info("  ./run_tests.sh wake")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ OpenWakeWord not installed: {e}")
        logger.error("Please install with: pip install openwakeword")
        return False
        
    except Exception as e:
        logger.error(f"❌ Error downloading models: {e}")
        return False

def main():
    """Main function"""
    setup_logging()
    logger = logging.getLogger("WakeWordDownloader")
    
    logger.info("🚀 Wake Word Model Downloader")
    logger.info("=" * 50)
    
    success = download_models()
    
    if success:
        logger.info("\n✅ Setup complete!")
        sys.exit(0)
    else:
        logger.error("\n❌ Setup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()