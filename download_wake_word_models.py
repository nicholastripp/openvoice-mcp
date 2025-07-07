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
        
        logger.info("🔄 Downloading OpenWakeWord models...")
        logger.info("This may take a few minutes on first run...")
        
        # Get the models directory
        models_dir = Path(openwakeword.__file__).parent / "resources" / "models"
        logger.info(f"Models directory: {models_dir}")
        
        # Create models directory if it doesn't exist
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Try the official download function first
        try:
            from openwakeword.utils import download_models as oww_download
            oww_download()
            logger.info("✅ Official download completed!")
        except Exception as e:
            logger.warning(f"Official download failed: {e}")
            logger.info("Attempting manual download...")
            
            # Manual download of essential models
            import urllib.request
            import ssl
            
            # Create SSL context for downloads
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Essential models to download
            models_to_download = {
                "alexa_v0.1.tflite": "https://github.com/dscripka/openWakeWord/raw/main/openwakeword/resources/models/alexa_v0.1.tflite",
                "hey_mycroft_v0.1.tflite": "https://github.com/dscripka/openWakeWord/raw/main/openwakeword/resources/models/hey_mycroft_v0.1.tflite",
                "hey_jarvis_v0.1.tflite": "https://github.com/dscripka/openWakeWord/raw/main/openwakeword/resources/models/hey_jarvis_v0.1.tflite",
                "ok_nabu_v0.1.tflite": "https://github.com/dscripka/openWakeWord/raw/main/openwakeword/resources/models/ok_nabu_v0.1.tflite"
            }
            
            for model_filename, url in models_to_download.items():
                model_path = models_dir / model_filename
                if not model_path.exists():
                    try:
                        logger.info(f"Downloading {model_filename}...")
                        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                        with urllib.request.urlopen(req, context=ssl_context) as response:
                            with open(model_path, 'wb') as f:
                                f.write(response.read())
                        logger.info(f"✅ Downloaded {model_filename}")
                    except Exception as download_error:
                        logger.warning(f"Failed to download {model_filename}: {download_error}")
                else:
                    logger.info(f"✅ {model_filename} already exists")
        
        # List available models
        logger.info("\n📋 Available wake word models:")
        
        model_files = list(models_dir.glob("*.tflite"))
        if model_files:
            for model_file in sorted(model_files):
                model_name = model_file.stem.replace("_v0.1", "")
                logger.info(f"  - {model_name} ({model_file.name})")
        else:
            logger.warning("No model files found in models directory")
            return False
            
        logger.info("\n💡 To use these models, update your config/config.yaml:")
        logger.info("wake_word:")
        logger.info("  model: \"alexa\"  # or hey_mycroft, hey_jarvis, ok_nabu")
        
        logger.info("\n🔧 You can now run the wake word test:")
        logger.info("  ./run_tests.sh wake")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ OpenWakeWord not installed: {e}")
        logger.error("Please install with: pip install openwakeword")
        return False
        
    except Exception as e:
        logger.error(f"❌ Error downloading models: {e}")
        import traceback
        logger.error(traceback.format_exc())
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