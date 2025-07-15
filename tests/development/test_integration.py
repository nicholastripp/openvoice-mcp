#!/usr/bin/env python3
"""
Integration test for the main application components
"""
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import load_config
from utils.logger import setup_logging, get_logger
from utils.dependencies import validate_audio_dependencies, validate_audio_device_access
from audio.capture import AudioCapture
from audio.playback import AudioPlayback
from openai_client.realtime import OpenAIRealtimeClient
from ha_client.conversation import HomeAssistantConversationClient

async def test_audio_capture_integration():
    """Test audio capture system integration"""
    logger = get_logger("IntegrationTest")
    
    try:
        # Load configuration
        config = load_config("config/config.yaml")
        
        # Test audio dependencies
        logger.info("Testing audio dependencies...")
        validate_audio_dependencies()
        input_devices, output_devices = validate_audio_device_access()
        logger.info(f"✅ Audio system validated: {len(input_devices)} input, {len(output_devices)} output devices")
        
        # Test audio capture initialization
        logger.info("Testing audio capture initialization...")
        audio_capture = AudioCapture(config.audio)
        await audio_capture.start()
        logger.info("✅ Audio capture started successfully")
        
        # Test audio playback initialization
        logger.info("Testing audio playback initialization...")
        audio_playback = AudioPlayback(config.audio)
        await audio_playback.start()
        logger.info("✅ Audio playback started successfully")
        
        # Test callback registration
        audio_captured = []
        
        def capture_callback(audio_data: bytes):
            audio_captured.append(len(audio_data))
            logger.debug(f"Audio callback received {len(audio_data)} bytes")
        
        audio_capture.add_callback(capture_callback)
        logger.info("✅ Audio callback registered")
        
        # Wait briefly to see if callbacks are working
        logger.info("Waiting 2 seconds for audio callbacks...")
        await asyncio.sleep(2)
        
        if audio_captured:
            logger.info(f"✅ Audio callbacks working: received {len(audio_captured)} audio chunks")
        else:
            logger.warning("⚠️ No audio callbacks received - microphone may be silent")
        
        # Cleanup
        await audio_capture.stop()
        await audio_playback.stop()
        logger.info("✅ Audio components cleaned up")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Audio integration test failed: {e}")
        return False

async def test_openai_integration():
    """Test OpenAI client integration"""
    logger = get_logger("IntegrationTest")
    
    try:
        # Load configuration
        config = load_config("config/config.yaml")
        
        # Test OpenAI client initialization
        logger.info("Testing OpenAI client initialization...")
        client = OpenAIRealtimeClient(config.openai, "You are a test assistant.")
        success = await client.connect()
        
        if success:
            logger.info("✅ OpenAI client connected successfully")
            await client.disconnect()
            logger.info("✅ OpenAI client disconnected successfully")
            return True
        else:
            logger.error("❌ OpenAI client connection failed")
            return False
        
    except Exception as e:
        logger.error(f"❌ OpenAI integration test failed: {e}")
        return False

async def test_ha_integration():
    """Test Home Assistant client integration"""
    logger = get_logger("IntegrationTest")
    
    try:
        # Load configuration
        config = load_config("config/config.yaml")
        
        # Test HA client initialization
        logger.info("Testing Home Assistant client initialization...")
        ha_client = HomeAssistantConversationClient(config.home_assistant)
        await ha_client.start()
        
        # Test basic API call
        api_status = await ha_client.rest_client.get_api_status()
        logger.info(f"✅ Home Assistant API status: {api_status.get('message', 'OK')}")
        
        await ha_client.stop()
        logger.info("✅ Home Assistant client stopped successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Home Assistant integration test failed: {e}")
        return False

async def main():
    """Run all integration tests"""
    setup_logging("INFO", console=True)
    logger = get_logger("IntegrationTest")
    
    logger.info("Starting integration tests...")
    
    # Test audio capture integration
    audio_success = await test_audio_capture_integration()
    
    # Test OpenAI integration
    openai_success = await test_openai_integration()
    
    # Test Home Assistant integration
    ha_success = await test_ha_integration()
    
    # Summary
    logger.info("Integration Test Results:")
    logger.info("=" * 40)
    logger.info(f"Audio System: {'✅ PASS' if audio_success else '❌ FAIL'}")
    logger.info(f"OpenAI Client: {'✅ PASS' if openai_success else '❌ FAIL'}")
    logger.info(f"Home Assistant: {'✅ PASS' if ha_success else '❌ FAIL'}")
    
    if all([audio_success, openai_success, ha_success]):
        logger.info("🎉 All integration tests passed!")
        return True
    else:
        logger.error("❌ Some integration tests failed")
        return False

if __name__ == "__main__":
    asyncio.run(main())