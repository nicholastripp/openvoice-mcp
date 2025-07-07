#!/usr/bin/env python3
"""
Full integration test script
Tests the complete voice assistant pipeline without actual audio I/O

Usage:
    ./venv/bin/python examples/test_full_integration.py
    
Note: Must be run from the project root using the virtual environment.
Requires all configuration files to be set up properly.
"""
import sys
import asyncio
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import load_config
from personality import PersonalityProfile
from openai_client.realtime import OpenAIRealtimeClient
from ha_client.conversation import HomeAssistantConversationClient
from function_bridge import FunctionCallBridge
from utils.logger import setup_logging, get_logger


async def test_integration(config_path, persona_path):
    """Test full integration without audio"""
    logger = get_logger("IntegrationTest")
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config(config_path)
        personality = PersonalityProfile(persona_path)
        
        logger.info(f"Assistant name: {personality.backstory.name}")
        logger.info(f"Personality traits: {personality.get_trait_summary()}")
        
        # Initialize Home Assistant client
        logger.info("Initializing Home Assistant client...")
        ha_client = HomeAssistantConversationClient(config.home_assistant)
        await ha_client.start()
        
        # Test HA connection
        api_status = await ha_client.rest_client.get_api_status()
        logger.info(f"HA API Status: {api_status.get('message', 'OK')}")
        
        # Initialize function bridge
        logger.info("Initializing function bridge...")
        function_bridge = FunctionCallBridge(ha_client)
        
        # Initialize OpenAI client
        logger.info("Initializing OpenAI client...")
        personality_prompt = personality.generate_prompt()
        logger.info(f"Generated personality prompt: {personality_prompt[:200]}...")
        
        openai_client = OpenAIRealtimeClient(config.openai, personality_prompt)
        
        # Register functions
        for func_def in function_bridge.get_function_definitions():
            openai_client.register_function(
                name=func_def["name"],
                handler=function_bridge.handle_function_call,
                description=func_def["description"],
                parameters=func_def["parameters"]
            )
            logger.info(f"Registered function: {func_def['name']}")
        
        # Connect to OpenAI
        logger.info("Connecting to OpenAI...")
        success = await openai_client.connect()
        if not success:
            logger.error("❌ Failed to connect to OpenAI")
            return False
        
        logger.info(f"✅ Connected to OpenAI (Session: {openai_client.session_id})")
        
        # Test commands that should trigger HA function calls
        test_commands = [
            "Hello, please introduce yourself",
            "Turn on the living room lights",
            "What's the temperature in the house?",
            "Turn off all the lights in the kitchen",
            "Set the thermostat to 72 degrees"
        ]
        
        logger.info("Testing voice commands...")
        for i, command in enumerate(test_commands, 1):
            logger.info(f"\n--- Test {i}: '{command}' ---")
            
            # Send text command
            await openai_client.send_text(command)
            
            # Wait for response
            await asyncio.sleep(3)
            
            logger.info(f"Command {i} completed")
        
        # Test personality update
        logger.info("\n--- Testing personality update ---")
        personality.update_trait("humor", 80)
        new_prompt = personality.generate_prompt()
        openai_client.update_personality(new_prompt)
        
        await openai_client.send_text("Tell me a joke about smart homes")
        await asyncio.sleep(3)
        
        # Cleanup
        logger.info("\nCleaning up...")
        await openai_client.disconnect()
        await ha_client.stop()
        
        logger.info("✅ Full integration test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}", exc_info=True)
        return False


async def test_function_bridge_standalone(config_path):
    """Test function bridge independently"""
    logger = get_logger("FunctionBridgeTest")
    
    try:
        logger.info("Testing function bridge standalone...")
        
        config = load_config(config_path)
        ha_client = HomeAssistantConversationClient(config.home_assistant)
        await ha_client.start()
        
        function_bridge = FunctionCallBridge(ha_client)
        
        # Test commands
        test_commands = [
            "turn on the lights",
            "what time is it",
            "invalid command xyz123"
        ]
        
        for command in test_commands:
            logger.info(f"Testing: '{command}'")
            
            args = {"command": command}
            result = await function_bridge.handle_function_call("control_home_assistant", args)
            
            logger.info(f"  Success: {result.get('success', False)}")
            logger.info(f"  Message: {result.get('message', 'No message')}")
            if 'error' in result:
                logger.info(f"  Error: {result['error']}")
            logger.info("")
        
        await ha_client.stop()
        logger.info("✅ Function bridge test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Function bridge test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test full integration")
    parser.add_argument("--config", default="config/config.yaml", help="Configuration file path")
    parser.add_argument("--persona", default="config/persona.ini", help="Personality file path")
    parser.add_argument("--bridge-only", action="store_true", help="Test function bridge only")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging("INFO", console=True)
    
    async def run_tests():
        if args.bridge_only:
            await test_function_bridge_standalone(args.config)
        else:
            await test_integration(args.config, args.persona)
    
    # Run tests
    asyncio.run(run_tests())


if __name__ == "__main__":
    main()