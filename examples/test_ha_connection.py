#!/usr/bin/env python3
"""
Test script for Home Assistant connection and Conversation API

Usage:
    ./venv/bin/python examples/test_ha_connection.py
    
Note: Must be run from the project root using the virtual environment.
Requires config/config.yaml to be configured with your HA settings.
"""
import sys
import asyncio
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import load_config
from ha_client.conversation import HomeAssistantConversationClient
from utils.logger import setup_logging, get_logger


async def test_connection(config_path):
    """Test basic connection to Home Assistant"""
    logger = get_logger("HATest")
    
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Create client
        ha_client = HomeAssistantConversationClient(config.home_assistant)
        
        # Test connection
        logger.info("Testing connection to Home Assistant...")
        await ha_client.start()
        
        logger.info("✅ Successfully connected to Home Assistant")
        
        # Test basic API call
        logger.info("Testing REST API...")
        api_status = await ha_client.rest_client.get_api_status()
        logger.info(f"API Status: {api_status.get('message', 'OK')}")
        
        # Test configuration
        ha_config = await ha_client.rest_client.get_config()
        logger.info(f"HA Version: {ha_config.get('version', 'Unknown')}")
        logger.info(f"Location: {ha_config.get('location_name', 'Unknown')}")
        
        # Test conversation API with simple commands
        test_commands = [
            "what time is it",
            "turn on the lights",
            "what's the weather like",
            "hello",
            "invalid command xyz123"
        ]
        
        logger.info("Testing Conversation API...")
        for command in test_commands:
            logger.info(f"Testing command: '{command}'")
            try:
                response = await ha_client.process_command(command)
                logger.info(f"  Response type: {response.response_type.value}")
                logger.info(f"  Speech: {response.speech_text[:100]}{'...' if len(response.speech_text) > 100 else ''}")
                if response.success_entities:
                    logger.info(f"  Success entities: {len(response.success_entities)}")
                if response.failed_entities:
                    logger.info(f"  Failed entities: {len(response.failed_entities)}")
                logger.info("")
            except Exception as e:
                logger.error(f"  Error: {e}")
        
        await ha_client.stop()
        logger.info("✅ Home Assistant test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Home Assistant test failed: {e}")
        return False
    
    return True


async def test_entities(config_path):
    """Test entity discovery and states"""
    logger = get_logger("HAEntities")
    
    try:
        config = load_config(config_path)
        ha_client = HomeAssistantConversationClient(config.home_assistant)
        await ha_client.start()
        
        # Get all states
        logger.info("Fetching all entity states...")
        states = await ha_client.rest_client.get_states()
        
        # Group by domain
        domains = {}
        for state in states:
            entity_id = state.get("entity_id", "")
            domain = entity_id.split(".")[0] if "." in entity_id else "unknown"
            
            if domain not in domains:
                domains[domain] = []
            domains[domain].append(state)
        
        # Show summary
        logger.info(f"Found {len(states)} entities across {len(domains)} domains:")
        for domain, entities in sorted(domains.items()):
            logger.info(f"  {domain}: {len(entities)} entities")
        
        # Show sample entities from common domains
        common_domains = ["light", "switch", "sensor", "climate", "media_player"]
        for domain in common_domains:
            if domain in domains:
                logger.info(f"\nSample {domain} entities:")
                for entity in domains[domain][:5]:  # Show first 5
                    entity_id = entity.get("entity_id", "")
                    state = entity.get("state", "")
                    friendly_name = entity.get("attributes", {}).get("friendly_name", entity_id)
                    logger.info(f"  {entity_id}: {state} ({friendly_name})")
        
        await ha_client.stop()
        return True
        
    except Exception as e:
        logger.error(f"❌ Entity test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Home Assistant connection")
    parser.add_argument("--config", default="config/config.yaml", help="Configuration file path")
    parser.add_argument("--entities", action="store_true", help="Test entity discovery")
    parser.add_argument("--conversation", action="store_true", help="Test conversation API only")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging("INFO", console=True)
    
    async def run_tests():
        if args.entities:
            await test_entities(args.config)
        elif args.conversation:
            await test_connection(args.config)
        else:
            # Run both tests
            success1 = await test_connection(args.config)
            if success1:
                await test_entities(args.config)
    
    # Run tests
    asyncio.run(run_tests())


if __name__ == "__main__":
    main()