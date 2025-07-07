#!/usr/bin/env python3
"""
Test script for Home Assistant connection and Conversation API

Usage:
    ./venv/bin/python examples/test_ha_connection.py
    
Note: Must be run from the project root using the virtual environment.
Requires config/config.yaml to be configured with your HA settings.
"""
print("DEBUG: Script starting...", flush=True)

import sys
import asyncio
import argparse
from pathlib import Path

print("DEBUG: Basic imports successful", flush=True)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
print("DEBUG: Path added to sys.path", flush=True)

try:
    from config import load_config
    print("DEBUG: config import successful", flush=True)
    from ha_client.conversation import HomeAssistantConversationClient
    print("DEBUG: ha_client import successful", flush=True) 
    from utils.logger import setup_logging, get_logger
    print("DEBUG: logger import successful", flush=True)
except Exception as e:
    print(f"DEBUG: Import failed: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)


async def test_connection(config_path):
    """Test basic connection to Home Assistant"""
    logger = get_logger("ha_voice_assistant.HATest")
    
    # Step 1: Load configuration
    try:
        logger.info("Step 1: Loading configuration...")
        config = load_config(config_path)
        logger.info("✅ Configuration loaded successfully")
        
        # Log connection details (masked for security)
        masked_url = config.home_assistant.url
        masked_token = f"{config.home_assistant.token[:8]}...{config.home_assistant.token[-4:]}" if len(config.home_assistant.token) > 12 else "****"
        logger.info(f"HA URL: {masked_url}")
        logger.info(f"HA Token: {masked_token}")
        logger.info(f"Timeout: {config.home_assistant.timeout}s")
        
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        logger.error(f"   Check that {config_path} exists and is valid YAML")
        return False
    
    # Step 2: Test basic connectivity
    try:
        logger.info("Step 2: Testing basic connectivity...")
        import aiohttp
        import asyncio
        
        # Parse URL for connectivity test
        from urllib.parse import urlparse
        parsed_url = urlparse(config.home_assistant.url)
        test_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        # Test basic HTTP connectivity (without auth)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            try:
                async with session.get(test_url) as response:
                    logger.info(f"✅ Basic connectivity OK (status: {response.status})")
            except aiohttp.ClientConnectorError as e:
                logger.error(f"❌ Connection failed: {e}")
                logger.error("   Check that the Home Assistant URL is correct and accessible")
                return False
            except asyncio.TimeoutError:
                logger.error("❌ Connection timeout")
                logger.error("   Home Assistant may be unreachable or very slow")
                return False
                
    except Exception as e:
        logger.error(f"❌ Connectivity test failed: {e}")
        return False
    
    # Step 3: Create and test HA client
    ha_client = None
    try:
        logger.info("Step 3: Creating Home Assistant client...")
        ha_client = HomeAssistantConversationClient(config.home_assistant)
        logger.info("✅ Client created successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to create HA client: {e}")
        return False
    
    # Step 4: Test authentication and API access
    try:
        logger.info("Step 4: Testing authentication and API access...")
        await ha_client.start()
        logger.info("✅ Successfully authenticated with Home Assistant")
        
    except aiohttp.ClientResponseError as e:
        if e.status == 401:
            logger.error("❌ Authentication failed (401 Unauthorized)")
            logger.error("   Check that your HA token is valid and has proper permissions")
        elif e.status == 403:
            logger.error("❌ Permission denied (403 Forbidden)")
            logger.error("   Your token may not have the required permissions")
        elif e.status == 404:
            logger.error("❌ API endpoint not found (404)")
            logger.error("   Check that your Home Assistant URL is correct")
        else:
            logger.error(f"❌ HTTP error {e.status}: {e.message}")
        return False
    except asyncio.TimeoutError:
        logger.error("❌ Authentication timeout")
        logger.error("   Home Assistant may be overloaded or slow to respond")
        return False
    except Exception as e:
        logger.error(f"❌ Authentication failed: {e}")
        logger.error(f"   Error type: {type(e).__name__}")
        return False
    
    # Step 5: Test basic API calls
    try:
        logger.info("Step 5: Testing basic API calls...")
        
        # Test API status
        api_status = await ha_client.rest_client.get_api_status()
        logger.info(f"✅ API Status: {api_status.get('message', 'OK')}")
        
        # Test configuration
        ha_config = await ha_client.rest_client.get_config()
        logger.info(f"✅ HA Version: {ha_config.get('version', 'Unknown')}")
        logger.info(f"✅ Location: {ha_config.get('location_name', 'Unknown')}")
        
    except Exception as e:
        logger.error(f"❌ API calls failed: {e}")
        if ha_client:
            await ha_client.stop()
        return False
    
    # Step 6: Test conversation API
    try:
        logger.info("Step 6: Testing Conversation API...")
        
        test_commands = [
            "what time is it",
            "hello"
        ]
        
        for command in test_commands:
            logger.info(f"Testing command: '{command}'")
            try:
                response = await ha_client.process_command(command)
                logger.info(f"  ✅ Response type: {response.response_type.value}")
                logger.info(f"  ✅ Speech: {response.speech_text[:100]}{'...' if len(response.speech_text) > 100 else ''}")
                if response.success_entities:
                    logger.info(f"  Success entities: {len(response.success_entities)}")
                if response.failed_entities:
                    logger.info(f"  Failed entities: {len(response.failed_entities)}")
                logger.info("")
            except Exception as e:
                logger.error(f"  ❌ Command failed: {e}")
        
        await ha_client.stop()
        logger.info("✅ Home Assistant test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Conversation API test failed: {e}")
        if ha_client:
            await ha_client.stop()
        return False
    
    return True


async def test_entities(config_path):
    """Test entity discovery and states"""
    logger = get_logger("ha_voice_assistant.HAEntities")
    
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
    print("DEBUG: main() function started", flush=True)
    parser = argparse.ArgumentParser(description="Test Home Assistant connection")
    parser.add_argument("--config", default="config/config.yaml", help="Configuration file path")
    parser.add_argument("--entities", action="store_true", help="Test entity discovery")
    parser.add_argument("--conversation", action="store_true", help="Test conversation API only")
    
    args = parser.parse_args()
    print("DEBUG: Arguments parsed", flush=True)
    
    # Setup logging
    try:
        setup_logging("INFO", console=True)
        # Ensure all loggers inherit from root logger
        logger = get_logger()  # Use default root logger
        print("DEBUG: Logging setup successful", flush=True)
    except Exception as e:
        print(f"DEBUG: Logging setup failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return
    
    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"❌ Configuration file not found: {args.config}")
        logger.error("Please create the configuration file:")
        logger.error(f"  cp {args.config}.example {args.config}")
        logger.error("Then edit it with your Home Assistant settings.")
        return
    
    async def run_tests():
        try:
            if args.entities:
                print("DEBUG: Running entities test", flush=True)
                await test_entities(args.config)
            elif args.conversation:
                print("DEBUG: Running conversation test", flush=True)
                await test_connection(args.config)
            else:
                # Run both tests
                print("DEBUG: Running connection test", flush=True)
                success1 = await test_connection(args.config)
                print(f"DEBUG: Connection test result: {success1}", flush=True)
                if success1:
                    print("DEBUG: Running entities test", flush=True)
                    await test_entities(args.config)
        except Exception as e:
            print(f"DEBUG: Exception in run_tests: {e}", flush=True)
            logger.error(f"❌ Test failed with unexpected error: {e}")
            logger.debug("Full traceback:", exc_info=True)
            import traceback
            traceback.print_exc()
    
    # Run tests
    try:
        print("DEBUG: About to run asyncio.run(run_tests())", flush=True)
        asyncio.run(run_tests())
        print("DEBUG: asyncio.run completed", flush=True)
    except Exception as e:
        print(f"DEBUG: asyncio.run failed: {e}", flush=True)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("DEBUG: __name__ == '__main__' - calling main()", flush=True)
    main()
    print("DEBUG: main() completed", flush=True)