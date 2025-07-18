#!/usr/bin/env python3
"""
Test device extraction from GetLiveContext responses

This script tests the device extraction functionality
to ensure we can properly parse GetLiveContext responses.
"""

import asyncio
import sys
import json
from pathlib import Path

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.main import VoiceAssistant
from src.personality import PersonalityProfile


async def test_device_extraction():
    """Test device extraction from GetLiveContext"""
    print("=" * 60)
    print("Testing Device Extraction from GetLiveContext")
    print("=" * 60)
    
    # Load configuration
    print("\n1. Loading configuration...")
    config = load_config("config/config.yaml")
    print("[OK] Configuration loaded")
    
    # Load personality
    print("\n2. Loading personality...")
    personality = PersonalityProfile("config/persona.ini")
    print("[OK] Personality loaded")
    
    # Create voice assistant instance
    print("\n3. Creating voice assistant instance...")
    assistant = VoiceAssistant(config, personality, skip_ha_check=True)
    print("[OK] Voice assistant created")
    
    # Test GetLiveContext response parsing
    print("\n4. Testing GetLiveContext response parser...")
    
    # Sample GetLiveContext response from the actual test
    test_response = """{"success": true, "result": "Live Context: An overview of the areas and the devices in this smart home:\\n- names: AJ\\n  domain: person\\n  state: not_home\\n- names: Attic\\n  domain: light\\n  state: 'on'\\n  areas: Attic\\n- names: Back Yard Lights\\n  domain: switch\\n  state: 'on'\\n  areas: Outside\\n- names: Living Room\\n  domain: sensor\\n  state: '73.04'\\n  areas: Living Room\\n  attributes:\\n    unit_of_measurement: '\u00b0F'\\n    device_class: temperature"}"""
    
    # Test the extraction
    try:
        devices = assistant._extract_devices_from_getlivecontext(test_response)
        
        print(f"\n[OK] Extracted {len(devices)} devices:")
        print("-" * 40)
        
        for i, device in enumerate(devices, 1):
            print(f"\n{i}. {device['entity_id']}")
            print(f"   State: {device['state']}")
            print(f"   Name: {device['attributes'].get('friendly_name', 'Unknown')}")
            if 'area' in device['attributes']:
                print(f"   Area: {device['attributes']['area']}")
            # Show other attributes
            other_attrs = {k: v for k, v in device['attributes'].items() 
                          if k not in ['friendly_name', 'area']}
            if other_attrs:
                print(f"   Attributes: {other_attrs}")
        
        print("\n" + "-" * 40)
        print(f"Successfully parsed {len(devices)} devices from GetLiveContext")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to parse response: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test actual MCP integration
    print("\n5. Testing live MCP device fetching...")
    
    try:
        # Connect to MCP
        await assistant.mcp_client.connect()
        print("[OK] Connected to MCP server")
        
        # Fetch devices
        print("\n6. Fetching devices via MCP...")
        await assistant._fetch_device_states()
        
        # Check device cache
        if assistant._device_cache:
            print(f"\n[OK] Device cache populated with {len(assistant._device_cache)} devices:")
            print("-" * 40)
            
            # Group by domain
            domains = {}
            for entity_id, state_info in assistant._device_cache.items():
                domain = entity_id.split('.')[0] if '.' in entity_id else 'unknown'
                if domain not in domains:
                    domains[domain] = []
                domains[domain].append((entity_id, state_info))
            
            # Show summary by domain
            for domain, entities in sorted(domains.items()):
                print(f"\n{domain}: {len(entities)} entities")
                # Show first 3 entities from each domain
                for entity_id, state_info in entities[:3]:
                    state = state_info.get('state', 'unknown')
                    name = state_info.get('attributes', {}).get('friendly_name', entity_id)
                    print(f"  - {entity_id}: {state} ({name})")
                if len(entities) > 3:
                    print(f"  ... and {len(entities) - 3} more")
            
            print("\n" + "-" * 40)
            print("Device extraction and caching successful!")
            
        else:
            print("\n[WARNING] Device cache is empty")
            print("This might indicate an issue with device fetching")
        
        # Disconnect
        await assistant.mcp_client.disconnect()
        print("\n[OK] Disconnected from MCP server")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] MCP test failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to disconnect
        try:
            await assistant.mcp_client.disconnect()
        except:
            pass
        
        return False


async def main():
    """Main entry point"""
    try:
        success = await test_device_extraction()
        if success:
            print("\n" + "=" * 60)
            print("Device Extraction Test PASSED")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("Device Extraction Test FAILED")
            print("=" * 60)
            
    except KeyboardInterrupt:
        print("\n[WARNING] Test interrupted")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\nTesting device extraction from GetLiveContext responses...\n")
    asyncio.run(main())