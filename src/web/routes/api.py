"""
API routes for testing and validation
"""
import logging
import asyncio
from typing import Dict

from aiohttp import web
import sounddevice as sd

logger = logging.getLogger(__name__)


async def test_openai(request: web.Request) -> web.Response:
    """Test OpenAI connection"""
    try:
        data = await request.json()
        api_key = data.get('api_key')
        
        if not api_key:
            return web.json_response({
                'status': 'error',
                'message': 'API key required'
            }, status=400)
        
        # Simple validation for now
        if api_key.startswith('sk-') and len(api_key) > 20:
            return web.json_response({
                'status': 'success',
                'message': 'API key format appears valid'
            })
        else:
            return web.json_response({
                'status': 'error',
                'message': 'Invalid API key format'
            })
            
    except Exception as e:
        logger.error(f"Error testing OpenAI: {e}")
        return web.json_response({
            'status': 'error',
            'message': str(e)
        }, status=500)


async def test_home_assistant(request: web.Request) -> web.Response:
    """Test Home Assistant connection"""
    try:
        data = await request.json()
        url = data.get('url')
        token = data.get('token')
        
        if not url or not token:
            return web.json_response({
                'status': 'error',
                'message': 'URL and token required'
            }, status=400)
        
        # Import here to avoid circular dependencies
        import aiohttp
        
        # Test connection
        async with aiohttp.ClientSession() as session:
            headers = {'Authorization': f'Bearer {token}'}
            async with session.get(f"{url}/api/", headers=headers) as resp:
                if resp.status == 200:
                    api_data = await resp.json()
                    return web.json_response({
                        'status': 'success',
                        'message': f"Connected to Home Assistant {api_data.get('version', 'unknown')}"
                    })
                else:
                    return web.json_response({
                        'status': 'error',
                        'message': f'Connection failed: {resp.status}'
                    })
                    
    except Exception as e:
        logger.error(f"Error testing HA: {e}")
        return web.json_response({
            'status': 'error',
            'message': str(e)
        }, status=500)


async def test_picovoice(request: web.Request) -> web.Response:
    """Test Picovoice access key"""
    try:
        data = await request.json()
        access_key = data.get('access_key')
        
        if not access_key:
            return web.json_response({
                'status': 'error',
                'message': 'Access key required'
            }, status=400)
        
        # Simple validation
        if len(access_key) > 20:
            return web.json_response({
                'status': 'success',
                'message': 'Access key format appears valid'
            })
        else:
            return web.json_response({
                'status': 'error',
                'message': 'Invalid access key format'
            })
            
    except Exception as e:
        logger.error(f"Error testing Picovoice: {e}")
        return web.json_response({
            'status': 'error',
            'message': str(e)
        }, status=500)


async def list_audio_devices(request: web.Request) -> web.Response:
    """List available audio devices"""
    try:
        devices = sd.query_devices()
        
        input_devices = []
        output_devices = []
        
        for i, device in enumerate(devices):
            device_info = {
                'id': i,
                'name': device['name'],
                'channels': device['max_input_channels'] or device['max_output_channels']
            }
            
            if device['max_input_channels'] > 0:
                input_devices.append(device_info)
            if device['max_output_channels'] > 0:
                output_devices.append(device_info)
        
        return web.json_response({
            'status': 'success',
            'input_devices': input_devices,
            'output_devices': output_devices
        })
        
    except Exception as e:
        logger.error(f"Error listing devices: {e}")
        return web.json_response({
            'status': 'error',
            'message': str(e)
        }, status=500)


async def test_audio_device(request: web.Request) -> web.Response:
    """Test specific audio device"""
    try:
        data = await request.json()
        device_id = data.get('device_id')
        device_type = data.get('type', 'input')
        
        # Simple test - just verify device exists
        devices = sd.query_devices()
        if 0 <= device_id < len(devices):
            device = devices[device_id]
            
            if device_type == 'input' and device['max_input_channels'] > 0:
                return web.json_response({
                    'status': 'success',
                    'message': f"Input device '{device['name']}' is available"
                })
            elif device_type == 'output' and device['max_output_channels'] > 0:
                return web.json_response({
                    'status': 'success',
                    'message': f"Output device '{device['name']}' is available"
                })
            else:
                return web.json_response({
                    'status': 'error',
                    'message': f"Device does not support {device_type}"
                })
        else:
            return web.json_response({
                'status': 'error',
                'message': 'Invalid device ID'
            })
            
    except Exception as e:
        logger.error(f"Error testing device: {e}")
        return web.json_response({
            'status': 'error',
            'message': str(e)
        }, status=500)


def api_routes(app: web.Application):
    """Set up API routes"""
    app.router.add_post('/api/test/openai', test_openai)
    app.router.add_post('/api/test/home_assistant', test_home_assistant)
    app.router.add_post('/api/test/picovoice', test_picovoice)
    app.router.add_get('/api/audio/devices', list_audio_devices)
    app.router.add_post('/api/audio/test', test_audio_device)