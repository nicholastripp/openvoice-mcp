"""
Setup wizard routes for first-time configuration
"""
import logging
from pathlib import Path

import aiohttp_jinja2
from aiohttp import web

logger = logging.getLogger(__name__)


@aiohttp_jinja2.template('setup/welcome.html')
async def welcome_page(request: web.Request) -> dict:
    """Welcome page for setup wizard"""
    return {
        'title': 'Welcome to HA Voice Assistant',
        'version': '1.1.0'
    }


@aiohttp_jinja2.template('setup/wizard.html')
async def wizard_page(request: web.Request) -> dict:
    """Main setup wizard page"""
    return {
        'title': 'Setup Your Voice Assistant',
        'step': 'api_keys'
    }


async def save_setup(request: web.Request) -> web.Response:
    """Save setup configuration"""
    try:
        data = await request.post()
        config_manager = request.app['config_manager']
        
        # Create .env file with API keys
        env_data = {
            'OPENAI_API_KEY': data.get('openai_api_key', ''),
            'HA_URL': data.get('ha_url', ''),
            'HA_TOKEN': data.get('ha_token', ''),
            'PICOVOICE_ACCESS_KEY': data.get('picovoice_key', '')
        }
        
        # Validate required fields
        missing = [k for k, v in env_data.items() if not v]
        if missing:
            return web.json_response({
                'status': 'error',
                'message': f'Missing required fields: {", ".join(missing)}'
            }, status=400)
        
        # Save to .env file
        await config_manager.save_env(env_data)
        
        # Mark setup as complete
        request.app['first_run'] = False
        
        return web.json_response({
            'status': 'success',
            'message': 'Configuration saved successfully',
            'redirect': '/status'
        })
        
    except Exception as e:
        logger.error(f"Error saving setup: {e}")
        return web.json_response({
            'status': 'error',
            'message': str(e)
        }, status=500)


def setup_wizard_routes(app: web.Application):
    """Set up wizard routes"""
    app.router.add_get('/setup', welcome_page)
    app.router.add_get('/setup/wizard', wizard_page)
    app.router.add_post('/setup/save', save_setup)