"""
Configuration editor routes
"""
import logging

import aiohttp_jinja2
from aiohttp import web

logger = logging.getLogger(__name__)


@aiohttp_jinja2.template('config/env.html')
async def env_editor(request: web.Request) -> dict:
    """Environment variables editor"""
    config_manager = request.app['config_manager']
    
    # Load current .env (without exposing secrets)
    env_vars = await config_manager.load_env_masked()
    
    return {
        'title': 'Environment Variables',
        'env_vars': env_vars
    }


@aiohttp_jinja2.template('config/yaml.html')
async def yaml_editor(request: web.Request) -> dict:
    """YAML configuration editor"""
    config_manager = request.app['config_manager']
    
    # Load current config.yaml
    config = await config_manager.load_yaml_config()
    
    # Scan for custom wake word models
    import os
    from pathlib import Path
    
    wake_words_dir = Path('config/wake_words')
    custom_wake_words = []
    
    if wake_words_dir.exists():
        for file in wake_words_dir.glob('*.ppn'):
            # Get filename without extension as the wake word name
            custom_wake_words.append(file.stem)
    
    return {
        'title': 'Configuration Settings',
        'config': config,
        'custom_wake_words': custom_wake_words
    }


@aiohttp_jinja2.template('config/audio_test.html')
async def audio_test(request: web.Request) -> dict:
    """Audio device testing page"""
    return {
        'title': 'Audio Device Testing'
    }


async def save_env(request: web.Request) -> web.Response:
    """Save environment variables"""
    try:
        data = await request.json()
        config_manager = request.app['config_manager']
        
        # Only update provided values
        await config_manager.update_env(data)
        
        return web.json_response({
            'status': 'success',
            'message': 'Environment variables updated'
        })
        
    except Exception as e:
        logger.error(f"Error saving env: {e}")
        return web.json_response({
            'status': 'error',
            'message': str(e)
        }, status=500)


async def save_yaml(request: web.Request) -> web.Response:
    """Save YAML configuration"""
    try:
        data = await request.json()
        config_manager = request.app['config_manager']
        
        # Validate and save config
        await config_manager.save_yaml_config(data)
        
        return web.json_response({
            'status': 'success',
            'message': 'Configuration saved',
            'restart_required': data.get('restart_required', True)
        })
        
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return web.json_response({
            'status': 'error',
            'message': str(e)
        }, status=500)


def config_routes(app: web.Application):
    """Set up configuration routes"""
    app.router.add_get('/config/env', env_editor)
    app.router.add_get('/config/yaml', yaml_editor)
    app.router.add_get('/config/audio', audio_test)
    app.router.add_post('/api/config/env', save_env)
    app.router.add_post('/api/config/yaml', save_yaml)