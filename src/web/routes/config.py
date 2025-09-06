"""
Configuration editor routes
"""
import logging
import sys
import os

import aiohttp_jinja2
from aiohttp import web

# Add parent directory to path to import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from openai_client.model_compatibility import ModelType, ModelCompatibility
from openai_client.voice_manager import VoiceManager

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
    
    # Get available models and voices from backend
    # Create a minimal config object for compatibility layer
    class MinimalConfig:
        def __init__(self):
            self.model = config.get('openai', {}).get('model', 'gpt-realtime')
            self.voice = config.get('openai', {}).get('voice', 'alloy')
            self.temperature = config.get('openai', {}).get('temperature', 0.8)
    
    minimal_config = MinimalConfig()
    
    # Get all available models
    available_models = [
        {
            'value': model_type.value,
            'name': model_type.value.replace('-', ' ').title().replace('Gpt', 'GPT')
        }
        for model_type in ModelType
    ]
    
    # Get all available voices from VoiceManager
    voice_manager = VoiceManager(minimal_config)
    available_voices = sorted(list(voice_manager.VOICE_PROFILES.keys()))
    
    # Scan for custom wake word models
    from pathlib import Path
    
    wake_words_dir = Path('config/wake_words')
    custom_wake_words = []
    
    if wake_words_dir.exists():
        for file in wake_words_dir.glob('*.ppn'):
            # Get full filename with extension for custom wake words
            custom_wake_words.append(file.name)
    
    return {
        'title': 'Configuration Settings',
        'config': config,
        'custom_wake_words': custom_wake_words,
        'available_models': available_models,
        'available_voices': available_voices
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