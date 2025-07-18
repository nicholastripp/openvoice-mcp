"""
Persona editor routes
"""
import logging

import aiohttp_jinja2
from aiohttp import web

logger = logging.getLogger(__name__)


@aiohttp_jinja2.template('persona/editor.html')
async def persona_editor(request: web.Request) -> dict:
    """Persona configuration editor"""
    config_manager = request.app['config_manager']
    
    # Load current persona.ini
    persona = await config_manager.load_persona_config()
    
    # Define personality traits with descriptions
    traits = {
        'helpfulness': 'How helpful and accommodating',
        'humor': 'Level of humor in responses',
        'formality': 'Formal vs casual speech',
        'patience': 'Patience with users',
        'verbosity': 'Amount of detail in responses',
        'warmth': 'Warm and friendly vs neutral',
        'curiosity': 'How curious and inquisitive',
        'confidence': 'Confidence in responses',
        'optimism': 'Optimistic vs pessimistic',
        'respectfulness': 'Respectful and polite'
    }
    
    return {
        'title': 'Personality Editor',
        'persona': persona,
        'traits': traits
    }


@aiohttp_jinja2.template('persona/preview.html')
async def persona_preview(request: web.Request) -> dict:
    """Preview persona with sample responses"""
    return {
        'title': 'Personality Preview'
    }


async def save_persona(request: web.Request) -> web.Response:
    """Save persona configuration"""
    try:
        data = await request.json()
        config_manager = request.app['config_manager']
        
        # Save persona config
        await config_manager.save_persona_config(data)
        
        return web.json_response({
            'status': 'success',
            'message': 'Personality saved'
        })
        
    except Exception as e:
        logger.error(f"Error saving persona: {e}")
        return web.json_response({
            'status': 'error',
            'message': str(e)
        }, status=500)


async def generate_preview(request: web.Request) -> web.Response:
    """Generate preview responses based on personality"""
    try:
        data = await request.json()
        
        # Sample responses based on personality traits
        samples = []
        
        # Generate different response styles
        if data.get('humor', 0) > 70:
            samples.append("Turning on the lights... Let there be light! And there was, thanks to electricity.")
        else:
            samples.append("I've turned on the lights for you.")
            
        if data.get('formality', 0) < 30:
            samples.append("Hey! The temperature's now 72 degrees. Pretty comfy!")
        else:
            samples.append("The current temperature has been set to 72 degrees Fahrenheit.")
            
        return web.json_response({
            'status': 'success',
            'samples': samples
        })
        
    except Exception as e:
        logger.error(f"Error generating preview: {e}")
        return web.json_response({
            'status': 'error',
            'message': str(e)
        }, status=500)


def persona_routes(app: web.Application):
    """Set up persona routes"""
    app.router.add_get('/persona', persona_editor)
    app.router.add_get('/persona/preview', persona_preview)
    app.router.add_post('/api/persona/save', save_persona)
    app.router.add_post('/api/persona/preview', generate_preview)