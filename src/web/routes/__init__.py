"""
Route handlers for the web UI
"""
from aiohttp import web

from .setup import setup_wizard_routes
from .config import config_routes
from .persona import persona_routes
from .status import status_routes
from .api import api_routes
from .auth import auth_routes


async def index_handler(request: web.Request) -> web.Response:
    """Main index page handler"""
    # If first run, redirect to setup wizard
    if request.app.get('first_run', False):
        raise web.HTTPFound('/setup')
    
    # Otherwise, redirect to status dashboard
    raise web.HTTPFound('/status')


def setup_routes(app: web.Application):
    """Set up all routes for the application"""
    # Index route
    app.router.add_get('/', index_handler)
    
    # Add route groups
    auth_routes(app)  # Auth routes should be first
    setup_wizard_routes(app)
    config_routes(app)
    persona_routes(app)
    status_routes(app)
    api_routes(app)