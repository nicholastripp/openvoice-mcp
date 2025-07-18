"""
Authentication routes for web UI
"""
import time
import secrets
import logging
from aiohttp import web
from ..auth import verify_password

logger = logging.getLogger(__name__)


async def login(request: web.Request) -> web.Response:
    """Handle login attempts"""
    # Rate limiting is applied by middleware before this handler
    
    try:
        data = await request.json()
        username = data.get('username', '')
        password = data.get('password', '')
    except:
        return web.json_response({'error': 'Invalid request'}, status=400)
    
    # Get auth config
    auth_config = request.app.get('auth_config', {})
    expected_username = auth_config.get('username', 'admin')
    expected_password_hash = auth_config.get('password_hash', '')
    
    logger.debug(f"Login attempt for user: {username}")
    
    # Verify credentials
    if username == expected_username and verify_password(password, expected_password_hash):
        # Create session
        session_token = secrets.token_urlsafe(32)
        if 'sessions' not in request.app:
            request.app['sessions'] = {}
        request.app['sessions'][session_token] = {
            'username': username,
            'created': time.time()
        }
        
        logger.info(f"Successful login for user: {username}")
        response = web.json_response({'status': 'success'})
        response.set_cookie(
            'session_token',
            session_token,
            max_age=auth_config.get('session_timeout', 3600),
            httponly=True,
            secure=True,
            samesite='Strict'
        )
        return response
    
    # Invalid credentials
    logger.warning(f"Failed login attempt for user: {username}")
    return web.json_response({'error': 'Invalid credentials'}, status=401)


async def logout(request: web.Request) -> web.Response:
    """Handle logout"""
    session_token = request.cookies.get('session_token')
    
    if session_token and 'sessions' in request.app:
        # Remove session
        request.app['sessions'].pop(session_token, None)
        logger.info("User logged out")
    
    response = web.json_response({'status': 'success'})
    response.del_cookie('session_token')
    return response


def auth_routes(app: web.Application):
    """Set up authentication routes"""
    app.router.add_post('/login', login)
    app.router.add_post('/logout', logout)