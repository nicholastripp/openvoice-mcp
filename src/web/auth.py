"""
Authentication middleware and utilities for the web UI
"""
import base64
import logging
import secrets
from typing import Optional, Callable

import bcrypt
from aiohttp import web

logger = logging.getLogger(__name__)


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt with salt.
    """
    # Generate salt and hash password
    salt = bcrypt.gensalt(rounds=12)  # 12 is current recommended default
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against a bcrypt hash"""
    try:
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    except Exception as e:
        logger.error(f"Password verification failed: {e}")
        return False


def extract_basic_auth(request: web.Request) -> Optional[tuple[str, str]]:
    """Extract username and password from Basic Auth header"""
    auth_header = request.headers.get('Authorization', '')
    
    if not auth_header.startswith('Basic '):
        return None
        
    try:
        # Decode base64
        encoded = auth_header[6:]  # Remove 'Basic '
        decoded = base64.b64decode(encoded).decode('utf-8')
        
        # Split username:password
        if ':' in decoded:
            username, password = decoded.split(':', 1)
            return username, password
            
    except Exception as e:
        logger.error(f"Failed to decode auth header: {e}")
        
    return None


def create_auth_middleware(auth_config: dict) -> Callable:
    """Create authentication middleware for aiohttp"""
    
    # Routes that don't require authentication
    public_routes = ['/login', '/static']
    
    @web.middleware
    async def auth_middleware(request: web.Request, handler: Callable) -> web.Response:
        """Check authentication for protected routes"""
        
        # Skip auth for public routes
        for public_route in public_routes:
            if request.path.startswith(public_route):
                return await handler(request)
        
        # Skip auth if disabled
        if not auth_config.get('enabled', True):
            return await handler(request)
        
        # Check for valid session
        session_token = request.cookies.get('session_token')
        if session_token and request.app.get('sessions', {}).get(session_token):
            # Valid session found
            return await handler(request)
        
        # Check Basic Auth
        auth_tuple = extract_basic_auth(request)
        if auth_tuple:
            username, password = auth_tuple
            
            # Debug logging (without exposing password)
            expected_username = auth_config.get('username')
            password_hash = auth_config.get('password_hash', '')
            
            logger.debug(f"Auth attempt for username: {username}")
            logger.debug(f"Expected username: {expected_username}")
            logger.debug(f"Password hash present: {bool(password_hash)}")
            
            if not password_hash:
                logger.error("No password hash configured! Run installer to set password.")
            
            # Verify credentials
            if (username == expected_username and 
                password_hash and
                verify_password(password, password_hash)):
                
                logger.info(f"Successful authentication for user: {username}")
                # Create session
                session_token = secrets.token_urlsafe(32)
                if 'sessions' not in request.app:
                    request.app['sessions'] = {}
                request.app['sessions'][session_token] = {
                    'username': username
                }
                
                # Process request and add session cookie
                response = await handler(request)
                response.set_cookie(
                    'session_token', 
                    session_token,
                    max_age=auth_config.get('session_timeout', 3600),
                    httponly=True,
                    secure=True,  # For HTTPS
                    samesite='Strict'
                )
                return response
            else:
                logger.warning(f"Authentication failed for username: {username}")
        
        # No valid auth - request credentials
        return web.Response(
            status=401,
            text='Authentication required',
            headers={
                'WWW-Authenticate': 'Basic realm="HA Voice Assistant"',
                'Content-Type': 'text/plain'
            }
        )
    
    return auth_middleware


def create_session_cleanup_task(app: web.Application, timeout: int):
    """Create a task to clean up expired sessions"""
    import asyncio
    import time
    
    async def cleanup_sessions():
        while True:
            await asyncio.sleep(300)  # Check every 5 minutes
            
            if 'sessions' not in app:
                continue
                
            current_time = time.time()
            expired = []
            
            # Find expired sessions
            for token, session in app['sessions'].items():
                if current_time - session.get('created', current_time) > timeout:
                    expired.append(token)
            
            # Remove expired sessions
            for token in expired:
                del app['sessions'][token]
                
            if expired:
                logger.info(f"Cleaned up {len(expired)} expired sessions")
    
    return cleanup_sessions