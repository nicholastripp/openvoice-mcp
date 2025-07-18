"""CSRF protection for web UI"""

import secrets
import hmac
from typing import Optional, Dict
from aiohttp import web
import json
import logging

logger = logging.getLogger(__name__)


class CSRFProtection:
    """CSRF protection using double-submit cookie pattern"""
    
    def __init__(self, secret_key: bytes):
        self.secret_key = secret_key
        self.cookie_name = 'csrf_token'
        self.header_name = 'X-CSRF-Token'
        self.form_field_name = 'csrf_token'
        
    def generate_token(self) -> str:
        """Generate a cryptographically secure CSRF token"""
        return secrets.token_urlsafe(32)
    
    def set_csrf_cookie(self, response: web.Response, token: str) -> None:
        """Set CSRF token cookie"""
        response.set_cookie(
            self.cookie_name,
            token,
            httponly=True,
            secure=True,
            samesite='Strict',
            max_age=3600  # 1 hour
        )
    
    async def get_token_from_request(self, request: web.Request) -> Optional[str]:
        """Extract CSRF token from request header or form data"""
        # Check header first
        token = request.headers.get(self.header_name)
        if token:
            return token
            
        # Check form data
        if request.content_type == 'application/x-www-form-urlencoded':
            try:
                data = await request.post()
                return data.get(self.form_field_name)
            except Exception:
                pass
                
        # Check JSON body
        if request.content_type == 'application/json':
            try:
                data = await request.json()
                return data.get(self.form_field_name)
            except Exception:
                pass
                
        return None
    
    def validate_csrf(self, request: web.Request, token: str) -> bool:
        """Validate CSRF token against cookie"""
        cookie_token = request.cookies.get(self.cookie_name)
        if not cookie_token or not token:
            return False
        return secrets.compare_digest(cookie_token, token)
    
    @web.middleware
    async def csrf_middleware(self, request: web.Request, handler):
        """CSRF protection middleware"""
        # For GET requests, ensure CSRF cookie is set
        if request.method in ['GET', 'HEAD', 'OPTIONS']:
            response = await handler(request)
            
            # Set CSRF cookie if not present
            if self.cookie_name not in request.cookies:
                token = self.generate_token()
                self.set_csrf_cookie(response, token)
                
            return response
            
        # Skip for WebSocket endpoints
        if request.path.startswith('/ws/'):
            return await handler(request)
            
        # Skip for API endpoints that use other auth (if needed)
        if request.path.startswith('/api/') and 'Authorization' in request.headers:
            return await handler(request)
            
        # Validate CSRF token for state-changing methods
        token = await self.get_token_from_request(request)
        if not self.validate_csrf(request, token):
            logger.warning(f"CSRF validation failed for {request.path} from {request.remote}")
            return web.json_response(
                {'error': 'Invalid or missing CSRF token'},
                status=403
            )
            
        return await handler(request)
    
    def get_csrf_token(self, request: web.Request) -> str:
        """Get or generate CSRF token for request"""
        # Check if token already exists in cookie
        token = request.cookies.get(self.cookie_name)
        if not token:
            token = self.generate_token()
        return token