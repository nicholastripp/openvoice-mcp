"""Security headers middleware for web UI"""

from aiohttp import web
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SecurityHeaders:
    """Security headers configuration and middleware"""
    
    def __init__(self, config: Optional[Dict[str, str]] = None):
        self.config = config or self.get_default_config()
        
    def get_default_config(self) -> Dict[str, str]:
        """Get default security headers"""
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains; preload',
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()',
            'Content-Security-Policy': self.get_default_csp()
        }
    
    def get_default_csp(self) -> str:
        """Get default Content Security Policy"""
        return (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "  # Required for some UI features
            "style-src 'self' 'unsafe-inline'; "  # Required for inline styles
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self' wss: ws:; "  # For WebSocket connections
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'; "
            "upgrade-insecure-requests"
        )
    
    def update_config(self, updates: Dict[str, str]):
        """Update security headers configuration"""
        self.config.update(updates)
        logger.info(f"Updated security headers configuration")
    
    @web.middleware
    async def security_headers_middleware(self, request: web.Request, handler):
        """Add security headers to all responses"""
        
        # Add a response preparation callback to remove Server header
        async def prepare_response(request, response):
            # Remove Server header if present
            response.headers.pop('Server', None)
        
        request.on_response_prepare.append(prepare_response)
        
        try:
            response = await handler(request)
        except web.HTTPException as ex:
            # Handle HTTP exceptions (redirects, errors, etc.)
            response = ex
            
        # Add security headers to all responses
        for header, value in self.config.items():
            if header not in response.headers:
                response.headers[header] = value
                
        # Remove sensitive headers that might leak information
        headers_to_remove = ['Server', 'X-Powered-By', 'X-AspNet-Version']
        for header in headers_to_remove:
            response.headers.pop(header, None)
            
        # Add cache control for sensitive pages
        if request.path.startswith(('/config', '/auth', '/api')):
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, private'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            
        return response
    
    def get_nonce(self) -> str:
        """Generate a nonce for inline scripts if needed"""
        import secrets
        return secrets.token_urlsafe(16)
    
    def get_csp_with_nonce(self, nonce: str) -> str:
        """Get CSP with nonce for inline scripts"""
        base_csp = self.config.get('Content-Security-Policy', self.get_default_csp())
        # Replace unsafe-inline with nonce
        return base_csp.replace(
            "script-src 'self' 'unsafe-inline'",
            f"script-src 'self' 'nonce-{nonce}'"
        )