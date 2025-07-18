"""Rate limiting for web UI"""

from collections import defaultdict
import time
import asyncio
from typing import Dict, List, Optional
from aiohttp import web
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """Sliding window rate limiter"""
    
    def __init__(self, max_requests: int = 5, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self._cleanup_task = None
        self._lock = asyncio.Lock()
        
    async def start_cleanup(self):
        """Start periodic cleanup of old requests"""
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
    async def stop_cleanup(self):
        """Stop cleanup task"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            
    async def _cleanup_loop(self):
        """Periodically clean up old request records"""
        while True:
            try:
                await asyncio.sleep(60)  # Cleanup every minute
                async with self._lock:
                    self._cleanup_old_requests()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in rate limiter cleanup: {e}")
            
    def _cleanup_old_requests(self):
        """Remove expired request records"""
        now = time.time()
        for identifier in list(self.requests.keys()):
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if now - req_time < self.window_seconds
            ]
            if not self.requests[identifier]:
                del self.requests[identifier]
    
    async def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed"""
        async with self._lock:
            now = time.time()
            
            # Clean old requests for this identifier
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if now - req_time < self.window_seconds
            ]
            
            # Check limit
            if len(self.requests[identifier]) >= self.max_requests:
                return False
                
            # Record request
            self.requests[identifier].append(now)
            return True
    
    def get_retry_after(self, identifier: str) -> int:
        """Get seconds until next request is allowed"""
        if not self.requests[identifier]:
            return 0
            
        oldest_request = min(self.requests[identifier])
        retry_after = int(self.window_seconds - (time.time() - oldest_request))
        return max(0, retry_after)


class RateLimitMiddleware:
    """Rate limiting middleware factory"""
    
    def __init__(self):
        # Different limits for different endpoints
        self.limiters = {
            'auth': RateLimiter(max_requests=5, window_seconds=60),      # 5 per minute
            'api': RateLimiter(max_requests=100, window_seconds=60),     # 100 per minute
            'config': RateLimiter(max_requests=20, window_seconds=60),   # 20 per minute
            'default': RateLimiter(max_requests=60, window_seconds=60)   # 60 per minute
        }
        
    async def start_cleanup(self):
        """Start cleanup tasks for all limiters"""
        for limiter in self.limiters.values():
            await limiter.start_cleanup()
            
    async def stop_cleanup(self):
        """Stop cleanup tasks for all limiters"""
        for limiter in self.limiters.values():
            await limiter.stop_cleanup()
        
    def get_limiter_for_path(self, path: str) -> RateLimiter:
        """Get appropriate rate limiter for path"""
        if path.startswith('/auth/') or path == '/login':
            return self.limiters['auth']
        elif path.startswith('/api/'):
            return self.limiters['api']
        elif path.startswith('/config/'):
            return self.limiters['config']
        return self.limiters['default']
    
    def get_identifier(self, request: web.Request) -> str:
        """Get identifier for rate limiting (IP address)"""
        # Try to get real IP from proxy headers
        forwarded = request.headers.get('X-Forwarded-For')
        if forwarded:
            # Take the first IP in the chain
            return forwarded.split(',')[0].strip()
            
        # Try X-Real-IP header
        real_ip = request.headers.get('X-Real-IP')
        if real_ip:
            return real_ip
            
        # Fall back to remote address
        if request.remote:
            return request.remote
            
        return 'unknown'
    
    @web.middleware
    async def rate_limit_middleware(self, request: web.Request, handler):
        """Rate limiting middleware"""
        # Skip rate limiting for static files
        if request.path.startswith('/static/'):
            return await handler(request)
            
        limiter = self.get_limiter_for_path(request.path)
        identifier = self.get_identifier(request)
        
        if not await limiter.is_allowed(identifier):
            retry_after = limiter.get_retry_after(identifier)
            logger.warning(
                f"Rate limit exceeded for {identifier} on {request.path} "
                f"(limit: {limiter.max_requests}/{limiter.window_seconds}s)"
            )
            
            return web.json_response(
                {
                    'error': 'Rate limit exceeded',
                    'retry_after': retry_after,
                    'message': f'Too many requests. Please try again in {retry_after} seconds.'
                },
                status=429,
                headers={'Retry-After': str(retry_after)}
            )
            
        return await handler(request)