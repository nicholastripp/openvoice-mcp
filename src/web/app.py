"""
Main web application for configuration and monitoring
"""
import asyncio
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any

import aiohttp_jinja2
import jinja2
from aiohttp import web

from .routes import setup_routes
from .utils.config_manager import ConfigManager
from .auth import create_auth_middleware, create_session_cleanup_task
from .certs import create_self_signed_cert, create_ssl_context
from .utils.csrf import CSRFProtection
from .utils.rate_limit import RateLimitMiddleware
from .utils.security_headers import SecurityHeaders

logger = logging.getLogger(__name__)


class WebApp:
    """Web application for configuration and monitoring"""
    
    def __init__(self, config_dir: Path, host: str = "127.0.0.1", port: int = 8080, 
                 auth_config: Optional[dict] = None, tls_config: Optional[dict] = None):
        self.config_dir = config_dir
        self.host = host
        self.port = port
        self.auth_config = auth_config or {}
        self.tls_config = tls_config or {}
        self.app: Optional[web.Application] = None
        self.runner: Optional[web.AppRunner] = None
        self.config_manager = ConfigManager(config_dir)
        self.start_time = None  # Will be set when server starts
        self.ssl_context = None  # Will be set if TLS is enabled
        self.assistant = None  # Will be set by voice assistant
        
    async def setup(self) -> web.Application:
        """Set up the web application"""
        middlewares = []
        
        # Initialize security components
        csrf_protection = CSRFProtection(secret_key=os.urandom(32))
        rate_limiter = RateLimitMiddleware()
        security_headers = SecurityHeaders()
        
        # Add security middleware in correct order
        # 1. Security headers (should be first to apply to all responses)
        middlewares.append(security_headers.security_headers_middleware)
        
        # 2. Rate limiting (before auth to prevent brute force)
        middlewares.append(rate_limiter.rate_limit_middleware)
        
        # 3. CSRF protection (after rate limiting)
        middlewares.append(csrf_protection.csrf_middleware)
        
        # 4. Authentication middleware if enabled
        if self.auth_config.get('enabled', True):
            auth_middleware = create_auth_middleware(self.auth_config)
            middlewares.append(auth_middleware)
            
        # Create app with client max size limit (10MB)
        self.app = web.Application(
            middlewares=middlewares,
            client_max_size=10 * 1024 * 1024  # 10MB limit
        )
        
        # Store auth config in app
        self.app['auth_config'] = self.auth_config
        
        # Store security components in app
        self.app['csrf'] = csrf_protection
        self.app['rate_limiter'] = rate_limiter
        self.app['security_headers'] = security_headers
        
        # Set up Jinja2 templates
        template_dir = Path(__file__).parent / "templates"
        
        # Context processor to inject CSRF token
        async def context_processor(request):
            csrf_token = ""
            if 'csrf' in request.app:
                csrf_token = request.app['csrf'].get_csrf_token(request)
            return {'csrf_token': csrf_token}
        
        aiohttp_jinja2.setup(
            self.app,
            loader=jinja2.FileSystemLoader(str(template_dir)),
            autoescape=jinja2.select_autoescape(['html', 'xml']),
            context_processors=[context_processor]
        )
        
        # Set up static file serving
        static_dir = Path(__file__).parent / "static"
        self.app.router.add_static('/static', static_dir, name='static')
        
        # Store config manager in app
        self.app['config_manager'] = self.config_manager
        
        # Initialize websocket list
        self.app['websockets'] = []
        
        # Set up routes
        setup_routes(self.app)
        
        # Check if first run (no .env file)
        env_path = self.config_dir.parent / ".env"
        self.app['first_run'] = not env_path.exists()
        
        # Set up TLS if enabled
        if self.tls_config.get('enabled', True):
            await self._setup_tls()
        
        # Start session cleanup task if auth is enabled
        if self.auth_config.get('enabled', True):
            cleanup_task = create_session_cleanup_task(
                self.app, 
                self.auth_config.get('session_timeout', 3600)
            )
            asyncio.create_task(cleanup_task())
        
        # Start rate limiter cleanup
        await rate_limiter.start_cleanup()
        
        # Set up shutdown handler
        async def on_shutdown(app):
            # Close any remaining WebSockets
            if 'websockets' in app:
                for ws in list(app['websockets']):
                    try:
                        await ws.close()
                    except Exception:
                        pass
            await rate_limiter.stop_cleanup()
            
        self.app.on_shutdown.append(on_shutdown)
        
        return self.app
        
    async def _setup_tls(self):
        """Set up TLS/HTTPS"""
        cert_file = self.tls_config.get('cert_file', '')
        key_file = self.tls_config.get('key_file', '')
        
        # Use self-signed cert if no cert specified
        if not cert_file or not key_file:
            cert_dir = self.config_dir / "certs"
            cert_file, key_file = create_self_signed_cert(cert_dir, self.host)
            
        cert_path = Path(cert_file)
        key_path = Path(key_file)
        
        if not cert_path.exists() or not key_path.exists():
            raise FileNotFoundError(f"Certificate files not found: {cert_file}, {key_file}")
            
        # Create SSL context
        self.ssl_context = create_ssl_context(cert_path, key_path)
        
    async def start(self):
        """Start the web server"""
        if not self.app:
            await self.setup()
            
        # Record start time
        import time
        self.start_time = time.time()
        self.app['start_time'] = self.start_time
            
        # Create runner
        self.runner = web.AppRunner(
            self.app,
            handle_signals=False,
            access_log=None  # We handle our own logging
        )
        await self.runner.setup()
        
        # Create site - server header will be handled by response middleware
        site = web.TCPSite(
            self.runner, 
            self.host, 
            self.port, 
            ssl_context=self.ssl_context
        )
            
        await site.start()
        
        protocol = "https" if self.ssl_context else "http"
        logger.info(f"Web UI started at {protocol}://{self.host}:{self.port}")
        if self.app['first_run']:
            logger.info("First run detected - setup wizard available")
            
    async def stop(self):
        """Stop the web server"""
        # Close all active WebSocket connections first
        if self.app and 'websockets' in self.app:
            logger.info(f"Closing {len(self.app['websockets'])} active WebSocket connections...")
            for ws in list(self.app['websockets']):  # Use list() to avoid modification during iteration
                try:
                    await ws.close()
                except Exception as e:
                    logger.debug(f"Error closing WebSocket: {e}")
            self.app['websockets'].clear()
        
        # Then cleanup the runner
        if self.runner:
            await self.runner.cleanup()
            logger.info("Web UI stopped")
            
    async def broadcast_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Broadcast an event to all connected WebSocket clients"""
        if not self.app or 'websockets' not in self.app:
            return
            
        # Prepare the message
        message = {
            'type': event_type,
            **data
        }
        
        # Send to all connected clients
        disconnected = []
        for ws in self.app['websockets']:
            try:
                await ws.send_json(message)
            except ConnectionResetError:
                disconnected.append(ws)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected.append(ws)
                
        # Clean up disconnected clients
        for ws in disconnected:
            if ws in self.app['websockets']:
                self.app['websockets'].remove(ws)
                
    def set_assistant(self, assistant) -> None:
        """Set the voice assistant reference for status queries"""
        self.assistant = assistant
        if self.app:
            self.app['assistant'] = assistant