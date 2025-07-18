"""
Main web application for configuration and monitoring
"""
import logging
import os
from pathlib import Path
from typing import Optional

import aiohttp_jinja2
import jinja2
from aiohttp import web

from .routes import setup_routes
from .utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class WebApp:
    """Web application for configuration and monitoring"""
    
    def __init__(self, config_dir: Path, host: str = "127.0.0.1", port: int = 8080):
        self.config_dir = config_dir
        self.host = host
        self.port = port
        self.app: Optional[web.Application] = None
        self.runner: Optional[web.AppRunner] = None
        self.config_manager = ConfigManager(config_dir)
        self.start_time = None  # Will be set when server starts
        
    async def setup(self) -> web.Application:
        """Set up the web application"""
        self.app = web.Application()
        
        # Set up Jinja2 templates
        template_dir = Path(__file__).parent / "templates"
        aiohttp_jinja2.setup(
            self.app,
            loader=jinja2.FileSystemLoader(str(template_dir)),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        # Set up static file serving
        static_dir = Path(__file__).parent / "static"
        self.app.router.add_static('/static', static_dir, name='static')
        
        # Store config manager in app
        self.app['config_manager'] = self.config_manager
        
        # Set up routes
        setup_routes(self.app)
        
        # Check if first run (no .env file)
        env_path = self.config_dir.parent / ".env"
        self.app['first_run'] = not env_path.exists()
        
        return self.app
        
    async def start(self):
        """Start the web server"""
        if not self.app:
            await self.setup()
            
        # Record start time
        import time
        self.start_time = time.time()
        self.app['start_time'] = self.start_time
            
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        
        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()
        
        logger.info(f"Web UI started at http://{self.host}:{self.port}")
        if self.app['first_run']:
            logger.info("First run detected - setup wizard available")
            
    async def stop(self):
        """Stop the web server"""
        if self.runner:
            await self.runner.cleanup()
            logger.info("Web UI stopped")