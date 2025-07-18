"""
Status dashboard routes
"""
import logging

import aiohttp_jinja2
from aiohttp import web

logger = logging.getLogger(__name__)


@aiohttp_jinja2.template('status/dashboard.html')
async def status_dashboard(request: web.Request) -> dict:
    """Main status dashboard"""
    return {
        'title': 'Voice Assistant Status',
        'ws_url': f"ws://{request.host}/ws/status"
    }


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    """WebSocket handler for real-time status updates"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    # Add to active connections
    if 'websockets' not in request.app:
        request.app['websockets'] = []
    request.app['websockets'].append(ws)
    
    try:
        # Send initial status
        await ws.send_json({
            'type': 'status',
            'state': 'idle',
            'connections': {
                'openai': True,
                'home_assistant': True,
                'wake_word': True
            }
        })
        
        # Keep connection alive
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                # Handle incoming messages if needed
                pass
            elif msg.type == web.WSMsgType.ERROR:
                logger.error(f'WebSocket error: {ws.exception()}')
                
    finally:
        # Remove from active connections
        request.app['websockets'].remove(ws)
        
    return ws


@aiohttp_jinja2.template('status/logs.html')
async def logs_viewer(request: web.Request) -> dict:
    """Log viewer page"""
    config_manager = request.app['config_manager']
    
    # Read last 100 lines of log
    logs = await config_manager.read_logs(lines=100)
    
    return {
        'title': 'System Logs',
        'logs': logs
    }


def status_routes(app: web.Application):
    """Set up status routes"""
    app.router.add_get('/status', status_dashboard)
    app.router.add_get('/status/logs', logs_viewer)
    app.router.add_get('/ws/status', websocket_handler)