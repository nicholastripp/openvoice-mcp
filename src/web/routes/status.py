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
    # Get app start time
    start_time = request.app.get('start_time', 0)
    
    return {
        'title': 'Voice Assistant Status',
        'ws_url': f"{'wss' if request.scheme == 'https' else 'ws'}://{request.host}/ws/status",
        'start_time': int(start_time * 1000)  # Convert to milliseconds for JavaScript
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
        # Get real status from assistant if available
        assistant = request.app.get('assistant')
        if assistant:
            # Import ConnectionState for comparison
            from openai_client.realtime import ConnectionState
            
            # Get real connection status
            # For OpenAI, return status string to reflect on-demand connection
            if assistant.openai_client:
                if assistant.openai_client.state == ConnectionState.CONNECTED:
                    openai_status = 'connected'
                else:
                    openai_status = 'ready'  # Configured but not connected (normal state)
            else:
                openai_status = 'not_configured'
            
            connections = {
                'openai': openai_status,
                'home_assistant': bool(assistant.mcp_client and assistant.mcp_client.is_connected),
                'wake_word': bool(assistant.wake_word_detector and assistant.wake_word_detector.is_running)
            }
            
            # Send real status
            await ws.send_json({
                'type': 'status',
                'state': assistant.session_state.value,
                'connections': connections
            })
        else:
            # Fallback to default status if assistant not available
            await ws.send_json({
                'type': 'status',
                'state': 'idle',
                'connections': {
                    'openai': 'not_configured',
                    'home_assistant': False,
                    'wake_word': False
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