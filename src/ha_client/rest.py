"""
Home Assistant REST API client
"""
import aiohttp
import asyncio
from typing import Dict, Any, Optional, List
from urllib.parse import urljoin

from config import HomeAssistantConfig
from utils.logger import get_logger


class HomeAssistantRestClient:
    """
    Client for Home Assistant REST API
    """
    
    def __init__(self, config: HomeAssistantConfig):
        self.config = config
        self.logger = get_logger("HARestClient")
        self.session: Optional[aiohttp.ClientSession] = None
        self.base_url = config.url.rstrip('/')
        
        # Setup headers
        self.headers = {
            "Authorization": f"Bearer {config.token}",
            "Content-Type": "application/json"
        }
    
    async def start(self) -> None:
        """Initialize the HTTP session"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=timeout
            )
            self.logger.info("HA REST client started")
    
    async def stop(self) -> None:
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
            self.logger.info("HA REST client stopped")
    
    async def get_api_status(self) -> Dict[str, Any]:
        """
        Get API status
        
        Returns:
            API status information
        """
        return await self._request("GET", "/api/")
    
    async def get_config(self) -> Dict[str, Any]:
        """
        Get Home Assistant configuration
        
        Returns:
            HA configuration
        """
        return await self._request("GET", "/api/config")
    
    async def get_states(self) -> List[Dict[str, Any]]:
        """
        Get all entity states
        
        Returns:
            List of all entity states
        """
        return await self._request("GET", "/api/states")
    
    async def get_state(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get state of specific entity
        
        Args:
            entity_id: Entity ID to query
            
        Returns:
            Entity state or None if not found
        """
        try:
            return await self._request("GET", f"/api/states/{entity_id}")
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                return None
            raise
    
    async def set_state(self, entity_id: str, state: str, attributes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Set state of an entity
        
        Args:
            entity_id: Entity ID to update
            state: New state value
            attributes: Optional attributes to set
            
        Returns:
            Updated entity state
        """
        data = {"state": state}
        if attributes:
            data["attributes"] = attributes
            
        return await self._request("POST", f"/api/states/{entity_id}", json=data)
    
    async def call_service(self, domain: str, service: str, entity_id: Optional[str] = None, 
                          service_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Call a Home Assistant service
        
        Args:
            domain: Service domain (e.g., 'light', 'switch')
            service: Service name (e.g., 'turn_on', 'turn_off')
            entity_id: Target entity ID (optional)
            service_data: Additional service data (optional)
            
        Returns:
            Service call result
        """
        data = {}
        if entity_id:
            data["entity_id"] = entity_id
        if service_data:
            data.update(service_data)
            
        return await self._request("POST", f"/api/services/{domain}/{service}", json=data)
    
    async def get_services(self) -> Dict[str, Any]:
        """
        Get available services
        
        Returns:
            Dictionary of available services
        """
        return await self._request("GET", "/api/services")
    
    async def get_events(self) -> List[Dict[str, Any]]:
        """
        Get event types
        
        Returns:
            List of event types
        """
        return await self._request("GET", "/api/events")
    
    async def fire_event(self, event_type: str, event_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fire an event
        
        Args:
            event_type: Type of event to fire
            event_data: Event data (optional)
            
        Returns:
            Event fire result
        """
        data = event_data or {}
        return await self._request("POST", f"/api/events/{event_type}", json=data)
    
    async def get_logbook(self, start_time: Optional[str] = None, end_time: Optional[str] = None,
                         entity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get logbook entries
        
        Args:
            start_time: Start time (ISO format, optional)
            end_time: End time (ISO format, optional) 
            entity: Filter by entity (optional)
            
        Returns:
            List of logbook entries
        """
        url = "/api/logbook"
        if start_time:
            url += f"/{start_time}"
            
        params = {}
        if end_time:
            params["end_time"] = end_time
        if entity:
            params["entity"] = entity
            
        return await self._request("GET", url, params=params)
    
    async def get_history(self, start_time: str, end_time: Optional[str] = None,
                         filter_entity_ids: Optional[List[str]] = None) -> List[List[Dict[str, Any]]]:
        """
        Get history data
        
        Args:
            start_time: Start time (ISO format)
            end_time: End time (ISO format, optional)
            filter_entity_ids: List of entity IDs to filter (optional)
            
        Returns:
            History data grouped by entity
        """
        url = f"/api/history/period/{start_time}"
        
        params = {}
        if end_time:
            params["end_time"] = end_time
        if filter_entity_ids:
            params["filter_entity_id"] = ",".join(filter_entity_ids)
            
        return await self._request("GET", url, params=params)
    
    async def render_template(self, template: str, variables: Optional[Dict[str, Any]] = None) -> str:
        """
        Render a Jinja2 template
        
        Args:
            template: Template string to render
            variables: Template variables (optional)
            
        Returns:
            Rendered template
        """
        data = {"template": template}
        if variables:
            data["variables"] = variables
            
        return await self._request("POST", "/api/template", json=data)
    
    async def check_config(self) -> Dict[str, Any]:
        """
        Check configuration validity
        
        Returns:
            Configuration check results
        """
        return await self._request("POST", "/api/config/core/check_config")
    
    async def get(self, endpoint: str, **kwargs) -> Any:
        """
        Convenience method for GET requests
        
        Args:
            endpoint: API endpoint
            **kwargs: Additional arguments for the request
            
        Returns:
            Response data
        """
        return await self._request("GET", endpoint, **kwargs)
    
    async def _request(self, method: str, endpoint: str, **kwargs) -> Any:
        """
        Make HTTP request to Home Assistant with retry logic
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional arguments for aiohttp
            
        Returns:
            Response data
            
        Raises:
            aiohttp.ClientError: On HTTP errors
            asyncio.TimeoutError: On timeout
        """
        if not self.session:
            await self.start()
            
        url = urljoin(self.base_url, endpoint)
        
        # Retry configuration
        max_retries = 3
        base_delay = 1.0  # seconds
        
        for attempt in range(max_retries):
            try:
                async with self.session.request(method, url, **kwargs) as response:
                    response.raise_for_status()
                    
                    # Handle different content types
                    content_type = response.headers.get('content-type', '')
                    
                    if 'application/json' in content_type:
                        return await response.json()
                    elif 'text/' in content_type:
                        return await response.text()
                    else:
                        return await response.read()
                        
            except aiohttp.ClientResponseError as e:
                # Don't retry on client errors (4xx), except for 429 (Too Many Requests)
                if 400 <= e.status < 500 and e.status != 429:
                    error_msg = f"HTTP {e.status} error for {method} {endpoint}"
                    if e.status == 401:
                        error_msg += "\n\nAuthentication failed. Your access token may be invalid or expired."
                        error_msg += "\nPlease check your Home Assistant access token in config.yaml"
                    elif e.status == 403:
                        error_msg += "\n\nAccess forbidden. Your token may not have the required permissions."
                    elif e.status == 404:
                        error_msg += "\n\nEndpoint not found. This may indicate an incompatible Home Assistant version."
                    
                    self.logger.error(error_msg)
                    raise
                    
                # Retry on server errors (5xx) and rate limiting
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(f"HTTP {e.status} error, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"HTTP {e.status} error for {method} {endpoint} after {max_retries} attempts")
                    raise
                    
            except aiohttp.ClientConnectorError as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    self.logger.warning(f"Connection error: {e}, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                else:
                    error_msg = f"Failed to connect to Home Assistant at {self.base_url}"
                    error_msg += f"\n\nConnection error: {e}"
                    error_msg += "\n\nPossible causes:"
                    error_msg += "\n  - Home Assistant is not running"
                    error_msg += "\n  - Network/firewall is blocking the connection"
                    error_msg += "\n  - Incorrect URL in config.yaml"
                    
                    self.logger.error(error_msg)
                    raise
                    
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Request timeout, retrying... (attempt {attempt + 1}/{max_retries})")
                else:
                    error_msg = f"Request to {endpoint} timed out after {self.config.timeout} seconds"
                    error_msg += "\n\nThis may indicate:"
                    error_msg += "\n  - Home Assistant is overloaded or unresponsive"
                    error_msg += "\n  - Network latency issues"
                    error_msg += "\n  - You may need to increase the timeout in config.yaml"
                    
                    self.logger.error(error_msg)
                    raise
                    
            except Exception as e:
                self.logger.error(f"Unexpected error for {method} {endpoint}: {e}")
                raise