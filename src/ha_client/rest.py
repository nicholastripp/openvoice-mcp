"""
Home Assistant REST API client
"""
import aiohttp
import asyncio
from typing import Dict, Any, Optional, List
from urllib.parse import urljoin

from ..config import HomeAssistantConfig
from ..utils.logger import get_logger


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
    
    async def _request(self, method: str, endpoint: str, **kwargs) -> Any:
        """
        Make HTTP request to Home Assistant
        
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
            self.logger.error(f"HTTP {e.status} error for {method} {endpoint}: {e.message}")
            raise
        except aiohttp.ClientError as e:
            self.logger.error(f"Client error for {method} {endpoint}: {e}")
            raise
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout for {method} {endpoint}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error for {method} {endpoint}: {e}")
            raise