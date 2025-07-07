"""
Home Assistant Conversation API client
"""
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .rest import HomeAssistantRestClient
from ..config import HomeAssistantConfig
from ..utils.logger import get_logger


class ResponseType(Enum):
    """Conversation response types"""
    ACTION_DONE = "action_done"
    QUERY_ANSWER = "query_answer"
    ERROR = "error"


@dataclass
class ConversationTarget:
    """Represents a conversation target"""
    type: str  # area, domain, device_class, device, entity, custom
    name: str
    id: Optional[str] = None


@dataclass
class ConversationResponse:
    """Parsed conversation response"""
    response_type: ResponseType
    language: str
    speech_text: str
    targets: list[ConversationTarget]
    success_entities: list[ConversationTarget]
    failed_entities: list[ConversationTarget]
    error_code: Optional[str] = None
    conversation_id: Optional[str] = None
    continue_conversation: bool = False


class HomeAssistantConversationClient:
    """
    Client for Home Assistant Conversation API
    Following Billy B-Assistant pattern for HA integration
    """
    
    def __init__(self, config: HomeAssistantConfig):
        self.config = config
        self.logger = get_logger("HAConversationClient")
        self.rest_client = HomeAssistantRestClient(config)
        
    async def start(self) -> None:
        """Initialize the conversation client"""
        await self.rest_client.start()
        
        # Test the connection
        try:
            await self.rest_client.get_api_status()
            self.logger.info("HA Conversation client started")
        except Exception as e:
            self.logger.error(f"Failed to connect to Home Assistant: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the conversation client"""
        await self.rest_client.stop()
        self.logger.info("HA Conversation client stopped")
    
    async def process_command(self, text: str, conversation_id: Optional[str] = None,
                            agent_id: Optional[str] = None) -> ConversationResponse:
        """
        Process a natural language command using HA's Conversation API
        
        This is the main method that follows the Billy B-Assistant pattern:
        - Send the full user request as-is to HA
        - HA processes it as a natural language query
        - Extract and parse the response
        
        Args:
            text: Natural language command/query
            conversation_id: Optional conversation ID for context
            agent_id: Optional conversation agent ID
            
        Returns:
            Parsed conversation response
            
        Raises:
            Exception: If the API call fails
        """
        self.logger.debug(f"Processing command: {text}")
        
        # Prepare request data
        data = {
            "text": text,
            "language": self.config.language
        }
        
        if conversation_id:
            data["conversation_id"] = conversation_id
        if agent_id:
            data["agent_id"] = agent_id
        
        try:
            # Call HA Conversation API
            response = await self.rest_client._request(
                "POST",
                "/api/conversation/process",
                json=data
            )
            
            # Parse and return structured response
            parsed_response = self._parse_response(response)
            
            self.logger.debug(f"Response type: {parsed_response.response_type.value}")
            self.logger.debug(f"Speech: {parsed_response.speech_text}")
            
            return parsed_response
            
        except Exception as e:
            self.logger.error(f"Error processing command '{text}': {e}")
            # Return error response
            return ConversationResponse(
                response_type=ResponseType.ERROR,
                language=self.config.language,
                speech_text=f"Sorry, I encountered an error: {str(e)}",
                targets=[],
                success_entities=[],
                failed_entities=[],
                error_code="api_error"
            )
    
    def _parse_response(self, response: Dict[str, Any]) -> ConversationResponse:
        """
        Parse raw HA conversation response into structured format
        
        Args:
            response: Raw response from HA Conversation API
            
        Returns:
            Parsed conversation response
        """
        # Extract basic info
        conversation_id = response.get("conversation_id")
        continue_conversation = response.get("continue_conversation", False)
        
        response_data = response.get("response", {})
        response_type_str = response_data.get("response_type", "error")
        language = response_data.get("language", self.config.language)
        
        # Parse response type
        try:
            response_type = ResponseType(response_type_str)
        except ValueError:
            response_type = ResponseType.ERROR
        
        # Extract speech text
        speech_data = response_data.get("speech", {})
        speech_text = ""
        
        if "plain" in speech_data:
            speech_text = speech_data["plain"].get("speech", "")
        elif "ssml" in speech_data:
            speech_text = speech_data["ssml"].get("speech", "")
        
        # Parse data section
        data_section = response_data.get("data", {})
        
        # Parse targets
        targets = []
        raw_targets = data_section.get("targets", [])
        for target_data in raw_targets:
            targets.append(ConversationTarget(
                type=target_data.get("type", "unknown"),
                name=target_data.get("name", ""),
                id=target_data.get("id")
            ))
        
        # Parse success entities
        success_entities = []
        raw_success = data_section.get("success", [])
        for entity_data in raw_success:
            success_entities.append(ConversationTarget(
                type=entity_data.get("type", "unknown"),
                name=entity_data.get("name", ""),
                id=entity_data.get("id")
            ))
        
        # Parse failed entities
        failed_entities = []
        raw_failed = data_section.get("failed", [])
        for entity_data in raw_failed:
            failed_entities.append(ConversationTarget(
                type=entity_data.get("type", "unknown"),
                name=entity_data.get("name", ""),
                id=entity_data.get("id")
            ))
        
        # Extract error code if present
        error_code = None
        if response_type == ResponseType.ERROR:
            error_code = data_section.get("code", "unknown")
        
        return ConversationResponse(
            response_type=response_type,
            language=language,
            speech_text=speech_text,
            targets=targets,
            success_entities=success_entities,
            failed_entities=failed_entities,
            error_code=error_code,
            conversation_id=conversation_id,
            continue_conversation=continue_conversation
        )
    
    async def prepare_language(self, language: Optional[str] = None) -> None:
        """
        Pre-load sentences for a language
        
        Args:
            language: Language code (defaults to configured language)
        """
        lang = language or self.config.language
        
        data = {"language": lang}
        
        try:
            await self.rest_client._request(
                "POST",
                "/api/conversation/prepare", 
                json=data
            )
            self.logger.debug(f"Prepared language: {lang}")
        except Exception as e:
            self.logger.warning(f"Failed to prepare language {lang}: {e}")
    
    def format_response_for_speech(self, response: ConversationResponse) -> str:
        """
        Format conversation response for natural speech
        
        This method follows the Billy B-Assistant pattern:
        - Extract the speech.plain.speech text
        - Return it for the AI to interpret and speak naturally
        
        Args:
            response: Conversation response
            
        Returns:
            Text suitable for speech synthesis
        """
        if response.speech_text:
            return response.speech_text
        
        # Fallback based on response type
        if response.response_type == ResponseType.ACTION_DONE:
            if response.success_entities:
                entity_names = [e.name for e in response.success_entities]
                return f"I've completed the action for {', '.join(entity_names)}"
            else:
                return "Action completed successfully"
                
        elif response.response_type == ResponseType.QUERY_ANSWER:
            return "I found the information you requested"
            
        elif response.response_type == ResponseType.ERROR:
            error_messages = {
                "no_intent_match": "I didn't understand that command",
                "no_valid_targets": "I couldn't find the device or area you mentioned",
                "failed_to_handle": "I couldn't complete that action",
                "unknown": "Something went wrong"
            }
            return error_messages.get(response.error_code, "I encountered an error")
        
        return "Task completed"
    
    def get_response_summary(self, response: ConversationResponse) -> Dict[str, Any]:
        """
        Get a summary of the conversation response for logging/debugging
        
        Args:
            response: Conversation response
            
        Returns:
            Summary dictionary
        """
        return {
            "type": response.response_type.value,
            "success": response.response_type != ResponseType.ERROR,
            "speech": response.speech_text,
            "targets_count": len(response.targets),
            "success_count": len(response.success_entities),
            "failed_count": len(response.failed_entities),
            "error_code": response.error_code,
            "continue_conversation": response.continue_conversation
        }