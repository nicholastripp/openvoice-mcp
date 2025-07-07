"""
Function calling bridge between OpenAI and Home Assistant
"""
import json
from typing import Dict, Any, Optional

from ha_client.conversation import HomeAssistantConversationClient, ConversationResponse, ResponseType
from utils.logger import get_logger


class FunctionCallBridge:
    """
    Bridge that handles OpenAI function calls and routes them to Home Assistant
    Following the Billy B-Assistant pattern of forwarding commands to HA's Conversation API
    """
    
    def __init__(self, ha_client: HomeAssistantConversationClient):
        self.ha_client = ha_client
        self.logger = get_logger("FunctionBridge")
        
        # Track conversation context
        self.conversation_id: Optional[str] = None
    
    def get_function_definitions(self) -> list[Dict[str, Any]]:
        """
        Get OpenAI function definitions for registration
        
        Returns:
            List of function definitions for OpenAI
        """
        return [
            {
                "type": "function",
                "name": "control_home_assistant",
                "description": (
                    "Control Home Assistant devices, check device states, or answer questions about the home. "
                    "This function can handle any smart home command including turning devices on/off, "
                    "setting brightness/temperature, checking status, and querying information."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": (
                                "The natural language command to send to Home Assistant. "
                                "Examples: 'turn on the living room lights', 'set thermostat to 72 degrees', "
                                "'what's the temperature in the bedroom', 'are any lights on in the kitchen'"
                            )
                        }
                    },
                    "required": ["command"]
                }
            }
        ]
    
    async def handle_function_call(self, function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle function calls from OpenAI
        
        Args:
            function_name: Name of the function being called
            arguments: Function arguments
            
        Returns:
            Function result to send back to OpenAI
        """
        self.logger.debug(f"Function call: {function_name} with args: {arguments}")
        
        if function_name == "control_home_assistant":
            return await self._handle_home_assistant_control(arguments)
        else:
            self.logger.error(f"Unknown function called: {function_name}")
            return {
                "success": False,
                "error": f"Unknown function: {function_name}",
                "message": "I don't know how to handle that request."
            }
    
    async def _handle_home_assistant_control(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle Home Assistant control function call
        
        Args:
            arguments: Function arguments containing the command
            
        Returns:
            Result of the HA operation
        """
        command = arguments.get("command", "")
        if not command:
            return {
                "success": False,
                "error": "missing_command",
                "message": "No command provided"
            }
        
        try:
            # Send command to Home Assistant Conversation API
            # Following Billy B-Assistant pattern: forward the full request as-is
            response = await self.ha_client.process_command(
                text=command,
                conversation_id=self.conversation_id
            )
            
            # Update conversation context
            if response.conversation_id:
                self.conversation_id = response.conversation_id
            
            # Convert HA response to OpenAI function result
            return self._convert_ha_response_to_function_result(response, command)
            
        except Exception as e:
            self.logger.error(f"Error processing HA command '{command}': {e}")
            return {
                "success": False,
                "error": "ha_api_error",
                "message": f"I encountered an error while trying to {command.lower()}. Please try again."
            }
    
    def _convert_ha_response_to_function_result(self, response: ConversationResponse, 
                                              original_command: str) -> Dict[str, Any]:
        """
        Convert HA conversation response to OpenAI function result
        
        Args:
            response: HA conversation response
            original_command: Original command sent to HA
            
        Returns:
            Function result for OpenAI
        """
        # Basic result structure
        result = {
            "success": response.response_type != ResponseType.ERROR,
            "command": original_command,
            "response_type": response.response_type.value,
            "message": response.speech_text or self._generate_fallback_message(response)
        }
        
        # Add detailed information based on response type
        if response.response_type == ResponseType.ACTION_DONE:
            result.update({
                "action": "completed",
                "targets": [{"type": t.type, "name": t.name, "id": t.id} for t in response.targets],
                "success_entities": [{"type": e.type, "name": e.name, "id": e.id} for e in response.success_entities],
                "failed_entities": [{"type": e.type, "name": e.name, "id": e.id} for e in response.failed_entities]
            })
            
            # Enhanced success message
            if response.success_entities:
                entity_names = [e.name for e in response.success_entities if e.name]
                if entity_names:
                    result["affected_devices"] = entity_names
            
        elif response.response_type == ResponseType.QUERY_ANSWER:
            result.update({
                "action": "query_answered",
                "answer": response.speech_text
            })
            
        elif response.response_type == ResponseType.ERROR:
            result.update({
                "action": "failed",
                "error_code": response.error_code,
                "error_details": self._get_error_explanation(response.error_code)
            })
        
        # Add conversation context if available
        if response.conversation_id:
            result["conversation_id"] = response.conversation_id
        
        if response.continue_conversation:
            result["continue_conversation"] = True
        
        return result
    
    def _generate_fallback_message(self, response: ConversationResponse) -> str:
        """
        Generate fallback message when HA doesn't provide speech text
        
        Args:
            response: HA conversation response
            
        Returns:
            Fallback message
        """
        if response.response_type == ResponseType.ACTION_DONE:
            if response.success_entities:
                return "I've completed the requested action successfully."
            else:
                return "The action has been completed."
                
        elif response.response_type == ResponseType.QUERY_ANSWER:
            return "I found the information you requested."
            
        elif response.response_type == ResponseType.ERROR:
            error_messages = {
                "no_intent_match": "I didn't understand that command. Could you please rephrase it?",
                "no_valid_targets": "I couldn't find the device or area you mentioned. Please check the name and try again.",
                "failed_to_handle": "I wasn't able to complete that action. There might be an issue with the device.",
                "unknown": "Something went wrong while processing your request."
            }
            return error_messages.get(response.error_code, "I encountered an error processing your request.")
        
        return "Task completed."
    
    def _get_error_explanation(self, error_code: Optional[str]) -> str:
        """
        Get detailed error explanation for troubleshooting
        
        Args:
            error_code: HA error code
            
        Returns:
            Detailed error explanation
        """
        explanations = {
            "no_intent_match": (
                "Home Assistant couldn't understand the command. The request might be too ambiguous, "
                "use unsupported language, or reference functionality that isn't available."
            ),
            "no_valid_targets": (
                "The specified device, area, or entity wasn't found in Home Assistant. "
                "Check that the device exists and is properly named."
            ),
            "failed_to_handle": (
                "Home Assistant understood the command but couldn't execute it. "
                "The device might be offline, misconfigured, or the action might not be supported."
            ),
            "unknown": (
                "An unexpected error occurred in Home Assistant while processing the request."
            )
        }
        
        return explanations.get(error_code, "No additional error information available.")
    
    def reset_conversation(self) -> None:
        """Reset conversation context"""
        self.conversation_id = None
        self.logger.debug("Conversation context reset")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get current conversation summary
        
        Returns:
            Summary of conversation state
        """
        return {
            "conversation_id": self.conversation_id,
            "has_context": self.conversation_id is not None
        }