"""
Model compatibility layer for OpenAI Realtime API migration.
Handles migration from gpt-4o-realtime-preview to gpt-realtime with fallback support.
"""
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ModelType(Enum):
    """Available model types for OpenAI Realtime API"""
    GPT_REALTIME = "gpt-realtime"
    GPT_4O_REALTIME_PREVIEW = "gpt-4o-realtime-preview"
    GPT_4O_MINI_REALTIME_PREVIEW = "gpt-4o-mini-realtime-preview"


@dataclass
class ModelCapabilities:
    """Capabilities and features for each model"""
    endpoint: str
    voices: List[str]
    features: List[str]
    pricing: Dict[str, float]  # per 1M tokens
    max_response_tokens: int
    supports_mcp: bool
    supports_image_input: bool
    supports_async_functions: bool


class ModelCompatibility:
    """Handles model compatibility and migration between OpenAI Realtime models"""
    
    # Model specifications and capabilities
    MODELS = {
        ModelType.GPT_REALTIME: ModelCapabilities(
            endpoint="wss://api.openai.com/v1/realtime",
            voices=["alloy", "ash", "ballad", "coral", "echo", 
                   "sage", "shimmer", "verse", "cedar", "marin"],
            features=["async_functions", "native_mcp", "image_input", 
                     "enhanced_function_calling", "improved_instruction_following"],
            pricing={"input": 32.0, "output": 64.0},  # $32/$64 per 1M tokens
            max_response_tokens=4096,
            supports_mcp=True,
            supports_image_input=True,
            supports_async_functions=True
        ),
        ModelType.GPT_4O_REALTIME_PREVIEW: ModelCapabilities(
            endpoint="wss://api.openai.com/v1/realtime",
            voices=["alloy", "ash", "ballad", "coral", "echo", 
                   "sage", "shimmer", "verse"],
            features=["basic_functions", "voice_activity_detection"],
            pricing={"input": 40.0, "output": 80.0},  # $40/$80 per 1M tokens
            max_response_tokens=4096,
            supports_mcp=False,
            supports_image_input=False,
            supports_async_functions=False
        ),
        ModelType.GPT_4O_MINI_REALTIME_PREVIEW: ModelCapabilities(
            endpoint="wss://api.openai.com/v1/realtime",
            voices=["alloy", "ash", "ballad", "coral", "echo", 
                   "sage", "shimmer", "verse"],
            features=["basic_functions", "voice_activity_detection"],
            pricing={"input": 10.0, "output": 20.0},  # Lower cost variant
            max_response_tokens=4096,
            supports_mcp=False,
            supports_image_input=False,
            supports_async_functions=False
        )
    }
    
    def __init__(self, config: Any, logger: Optional[logging.Logger] = None):
        """
        Initialize model compatibility handler.
        
        Args:
            config: Application configuration object
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.current_model = None
        self.fallback_attempts = 0
        self.max_fallback_attempts = 3
        
    def get_model_type(self, model_name: str) -> Optional[ModelType]:
        """
        Convert model name string to ModelType enum.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelType enum or None if not found
        """
        for model_type in ModelType:
            if model_type.value == model_name:
                return model_type
        return None
    
    def select_model(self) -> str:
        """
        Select the appropriate model based on configuration.
        
        Returns:
            Selected model name
        """
        model_selection = self.config.model_selection
        
        if model_selection == "legacy":
            model = self.config.legacy_model
            self.logger.info(f"Using legacy model: {model}")
        elif model_selection == "new":
            model = self.config.model
            self.logger.info(f"Using new model: {model}")
        else:  # auto
            # Start with new model, will fallback if connection fails
            model = self.config.model
            self.logger.info(f"Auto-selecting model, starting with: {model}")
        
        self.current_model = self.get_model_type(model)
        return model
    
    def get_capabilities(self, model: Optional[str] = None) -> Optional[ModelCapabilities]:
        """
        Get capabilities for a specific model.
        
        Args:
            model: Model name (uses current model if not specified)
            
        Returns:
            ModelCapabilities or None if model not found
        """
        if model:
            model_type = self.get_model_type(model)
        else:
            model_type = self.current_model
            
        if model_type:
            return self.MODELS.get(model_type)
        return None
    
    def is_voice_available(self, voice: str, model: Optional[str] = None) -> bool:
        """
        Check if a voice is available for the specified model.
        
        Args:
            voice: Voice name to check
            model: Model name (uses current model if not specified)
            
        Returns:
            True if voice is available, False otherwise
        """
        capabilities = self.get_capabilities(model)
        if capabilities:
            return voice in capabilities.voices
        return False
    
    def get_compatible_voice(self, preferred_voice: str, model: Optional[str] = None) -> str:
        """
        Get a compatible voice for the model, with fallback logic.
        
        Args:
            preferred_voice: Preferred voice name
            model: Model name (uses current model if not specified)
            
        Returns:
            Compatible voice name
        """
        if self.is_voice_available(preferred_voice, model):
            return preferred_voice
        
        # Try fallback voice
        fallback = self.config.voice_fallback
        if self.is_voice_available(fallback, model):
            self.logger.warning(
                f"Voice '{preferred_voice}' not available, using fallback: '{fallback}'"
            )
            return fallback
        
        # Default to alloy (available in all models)
        self.logger.warning(
            f"Neither '{preferred_voice}' nor '{fallback}' available, using 'alloy'"
        )
        return "alloy"
    
    def should_fallback(self, error: Exception) -> bool:
        """
        Determine if we should fallback to legacy model based on error.
        
        Args:
            error: Exception that occurred
            
        Returns:
            True if should fallback, False otherwise
        """
        if self.config.model_selection != "auto":
            return False  # No fallback unless in auto mode
        
        if self.fallback_attempts >= self.max_fallback_attempts:
            return False  # Too many attempts
        
        # Check for specific error patterns that indicate fallback
        error_msg = str(error).lower()
        fallback_indicators = [
            "model not found",
            "invalid model",
            "not available",
            "unauthorized",
            "rate limit",
            "connection failed"
        ]
        
        for indicator in fallback_indicators:
            if indicator in error_msg:
                self.fallback_attempts += 1
                return True
        
        return False
    
    def get_fallback_model(self) -> Optional[str]:
        """
        Get the fallback model to use.
        
        Returns:
            Fallback model name or None if no fallback available
        """
        if self.current_model == self.get_model_type(self.config.model):
            # Fallback from new to legacy
            return self.config.legacy_model
        return None
    
    def get_session_config(self, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Get session configuration for the specified model.
        
        Args:
            model: Model name (uses current model if not specified)
            
        Returns:
            Session configuration dictionary
        """
        capabilities = self.get_capabilities(model)
        if not capabilities:
            raise ValueError(f"Unknown model: {model}")
        
        # Base session configuration
        session_config = {
            "model": model or self.current_model.value,
            "voice": self.get_compatible_voice(self.config.voice, model),
            "instructions": "You are a helpful voice assistant for Home Assistant. You can control smart home devices, answer questions, and help with various tasks. Always be concise and clear in your responses.",
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500
            },
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "temperature": self.config.temperature,
            "max_response_output_tokens": capabilities.max_response_tokens
        }
        
        # Add model-specific features
        if capabilities.supports_mcp:
            # Add MCP configuration if supported
            if hasattr(self.config, 'mcp_servers') and self.config.mcp_servers:
                session_config["mcp_servers"] = self.config.mcp_servers
        
        # Add tools configuration
        session_config["tools"] = []
        session_config["tool_choice"] = "auto"
        
        return session_config
    
    def calculate_cost(self, tokens: Dict[str, int], model: Optional[str] = None) -> float:
        """
        Calculate cost for token usage based on model pricing.
        
        Args:
            tokens: Dictionary with 'input' and 'output' token counts
            model: Model name (uses current model if not specified)
            
        Returns:
            Total cost in dollars
        """
        capabilities = self.get_capabilities(model)
        if not capabilities:
            return 0.0
        
        input_cost = (tokens.get('input', 0) / 1_000_000) * capabilities.pricing['input']
        output_cost = (tokens.get('output', 0) / 1_000_000) * capabilities.pricing['output']
        
        return input_cost + output_cost
    
    def get_performance_improvements(self) -> Dict[str, Any]:
        """
        Get expected performance improvements when using new model.
        
        Returns:
            Dictionary of performance metrics and improvements
        """
        return {
            "big_bench_audio": {
                "old": "65.6%",
                "new": "82.8%",
                "improvement": "+26%"
            },
            "instruction_following": {
                "old": "20.6%",
                "new": "30.5%",
                "improvement": "+48%"
            },
            "function_calling": {
                "old": "49.7%",
                "new": "66.5%",
                "improvement": "+34%"
            },
            "cost_reduction": {
                "old": "$40/$80",
                "new": "$32/$64",
                "improvement": "-20%"
            },
            "new_features": [
                "Native MCP server support",
                "Image input capabilities",
                "Asynchronous function calling",
                "Two new voices: Cedar and Marin",
                "Enhanced instruction following"
            ]
        }
    
    def validate_migration(self) -> Dict[str, bool]:
        """
        Validate that migration is properly configured.
        
        Returns:
            Dictionary of validation results
        """
        results = {
            "config_valid": True,
            "model_available": False,
            "voice_compatible": False,
            "features_ready": True
        }
        
        # Check model configuration
        try:
            model = self.select_model()
            results["model_available"] = model in [m.value for m in ModelType]
        except Exception as e:
            self.logger.error(f"Model selection failed: {e}")
            results["config_valid"] = False
        
        # Check voice compatibility
        if results["model_available"]:
            voice = self.config.voice
            results["voice_compatible"] = self.is_voice_available(voice)
        
        return results