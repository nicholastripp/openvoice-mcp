"""
Configuration management for HA Realtime Voice Assistant
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv


@dataclass
class OpenAIConfig:
    """OpenAI API configuration"""
    api_key: str
    voice: str = "alloy"
    model: str = "gpt-4o-realtime-preview"
    temperature: float = 0.8
    language: str = "en"


@dataclass
class HomeAssistantConfig:
    """Home Assistant API configuration"""
    url: str
    token: str
    language: str = "en"
    timeout: int = 10


@dataclass
class AudioConfig:
    """Audio configuration"""
    input_device: str = "default"
    output_device: str = "default"
    sample_rate: int = 48000
    channels: int = 1
    chunk_size: int = 1200
    input_volume: float = 1.0
    output_volume: float = 1.0
    feedback_prevention: bool = True
    feedback_threshold: float = 0.1
    mute_during_response: bool = True


@dataclass
class WakeWordConfig:
    """Wake word configuration"""
    enabled: bool = True
    model: str = "alexa"
    sensitivity: float = 0.0001  # Reasonable default threshold for wake word detection
    timeout: float = 5.0
    vad_enabled: bool = True
    cooldown: float = 2.0
    
    # Model download settings
    auto_download: bool = True
    download_timeout: int = 300
    retry_downloads: int = 3
    
    # Noise suppression settings
    speex_noise_suppression: bool = True  # Enable if speexdsp_ns is available


@dataclass
class SessionConfig:
    """Session configuration"""
    timeout: int = 30
    auto_end_silence: float = 3.0
    max_duration: int = 300
    interrupt_threshold: float = 0.5
    auto_end_after_response: bool = True
    response_cooldown_delay: float = 2.0
    
    # Multi-turn conversation settings
    conversation_mode: str = "single_turn"  # "single_turn" or "multi_turn"
    multi_turn_timeout: float = 30.0  # seconds to wait for follow-up questions
    multi_turn_max_turns: int = 10  # maximum conversation turns per session
    multi_turn_end_phrases: list = None  # phrases to end conversation
    
    def __post_init__(self):
        # Set default end phrases if not provided
        if self.multi_turn_end_phrases is None:
            self.multi_turn_end_phrases = ["goodbye", "stop", "that's all", "thank you", "bye"]


@dataclass
class SystemConfig:
    """System configuration"""
    log_level: str = "INFO"
    log_file: str = "logs/assistant.log"
    led_gpio: Optional[int] = None
    daemon: bool = False


@dataclass
class AdvancedConfig:
    """Advanced configuration options"""
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 10
    audio_buffer_size: int = 8192
    cost_tracking: bool = True
    debug_audio: bool = False


@dataclass
class AppConfig:
    """Main application configuration"""
    openai: OpenAIConfig
    home_assistant: HomeAssistantConfig
    audio: AudioConfig = field(default_factory=AudioConfig)
    wake_word: WakeWordConfig = field(default_factory=WakeWordConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)


def load_config(config_path: str = "config/config.yaml") -> AppConfig:
    """
    Load configuration from YAML file and environment variables.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        AppConfig instance
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If required configuration is missing
    """
    # Load environment variables
    load_dotenv()
    
    # Load YAML config
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Expand environment variables in strings
    config_data = _expand_env_vars(config_data)
    
    try:
        # Create configuration objects
        openai_config = OpenAIConfig(**config_data.get("openai", {}))
        ha_config = HomeAssistantConfig(**config_data.get("home_assistant", {}))
        
        # Validate required fields
        if not openai_config.api_key:
            raise ValueError("OpenAI API key is required")
        if not ha_config.url:
            raise ValueError("Home Assistant URL is required")
        if not ha_config.token:
            raise ValueError("Home Assistant token is required")
        
        # Create optional configuration objects
        audio_config = AudioConfig(**config_data.get("audio", {}))
        wake_word_config = WakeWordConfig(**config_data.get("wake_word", {}))
        session_config = SessionConfig(**config_data.get("session", {}))
        system_config = SystemConfig(**config_data.get("system", {}))
        advanced_config = AdvancedConfig(**config_data.get("advanced", {}))
        
        return AppConfig(
            openai=openai_config,
            home_assistant=ha_config,
            audio=audio_config,
            wake_word=wake_word_config,
            session=session_config,
            system=system_config,
            advanced=advanced_config
        )
        
    except TypeError as e:
        raise ValueError(f"Invalid configuration: {e}")


def _expand_env_vars(obj: Any) -> Any:
    """
    Recursively expand environment variables in configuration.
    
    Supports ${VAR_NAME} syntax.
    """
    if isinstance(obj, dict):
        return {key: _expand_env_vars(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        return os.path.expandvars(obj)
    else:
        return obj