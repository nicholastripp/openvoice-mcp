"""
Configuration management for HA Realtime Voice Assistant
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
from urllib.parse import urlparse


@dataclass
class OpenAIConfig:
    """OpenAI API configuration"""
    api_key: str
    voice: str = "alloy"
    model: str = "gpt-4o-realtime-preview"
    temperature: float = 0.8
    language: str = "en"


@dataclass
class MCPConfig:
    """Model Context Protocol configuration"""
    sse_endpoint: str = "/mcp_server/sse"
    auth_method: str = "token"  # "token" or "oauth"
    connection_timeout: int = 30
    reconnect_attempts: int = 3
    ssl_verify: bool = True  # Whether to verify SSL certificates
    ssl_ca_bundle: Optional[str] = None  # Path to custom CA bundle


@dataclass
class HomeAssistantConfig:
    """Home Assistant API configuration"""
    url: str
    token: str
    language: str = "en"
    timeout: int = 10
    mcp: MCPConfig = field(default_factory=MCPConfig)


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
    
    # Automatic Gain Control (AGC) settings
    agc_enabled: bool = False  # Disabled by default for backward compatibility
    agc_target_rms: float = 0.3  # Target RMS level (0-1 range, 0.3 = 30% of max)
    agc_max_gain: float = 3.0    # Maximum gain multiplier allowed by AGC
    agc_min_gain: float = 0.1    # Minimum gain multiplier allowed by AGC
    agc_attack_time: float = 0.5  # Seconds to decrease gain (fast response to clipping)
    agc_release_time: float = 2.0 # Seconds to increase gain (slow response to quiet)
    agc_clipping_threshold: float = 0.05  # Maximum acceptable clipping ratio (5%)


@dataclass
class WakeWordConfig:
    """Wake word configuration for Porcupine"""
    enabled: bool = True
    model: str = "picovoice"  # Porcupine built-in keyword
    sensitivity: float = 1.0  # Detection sensitivity (0.0-1.0)
    timeout: float = 5.0
    vad_enabled: bool = True
    cooldown: float = 2.0
    test_mode: bool = False  # Test wake word detection without OpenAI connection
    confirmation_beep_enabled: bool = False  # Play beep after wake word detection
    
    # Audio gain settings
    audio_gain: float = 1.0  # Audio amplification factor (1.0-5.0, default 1.0 to prevent clipping)
    audio_gain_mode: str = "fixed"  # Gain mode: "fixed" or "dynamic"
    
    # Porcupine settings
    porcupine_access_key: Optional[str] = None  # Picovoice access key
    
    # High-pass filter settings (required for Porcupine)
    highpass_filter_enabled: bool = True   # Required for Porcupine to work properly
    highpass_filter_cutoff: float = 80.0   # Hz - removes DC offset and low frequency noise


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
    multi_turn_stuck_multiplier: float = 4.0  # multiplier for stuck detection (4x timeout = 2 minutes with 30s base)
    
    def __post_init__(self):
        # Set default end phrases if not provided
        if self.multi_turn_end_phrases is None:
            self.multi_turn_end_phrases = ["goodbye", "stop", "that's all", "thank you", "bye"]


@dataclass
class SystemConfig:
    """System configuration"""
    log_level: str = "INFO"
    log_file: str = "logs/assistant.log"
    console_log_level: str = "INFO"
    log_to_file: bool = True
    log_max_size_mb: int = 10
    log_backup_count: int = 3
    led_gpio: Optional[int] = None
    daemon: bool = False


@dataclass
class WebUIConfig:
    """Web UI configuration"""
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8080


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
    web_ui: WebUIConfig = field(default_factory=WebUIConfig)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)


def _validate_url(url: str, service_name: str) -> None:
    """
    Validate URL format and provide helpful error messages.
    
    Args:
        url: The URL to validate
        service_name: Name of the service (for error messages)
        
    Raises:
        ValueError: If URL is invalid with helpful error message
    """
    try:
        parsed = urlparse(url)
        
        # Check for missing scheme
        if not parsed.scheme:
            raise ValueError(
                f"{service_name} URL is missing the protocol (http:// or https://).\n"
                f"Current value: '{url}'\n"
                f"Expected format: http://your-homeassistant-ip:8123 or https://your-domain.com"
            )
        
        # Check for missing netloc (domain/IP)
        if not parsed.netloc:
            raise ValueError(
                f"{service_name} URL is missing the host/domain.\n"
                f"Current value: '{url}'\n"
                f"Expected format: http://192.168.1.100:8123 or http://homeassistant.local:8123"
            )
        
        # Check for invalid scheme
        if parsed.scheme not in ['http', 'https']:
            raise ValueError(
                f"{service_name} URL must use http:// or https:// protocol.\n"
                f"Current value: '{url}'\n"
                f"Found protocol: '{parsed.scheme}://'"
            )
            
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(
            f"Invalid {service_name} URL format: '{url}'\n"
            f"Expected format: http://your-homeassistant-ip:8123\n"
            f"Error: {str(e)}"
        )


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
        
        # Validate required fields with helpful error messages
        if not openai_config.api_key:
            raise ValueError(
                "OpenAI API key is required.\n"
                "Please set the OPENAI_API_KEY environment variable or add it to config.yaml:\n"
                "  openai:\n"
                "    api_key: 'your-api-key-here'"
            )
        
        if not ha_config.url:
            raise ValueError(
                "Home Assistant URL is required.\n"
                "Please update config/config.yaml with your Home Assistant URL:\n"
                "  home_assistant:\n"
                "    url: 'http://your-homeassistant-ip:8123'"
            )
        
        if not ha_config.token:
            raise ValueError(
                "Home Assistant access token is required.\n"
                "Please set the HA_TOKEN environment variable or add it to config.yaml:\n"
                "  home_assistant:\n"
                "    token: 'your-long-lived-access-token'\n\n"
                "To create a token:\n"
                "1. Go to your Home Assistant profile (http://your-ha-ip:8123/profile)\n"
                "2. Scroll down to 'Long-Lived Access Tokens'\n"
                "3. Click 'Create Token' and copy the generated token"
            )
        
        # Validate URL format
        _validate_url(ha_config.url, "Home Assistant")
        
        # Create optional configuration objects
        audio_config = AudioConfig(**config_data.get("audio", {}))
        wake_word_config = WakeWordConfig(**config_data.get("wake_word", {}))
        
        # Create MCP config from home_assistant section
        mcp_data = config_data.get("home_assistant", {}).get("mcp", {})
        mcp_config = MCPConfig(**mcp_data)
        ha_config.mcp = mcp_config
        
        # Validate wake word gain settings
        if wake_word_config.audio_gain < 1.0 or wake_word_config.audio_gain > 5.0:
            raise ValueError(f"wake_word.audio_gain must be between 1.0 and 5.0, got {wake_word_config.audio_gain}")
        if wake_word_config.audio_gain_mode not in ["fixed", "dynamic"]:
            raise ValueError(f"wake_word.audio_gain_mode must be 'fixed' or 'dynamic', got '{wake_word_config.audio_gain_mode}'")
        
        session_config = SessionConfig(**config_data.get("session", {}))
        system_config = SystemConfig(**config_data.get("system", {}))
        web_ui_config = WebUIConfig(**config_data.get("web_ui", {}))
        advanced_config = AdvancedConfig(**config_data.get("advanced", {}))
        
        return AppConfig(
            openai=openai_config,
            home_assistant=ha_config,
            audio=audio_config,
            wake_word=wake_word_config,
            session=session_config,
            system=system_config,
            web_ui=web_ui_config,
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