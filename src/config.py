"""
Configuration management for OpenVoice MCP
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
    model: str = "gpt-realtime"  # New production model (changed from gpt-4o-realtime-preview)
    legacy_model: str = "gpt-4o-realtime-preview"  # Fallback for compatibility
    model_selection: str = "auto"  # auto, new, legacy - controls model selection
    temperature: float = 0.8
    language: str = "en"
    
    # Available voices for each model
    VOICES = {
        "gpt-realtime": ["alloy", "ash", "ballad", "coral", "echo", 
                        "sage", "shimmer", "verse", "cedar", "marin"],
        "gpt-4o-realtime-preview": ["alloy", "ash", "ballad", "coral", 
                                     "echo", "sage", "shimmer", "verse"]
    }
    
    # Voice fallback configuration
    voice_fallback: str = "alloy"  # Fallback if selected voice unavailable
    auto_select_voice: bool = True  # Auto-select based on model capabilities


@dataclass
class MCPConfig:
    """Model Context Protocol configuration"""
    sse_endpoint: str = "/mcp_server/sse"
    auth_method: str = "token"  # "token" or "oauth"
    connection_timeout: int = 30
    reconnect_attempts: int = 3
    ssl_verify: bool = True  # Whether to verify SSL certificates
    ssl_ca_bundle: Optional[str] = None  # Path to custom CA bundle
    max_reconnect_attempts: int = 5  # Maximum number of reconnection attempts
    reconnect_base_delay: float = 1.0  # Base delay between reconnection attempts
    reconnect_max_delay: float = 60.0  # Maximum delay between reconnection attempts
    sse_read_timeout: int = 300  # SSE read timeout in seconds
    
    # Native MCP integration settings
    native_mode: bool = False  # Enable native MCP support through OpenAI
    endpoint: str = "/mcp_server/sse"  # MCP server endpoint path
    approval_mode: str = "never"  # Tool approval: "never", "always", "on_error"
    approval_timeout: int = 5000  # Milliseconds to wait for approval
    enable_fallback: bool = True  # Fallback to bridge mode on native failure
    performance_tracking: bool = True  # Track performance metrics
    cache_tool_definitions: bool = True  # Cache discovered tools
    tool_timeout: int = 30000  # Tool execution timeout in milliseconds


@dataclass
class MCPServerConfig:
    """Configuration for individual MCP servers (native or client-side)"""
    name: str
    mode: str = "native"  # "native" (OpenAI handles) or "client" (local client-side)

    # Native mode fields (for remote servers)
    server_url: Optional[str] = None
    authorization: Optional[str] = None
    description: Optional[str] = None
    require_approval: str = "always"  # "always", "never", or dict with tool-specific rules
    allowed_tools: Optional[list] = None  # Filter which tools to expose from this server

    # Client mode fields (for local/stdio servers - uses existing ha-realtime-assist code)
    transport: Optional[str] = None  # "stdio", "sse", etc.
    command: Optional[str] = None  # Command to run for stdio transport
    args: Optional[list] = None  # Arguments for the command
    env: Optional[Dict[str, str]] = None  # Environment variables
    timeout: int = 30  # Connection timeout

    # Shared fields
    enabled: bool = True  # Whether this server is active
    priority: int = 100  # Server priority for tool routing (lower = higher priority)

    def __post_init__(self):
        """Validate configuration based on mode"""
        if self.mode == "native":
            if not self.server_url:
                raise ValueError(f"MCP server '{self.name}' in native mode requires server_url")
        elif self.mode == "client":
            if not self.transport:
                raise ValueError(f"MCP server '{self.name}' in client mode requires transport")
            if self.transport == "stdio" and not self.command:
                raise ValueError(f"MCP server '{self.name}' with stdio transport requires command")
        else:
            raise ValueError(f"MCP server '{self.name}' has invalid mode: '{self.mode}'. Must be 'native' or 'client'")


@dataclass
class HomeAssistantConfig:
    """Home Assistant API configuration (optional)"""
    url: Optional[str] = None
    token: Optional[str] = None
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
    
    # Language settings
    language: str = "en"  # Language code (en, de, es, fr, it, nl)
    
    # Multi-turn conversation settings
    conversation_mode: str = "multi_turn"  # "single_turn" or "multi_turn"
    multi_turn_timeout: float = 300.0  # Safety fallback timeout (5 minutes)
    multi_turn_max_turns: int = 10  # maximum conversation turns per session
    multi_turn_end_phrases: list = None  # phrases to end conversation (deprecated - use multi_turn_end_phrases_dict)
    multi_turn_end_phrases_dict: dict = None  # language-specific end phrases
    multi_turn_stuck_multiplier: float = 4.0  # multiplier for stuck detection
    extended_silence_threshold: float = 8.0  # seconds of silence before ending conversation
    
    def __post_init__(self):
        # Set default multi-language end phrases if not provided
        if self.multi_turn_end_phrases_dict is None:
            self.multi_turn_end_phrases_dict = {
                "en": ["stop", "thank you", "goodbye", "that's all", "bye", "end session", "exit", 
                       "that's it", "done", "finished", "no more", "nothing else"],
                "de": ["stopp", "stop", "ende", "danke", "tschüss", "beenden", "fertig", "schluss",
                       "das war's", "das wars", "auf wiedersehen", "nichts mehr"],
                "es": ["parar", "detener", "gracias", "adiós", "terminar", "fin", "hasta luego",
                       "basta", "finalizar", "nada más", "eso es todo"],
                "fr": ["arrêter", "stop", "merci", "au revoir", "terminer", "fin", "c'est tout",
                       "fini", "terminé", "plus rien", "c'est fini"],
                "it": ["ferma", "stop", "grazie", "arrivederci", "fine", "basta", "termina",
                       "finito", "chiudi", "niente altro", "è tutto"],
                "nl": ["stop", "stoppen", "bedankt", "tot ziens", "klaar", "einde", "genoeg",
                       "af", "afgelopen", "niets meer", "dat is alles"]
            }
        
        # For backward compatibility, if old multi_turn_end_phrases is set, use it for English
        if self.multi_turn_end_phrases and not self.multi_turn_end_phrases_dict.get("en"):
            self.multi_turn_end_phrases_dict["en"] = self.multi_turn_end_phrases


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
class WebUIAuthConfig:
    """Web UI authentication configuration"""
    enabled: bool = True
    username: str = "admin"
    password_hash: str = ""
    session_timeout: int = 3600


@dataclass
class WebUITLSConfig:
    """Web UI TLS/HTTPS configuration"""
    enabled: bool = True
    cert_file: str = ""
    key_file: str = ""


@dataclass
class WebUIConfig:
    """Web UI configuration"""
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8443
    auth: WebUIAuthConfig = field(default_factory=WebUIAuthConfig)
    tls: WebUITLSConfig = field(default_factory=WebUITLSConfig)


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
    home_assistant: Optional[HomeAssistantConfig] = None  # Optional - system can run as pure voice UI
    audio: AudioConfig = field(default_factory=AudioConfig)
    wake_word: WakeWordConfig = field(default_factory=WakeWordConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    web_ui: WebUIConfig = field(default_factory=WebUIConfig)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)
    mcp_servers: Optional[Dict[str, MCPServerConfig]] = None  # Optional multi-server MCP configuration


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

        # Create Home Assistant config only if section exists (optional)
        ha_config = None
        if "home_assistant" in config_data:
            ha_config = HomeAssistantConfig(**config_data["home_assistant"])

            # Validate HA config if provided
            if ha_config.url and ha_config.token:
                # Validate URL format
                _validate_url(ha_config.url, "Home Assistant")
            elif ha_config.url or ha_config.token:
                # If only one is provided, warn user
                raise ValueError(
                    "Home Assistant configuration is incomplete.\n"
                    "Both 'url' and 'token' must be provided if using Home Assistant integration.\n"
                    "Either provide both, or omit the home_assistant section entirely for pure voice UI mode."
                )

        # Validate model selection
        if openai_config.model_selection not in ["auto", "new", "legacy"]:
            raise ValueError(
                f"Invalid model_selection: '{openai_config.model_selection}'. "
                "Must be 'auto', 'new', or 'legacy'"
            )

        # Determine actual model to use based on selection
        if openai_config.model_selection == "legacy":
            actual_model = openai_config.legacy_model
        elif openai_config.model_selection == "new":
            actual_model = openai_config.model
        else:  # auto
            actual_model = openai_config.model  # Default to new, will fallback if needed

        # Validate voice availability for selected model
        available_voices = openai_config.VOICES.get(actual_model, [])
        if openai_config.voice not in available_voices:
            print(f"Warning: Voice '{openai_config.voice}' not available for model '{actual_model}'")
            if openai_config.auto_select_voice:
                # Auto-select compatible voice
                if openai_config.voice_fallback in available_voices:
                    print(f"Using fallback voice: '{openai_config.voice_fallback}'")
                    openai_config.voice = openai_config.voice_fallback
                else:
                    print(f"Using default voice: 'alloy'")
                    openai_config.voice = "alloy"

        # Validate required fields with helpful error messages
        if not openai_config.api_key:
            raise ValueError(
                "OpenAI API key is required.\n"
                "Please set the OPENAI_API_KEY environment variable or add it to config.yaml:\n"
                "  openai:\n"
                "    api_key: 'your-api-key-here'"
            )
        
        # Create optional configuration objects
        audio_config = AudioConfig(**config_data.get("audio", {}))
        wake_word_config = WakeWordConfig(**config_data.get("wake_word", {}))

        # Create MCP config from home_assistant section if it exists
        if ha_config:
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
        
        # Create WebUIConfig with nested dataclasses
        web_ui_data = config_data.get("web_ui", {})
        if "auth" in web_ui_data:
            web_ui_data["auth"] = WebUIAuthConfig(**web_ui_data["auth"])
        if "tls" in web_ui_data:
            web_ui_data["tls"] = WebUITLSConfig(**web_ui_data["tls"])
        web_ui_config = WebUIConfig(**web_ui_data)
        
        advanced_config = AdvancedConfig(**config_data.get("advanced", {}))

        # Parse MCP servers configuration (optional)
        mcp_servers = None
        if "mcp_servers" in config_data:
            mcp_servers = {}
            for server_name, server_data in config_data["mcp_servers"].items():
                try:
                    mcp_servers[server_name] = MCPServerConfig(name=server_name, **server_data)
                except Exception as e:
                    raise ValueError(f"Invalid configuration for MCP server '{server_name}': {e}")

        return AppConfig(
            openai=openai_config,
            home_assistant=ha_config,
            audio=audio_config,
            wake_word=wake_word_config,
            session=session_config,
            system=system_config,
            web_ui=web_ui_config,
            advanced=advanced_config,
            mcp_servers=mcp_servers
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