"""
Logging utilities for HA Realtime Voice Assistant
"""
import logging
import logging.handlers
import sys
import io
from pathlib import Path
from typing import Optional

from .text_utils import sanitize_unicode_text


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Set up logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console: Whether to log to console
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("ha_voice_assistant")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter (will be wrapped in UnicodeFormatter)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler with UTF-8 encoding
    if console:
        # Force UTF-8 encoding for console output
        # Wrap stdout to ensure UTF-8 encoding on systems that default to latin-1
        utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        console_handler = logging.StreamHandler(utf8_stdout)
        console_handler.setFormatter(UnicodeFormatter(formatter._fmt, formatter.datefmt))
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use rotating file handler to prevent huge log files
        # Force UTF-8 encoding for file output
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(UnicodeFormatter(formatter._fmt, formatter.datefmt))
        logger.addHandler(file_handler)
    
    return logger


class UnicodeFormatter(logging.Formatter):
    """
    Custom formatter that handles Unicode characters safely.
    
    Sanitizes Unicode text in log messages to prevent encoding errors
    on systems with limited Unicode support.
    """
    
    def format(self, record):
        try:
            # Get the original formatted message
            formatted = super().format(record)
            # Sanitize Unicode characters
            return sanitize_unicode_text(formatted)
        except Exception as e:
            # If formatting fails, return a safe fallback
            safe_msg = sanitize_unicode_text(str(record.getMessage()))
            safe_name = sanitize_unicode_text(str(record.name))
            safe_level = sanitize_unicode_text(str(record.levelname))
            return f"{safe_name} - {safe_level} - {safe_msg} [Unicode formatting error: {str(e)}]"


def get_logger(name: str = "ha_voice_assistant") -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


def safe_log(logger, level, message, *args, **kwargs):
    """
    Safely log a message with Unicode handling.
    
    Args:
        logger: Logger instance
        level: Log level (info, debug, warning, error, critical)
        message: Message template
        *args: Arguments for message formatting
        **kwargs: Keyword arguments for message formatting
    """
    try:
        # Sanitize all arguments
        safe_args = [sanitize_unicode_text(str(arg)) for arg in args]
        safe_kwargs = {k: sanitize_unicode_text(str(v)) for k, v in kwargs.items()}
        safe_message = sanitize_unicode_text(str(message))
        
        # Get the log method
        log_method = getattr(logger, level.lower())
        
        # Log with sanitized arguments
        if safe_args or safe_kwargs:
            log_method(safe_message, *safe_args, **safe_kwargs)
        else:
            log_method(safe_message)
            
    except Exception as e:
        # Fallback logging if all else fails
        fallback_msg = f"Logging error: {str(e)} | Original: {sanitize_unicode_text(str(message))}"
        logger.error(fallback_msg)