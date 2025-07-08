"""
Text processing utilities for Unicode handling and text sanitization
"""
import unicodedata
import re


def sanitize_unicode_text(text: str, fallback_encoding: str = "ascii") -> str:
    """
    Sanitize Unicode text to ensure compatibility across different environments.
    
    Converts problematic Unicode characters to ASCII equivalents and handles
    encoding issues that may occur on systems with limited Unicode support.
    
    Args:
        text: Input text that may contain Unicode characters
        fallback_encoding: Encoding to fall back to if Unicode handling fails
        
    Returns:
        Sanitized text safe for logging and display
    """
    if not isinstance(text, str):
        return str(text)
    
    try:
        # First, try to normalize Unicode characters
        normalized = unicodedata.normalize('NFKC', text)
        
        # Replace common problematic Unicode characters with ASCII equivalents
        replacements = {
            # Smart quotes
            '\u201c': '"',  # Left double quotation mark
            '\u201d': '"',  # Right double quotation mark
            '\u2018': "'",  # Left single quotation mark
            '\u2019': "'",  # Right single quotation mark
            '\u201a': "'",  # Single low-9 quotation mark
            '\u201e': '"',  # Double low-9 quotation mark
            
            # Dashes
            '\u2013': '-',  # En dash
            '\u2014': '--', # Em dash
            '\u2015': '--', # Horizontal bar
            
            # Other common characters
            '\u2026': '...', # Horizontal ellipsis
            '\u00a0': ' ',   # Non-breaking space
            '\u2022': '*',   # Bullet point
            '\u2023': '>',   # Triangular bullet
            
            # Degree and other symbols
            '\u00b0': 'deg', # Degree symbol
            '\u00b1': '+/-', # Plus-minus sign
            '\u00d7': 'x',   # Multiplication sign
            '\u00f7': '/',   # Division sign
        }
        
        # Apply replacements
        for unicode_char, ascii_replacement in replacements.items():
            normalized = normalized.replace(unicode_char, ascii_replacement)
        
        # Try to encode/decode to catch any remaining problematic characters
        try:
            # Test if the text can be encoded in the fallback encoding
            normalized.encode(fallback_encoding)
            return normalized
        except UnicodeEncodeError:
            # If we still have problematic characters, use more aggressive fallbacks
            return _aggressive_unicode_fallback(normalized, fallback_encoding)
            
    except Exception:
        # If all else fails, use aggressive fallback
        return _aggressive_unicode_fallback(text, fallback_encoding)


def _aggressive_unicode_fallback(text: str, encoding: str = "ascii") -> str:
    """
    Aggressive fallback for Unicode text that can't be handled normally.
    
    This method will replace or remove any characters that can't be encoded
    in the target encoding.
    """
    try:
        # Try to encode with error handling
        encoded = text.encode(encoding, errors='replace')
        decoded = encoded.decode(encoding)
        
        # Clean up any replacement characters
        decoded = decoded.replace('\ufffd', '?')  # Unicode replacement character
        
        return decoded
    except Exception:
        # Last resort: keep only ASCII characters
        return ''.join(char for char in text if ord(char) < 128)


def clean_entity_name(entity_name: str) -> str:
    """
    Clean entity names for safe logging and display.
    
    Specifically designed for Home Assistant entity names that may contain
    Unicode characters in friendly names.
    
    Args:
        entity_name: Entity name or friendly name from Home Assistant
        
    Returns:
        Cleaned name safe for logging
    """
    if not entity_name:
        return entity_name
    
    # Sanitize Unicode
    cleaned = sanitize_unicode_text(entity_name)
    
    # Additional cleaning for entity names
    # Remove any remaining problematic characters
    cleaned = re.sub(r'[^\w\s\-_.,()[\]{}]', '', cleaned)
    
    # Normalize whitespace
    cleaned = ' '.join(cleaned.split())
    
    return cleaned


def safe_format_log_message(template: str, *args, **kwargs) -> str:
    """
    Safely format log messages with Unicode arguments.
    
    Ensures that all arguments are Unicode-safe before formatting.
    
    Args:
        template: Log message template
        *args: Positional arguments for formatting
        **kwargs: Keyword arguments for formatting
        
    Returns:
        Safely formatted log message
    """
    try:
        # Sanitize all arguments
        safe_args = [sanitize_unicode_text(str(arg)) for arg in args]
        safe_kwargs = {k: sanitize_unicode_text(str(v)) for k, v in kwargs.items()}
        
        # Format the message
        return template.format(*safe_args, **safe_kwargs)
    except Exception as e:
        # If formatting fails, return a safe fallback
        return f"Log formatting error: {sanitize_unicode_text(str(e))} | Template: {sanitize_unicode_text(template)}"


# Convenience function for common use case
def safe_str(obj) -> str:
    """Convert any object to a Unicode-safe string."""
    return sanitize_unicode_text(str(obj))