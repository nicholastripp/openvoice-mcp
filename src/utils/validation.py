"""
Input validation utilities for security
"""
import re
import os
from typing import Optional, List, Any, Dict
from pathlib import Path


class InputValidator:
    """Centralized input validation for security-critical operations"""
    
    @staticmethod
    def validate_hostname(hostname: str, max_length: int = 253) -> str:
        """
        Validate hostname for use in certificates and network operations.
        
        Args:
            hostname: The hostname to validate
            max_length: Maximum allowed length (default 253 for DNS)
            
        Returns:
            Validated hostname
            
        Raises:
            ValueError: If hostname is invalid
        """
        if not hostname or len(hostname) > max_length:
            raise ValueError(f"Invalid hostname length: {len(hostname) if hostname else 0}")
        
        # RFC 1123 compliant hostname pattern
        pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9\-\.]{0,251}[a-zA-Z0-9])?$'
        if not re.match(pattern, hostname):
            raise ValueError(f"Invalid hostname format")
        
        # Check for consecutive special characters
        if '..' in hostname or '--' in hostname or '.-' in hostname or '-.' in hostname:
            raise ValueError("Invalid hostname: consecutive special characters")
        
        # Check each label in the hostname
        labels = hostname.split('.')
        for label in labels:
            if len(label) > 63:
                raise ValueError(f"Hostname label too long: {label}")
            if label.startswith('-') or label.endswith('-'):
                raise ValueError(f"Invalid hostname label: {label}")
        
        return hostname
    
    @staticmethod
    def validate_filename(filename: str, allowed_extensions: Optional[List[str]] = None,
                         max_length: int = 255) -> str:
        """
        Validate and sanitize filename to prevent path traversal.
        
        Args:
            filename: The filename to validate
            allowed_extensions: List of allowed file extensions (with dots)
            max_length: Maximum filename length
            
        Returns:
            Sanitized filename
            
        Raises:
            ValueError: If filename is invalid
        """
        if not filename:
            raise ValueError("Filename cannot be empty")
        
        # First check for path traversal patterns BEFORE extracting basename
        if '..' in filename or '../' in filename or '..\\' in filename:
            raise ValueError("Invalid filename: contains directory traversal pattern")
        
        # Check for absolute paths
        if filename.startswith('/') or filename.startswith('\\'):
            raise ValueError("Invalid filename: absolute paths not allowed")
        
        # Check for Windows drive letters
        if len(filename) > 1 and filename[1] == ':':
            raise ValueError("Invalid filename: Windows drive paths not allowed")
        
        # Extract basename to remove any directory components
        base_name = os.path.basename(filename)
        
        # After extraction, ensure we still have a valid filename
        if not base_name or base_name != filename:
            raise ValueError("Invalid filename: contains path separators")
        
        # Reject hidden files
        if base_name.startswith('.'):
            raise ValueError("Invalid filename: hidden files not allowed")
        
        # Check for null bytes and other control characters
        if '\x00' in base_name:
            raise ValueError("Invalid filename: contains null byte")
        
        # Check for other dangerous characters (tabs, newlines, etc.)
        if any(ord(char) < 32 for char in base_name):
            raise ValueError("Invalid filename: contains control characters")
        
        # Check file extension if specified
        if allowed_extensions:
            if not any(base_name.endswith(ext) for ext in allowed_extensions):
                raise ValueError(f"Invalid file extension. Allowed: {allowed_extensions}")
        
        # Remove potentially dangerous characters
        # Allow only alphanumeric, dots, hyphens, underscores
        safe_name = re.sub(r'[^a-zA-Z0-9\.\-_]', '_', base_name)
        
        # Ensure the name isn't just dots and underscores
        if re.match(r'^[\._]+$', safe_name):
            raise ValueError("Invalid filename: no valid characters")
        
        # Limit length
        if len(safe_name) > max_length:
            # Preserve extension if possible
            name_part, ext = os.path.splitext(safe_name)
            max_name_length = max_length - len(ext)
            if max_name_length > 0:
                safe_name = name_part[:max_name_length] + ext
            else:
                safe_name = safe_name[:max_length]
        
        return safe_name
    
    @staticmethod
    def validate_path_within_directory(file_path: Path, allowed_directory: Path) -> Path:
        """
        Validate that a file path is within an allowed directory.
        
        Args:
            file_path: The file path to validate
            allowed_directory: The directory that should contain the file
            
        Returns:
            Resolved file path
            
        Raises:
            ValueError: If path escapes the allowed directory
        """
        # Resolve both paths to absolute paths
        resolved_file = file_path.resolve()
        resolved_dir = allowed_directory.resolve()
        
        # Check if the file is within the allowed directory
        try:
            resolved_file.relative_to(resolved_dir)
        except ValueError:
            raise ValueError(f"Path traversal detected: file is outside allowed directory")
        
        return resolved_file
    
    @staticmethod
    def validate_alphanumeric(value: str, max_length: int = 255,
                            allow_spaces: bool = False,
                            allow_special: str = "") -> str:
        """
        Validate alphanumeric input with optional allowed characters.
        
        Args:
            value: The string to validate
            max_length: Maximum allowed length
            allow_spaces: Whether to allow spaces
            allow_special: Additional special characters to allow
            
        Returns:
            Validated string
            
        Raises:
            ValueError: If input is invalid
        """
        if not value or len(value) > max_length:
            raise ValueError(f"Invalid input length: {len(value) if value else 0}")
        
        # Build pattern based on allowed characters
        pattern_parts = ['a-zA-Z0-9']
        if allow_spaces:
            pattern_parts.append(r'\s')
        if allow_special:
            # Escape special regex characters
            escaped_special = re.escape(allow_special)
            pattern_parts.append(escaped_special)
        
        pattern = f'^[{"".join(pattern_parts)}]+$'
        
        if not re.match(pattern, value):
            raise ValueError("Input contains invalid characters")
        
        return value
    
    @staticmethod
    def validate_command_argument(arg: str, max_length: int = 1024) -> str:
        """
        Validate a command-line argument for subprocess execution.
        
        Args:
            arg: The argument to validate
            max_length: Maximum allowed length
            
        Returns:
            Validated argument
            
        Raises:
            ValueError: If argument contains dangerous patterns
        """
        if not arg:
            return arg
        
        if len(arg) > max_length:
            raise ValueError(f"Argument too long: {len(arg)} characters")
        
        # Check for shell metacharacters and injection attempts
        dangerous_patterns = [
            ';', '|', '&', '$', '`', '\n', '\r', 
            '$(', '${', '<(', '>(', 
            '&&', '||', '>>', '<<',
            '../', '..\\', 
        ]
        
        for pattern in dangerous_patterns:
            if pattern in arg:
                raise ValueError(f"Argument contains dangerous pattern: {pattern}")
        
        # Check for null bytes
        if '\x00' in arg:
            raise ValueError("Argument contains null byte")
        
        return arg
    
    @staticmethod
    def sanitize_json_input(data: Any, max_depth: int = 10) -> Any:
        """
        Recursively sanitize JSON input to prevent injection attacks.
        
        Args:
            data: The JSON data to sanitize
            max_depth: Maximum nesting depth allowed
            
        Returns:
            Sanitized data
            
        Raises:
            ValueError: If data contains suspicious patterns
        """
        def _sanitize(obj: Any, depth: int = 0) -> Any:
            if depth > max_depth:
                raise ValueError(f"JSON nesting too deep: {depth}")
            
            if isinstance(obj, dict):
                sanitized = {}
                for key, value in obj.items():
                    # Validate dictionary keys
                    if not isinstance(key, str):
                        raise ValueError(f"Non-string dictionary key: {type(key)}")
                    if len(key) > 256:
                        raise ValueError(f"Dictionary key too long: {len(key)}")
                    # Recursively sanitize values
                    sanitized[key] = _sanitize(value, depth + 1)
                return sanitized
            
            elif isinstance(obj, list):
                return [_sanitize(item, depth + 1) for item in obj]
            
            elif isinstance(obj, str):
                # Check for suspicious patterns in strings
                if len(obj) > 10000:  # Reasonable limit for string values
                    raise ValueError(f"String value too long: {len(obj)}")
                # Remove null bytes
                if '\x00' in obj:
                    obj = obj.replace('\x00', '')
                return obj
            
            elif isinstance(obj, (int, float, bool, type(None))):
                return obj
            
            else:
                # Reject unknown types
                raise ValueError(f"Unsupported JSON type: {type(obj)}")
        
        return _sanitize(data)
    
    @staticmethod
    def validate_url(url: str, allowed_schemes: Optional[List[str]] = None,
                    allowed_hosts: Optional[List[str]] = None) -> str:
        """
        Validate URL for safety.
        
        Args:
            url: The URL to validate
            allowed_schemes: List of allowed URL schemes (default: ['http', 'https'])
            allowed_hosts: List of allowed hostnames/domains
            
        Returns:
            Validated URL
            
        Raises:
            ValueError: If URL is invalid or not allowed
        """
        from urllib.parse import urlparse
        
        if not url:
            raise ValueError("URL cannot be empty")
        
        if allowed_schemes is None:
            allowed_schemes = ['http', 'https']
        
        try:
            parsed = urlparse(url)
        except Exception as e:
            raise ValueError(f"Invalid URL format: {e}")
        
        # Check scheme
        if parsed.scheme not in allowed_schemes:
            raise ValueError(f"URL scheme not allowed: {parsed.scheme}")
        
        # Check for missing host
        if not parsed.netloc:
            raise ValueError("URL missing host")
        
        # Check against allowed hosts if specified
        if allowed_hosts:
            hostname = parsed.netloc.split(':')[0].lower()
            if not any(hostname == h.lower() or hostname.endswith('.' + h.lower()) 
                      for h in allowed_hosts):
                raise ValueError(f"Host not allowed: {hostname}")
        
        # Check for localhost/private IPs (often a security risk)
        private_patterns = ['localhost', '127.0.0.1', '0.0.0.0', '::1', '169.254.']
        hostname = parsed.netloc.split(':')[0].lower()
        if any(pattern in hostname for pattern in private_patterns):
            # This might be intentional, so just log a warning
            pass  # Consider logging this
        
        return url


# Convenience functions for common validations
def validate_safe_filename(filename: str, extension: Optional[str] = None) -> str:
    """Convenience function to validate a filename with optional extension check."""
    extensions = [extension] if extension else None
    return InputValidator.validate_filename(filename, extensions)


def validate_safe_path(path: str, base_dir: str) -> Path:
    """Convenience function to validate a path is within a base directory."""
    file_path = Path(path)
    base_path = Path(base_dir)
    return InputValidator.validate_path_within_directory(file_path, base_path)


def sanitize_for_log(message: str, max_length: int = 1000) -> str:
    """Sanitize a string for safe logging (prevent log injection)."""
    if not message:
        return ""
    
    # Truncate if too long
    if len(message) > max_length:
        message = message[:max_length] + "..."
    
    # Remove newlines and carriage returns to prevent log injection
    message = message.replace('\n', '\\n').replace('\r', '\\r')
    
    # Remove null bytes
    message = message.replace('\x00', '')
    
    return message