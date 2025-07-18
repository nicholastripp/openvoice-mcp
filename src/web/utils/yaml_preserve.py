"""
YAML configuration updater that preserves comments and formatting
"""
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class YamlPreserver:
    """Updates YAML files while preserving comments, formatting, and structure"""
    
    def __init__(self, template_path: Path):
        """
        Initialize with a template file (config.yaml.example)
        
        Args:
            template_path: Path to the template YAML file with all settings and comments
        """
        self.template_path = template_path
        self._load_template()
    
    def _load_template(self):
        """Load the template file content"""
        if not self.template_path.exists():
            raise FileNotFoundError(f"Template file not found: {self.template_path}")
        
        with open(self.template_path, 'r') as f:
            self.template_lines = f.readlines()
    
    def update_config(self, config_path: Path, updates: Dict[str, Any]) -> None:
        """
        Update a config file with new values while preserving structure
        
        Args:
            config_path: Path to the config file to update
            updates: Dictionary of updates (can be nested)
        """
        # If config doesn't exist, copy from template
        if not config_path.exists():
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.template_path, 'r') as src, open(config_path, 'w') as dst:
                dst.write(src.read())
        
        # Read current config
        with open(config_path, 'r') as f:
            lines = f.readlines()
        
        # Update lines with new values
        updated_lines = self._update_lines(lines, updates)
        
        # Write back
        with open(config_path, 'w') as f:
            f.writelines(updated_lines)
        
        logger.info(f"Updated config file: {config_path}")
    
    def _update_lines(self, lines: List[str], updates: Dict[str, Any], 
                      parent_key: str = "") -> List[str]:
        """
        Update YAML lines with new values
        
        Args:
            lines: Original lines from the file
            updates: Dictionary of updates
            parent_key: Parent key for nested updates (e.g., "audio.")
        """
        updated_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                updated_lines.append(line)
                i += 1
                continue
            
            # Check if this line contains a key we need to update
            updated = False
            for key, value in updates.items():
                full_key = parent_key + key
                
                if isinstance(value, dict):
                    # Handle nested dictionaries
                    if self._is_section_header(line, key):
                        logger.debug(f"Found section header '{key}' at line {i}")
                        updated_lines.append(line)
                        # Find the end of this section and update it
                        section_lines, section_end = self._extract_section(lines[i+1:])
                        logger.debug(f"Section '{key}' spans {len(section_lines)} lines, ends at offset {section_end}")
                        updated_section = self._update_lines(
                            section_lines, value, full_key + "."
                        )
                        updated_lines.extend(updated_section)
                        i += section_end + 1
                        updated = True
                        break
                else:
                    # Handle simple key-value pairs
                    if self._is_key_value_line(line, key):
                        # Preserve indentation and update value
                        indent = len(line) - len(line.lstrip())
                        comment_match = re.search(r'#.*$', line)
                        comment = comment_match.group() if comment_match else ""
                        
                        # Format the value appropriately
                        formatted_value = self._format_value(value)
                        
                        # Build the updated line
                        updated_line = " " * indent + f"{key}: {formatted_value}"
                        if comment:
                            # Preserve inline comments with proper spacing
                            updated_line += "  " + comment
                        updated_line += "\n"
                        
                        updated_lines.append(updated_line)
                        updated = True
                        break
            
            if not updated:
                updated_lines.append(line)
            
            i += 1
        
        return updated_lines
    
    def _is_section_header(self, line: str, key: str) -> bool:
        """Check if a line is a section header for the given key"""
        stripped = line.strip()
        return bool(re.match(f"^{re.escape(key)}:\\s*$", stripped))
    
    def _is_key_value_line(self, line: str, key: str) -> bool:
        """Check if a line contains a key-value pair for the given key"""
        stripped = line.strip()
        return bool(re.match(f"^{re.escape(key)}:\\s*", stripped))
    
    def _extract_section(self, lines: List[str]) -> Tuple[List[str], int]:
        """
        Extract lines belonging to a YAML section
        
        Returns:
            Tuple of (section_lines, end_index)
        """
        if not lines:
            return [], 0
        
        # Determine the indentation level of the section
        first_content_line = None
        for line in lines:
            if line.strip() and not line.strip().startswith('#'):
                first_content_line = line
                break
        
        if not first_content_line:
            return lines, len(lines)
        
        section_indent = len(first_content_line) - len(first_content_line.lstrip())
        section_lines = []
        
        for i, line in enumerate(lines):
            # Skip empty lines and comments at the beginning
            if i == 0 or not line.strip() or line.strip().startswith('#'):
                section_lines.append(line)
                continue
            
            # Check indentation for content lines
            line_indent = len(line) - len(line.lstrip())
            
            # For top-level sections (indent 0), check if this is a new section
            if section_indent == 0 and line_indent == 0:
                # Check if this line is a new section header (ends with ':')
                stripped = line.strip()
                if ':' in stripped and not any(c in stripped.split(':')[0] for c in ['"', "'", ' ']):
                    # This is a new top-level section, don't include it
                    return section_lines, i
            
            # For any indent level, if we hit a line with less indentation, we're done
            elif line_indent < section_indent:
                return section_lines, i
            
            section_lines.append(line)
        
        return section_lines, len(lines)
    
    def _format_value(self, value: Any) -> str:
        """Format a value for YAML output"""
        if isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, str):
            # Handle special OpenAI voice values with spaces
            if value in ['hey google', 'hey siri', 'ok google']:
                return f'"{value}"'
            # Quote strings that need it
            elif any(c in value for c in [':', '#', '@', '|', '>', '-', '*', '&', '!', '%', '`', '?', '"', "'"]):
                # Use double quotes and escape any internal quotes
                escaped = value.replace('"', '\\"')
                return f'"{escaped}"'
            # Check if string looks like a number or boolean
            elif value.lower() in ['true', 'false', 'yes', 'no', 'on', 'off']:
                return f'"{value}"'
            elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                return f'"{value}"'
            else:
                return value
        elif isinstance(value, (int, float)):
            return str(value)
        elif value is None:
            return "null"
        else:
            # For complex types, convert to string
            return str(value)
    
    def merge_with_template(self, config_path: Path, updates: Dict[str, Any]) -> None:
        """
        Merge updates with template, ensuring all template settings are present
        
        This is useful for ensuring config files have all required settings
        """
        # Start with template
        lines = self.template_lines.copy()
        
        # Apply updates
        updated_lines = self._update_lines(lines, updates)
        
        # Write result
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            f.writelines(updated_lines)
        
        logger.info(f"Merged config with template: {config_path}")