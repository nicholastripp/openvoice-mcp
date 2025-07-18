"""
Configuration file management utilities
"""
import asyncio
import configparser
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration files for the web UI"""
    
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.env_path = config_dir.parent / ".env"
        self.yaml_path = config_dir / "config.yaml"
        self.persona_path = config_dir / "persona.ini"
        self.logs_dir = config_dir.parent / "logs"
        
    async def load_env_masked(self) -> Dict[str, str]:
        """Load .env file with masked sensitive values"""
        env_vars = {}
        
        if self.env_path.exists():
            with open(self.env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        # Mask sensitive values
                        if value and key in ['OPENAI_API_KEY', 'HA_TOKEN', 'PICOVOICE_ACCESS_KEY']:
                            env_vars[key] = value[:4] + '...' + value[-4:] if len(value) > 8 else '****'
                        else:
                            env_vars[key] = value
                            
        return env_vars
        
    async def save_env(self, data: Dict[str, str]):
        """Save environment variables to .env file"""
        # Create backup
        if self.env_path.exists():
            backup_path = self.env_path.with_suffix('.env.backup')
            shutil.copy2(self.env_path, backup_path)
            
        # Write new .env file
        with open(self.env_path, 'w') as f:
            f.write("# Home Assistant Voice Assistant Configuration\n")
            f.write(f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for key, value in data.items():
                f.write(f"{key}={value}\n")
                
        logger.info("Saved .env configuration")
        
    async def update_env(self, updates: Dict[str, str]):
        """Update specific environment variables"""
        # Read existing
        existing = {}
        if self.env_path.exists():
            with open(self.env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        existing[key] = value
                        
        # Update with new values (only if not empty)
        for key, value in updates.items():
            if value:  # Only update if value provided
                existing[key] = value
                
        # Save back
        await self.save_env(existing)
        
    async def load_yaml_config(self) -> Dict[str, Any]:
        """Load YAML configuration"""
        if self.yaml_path.exists():
            with open(self.yaml_path, 'r') as f:
                return yaml.safe_load(f) or {}
        
        # Return default structure if file doesn't exist
        return {
            'openai': {'voice': 'alloy', 'temperature': 0.8},
            'audio': {'input_device': 'default', 'output_device': 'default'},
            'wake_word': {'enabled': True, 'sensitivity': 1.0}
        }
        
    async def save_yaml_config(self, data: Dict[str, Any]):
        """Save YAML configuration"""
        # Create backup
        if self.yaml_path.exists():
            backup_path = self.yaml_path.with_suffix('.yaml.backup')
            shutil.copy2(self.yaml_path, backup_path)
            
        # Save with comments
        with open(self.yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            
        logger.info("Saved YAML configuration")
        
    async def load_persona_config(self) -> Dict[str, Any]:
        """Load persona configuration"""
        config = configparser.ConfigParser()
        
        if self.persona_path.exists():
            config.read(self.persona_path)
            
        # Convert to dict
        result = {}
        for section in config.sections():
            result[section] = dict(config.items(section))
            
        # Ensure personality traits are integers
        if 'PERSONALITY' in result:
            for trait, value in result['PERSONALITY'].items():
                try:
                    result['PERSONALITY'][trait] = int(value)
                except ValueError:
                    result['PERSONALITY'][trait] = 50  # Default
                    
        return result
        
    async def save_persona_config(self, data: Dict[str, Any]):
        """Save persona configuration"""
        # Create backup
        if self.persona_path.exists():
            backup_path = self.persona_path.with_suffix('.ini.backup')
            shutil.copy2(self.persona_path, backup_path)
            
        config = configparser.ConfigParser()
        
        # Add sections
        for section, values in data.items():
            config[section] = values
            
        # Write to file
        with open(self.persona_path, 'w') as f:
            config.write(f)
            
        logger.info("Saved persona configuration")
        
    async def read_logs(self, lines: int = 100) -> str:
        """Read last N lines from log file"""
        log_file = self.logs_dir / "voice_assistant.log"
        
        if not log_file.exists():
            return "No logs available"
            
        # Use tail-like approach for efficiency
        with open(log_file, 'rb') as f:
            # Go to end of file
            f.seek(0, 2)
            file_length = f.tell()
            
            # Read backwards to find N lines
            block_size = 1024
            blocks = []
            lines_found = 0
            
            while lines_found < lines and file_length > 0:
                block_start = max(0, file_length - block_size)
                f.seek(block_start)
                block = f.read(min(block_size, file_length))
                blocks.append(block)
                lines_found += block.count(b'\n')
                file_length = block_start
                
            # Join blocks and get last N lines
            text = b''.join(reversed(blocks)).decode('utf-8', errors='ignore')
            return '\n'.join(text.splitlines()[-lines:])