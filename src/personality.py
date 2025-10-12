"""
Personality system for OpenVoice MCP
Based on Billy B-Assistant personality framework
"""
import configparser
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PersonalityTraits:
    """Personality traits configuration"""
    helpfulness: int = 90
    humor: int = 30
    formality: int = 50
    patience: int = 85
    verbosity: int = 60
    warmth: int = 70
    curiosity: int = 40
    confidence: int = 80
    optimism: int = 75
    respectfulness: int = 95


@dataclass
class Backstory:
    """Assistant backstory and identity"""
    name: str = "Home Assistant"
    role: str = "helpful home automation assistant"
    personality: str = "friendly and efficient"
    origin: str = "I am your Home Assistant voice companion"
    purpose: str = "I help you control your smart home devices through natural conversation"
    specialties: str = "home automation, device control, status queries"


@dataclass
class MetaConfig:
    """Meta configuration for personality"""
    instructions: str = ""
    context: str = "You are integrated with Home Assistant and can control smart home devices, check device states, and answer questions about the home."
    style_notes: str = "Keep responses concise but friendly. Always confirm actions you take. Be helpful and conversational."


class PersonalityProfile:
    """Manages personality configuration and prompt generation"""
    
    def __init__(self, persona_file: str = "config/persona.ini"):
        self.persona_file = Path(persona_file)
        self.traits = PersonalityTraits()
        self.backstory = Backstory()
        self.meta = MetaConfig()
        
        if self.persona_file.exists():
            self.load_from_file()
    
    def load_from_file(self) -> None:
        """Load personality from INI file"""
        config = configparser.ConfigParser()
        config.read(self.persona_file)
        
        # Load personality traits
        if 'PERSONALITY' in config:
            personality_data = dict(config['PERSONALITY'])
            # Convert string values to integers
            for key, value in personality_data.items():
                if hasattr(self.traits, key):
                    setattr(self.traits, key, int(value))
        
        # Load backstory
        if 'BACKSTORY' in config:
            backstory_data = dict(config['BACKSTORY'])
            for key, value in backstory_data.items():
                if hasattr(self.backstory, key):
                    setattr(self.backstory, key, value)
        
        # Load meta configuration
        if 'META' in config:
            meta_data = dict(config['META'])
            for key, value in meta_data.items():
                if hasattr(self.meta, key):
                    setattr(self.meta, key, value)
    
    def save_to_file(self) -> None:
        """Save personality to INI file"""
        config = configparser.ConfigParser()
        
        # Save personality traits
        config['PERSONALITY'] = {}
        for key, value in self.traits.__dict__.items():
            config['PERSONALITY'][key] = str(value)
        
        # Save backstory
        config['BACKSTORY'] = {}
        for key, value in self.backstory.__dict__.items():
            config['BACKSTORY'][key] = value
        
        # Save meta configuration
        config['META'] = {}
        for key, value in self.meta.__dict__.items():
            config['META'][key] = value
        
        # Create directory if it doesn't exist
        self.persona_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.persona_file, 'w') as f:
            config.write(f)
    
    def generate_prompt(self) -> str:
        """
        Generate OpenAI system prompt based on personality traits.
        
        Returns:
            Generated system prompt string
        """
        # Use custom instructions if provided
        if self.meta.instructions.strip():
            base_prompt = self.meta.instructions
        else:
            # Generate prompt based on personality traits
            base_prompt = self._generate_base_prompt()
        
        # Add context and style notes
        full_prompt = f"{base_prompt}\n\n{self.meta.context}"
        if self.meta.style_notes:
            full_prompt += f"\n\nStyle guidelines: {self.meta.style_notes}"
        
        return full_prompt
    
    def _generate_base_prompt(self) -> str:
        """Generate base prompt from personality traits"""
        traits = self.traits
        backstory = self.backstory
        
        # Start with basic identity
        prompt = f"You are {backstory.name}, {backstory.role}. {backstory.personality}."
        
        # Add personality modifiers based on traits
        personality_parts = []
        
        if traits.helpfulness >= 80:
            personality_parts.append("extremely helpful and accommodating")
        elif traits.helpfulness >= 60:
            personality_parts.append("helpful")
        
        if traits.warmth >= 70:
            personality_parts.append("warm and friendly")
        elif traits.warmth >= 40:
            personality_parts.append("friendly")
        
        if traits.humor >= 60:
            personality_parts.append("with a good sense of humor")
        elif traits.humor >= 30:
            personality_parts.append("occasionally humorous")
        
        if traits.patience >= 80:
            personality_parts.append("very patient")
        elif traits.patience >= 60:
            personality_parts.append("patient")
        
        if traits.confidence >= 80:
            personality_parts.append("confident")
        
        if personality_parts:
            prompt += f" You are {', '.join(personality_parts)}."
        
        # Add communication style
        if traits.formality <= 30:
            prompt += " You speak in a casual, relaxed manner."
        elif traits.formality >= 70:
            prompt += " You speak in a formal, professional manner."
        else:
            prompt += " You speak in a conversational, approachable manner."
        
        if traits.verbosity <= 40:
            prompt += " You keep your responses brief and to the point."
        elif traits.verbosity >= 70:
            prompt += " You provide detailed, comprehensive responses."
        else:
            prompt += " You provide appropriate detail in your responses."
        
        # Add purpose and specialties
        prompt += f" {backstory.purpose}"
        if backstory.specialties:
            prompt += f" You specialize in {backstory.specialties}."
        
        return prompt
    
    def update_trait(self, trait_name: str, value: int) -> bool:
        """
        Update a personality trait.
        
        Args:
            trait_name: Name of the trait to update
            value: New value (0-100)
            
        Returns:
            True if successful, False if trait doesn't exist
        """
        if hasattr(self.traits, trait_name) and 0 <= value <= 100:
            setattr(self.traits, trait_name, value)
            return True
        return False
    
    def get_trait_summary(self) -> Dict[str, int]:
        """Get summary of all personality traits"""
        return self.traits.__dict__.copy()