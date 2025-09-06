"""
Voice management system for OpenAI Realtime API.
Handles voice selection, compatibility, and characteristics for different models.
"""
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class VoiceCharacteristic(Enum):
    """Voice characteristic categories"""
    GENDER = "gender"
    TONE = "tone"
    STYLE = "style"
    ACCENT = "accent"


@dataclass
class VoiceProfile:
    """Profile for each available voice"""
    name: str
    gender: str
    tone: str
    style: str
    accent: str
    models: List[str]  # Models that support this voice
    description: str
    recommended_for: List[str]


class VoiceManager:
    """Manages voice selection and compatibility for OpenAI Realtime models"""
    
    # Voice profiles with characteristics
    VOICE_PROFILES = {
        "alloy": VoiceProfile(
            name="alloy",
            gender="neutral",
            tone="neutral",
            style="balanced",
            accent="american",
            models=["gpt-realtime", "gpt-4o-realtime-preview", "gpt-4o-mini-realtime-preview"],
            description="Neutral and balanced voice suitable for general use",
            recommended_for=["general", "information", "assistance"]
        ),
        "ash": VoiceProfile(
            name="ash",
            gender="neutral",
            tone="calm",
            style="soft",
            accent="american",
            models=["gpt-realtime", "gpt-4o-realtime-preview", "gpt-4o-mini-realtime-preview"],
            description="Calm and soft-spoken voice",
            recommended_for=["meditation", "relaxation", "gentle_guidance"]
        ),
        "ballad": VoiceProfile(
            name="ballad",
            gender="neutral",
            tone="warm",
            style="expressive",
            accent="american",
            models=["gpt-realtime", "gpt-4o-realtime-preview", "gpt-4o-mini-realtime-preview"],
            description="Warm and expressive voice with storytelling quality",
            recommended_for=["storytelling", "narration", "engaging_content"]
        ),
        "coral": VoiceProfile(
            name="coral",
            gender="feminine",
            tone="friendly",
            style="conversational",
            accent="american",
            models=["gpt-realtime", "gpt-4o-realtime-preview", "gpt-4o-mini-realtime-preview"],
            description="Friendly and conversational feminine voice",
            recommended_for=["conversation", "customer_service", "friendly_assistance"]
        ),
        "echo": VoiceProfile(
            name="echo",
            gender="masculine",
            tone="smooth",
            style="professional",
            accent="american",
            models=["gpt-realtime", "gpt-4o-realtime-preview", "gpt-4o-mini-realtime-preview"],
            description="Smooth and professional masculine voice",
            recommended_for=["business", "professional", "formal_content"]
        ),
        "sage": VoiceProfile(
            name="sage",
            gender="neutral",
            tone="wise",
            style="thoughtful",
            accent="american",
            models=["gpt-realtime", "gpt-4o-realtime-preview", "gpt-4o-mini-realtime-preview"],
            description="Wise and thoughtful voice with depth",
            recommended_for=["education", "guidance", "thoughtful_content"]
        ),
        "shimmer": VoiceProfile(
            name="shimmer",
            gender="feminine",
            tone="energetic",
            style="upbeat",
            accent="american",
            models=["gpt-realtime", "gpt-4o-realtime-preview", "gpt-4o-mini-realtime-preview"],
            description="Energetic and upbeat feminine voice",
            recommended_for=["entertainment", "energy", "positive_content"]
        ),
        "verse": VoiceProfile(
            name="verse",
            gender="masculine",
            tone="dynamic",
            style="versatile",
            accent="american",
            models=["gpt-realtime", "gpt-4o-realtime-preview", "gpt-4o-mini-realtime-preview"],
            description="Dynamic and versatile masculine voice",
            recommended_for=["varied_content", "adaptable", "multi_purpose"]
        ),
        # New voices exclusive to gpt-realtime
        "cedar": VoiceProfile(
            name="cedar",
            gender="masculine",
            tone="rich",
            style="authoritative",
            accent="american",
            models=["gpt-realtime"],
            description="Rich and authoritative masculine voice with excellent clarity",
            recommended_for=["announcements", "authority", "clear_communication"]
        ),
        "marin": VoiceProfile(
            name="marin",
            gender="feminine",
            tone="crisp",
            style="articulate",
            accent="american",
            models=["gpt-realtime"],
            description="Crisp and articulate feminine voice with precise pronunciation",
            recommended_for=["precision", "clarity", "technical_content"]
        )
    }
    
    def __init__(self, config: Any, logger: Optional[logging.Logger] = None):
        """
        Initialize voice manager.
        
        Args:
            config: Application configuration object
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.current_voice = None
        self.voice_history = []
        
    def get_available_voices(self, model: str) -> List[str]:
        """
        Get list of available voices for a specific model.
        
        Args:
            model: Model name
            
        Returns:
            List of available voice names
        """
        available = []
        for voice_name, profile in self.VOICE_PROFILES.items():
            if model in profile.models:
                available.append(voice_name)
        return available
    
    def get_voice_profile(self, voice: str) -> Optional[VoiceProfile]:
        """
        Get profile for a specific voice.
        
        Args:
            voice: Voice name
            
        Returns:
            VoiceProfile or None if not found
        """
        return self.VOICE_PROFILES.get(voice)
    
    def is_voice_compatible(self, voice: str, model: str) -> bool:
        """
        Check if a voice is compatible with a model.
        
        Args:
            voice: Voice name
            model: Model name
            
        Returns:
            True if compatible, False otherwise
        """
        profile = self.get_voice_profile(voice)
        if profile:
            return model in profile.models
        return False
    
    def select_voice(self, preferred: str, model: str, 
                    use_case: Optional[str] = None) -> str:
        """
        Select appropriate voice based on preferences and compatibility.
        
        Args:
            preferred: Preferred voice name
            model: Model name
            use_case: Optional use case for recommendation
            
        Returns:
            Selected voice name
        """
        # Check if preferred voice is available
        if self.is_voice_compatible(preferred, model):
            self.logger.info(f"Using preferred voice: {preferred}")
            self.current_voice = preferred
            return preferred
        
        self.logger.warning(
            f"Voice '{preferred}' not compatible with model '{model}'"
        )
        
        # Try fallback voice
        fallback = self.config.voice_fallback
        if fallback and self.is_voice_compatible(fallback, model):
            self.logger.info(f"Using fallback voice: {fallback}")
            self.current_voice = fallback
            return fallback
        
        # Select based on use case if provided
        if use_case:
            recommended = self.get_recommended_voice(model, use_case)
            if recommended:
                self.logger.info(
                    f"Using recommended voice for '{use_case}': {recommended}"
                )
                self.current_voice = recommended
                return recommended
        
        # Default to first available voice
        available = self.get_available_voices(model)
        if available:
            default = available[0]
            self.logger.info(f"Using default voice: {default}")
            self.current_voice = default
            return default
        
        # Fallback to alloy (should always be available)
        self.logger.error("No compatible voices found, using 'alloy'")
        self.current_voice = "alloy"
        return "alloy"
    
    def get_recommended_voice(self, model: str, use_case: str) -> Optional[str]:
        """
        Get recommended voice for a specific use case.
        
        Args:
            model: Model name
            use_case: Use case identifier
            
        Returns:
            Recommended voice name or None
        """
        use_case_lower = use_case.lower()
        
        for voice_name, profile in self.VOICE_PROFILES.items():
            # Check if voice is compatible with model
            if model not in profile.models:
                continue
            
            # Check if use case matches recommendations
            for recommendation in profile.recommended_for:
                if recommendation.lower() in use_case_lower or \
                   use_case_lower in recommendation.lower():
                    return voice_name
        
        return None
    
    def get_voice_by_characteristics(self, model: str, 
                                    gender: Optional[str] = None,
                                    tone: Optional[str] = None,
                                    style: Optional[str] = None) -> Optional[str]:
        """
        Find voice by characteristics.
        
        Args:
            model: Model name
            gender: Desired gender characteristic
            tone: Desired tone characteristic
            style: Desired style characteristic
            
        Returns:
            Matching voice name or None
        """
        candidates = []
        
        for voice_name, profile in self.VOICE_PROFILES.items():
            # Check model compatibility
            if model not in profile.models:
                continue
            
            # Score based on matching characteristics
            score = 0
            if gender and profile.gender == gender:
                score += 1
            if tone and profile.tone == tone:
                score += 1
            if style and profile.style == style:
                score += 1
            
            if score > 0:
                candidates.append((voice_name, score))
        
        # Return best match
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        return None
    
    def migrate_voice_preference(self, old_model: str, new_model: str, 
                                current_voice: str) -> str:
        """
        Migrate voice preference when switching models.
        
        Args:
            old_model: Previous model name
            new_model: New model name
            current_voice: Currently selected voice
            
        Returns:
            Migrated voice selection
        """
        # If current voice works with new model, keep it
        if self.is_voice_compatible(current_voice, new_model):
            self.logger.info(
                f"Voice '{current_voice}' compatible with new model"
            )
            return current_voice
        
        # Find voice with similar characteristics
        profile = self.get_voice_profile(current_voice)
        if profile:
            similar = self.get_voice_by_characteristics(
                new_model,
                gender=profile.gender,
                tone=profile.tone,
                style=profile.style
            )
            if similar:
                self.logger.info(
                    f"Migrating from '{current_voice}' to similar voice '{similar}'"
                )
                return similar
        
        # Use configured fallback
        return self.select_voice(
            self.config.voice_fallback, 
            new_model
        )
    
    def log_voice_usage(self, voice: str, duration: float, 
                       satisfaction: Optional[float] = None):
        """
        Log voice usage for analytics.
        
        Args:
            voice: Voice name used
            duration: Duration of usage in seconds
            satisfaction: Optional satisfaction score (0-1)
        """
        usage_entry = {
            "voice": voice,
            "duration": duration,
            "satisfaction": satisfaction,
            "timestamp": None  # Would be set to current time
        }
        
        self.voice_history.append(usage_entry)
        
        # Keep only last 100 entries
        if len(self.voice_history) > 100:
            self.voice_history = self.voice_history[-100:]
    
    def get_voice_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about voice usage.
        
        Returns:
            Dictionary of voice usage statistics
        """
        if not self.voice_history:
            return {"message": "No voice usage data available"}
        
        stats = {
            "total_sessions": len(self.voice_history),
            "voice_usage": {},
            "average_duration": 0,
            "average_satisfaction": None
        }
        
        total_duration = 0
        total_satisfaction = 0
        satisfaction_count = 0
        
        for entry in self.voice_history:
            voice = entry["voice"]
            duration = entry["duration"]
            
            # Count voice usage
            if voice not in stats["voice_usage"]:
                stats["voice_usage"][voice] = {
                    "count": 0,
                    "total_duration": 0
                }
            
            stats["voice_usage"][voice]["count"] += 1
            stats["voice_usage"][voice]["total_duration"] += duration
            total_duration += duration
            
            # Track satisfaction
            if entry["satisfaction"] is not None:
                total_satisfaction += entry["satisfaction"]
                satisfaction_count += 1
        
        # Calculate averages
        stats["average_duration"] = total_duration / len(self.voice_history)
        if satisfaction_count > 0:
            stats["average_satisfaction"] = total_satisfaction / satisfaction_count
        
        # Find most used voice
        most_used = max(
            stats["voice_usage"].items(),
            key=lambda x: x[1]["count"]
        )
        stats["most_used_voice"] = most_used[0]
        
        return stats
    
    def validate_voice_configuration(self) -> Dict[str, bool]:
        """
        Validate voice configuration.
        
        Returns:
            Dictionary of validation results
        """
        results = {
            "primary_voice_valid": False,
            "fallback_voice_valid": False,
            "new_voices_available": False
        }
        
        # Get current model
        model = self.config.model
        
        # Check primary voice
        primary = self.config.voice
        results["primary_voice_valid"] = self.is_voice_compatible(primary, model)
        
        # Check fallback voice
        fallback = self.config.voice_fallback
        if fallback:
            results["fallback_voice_valid"] = self.is_voice_compatible(fallback, model)
        
        # Check if new voices are available (cedar, marin)
        available = self.get_available_voices(model)
        results["new_voices_available"] = any(
            v in ["cedar", "marin"] for v in available
        )
        
        return results