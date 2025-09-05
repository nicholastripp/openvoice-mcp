#!/usr/bin/env python3
"""
Test suite for OpenAI Realtime API model migration.
Tests model compatibility, voice selection, and performance tracking.
"""
import sys
import os
import asyncio
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import OpenAIConfig
from openai_client.model_compatibility import ModelCompatibility, ModelType
from openai_client.voice_manager import VoiceManager
from openai_client.performance_metrics import PerformanceMetrics


class TestModelCompatibility(unittest.TestCase):
    """Test model compatibility and migration logic"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = Mock()
        self.config.openai = Mock()
        self.config.openai.model = "gpt-realtime"
        self.config.openai.legacy_model = "gpt-4o-realtime-preview"
        self.config.openai.model_selection = "auto"
        self.config.openai.voice = "alloy"
        self.config.openai.voice_fallback = "alloy"
        self.config.openai.temperature = 0.8
        
        self.compat = ModelCompatibility(self.config)
    
    def test_model_selection_auto(self):
        """Test automatic model selection"""
        self.config.openai.model_selection = "auto"
        selected = self.compat.select_model()
        self.assertEqual(selected, "gpt-realtime")
    
    def test_model_selection_legacy(self):
        """Test forcing legacy model"""
        self.config.openai.model_selection = "legacy"
        selected = self.compat.select_model()
        self.assertEqual(selected, "gpt-4o-realtime-preview")
    
    def test_model_selection_new(self):
        """Test forcing new model"""
        self.config.openai.model_selection = "new"
        selected = self.compat.select_model()
        self.assertEqual(selected, "gpt-realtime")
    
    def test_voice_availability_new_model(self):
        """Test voice availability for new model"""
        # Standard voices
        self.assertTrue(self.compat.is_voice_available("alloy", "gpt-realtime"))
        self.assertTrue(self.compat.is_voice_available("echo", "gpt-realtime"))
        
        # New exclusive voices
        self.assertTrue(self.compat.is_voice_available("cedar", "gpt-realtime"))
        self.assertTrue(self.compat.is_voice_available("marin", "gpt-realtime"))
    
    def test_voice_availability_legacy_model(self):
        """Test voice availability for legacy model"""
        # Standard voices
        self.assertTrue(self.compat.is_voice_available("alloy", "gpt-4o-realtime-preview"))
        self.assertTrue(self.compat.is_voice_available("echo", "gpt-4o-realtime-preview"))
        
        # New voices should not be available
        self.assertFalse(self.compat.is_voice_available("cedar", "gpt-4o-realtime-preview"))
        self.assertFalse(self.compat.is_voice_available("marin", "gpt-4o-realtime-preview"))
    
    def test_compatible_voice_selection(self):
        """Test getting compatible voice with fallback"""
        # Cedar not available for legacy model, should fallback
        voice = self.compat.get_compatible_voice("cedar", "gpt-4o-realtime-preview")
        self.assertEqual(voice, "alloy")  # Falls back to configured fallback
        
        # Cedar available for new model
        voice = self.compat.get_compatible_voice("cedar", "gpt-realtime")
        self.assertEqual(voice, "cedar")
    
    def test_fallback_detection(self):
        """Test error-based fallback detection"""
        self.config.openai.model_selection = "auto"
        
        # Should trigger fallback
        error1 = Exception("Model not found")
        self.assertTrue(self.compat.should_fallback(error1))
        
        error2 = Exception("Invalid model specified")
        self.assertTrue(self.compat.should_fallback(error2))
        
        # Should not trigger fallback
        error3 = Exception("Network timeout")
        self.assertFalse(self.compat.should_fallback(error3))
        
        # No fallback in legacy mode
        self.config.openai.model_selection = "legacy"
        self.assertFalse(self.compat.should_fallback(error1))
    
    def test_session_config_generation(self):
        """Test session configuration for different models"""
        self.compat.current_model = ModelType.GPT_REALTIME
        config = self.compat.get_session_config("gpt-realtime")
        
        self.assertEqual(config["model"], "gpt-realtime")
        self.assertIn("voice", config)
        self.assertIn("instructions", config)
        self.assertIn("turn_detection", config)
        self.assertEqual(config["input_audio_format"], "pcm16")
        self.assertEqual(config["output_audio_format"], "pcm16")
    
    def test_cost_calculation(self):
        """Test cost calculation for different models"""
        tokens = {"input": 1000, "output": 500}
        
        # New model cost (cheaper)
        cost_new = self.compat.calculate_cost(tokens, "gpt-realtime")
        expected_new = (1000 * 32 / 1_000_000) + (500 * 64 / 1_000_000)
        self.assertAlmostEqual(cost_new, expected_new, places=6)
        
        # Legacy model cost (more expensive)
        cost_legacy = self.compat.calculate_cost(tokens, "gpt-4o-realtime-preview")
        expected_legacy = (1000 * 40 / 1_000_000) + (500 * 80 / 1_000_000)
        self.assertAlmostEqual(cost_legacy, expected_legacy, places=6)
        
        # Verify new model is cheaper
        self.assertLess(cost_new, cost_legacy)
    
    def test_performance_improvements(self):
        """Test performance improvement metrics"""
        improvements = self.compat.get_performance_improvements()
        
        self.assertIn("big_bench_audio", improvements)
        self.assertIn("instruction_following", improvements)
        self.assertIn("function_calling", improvements)
        self.assertIn("cost_reduction", improvements)
        self.assertIn("new_features", improvements)
        
        # Verify improvements are positive
        self.assertEqual(improvements["cost_reduction"]["improvement"], "-20%")
        self.assertEqual(improvements["function_calling"]["improvement"], "+34%")


class TestVoiceManager(unittest.TestCase):
    """Test voice management and migration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = Mock()
        self.config.openai = Mock()
        self.config.openai.voice = "alloy"
        self.config.openai.voice_fallback = "echo"
        
        self.voice_mgr = VoiceManager(self.config)
    
    def test_get_available_voices(self):
        """Test getting available voices for models"""
        # New model should have all voices including cedar and marin
        voices_new = self.voice_mgr.get_available_voices("gpt-realtime")
        self.assertIn("cedar", voices_new)
        self.assertIn("marin", voices_new)
        self.assertIn("alloy", voices_new)
        self.assertEqual(len(voices_new), 10)  # 8 standard + 2 new
        
        # Legacy model should not have new voices
        voices_legacy = self.voice_mgr.get_available_voices("gpt-4o-realtime-preview")
        self.assertNotIn("cedar", voices_legacy)
        self.assertNotIn("marin", voices_legacy)
        self.assertIn("alloy", voices_legacy)
        self.assertEqual(len(voices_legacy), 8)  # 8 standard only
    
    def test_voice_selection_with_fallback(self):
        """Test voice selection with fallback logic"""
        # Select cedar for new model - should work
        voice = self.voice_mgr.select_voice("cedar", "gpt-realtime")
        self.assertEqual(voice, "cedar")
        
        # Select cedar for legacy model - should fallback
        voice = self.voice_mgr.select_voice("cedar", "gpt-4o-realtime-preview")
        self.assertEqual(voice, "echo")  # Configured fallback
    
    def test_voice_recommendation_by_use_case(self):
        """Test voice recommendation based on use case"""
        # Test authority use case
        voice = self.voice_mgr.get_recommended_voice("gpt-realtime", "authority")
        self.assertEqual(voice, "cedar")  # Cedar is recommended for authority
        
        # Test precision use case
        voice = self.voice_mgr.get_recommended_voice("gpt-realtime", "precision")
        self.assertEqual(voice, "marin")  # Marin is recommended for precision
        
        # Test general use case
        voice = self.voice_mgr.get_recommended_voice("gpt-realtime", "general")
        self.assertEqual(voice, "alloy")  # Alloy is recommended for general
    
    def test_voice_by_characteristics(self):
        """Test finding voice by characteristics"""
        # Find masculine, rich voice
        voice = self.voice_mgr.get_voice_by_characteristics(
            "gpt-realtime",
            gender="masculine",
            tone="rich"
        )
        self.assertEqual(voice, "cedar")
        
        # Find feminine, crisp voice
        voice = self.voice_mgr.get_voice_by_characteristics(
            "gpt-realtime",
            gender="feminine",
            tone="crisp"
        )
        self.assertEqual(voice, "marin")
    
    def test_voice_migration(self):
        """Test voice migration between models"""
        # Migrate cedar from new to legacy model
        migrated = self.voice_mgr.migrate_voice_preference(
            "gpt-realtime",
            "gpt-4o-realtime-preview",
            "cedar"
        )
        # Should find similar masculine voice
        self.assertIn(migrated, ["echo", "verse"])  # Both are masculine
        
        # Migrate standard voice - should keep it
        migrated = self.voice_mgr.migrate_voice_preference(
            "gpt-realtime",
            "gpt-4o-realtime-preview",
            "alloy"
        )
        self.assertEqual(migrated, "alloy")  # Compatible with both


class TestPerformanceMetrics(unittest.TestCase):
    """Test performance metrics tracking"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = Mock()
        self.config.openai = Mock()
        
        # Use temp directory for metrics
        self.metrics = PerformanceMetrics(self.config, metrics_dir="/tmp/test_metrics")
    
    def test_session_tracking(self):
        """Test session start and end tracking"""
        # Start session
        session = self.metrics.start_session("test_123", "gpt-realtime")
        self.assertIsNotNone(session)
        self.assertEqual(session.session_id, "test_123")
        self.assertEqual(session.model, "gpt-realtime")
        
        # Record some metrics
        self.metrics.record_tokens(100, 50)
        self.metrics.record_response(0.150, is_first=True)
        self.metrics.record_function_call(True, 0.200)
        
        # End session
        ended = self.metrics.end_session()
        self.assertIsNotNone(ended)
        self.assertEqual(ended.input_tokens, 100)
        self.assertEqual(ended.output_tokens, 50)
        self.assertEqual(ended.functions_called, 1)
        self.assertEqual(ended.function_success_rate, 1.0)
    
    def test_cost_tracking(self):
        """Test cost calculation and tracking"""
        self.metrics.start_session("cost_test", "gpt-realtime")
        
        # Record token usage
        self.metrics.record_tokens(1000, 500)
        
        # Check cost calculation
        session = self.metrics.current_session
        expected_cost = (1000 * 32 / 1_000_000) + (500 * 64 / 1_000_000)
        self.assertAlmostEqual(session.estimated_cost, expected_cost, places=6)
    
    def test_model_comparison(self):
        """Test model comparison metrics"""
        # Create mock session history
        from openai_client.performance_metrics import SessionMetrics
        
        # Add legacy model sessions
        for i in range(3):
            session = SessionMetrics(
                session_id=f"legacy_{i}",
                model="gpt-4o-realtime-preview",
                start_time=1000 + i,
                end_time=1100 + i,
                average_response_latency=150,
                estimated_cost=0.040,
                function_success_rate=0.50
            )
            self.metrics.session_history.append(session)
        
        # Add new model sessions
        for i in range(3):
            session = SessionMetrics(
                session_id=f"new_{i}",
                model="gpt-realtime",
                start_time=2000 + i,
                end_time=2100 + i,
                average_response_latency=120,
                estimated_cost=0.032,
                function_success_rate=0.67
            )
            self.metrics.session_history.append(session)
        
        # Compare models
        comparison = self.metrics.compare_models(
            "gpt-4o-realtime-preview",
            "gpt-realtime"
        )
        
        # Verify improvements
        self.assertGreater(comparison.latency_improvement, 0)
        self.assertGreater(comparison.cost_reduction, 0)
        self.assertGreater(comparison.function_accuracy_improvement, 0)
    
    def test_statistics_generation(self):
        """Test statistics generation"""
        # Start and end a session with metrics
        self.metrics.start_session("stats_test", "gpt-realtime")
        self.metrics.record_tokens(500, 250)
        self.metrics.record_response(0.100)
        self.metrics.end_session()
        
        # Get statistics
        stats = self.metrics.get_statistics()
        
        self.assertEqual(stats["total_sessions"], 1)
        self.assertIn("gpt-realtime", stats["model_usage"])
        self.assertEqual(stats["total_tokens"]["input"], 500)
        self.assertEqual(stats["total_tokens"]["output"], 250)


class TestIntegration(unittest.TestCase):
    """Integration tests for model migration"""
    
    @patch('openai_client.realtime.websockets.connect')
    async def test_websocket_connection_with_new_model(self, mock_connect):
        """Test WebSocket connection with new model"""
        from openai_client.realtime import OpenAIRealtimeClient
        
        # Create config
        config = Mock()
        config.api_key = "test_key"
        config.voice = "cedar"
        config.model = "gpt-realtime"
        config.legacy_model = "gpt-4o-realtime-preview"
        config.model_selection = "new"
        config.voice_fallback = "alloy"
        config.auto_select_voice = True
        config.temperature = 0.8
        config.language = "en"
        
        # Mock WebSocket
        mock_ws = MagicMock()
        mock_connect.return_value = mock_ws
        
        # Create client
        client = OpenAIRealtimeClient(config)
        
        # Attempt connection
        result = await client.connect()
        
        # Verify model selection
        self.assertEqual(client.selected_model, "gpt-realtime")
        
        # Verify WebSocket URL contains correct model
        call_args = mock_connect.call_args
        url = call_args[0][0] if call_args else None
        if url:
            self.assertIn("model=gpt-realtime", url)
    
    def test_configuration_validation(self):
        """Test configuration validation for migration"""
        from config import load_config, OpenAIConfig
        
        # Create test config
        config_data = {
            "api_key": "test_key",
            "voice": "cedar",
            "model": "gpt-realtime",
            "legacy_model": "gpt-4o-realtime-preview",
            "model_selection": "auto",
            "voice_fallback": "alloy",
            "auto_select_voice": True,
            "temperature": 0.8,
            "language": "en"
        }
        
        # Create OpenAI config
        openai_config = OpenAIConfig(**config_data)
        
        # Verify new fields
        self.assertEqual(openai_config.model, "gpt-realtime")
        self.assertEqual(openai_config.legacy_model, "gpt-4o-realtime-preview")
        self.assertEqual(openai_config.model_selection, "auto")
        self.assertEqual(openai_config.voice_fallback, "alloy")
        self.assertTrue(openai_config.auto_select_voice)
        
        # Verify voice lists
        self.assertIn("cedar", openai_config.VOICES["gpt-realtime"])
        self.assertIn("marin", openai_config.VOICES["gpt-realtime"])
        self.assertNotIn("cedar", openai_config.VOICES["gpt-4o-realtime-preview"])


def run_async_test(coro):
    """Helper to run async tests"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)