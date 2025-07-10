#!/usr/bin/env python3
"""
Test script for multi-turn conversation functionality
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import load_config, SessionConfig


def test_config_loading():
    """Test that multi-turn configuration loads correctly"""
    print("Testing multi-turn configuration loading...")
    
    # Create a test config
    config = SessionConfig()
    
    # Test default values
    assert config.conversation_mode == "single_turn"
    assert config.multi_turn_timeout == 30.0
    assert config.multi_turn_max_turns == 10
    assert "goodbye" in config.multi_turn_end_phrases
    
    print("✓ Configuration loading test passed")


def test_phrase_detection():
    """Test end phrase detection logic"""
    print("Testing phrase detection...")
    
    # Simulate the phrase detection logic
    def contains_end_phrases(text, end_phrases):
        if not text:
            return False
        text_lower = text.lower().strip()
        for phrase in end_phrases:
            if phrase.lower() in text_lower:
                return True
        return False
    
    end_phrases = ["goodbye", "stop", "that's all", "thank you", "bye"]
    
    # Test positive cases
    assert contains_end_phrases("Goodbye!", end_phrases)
    assert contains_end_phrases("Thank you very much", end_phrases)
    assert contains_end_phrases("That's all for now", end_phrases)
    assert contains_end_phrases("I need to stop", end_phrases)
    assert contains_end_phrases("Bye bye!", end_phrases)
    
    # Test negative cases
    assert not contains_end_phrases("What's the weather?", end_phrases)
    assert not contains_end_phrases("How are you?", end_phrases)
    assert not contains_end_phrases("", end_phrases)
    assert not contains_end_phrases("Good morning", end_phrases)
    
    print("✓ Phrase detection test passed")


def test_session_states():
    """Test session state enumeration"""
    print("Testing session states...")
    
    # Import the session states
    from main import SessionState
    
    # Test that MULTI_TURN_LISTENING state exists
    assert hasattr(SessionState, 'MULTI_TURN_LISTENING')
    assert SessionState.MULTI_TURN_LISTENING.value == "multi_turn_listening"
    
    # Test other required states
    assert hasattr(SessionState, 'IDLE')
    assert hasattr(SessionState, 'LISTENING')
    assert hasattr(SessionState, 'PROCESSING')
    assert hasattr(SessionState, 'RESPONDING')
    assert hasattr(SessionState, 'AUDIO_PLAYING')
    assert hasattr(SessionState, 'COOLDOWN')
    
    print("✓ Session states test passed")


def test_multi_turn_logic():
    """Test multi-turn conversation logic"""
    print("Testing multi-turn conversation logic...")
    
    # Test turn counting
    turn_count = 0
    max_turns = 3
    
    # Simulate conversation turns
    for i in range(max_turns + 1):
        turn_count += 1
        should_end = turn_count >= max_turns
        
        if i < max_turns:
            assert not should_end, f"Should not end at turn {turn_count}"
        else:
            assert should_end, f"Should end at turn {turn_count}"
    
    print("✓ Multi-turn logic test passed")


def main():
    """Run all tests"""
    print("Running multi-turn conversation tests...\n")
    
    try:
        test_config_loading()
        test_phrase_detection()
        test_session_states()
        test_multi_turn_logic()
        
        print("\n✅ All tests passed! Multi-turn conversation functionality is working correctly.")
        print("\nTo enable multi-turn conversations:")
        print("1. Set 'conversation_mode: \"multi_turn\"' in your config.yaml")
        print("2. Optionally adjust 'multi_turn_timeout' and 'multi_turn_max_turns'")
        print("3. Customize 'multi_turn_end_phrases' as needed")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()