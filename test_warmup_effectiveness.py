#!/usr/bin/env python3
"""
Test script to verify OpenWakeWord warm-up effectiveness

This script tests whether the warm-up mechanism successfully enables
consistent non-zero predictions from OpenWakeWord.
"""
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.logger import setup_logging, get_logger

try:
    import openwakeword
    from openwakeword import Model as WakeWordModel
    OPENWAKEWORD_AVAILABLE = True
except ImportError:
    OPENWAKEWORD_AVAILABLE = False
    WakeWordModel = None


def test_without_warmup():
    """Test OpenWakeWord behavior without warm-up"""
    print("="*60)
    print("TEST: OpenWakeWord WITHOUT Warm-up")
    print("="*60)
    
    if not OPENWAKEWORD_AVAILABLE:
        print("[ERROR] OpenWakeWord not available")
        return False
    
    try:
        model = WakeWordModel(wakeword_models=['alexa_v0.1'])
        print("[OK] Model created")
        
        # Test immediate predictions
        print("\nImmediate predictions (no warm-up):")
        zero_count = 0
        for i in range(10):
            audio = np.random.normal(0, 0.05, 1280).astype(np.float32)
            predictions = model.predict(audio)
            max_conf = max(predictions.values()) if predictions else 0.0
            
            if max_conf == 0.0:
                zero_count += 1
            
            print(f"  Prediction {i+1}: max={max_conf:.8f} {'[ZERO]' if max_conf == 0.0 else '[NON-ZERO]'}")
        
        print(f"\nResults: {zero_count}/10 predictions were zero")
        return zero_count < 5  # Success if less than half are zero
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return False


def test_with_warmup():
    """Test OpenWakeWord behavior with warm-up"""
    print("\n" + "="*60)
    print("TEST: OpenWakeWord WITH Warm-up")
    print("="*60)
    
    if not OPENWAKEWORD_AVAILABLE:
        print("[ERROR] OpenWakeWord not available")
        return False
    
    try:
        model = WakeWordModel(wakeword_models=['alexa_v0.1'])
        print("[OK] Model created")
        
        # Warm-up phase
        print("\nWarming up model with 10 chunks...")
        warmup_results = []
        for i in range(10):
            if i < 3:
                audio = np.zeros(1280, dtype=np.float32)
            elif i < 6:
                audio = np.random.normal(0, 0.01, 1280).astype(np.float32)
            else:
                audio = np.random.normal(0, 0.05, 1280).astype(np.float32)
            
            predictions = model.predict(audio)
            max_conf = max(predictions.values()) if predictions else 0.0
            warmup_results.append(max_conf)
            
            if i % 3 == 0 or max_conf > 0.0:
                print(f"  Warm-up {i+1}: max={max_conf:.8f}")
        
        non_zero_warmup = sum(1 for r in warmup_results if r > 0.0)
        print(f"Warm-up complete: {non_zero_warmup}/10 non-zero predictions")
        
        # Test post-warmup predictions
        print("\nPost-warmup predictions:")
        zero_count = 0
        for i in range(10):
            audio = np.random.normal(0, 0.05, 1280).astype(np.float32)
            predictions = model.predict(audio)
            max_conf = max(predictions.values()) if predictions else 0.0
            
            if max_conf == 0.0:
                zero_count += 1
            
            print(f"  Prediction {i+1}: max={max_conf:.8f} {'[ZERO]' if max_conf == 0.0 else '[NON-ZERO]'}")
        
        print(f"\nResults: {zero_count}/10 predictions were zero")
        return zero_count == 0  # Success if no zeros after warm-up
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return False


def test_warmup_persistence():
    """Test if warm-up effect persists"""
    print("\n" + "="*60)
    print("TEST: Warm-up Persistence")
    print("="*60)
    
    if not OPENWAKEWORD_AVAILABLE:
        print("[ERROR] OpenWakeWord not available")
        return False
    
    try:
        model = WakeWordModel(wakeword_models=['alexa_v0.1'])
        print("[OK] Model created")
        
        # Warm-up
        print("\nWarming up model...")
        for i in range(10):
            audio = np.random.normal(0, 0.05, 1280).astype(np.float32)
            model.predict(audio)
        print("Warm-up complete")
        
        # Test predictions over time
        print("\nTesting persistence (50 predictions):")
        results = []
        for i in range(50):
            audio = np.random.normal(0, 0.05, 1280).astype(np.float32)
            predictions = model.predict(audio)
            max_conf = max(predictions.values()) if predictions else 0.0
            results.append(max_conf)
            
            if i % 10 == 0:
                recent_zeros = sum(1 for r in results[-10:] if r == 0.0)
                print(f"  After {i+1} predictions: {recent_zeros}/10 recent were zero")
        
        total_zeros = sum(1 for r in results if r == 0.0)
        print(f"\nResults: {total_zeros}/50 predictions were zero")
        return total_zeros < 5  # Success if less than 10% are zero
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return False


def main():
    """Run warm-up effectiveness tests"""
    setup_logging("INFO", console=True)
    
    print("OpenWakeWord Warm-up Effectiveness Test")
    print("This test verifies that warm-up improves prediction consistency")
    print()
    
    if not OPENWAKEWORD_AVAILABLE:
        print("[ERROR] OpenWakeWord not available - cannot run tests")
        return
    
    # Run tests
    results = []
    
    # Test 1: Without warm-up
    success1 = test_without_warmup()
    results.append(("Without Warm-up", success1))
    
    # Test 2: With warm-up
    success2 = test_with_warmup()
    results.append(("With Warm-up", success2))
    
    # Test 3: Persistence
    success3 = test_warmup_persistence()
    results.append(("Warm-up Persistence", success3))
    
    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"[{status}] {test_name}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] Warm-up mechanism is effective!")
    else:
        print("[WARNING] Some tests failed - warm-up may need adjustment")


if __name__ == "__main__":
    main()