#!/usr/bin/env python3
"""
Basic OpenWakeWord test - minimal test to get ANY non-zero prediction

This is the simplest possible test to verify OpenWakeWord functionality.
"""
import sys
import numpy as np
from pathlib import Path

# Add src to path for utilities
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.logger import setup_logging, get_logger

try:
    import openwakeword
    from openwakeword import Model as WakeWordModel
    OPENWAKEWORD_AVAILABLE = True
    print("[INFO] OpenWakeWord imported successfully")
except ImportError as e:
    OPENWAKEWORD_AVAILABLE = False
    print(f"[ERROR] OpenWakeWord import failed: {e}")
    WakeWordModel = None


def test_basic_functionality():
    """Test the most basic OpenWakeWord functionality"""
    print("="*60)
    print("BASIC OPENWAKEWORD TEST")
    print("="*60)
    
    if not OPENWAKEWORD_AVAILABLE:
        print("[ERROR] OpenWakeWord not available")
        return False
    
    print("[INFO] Starting basic OpenWakeWord test...")
    
    try:
        print("[STEP 1] Creating model...")
        model = WakeWordModel(wakeword_models=['alexa_v0.1'])
        print("[SUCCESS] Model created successfully")
        
        print(f"[INFO] Available models: {list(model.models.keys())}")
        
        print("[STEP 2] Testing with different audio inputs...")
        
        # Test 1: Silence
        print("  [TEST 1] Testing with silence...")
        silence = np.zeros(1280, dtype=np.float32)
        pred1 = model.predict(silence)
        max1 = max(pred1.values()) if pred1 else 0.0
        print(f"    Silence: {pred1} (max: {max1:.8f})")
        
        # Test 2: Low noise
        print("  [TEST 2] Testing with low noise...")
        noise_low = np.random.normal(0, 0.01, 1280).astype(np.float32)
        pred2 = model.predict(noise_low)
        max2 = max(pred2.values()) if pred2 else 0.0
        print(f"    Low noise: {pred2} (max: {max2:.8f})")
        
        # Test 3: Medium noise
        print("  [TEST 3] Testing with medium noise...")
        noise_med = np.random.normal(0, 0.05, 1280).astype(np.float32)
        pred3 = model.predict(noise_med)
        max3 = max(pred3.values()) if pred3 else 0.0
        print(f"    Medium noise: {pred3} (max: {max3:.8f})")
        
        # Test 4: High noise
        print("  [TEST 4] Testing with high noise...")
        noise_high = np.random.normal(0, 0.1, 1280).astype(np.float32)
        pred4 = model.predict(noise_high)
        max4 = max(pred4.values()) if pred4 else 0.0
        print(f"    High noise: {pred4} (max: {max4:.8f})")
        
        # Test 5: Sine wave
        print("  [TEST 5] Testing with sine wave...")
        sine = np.sin(2 * np.pi * 440 * np.arange(1280) / 16000).astype(np.float32)
        pred5 = model.predict(sine)
        max5 = max(pred5.values()) if pred5 else 0.0
        print(f"    Sine wave: {pred5} (max: {max5:.8f})")
        
        # Check results
        all_max = [max1, max2, max3, max4, max5]
        max_overall = max(all_max)
        
        print(f"\n[SUMMARY] Results:")
        print(f"  All prediction maximums: {[f'{m:.8f}' for m in all_max]}")
        print(f"  Overall maximum: {max_overall:.8f}")
        
        if max_overall > 0.0:
            print(f"[SUCCESS] Got non-zero predictions! OpenWakeWord is working.")
            return True
        else:
            print(f"[PROBLEM] All predictions are exactly zero. OpenWakeWord is not working properly.")
            return False
            
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sequential_predictions():
    """Test sequential predictions to see if they vary"""
    print("="*60)
    print("SEQUENTIAL PREDICTIONS TEST")
    print("="*60)
    
    if not OPENWAKEWORD_AVAILABLE:
        print("[ERROR] OpenWakeWord not available")
        return False
    
    try:
        print("[INFO] Testing sequential predictions...")
        model = WakeWordModel(wakeword_models=['alexa_v0.1'])
        
        predictions_list = []
        for i in range(10):
            # Generate different random audio each time
            audio = np.random.normal(0, 0.05, 1280).astype(np.float32)
            pred = model.predict(audio)
            max_conf = max(pred.values()) if pred else 0.0
            predictions_list.append(max_conf)
            
            print(f"  Prediction {i+1}: {pred} (max: {max_conf:.8f})")
        
        # Check if predictions vary
        unique_predictions = set(predictions_list)
        print(f"\n[SUMMARY] Sequential predictions:")
        print(f"  All predictions: {[f'{p:.8f}' for p in predictions_list]}")
        print(f"  Unique values: {len(unique_predictions)}")
        
        if len(unique_predictions) > 1:
            print(f"[SUCCESS] Predictions vary - model is responding to different inputs")
            return True
        else:
            print(f"[PROBLEM] All predictions are identical - model may be stuck")
            return False
            
    except Exception as e:
        print(f"[ERROR] Sequential test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_reset():
    """Test if model reset affects predictions"""
    print("="*60)
    print("MODEL RESET TEST")
    print("="*60)
    
    if not OPENWAKEWORD_AVAILABLE:
        print("[ERROR] OpenWakeWord not available")
        return False
    
    try:
        print("[INFO] Testing model reset behavior...")
        model = WakeWordModel(wakeword_models=['alexa_v0.1'])
        
        # Test before reset
        audio = np.random.normal(0, 0.05, 1280).astype(np.float32)
        pred_before = model.predict(audio)
        max_before = max(pred_before.values()) if pred_before else 0.0
        print(f"  Before reset: {pred_before} (max: {max_before:.8f})")
        
        # Reset model
        print("[INFO] Resetting model...")
        model.reset()
        
        # Test after reset
        pred_after = model.predict(audio)
        max_after = max(pred_after.values()) if pred_after else 0.0
        print(f"  After reset: {pred_after} (max: {max_after:.8f})")
        
        # Test a few more predictions after reset
        for i in range(3):
            audio = np.random.normal(0, 0.05, 1280).astype(np.float32)
            pred = model.predict(audio)
            max_conf = max(pred.values()) if pred else 0.0
            print(f"  Post-reset {i+1}: {pred} (max: {max_conf:.8f})")
        
        print(f"[INFO] Model reset test completed")
        return True
        
    except Exception as e:
        print(f"[ERROR] Reset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run basic tests"""
    setup_logging("INFO", console=True)
    
    print("Starting basic OpenWakeWord tests...")
    print("This will help identify if OpenWakeWord works at all.")
    print()
    
    if not OPENWAKEWORD_AVAILABLE:
        print("[ERROR] OpenWakeWord not available - cannot run tests")
        return
    
    # Run tests
    results = []
    
    results.append(("Basic Functionality", test_basic_functionality()))
    results.append(("Sequential Predictions", test_sequential_predictions()))
    results.append(("Model Reset", test_model_reset()))
    
    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"[{status}] {test_name}")
    
    passed = sum(results)
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] All tests passed! OpenWakeWord is working correctly.")
        print("The issue is likely in our audio processing pipeline.")
    elif passed > 0:
        print("[PARTIAL] Some tests passed. OpenWakeWord works but may have issues.")
    else:
        print("[CRITICAL] All tests failed. OpenWakeWord is not working at all.")


if __name__ == "__main__":
    main()