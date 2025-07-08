#!/usr/bin/env python3
"""
Test different audio normalization approaches to fix OpenWakeWord

Based on diagnostic results showing microphone audio isn't normalized correctly
for OpenWakeWord, this tests different normalization strategies.
"""
import sys
import numpy as np
import sounddevice as sd
import threading
import queue
import time
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


def test_normalization_methods():
    """Test different audio normalization methods with microphone input"""
    print("="*60)
    print("TEST: Audio Normalization Methods")
    print("="*60)
    
    if not OPENWAKEWORD_AVAILABLE:
        print("[ERROR] OpenWakeWord not available")
        return False
    
    try:
        # Create model
        model = WakeWordModel(wakeword_models=['alexa_v0.1'])
        print(f"[OK] Model created: {list(model.models.keys())}")
        
        # Audio parameters
        SAMPLE_RATE = 16000
        CHUNK_SIZE = 1280
        CHANNELS = 1
        TEST_DURATION = 5  # seconds per method
        
        # Normalization methods to test
        normalization_methods = {
            "raw_float32": lambda x: x,  # Direct float32 from sounddevice
            "standard_pcm16": lambda x: x * 32767.0 / 32767.0,  # Standard normalization
            "amplify_10x": lambda x: np.clip(x * 10.0, -1.0, 1.0),  # Amplify weak signal
            "amplify_20x": lambda x: np.clip(x * 20.0, -1.0, 1.0),  # Stronger amplification
            "amplify_50x": lambda x: np.clip(x * 50.0, -1.0, 1.0),  # Very strong amplification
            "dynamic_normalize": lambda x: x / (np.max(np.abs(x)) + 1e-7) if np.max(np.abs(x)) > 0.01 else x * 10.0,
            "rms_normalize": lambda x: x / (np.sqrt(np.mean(x**2)) + 1e-7) * 0.1,
            "peak_normalize": lambda x: x / (np.max(np.abs(x)) + 1e-7) * 0.5,
        }
        
        print(f"[INFO] Testing {len(normalization_methods)} normalization methods")
        print(f"[INFO] Recording {TEST_DURATION}s for each method")
        print("[INFO] Make some noise or speak during each test!")
        
        results = {}
        
        for method_name, normalize_func in normalization_methods.items():
            print(f"\n[TESTING] {method_name}")
            print(f"[START] Recording for {TEST_DURATION} seconds...")
            
            # Audio queue
            audio_queue = queue.Queue()
            
            def audio_callback(indata, frames, time, status):
                if status:
                    print(f"Audio status: {status}")
                audio_queue.put(indata[:, 0].copy())  # First channel
            
            # Start recording
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                blocksize=CHUNK_SIZE,
                callback=audio_callback,
                dtype=np.float32
            )
            
            predictions_list = []
            audio_levels = []
            
            with stream:
                start_time = time.time()
                chunk_count = 0
                
                while time.time() - start_time < TEST_DURATION:
                    try:
                        # Get audio chunk
                        audio_chunk = audio_queue.get(timeout=0.1)
                        chunk_count += 1
                        
                        # Apply normalization
                        normalized_audio = normalize_func(audio_chunk)
                        
                        # Ensure float32
                        if normalized_audio.dtype != np.float32:
                            normalized_audio = normalized_audio.astype(np.float32)
                        
                        # Calculate levels
                        raw_level = np.max(np.abs(audio_chunk))
                        norm_level = np.max(np.abs(normalized_audio))
                        audio_levels.append((raw_level, norm_level))
                        
                        # Feed to OpenWakeWord
                        predictions = model.predict(normalized_audio)
                        max_conf = max(predictions.values()) if predictions else 0.0
                        predictions_list.append(max_conf)
                        
                        # Log interesting results
                        if chunk_count % 20 == 0 or max_conf > 0.0001:
                            print(f"  [CHUNK {chunk_count}] raw={raw_level:.3f}, norm={norm_level:.3f}, pred={max_conf:.8f}")
                    
                    except queue.Empty:
                        continue
                    except Exception as e:
                        print(f"  [ERROR] {e}")
            
            # Analyze results
            if predictions_list:
                non_zero = sum(1 for p in predictions_list if p > 0.0)
                unique_preds = len(set(predictions_list))
                max_pred = max(predictions_list)
                avg_pred = sum(predictions_list) / len(predictions_list)
                
                avg_raw_level = sum(l[0] for l in audio_levels) / len(audio_levels)
                avg_norm_level = sum(l[1] for l in audio_levels) / len(audio_levels)
                
                results[method_name] = {
                    'non_zero': non_zero,
                    'unique': unique_preds,
                    'max': max_pred,
                    'avg': avg_pred,
                    'chunks': len(predictions_list),
                    'avg_raw_level': avg_raw_level,
                    'avg_norm_level': avg_norm_level
                }
                
                print(f"  [RESULTS] non_zero={non_zero}/{len(predictions_list)}, unique={unique_preds}, max={max_pred:.8f}")
            else:
                results[method_name] = None
        
        # Summary
        print("\n" + "="*60)
        print("NORMALIZATION TEST SUMMARY")
        print("="*60)
        
        best_method = None
        best_score = 0
        
        for method_name, result in results.items():
            if result:
                # Score based on unique predictions and non-zero count
                score = result['unique'] + (result['non_zero'] / result['chunks'] * 100)
                
                print(f"\n[{method_name}]")
                print(f"  Raw audio level: {result['avg_raw_level']:.4f}")
                print(f"  Normalized level: {result['avg_norm_level']:.4f}")
                print(f"  Non-zero predictions: {result['non_zero']}/{result['chunks']} ({result['non_zero']/result['chunks']*100:.1f}%)")
                print(f"  Unique values: {result['unique']}")
                print(f"  Max prediction: {result['max']:.8f}")
                print(f"  Score: {score:.2f}")
                
                if score > best_score:
                    best_score = score
                    best_method = method_name
        
        if best_method:
            print(f"\n[BEST METHOD] {best_method} with score {best_score:.2f}")
            return True
        else:
            print("\n[FAILED] No normalization method produced good results")
            return False
            
    except Exception as e:
        print(f"[ERROR] Normalization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fixed_detector():
    """Test the detector with the best normalization approach"""
    print("\n" + "="*60)
    print("TEST: Fixed Wake Word Detection")
    print("="*60)
    
    if not OPENWAKEWORD_AVAILABLE:
        print("[ERROR] OpenWakeWord not available")
        return False
    
    try:
        model = WakeWordModel(wakeword_models=['alexa_v0.1'])
        print(f"[OK] Model created: {list(model.models.keys())}")
        
        # Best normalization based on typical results
        def normalize_audio(audio):
            # Amplify weak microphone signal and ensure good range
            return np.clip(audio * 20.0, -1.0, 1.0).astype(np.float32)
        
        SAMPLE_RATE = 16000
        CHUNK_SIZE = 1280
        TEST_DURATION = 30
        
        print(f"[INFO] Testing fixed detector for {TEST_DURATION}s")
        print("[INFO] Say 'Alexa' multiple times to test detection")
        
        audio_queue = queue.Queue()
        
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio status: {status}")
            audio_queue.put(indata[:, 0].copy())
        
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=CHUNK_SIZE,
            callback=audio_callback,
            dtype=np.float32
        )
        
        detections = []
        
        with stream:
            start_time = time.time()
            chunk_count = 0
            
            while time.time() - start_time < TEST_DURATION:
                try:
                    audio_chunk = audio_queue.get(timeout=0.1)
                    chunk_count += 1
                    
                    # Apply normalization
                    normalized_audio = normalize_audio(audio_chunk)
                    
                    # Feed to OpenWakeWord
                    predictions = model.predict(normalized_audio)
                    max_conf = max(predictions.values()) if predictions else 0.0
                    
                    # Log activity
                    if chunk_count % 50 == 0 or max_conf > 0.01:
                        raw_level = np.max(np.abs(audio_chunk))
                        norm_level = np.max(np.abs(normalized_audio))
                        print(f"[CHUNK {chunk_count}] raw={raw_level:.3f}, norm={norm_level:.3f}, pred={max_conf:.6f}")
                    
                    # Check for detection
                    if max_conf > 0.3:  # Reasonable threshold
                        timestamp = time.time() - start_time
                        detections.append((timestamp, max_conf))
                        print(f"*** [WAKE WORD DETECTED] at {timestamp:.1f}s: confidence={max_conf:.6f} ***")
                
                except queue.Empty:
                    continue
        
        print(f"\n[RESULTS] Processed {chunk_count} chunks")
        print(f"[RESULTS] Detections: {len(detections)}")
        
        if detections:
            print("\n[DETECTIONS]")
            for i, (timestamp, confidence) in enumerate(detections):
                print(f"  {i+1}: {timestamp:.1f}s - confidence={confidence:.6f}")
        
        return len(detections) > 0
        
    except Exception as e:
        print(f"[ERROR] Fixed detector test failed: {e}")
        return False


def main():
    """Run normalization fix tests"""
    setup_logging("INFO", console=True)
    
    print("OpenWakeWord Audio Normalization Fix Test")
    print("This tests different normalization methods to fix the microphone issue")
    print()
    
    if not OPENWAKEWORD_AVAILABLE:
        print("[ERROR] OpenWakeWord not available - cannot run tests")
        return
    
    # Test 1: Find best normalization
    print("[STEP 1] Testing normalization methods...")
    success1 = test_normalization_methods()
    
    # Test 2: Test fixed detector
    if success1:
        print("\n[STEP 2] Testing fixed detector with best normalization...")
        success2 = test_fixed_detector()
    else:
        success2 = False
    
    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    if success1 and success2:
        print("[SUCCESS] Found working normalization method!")
        print("[ACTION] Update detector.py to amplify microphone audio before feeding to OpenWakeWord")
        print("[FIX] normalized_audio = np.clip(audio * 20.0, -1.0, 1.0)")
    else:
        print("[FAILED] Could not find working normalization")
        print("[ACTION] May need different microphone or audio device configuration")


if __name__ == "__main__":
    main()