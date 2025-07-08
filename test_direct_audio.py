#!/usr/bin/env python3
"""
Test OpenWakeWord with direct audio input bypassing our processing pipeline

This test directly feeds microphone audio to OpenWakeWord without our
conversion/resampling to identify where the audio processing issue lies.
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


def test_direct_16khz_input():
    """Test OpenWakeWord with direct 16kHz microphone input"""
    print("="*60)
    print("TEST: Direct 16kHz Microphone Input")
    print("="*60)
    
    if not OPENWAKEWORD_AVAILABLE:
        print("[ERROR] OpenWakeWord not available")
        return False
    
    try:
        # Create model
        model = WakeWordModel(wakeword_models=['alexa_v0.1'])
        print(f"[OK] Model created: {list(model.models.keys())}")
        
        # Audio parameters - exactly what OpenWakeWord expects
        SAMPLE_RATE = 16000  # OpenWakeWord native rate
        CHUNK_SIZE = 1280    # 80ms at 16kHz
        CHANNELS = 1
        
        print(f"[INFO] Recording at {SAMPLE_RATE}Hz, {CHUNK_SIZE} samples per chunk")
        print("[INFO] This bypasses all our audio processing - direct to OpenWakeWord")
        
        # Audio queue for threading
        audio_queue = queue.Queue()
        stop_recording = threading.Event()
        
        def audio_callback(indata, frames, time, status):
            """Direct audio callback - minimal processing"""
            if status:
                print(f"Audio status: {status}")
            
            # Convert to float32 and normalize - minimal processing
            audio_float = indata[:, 0].astype(np.float32)  # Take first channel, convert to float32
            audio_queue.put(audio_float.copy())
        
        # Start recording at 16kHz directly
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            blocksize=CHUNK_SIZE,
            callback=audio_callback,
            dtype=np.float32
        )
        
        print(f"[START] Recording for 30 seconds - say 'Alexa' multiple times")
        print(f"[INFO] Using device: {sd.query_devices(sd.default.device[0])['name']}")
        
        detections = []
        chunk_count = 0
        
        with stream:
            start_time = time.time()
            
            while time.time() - start_time < 30:
                try:
                    # Get audio chunk
                    audio_chunk = audio_queue.get(timeout=0.1)
                    chunk_count += 1
                    
                    # Ensure exactly 1280 samples
                    if len(audio_chunk) != CHUNK_SIZE:
                        print(f"[WARNING] Unexpected chunk size: {len(audio_chunk)}, expected {CHUNK_SIZE}")
                        if len(audio_chunk) > CHUNK_SIZE:
                            audio_chunk = audio_chunk[:CHUNK_SIZE]
                        else:
                            # Pad with zeros
                            audio_chunk = np.pad(audio_chunk, (0, CHUNK_SIZE - len(audio_chunk)), mode='constant')
                    
                    # Calculate audio level
                    audio_level = np.max(np.abs(audio_chunk))
                    
                    # Feed directly to OpenWakeWord
                    predictions = model.predict(audio_chunk)
                    max_conf = max(predictions.values()) if predictions else 0.0
                    
                    # Log significant activity
                    if chunk_count % 50 == 0 or audio_level > 0.02 or max_conf > 0.01:
                        print(f"[CHUNK {chunk_count}] level={audio_level:.3f}, pred={predictions}, max={max_conf:.6f}")
                    
                    # Check for detections
                    if max_conf > 0.1:  # Lower threshold for testing
                        detection_time = time.time() - start_time
                        detections.append((detection_time, max_conf, audio_level))
                        print(f"*** [DETECTION] at {detection_time:.1f}s: confidence={max_conf:.6f}, audio_level={audio_level:.3f} ***")
                
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"[ERROR] Processing chunk {chunk_count}: {e}")
        
        print(f"\n[RESULTS] Processed {chunk_count} chunks")
        print(f"[RESULTS] Detections: {len(detections)}")
        
        if detections:
            print("[DETECTIONS]")
            for i, (timestamp, confidence, level) in enumerate(detections):
                print(f"  {i+1}: {timestamp:.1f}s - confidence={confidence:.6f}, level={level:.3f}")
        else:
            print("[NO DETECTIONS] OpenWakeWord did not detect any wake words")
        
        return len(detections) > 0
        
    except Exception as e:
        print(f"[ERROR] Direct audio test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_synthetic_audio():
    """Test OpenWakeWord with synthetic audio patterns"""
    print("\n" + "="*60)
    print("TEST: Synthetic Audio Patterns")
    print("="*60)
    
    if not OPENWAKEWORD_AVAILABLE:
        print("[ERROR] OpenWakeWord not available")
        return False
    
    try:
        model = WakeWordModel(wakeword_models=['alexa_v0.1'])
        print("[OK] Model created for synthetic testing")
        
        # Test different synthetic patterns
        test_patterns = [
            ("Silence", np.zeros(1280, dtype=np.float32)),
            ("Low sine 440Hz", np.sin(2 * np.pi * 440 * np.arange(1280) / 16000).astype(np.float32) * 0.1),
            ("High sine 1000Hz", np.sin(2 * np.pi * 1000 * np.arange(1280) / 16000).astype(np.float32) * 0.2),
            ("White noise low", np.random.normal(0, 0.05, 1280).astype(np.float32)),
            ("White noise high", np.random.normal(0, 0.2, 1280).astype(np.float32)),
            ("Chirp", np.sin(2 * np.pi * np.linspace(100, 2000, 1280) * np.arange(1280) / 16000).astype(np.float32) * 0.15),
        ]
        
        print("[INFO] Testing synthetic patterns to verify model responsiveness...")
        
        varying_predictions = False
        all_predictions = []
        
        for name, audio_pattern in test_patterns:
            predictions = model.predict(audio_pattern)
            max_conf = max(predictions.values()) if predictions else 0.0
            all_predictions.append(max_conf)
            
            print(f"[PATTERN] {name:15} -> {predictions} (max: {max_conf:.8f})")
        
        # Check if predictions vary
        unique_predictions = len(set(all_predictions))
        print(f"\n[ANALYSIS] {unique_predictions} unique prediction values out of {len(all_predictions)} tests")
        
        if unique_predictions > 1:
            print("[SUCCESS] Model produces varying predictions for different inputs")
            varying_predictions = True
        else:
            print("[PROBLEM] Model returns identical predictions for all inputs")
        
        return varying_predictions
        
    except Exception as e:
        print(f"[ERROR] Synthetic audio test failed: {e}")
        return False


def test_audio_format_compatibility():
    """Test different audio format variations"""
    print("\n" + "="*60)
    print("TEST: Audio Format Compatibility")
    print("="*60)
    
    if not OPENWAKEWORD_AVAILABLE:
        print("[ERROR] OpenWakeWord not available")
        return False
    
    try:
        model = WakeWordModel(wakeword_models=['alexa_v0.1'])
        print("[OK] Model created for format testing")
        
        # Generate base audio
        base_audio = np.sin(2 * np.pi * 440 * np.arange(1280) / 16000) * 0.1
        
        format_tests = [
            ("float32 [-1,1]", base_audio.astype(np.float32)),
            ("float64 converted", base_audio.astype(np.float64).astype(np.float32)),
            ("int16 converted", (base_audio * 32767).astype(np.int16).astype(np.float32) / 32767.0),
            ("Normalized [0,1]", np.abs(base_audio).astype(np.float32)),
            ("Scaled x2", (base_audio * 2).astype(np.float32)),
            ("Scaled x0.5", (base_audio * 0.5).astype(np.float32)),
            ("Clipped [-1,1]", np.clip(base_audio * 2, -1.0, 1.0).astype(np.float32)),
        ]
        
        print("[INFO] Testing audio format variations...")
        
        working_formats = []
        
        for name, audio_data in format_tests:
            try:
                predictions = model.predict(audio_data)
                max_conf = max(predictions.values()) if predictions else 0.0
                
                print(f"[FORMAT] {name:15} -> max={max_conf:.8f}, range=[{np.min(audio_data):.3f}, {np.max(audio_data):.3f}]")
                
                if max_conf > 0.0:
                    working_formats.append(name)
                    
            except Exception as e:
                print(f"[FORMAT] {name:15} -> ERROR: {e}")
        
        print(f"\n[ANALYSIS] {len(working_formats)}/{len(format_tests)} formats produced non-zero predictions")
        
        if working_formats:
            print("[WORKING FORMATS]")
            for fmt in working_formats:
                print(f"  - {fmt}")
        
        return len(working_formats) > 0
        
    except Exception as e:
        print(f"[ERROR] Format compatibility test failed: {e}")
        return False


def main():
    """Run direct audio tests"""
    setup_logging("INFO", console=True)
    
    print("OpenWakeWord Direct Audio Pipeline Test")
    print("This bypasses our audio processing to isolate the issue")
    print()
    
    if not OPENWAKEWORD_AVAILABLE:
        print("[ERROR] OpenWakeWord not available - cannot run tests")
        return
    
    # Run tests
    results = []
    
    # Test 1: Synthetic audio to verify model works
    success1 = test_synthetic_audio()
    results.append(("Synthetic Audio", success1))
    
    # Test 2: Audio format compatibility
    success2 = test_audio_format_compatibility()
    results.append(("Format Compatibility", success2))
    
    # Test 3: Direct microphone input (only if model works with synthetic)
    if success1:
        print("\n[INFO] Model works with synthetic audio - testing direct microphone...")
        success3 = test_direct_16khz_input()
        results.append(("Direct Microphone", success3))
    else:
        print("\n[SKIP] Model doesn't work with synthetic audio - skipping microphone test")
        results.append(("Direct Microphone", False))
    
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
    
    # Diagnosis
    if results[0][1]:  # Synthetic audio works
        if results[2][1]:  # Direct microphone works
            print("\n[DIAGNOSIS] OpenWakeWord works correctly - issue is in our audio processing pipeline")
        else:
            print("\n[DIAGNOSIS] Issue is with microphone audio format or device compatibility")
    else:
        print("\n[DIAGNOSIS] Fundamental issue with OpenWakeWord model or installation")


if __name__ == "__main__":
    main()