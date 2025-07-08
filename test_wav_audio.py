#!/usr/bin/env python3
"""
Test OpenWakeWord with WAV file input to isolate audio processing issues

This test uses pre-recorded audio files to test OpenWakeWord without
any microphone input variability.
"""
import sys
import numpy as np
import wave
import tempfile
import os
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


def generate_test_wav(filename, duration=2.0, sample_rate=16000):
    """Generate a test WAV file with synthetic audio"""
    samples = int(duration * sample_rate)
    
    # Generate a complex audio pattern
    t = np.arange(samples) / sample_rate
    
    # Combine multiple frequencies to simulate speech-like audio
    audio = (
        0.3 * np.sin(2 * np.pi * 300 * t) +      # Base frequency
        0.2 * np.sin(2 * np.pi * 600 * t) +      # Harmonic
        0.1 * np.sin(2 * np.pi * 1200 * t) +     # Higher harmonic
        0.05 * np.random.normal(0, 1, samples)   # Noise
    )
    
    # Normalize to [-1, 1]
    audio = audio / np.max(np.abs(audio))
    
    # Convert to int16 for WAV file
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Write WAV file
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample (int16)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    
    print(f"[CREATED] {filename}: {duration}s, {sample_rate}Hz, {samples} samples")
    return filename


def read_wav_file(filename):
    """Read WAV file and return audio data in OpenWakeWord format"""
    with wave.open(filename, 'r') as wav_file:
        # Get WAV file parameters
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        framerate = wav_file.getframerate()
        frames = wav_file.getnframes()
        
        print(f"[WAV INFO] {filename}: {channels}ch, {sample_width}bytes, {framerate}Hz, {frames} frames")
        
        # Read all frames
        audio_bytes = wav_file.readframes(frames)
        
        # Convert to numpy array
        if sample_width == 2:  # int16
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        
        # Take first channel if stereo
        if channels > 1:
            audio_array = audio_array[::channels]
        
        # Convert to float32 and normalize
        audio_float = audio_array.astype(np.float32) / 32767.0
        
        return audio_float, framerate


def process_wav_with_openwakeword(wav_filename):
    """Process a WAV file through OpenWakeWord chunk by chunk"""
    print(f"\n[PROCESSING] {wav_filename}")
    
    if not OPENWAKEWORD_AVAILABLE:
        print("[ERROR] OpenWakeWord not available")
        return False
    
    try:
        # Load model
        model = WakeWordModel(wakeword_models=['alexa_v0.1'])
        print(f"[OK] Model loaded: {list(model.models.keys())}")
        
        # Read WAV file
        audio_data, file_sample_rate = read_wav_file(wav_filename)
        print(f"[INFO] Audio loaded: {len(audio_data)} samples at {file_sample_rate}Hz")
        
        # Resample to 16kHz if needed
        if file_sample_rate != 16000:
            from scipy import signal
            target_length = int(len(audio_data) * 16000 / file_sample_rate)
            audio_data = signal.resample(audio_data, target_length)
            print(f"[RESAMPLE] {file_sample_rate}Hz -> 16kHz: {len(audio_data)} samples")
        
        # Process in 80ms chunks (1280 samples at 16kHz)
        chunk_size = 1280
        num_chunks = len(audio_data) // chunk_size
        detections = []
        predictions_list = []
        
        print(f"[INFO] Processing {num_chunks} chunks of {chunk_size} samples each")
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            
            chunk = audio_data[start_idx:end_idx]
            
            # Ensure chunk is exactly the right size
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
            
            # Process chunk
            predictions = model.predict(chunk)
            max_conf = max(predictions.values()) if predictions else 0.0
            predictions_list.append(max_conf)
            
            # Calculate audio level for this chunk
            audio_level = np.max(np.abs(chunk))
            
            # Log interesting chunks
            if i % 20 == 0 or max_conf > 0.01 or audio_level > 0.1:
                time_sec = i * chunk_size / 16000
                print(f"[CHUNK {i:3d}] {time_sec:5.2f}s: level={audio_level:.3f}, pred={max_conf:.6f}")
            
            # Check for detections
            if max_conf > 0.1:  # Lower threshold for testing
                time_sec = i * chunk_size / 16000
                detections.append((time_sec, max_conf, audio_level))
                print(f"*** [DETECTION] at {time_sec:.2f}s: confidence={max_conf:.6f} ***")
        
        # Analysis
        print(f"\n[RESULTS] Processed {num_chunks} chunks")
        print(f"[RESULTS] Detections: {len(detections)}")
        
        # Prediction statistics
        non_zero_preds = [p for p in predictions_list if p > 0.0]
        unique_preds = len(set(predictions_list))
        
        print(f"[STATS] Non-zero predictions: {len(non_zero_preds)}/{len(predictions_list)}")
        print(f"[STATS] Unique prediction values: {unique_preds}")
        print(f"[STATS] Max prediction: {max(predictions_list):.8f}")
        print(f"[STATS] Min prediction: {min(predictions_list):.8f}")
        
        if detections:
            print("[DETECTIONS]")
            for i, (timestamp, confidence, level) in enumerate(detections):
                print(f"  {i+1}: {timestamp:.2f}s - confidence={confidence:.6f}, level={level:.3f}")
        
        # Check if model is responding to audio variations
        if unique_preds > 1:
            print("[SUCCESS] Model produces varying predictions")
        else:
            print("[PROBLEM] Model returns identical predictions for all chunks")
        
        return len(detections) > 0 or unique_preds > 1
        
    except Exception as e:
        print(f"[ERROR] WAV processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_wav_formats():
    """Test OpenWakeWord with different WAV file formats"""
    print("="*60)
    print("TEST: Different WAV File Formats")
    print("="*60)
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        test_files = []
        
        # Generate different test WAV files
        test_configs = [
            ("simple_16k.wav", 2.0, 16000),    # Native 16kHz
            ("high_48k.wav", 2.0, 48000),      # High sample rate
            ("short_16k.wav", 0.5, 16000),     # Short duration
            ("long_16k.wav", 5.0, 16000),      # Long duration
        ]
        
        # Create test files
        for filename, duration, sample_rate in test_configs:
            filepath = os.path.join(temp_dir, filename)
            generate_test_wav(filepath, duration, sample_rate)
            test_files.append((filepath, filename))
        
        # Test each file
        results = []
        for filepath, filename in test_files:
            print(f"\n{'-'*40}")
            success = process_wav_with_openwakeword(filepath)
            results.append((filename, success))
        
        # Summary
        print(f"\n{'='*60}")
        print("WAV FORMAT TEST SUMMARY")
        print(f"{'='*60}")
        
        for filename, success in results:
            status = "PASS" if success else "FAIL"
            print(f"[{status}] {filename}")
        
        passed = sum(1 for _, success in results if success)
        total = len(results)
        print(f"\nOverall: {passed}/{total} WAV tests passed")
        
        return passed > 0


def main():
    """Run WAV file tests"""
    setup_logging("INFO", console=True)
    
    print("OpenWakeWord WAV File Test")
    print("This tests OpenWakeWord with pre-recorded audio files")
    print("to isolate microphone input issues")
    print()
    
    if not OPENWAKEWORD_AVAILABLE:
        print("[ERROR] OpenWakeWord not available - cannot run tests")
        return
    
    try:
        success = test_different_wav_formats()
        
        print(f"\n{'='*60}")
        print("FINAL DIAGNOSIS")
        print(f"{'='*60}")
        
        if success:
            print("[CONCLUSION] OpenWakeWord can process audio files successfully")
            print("[NEXT STEP] The issue is likely in our microphone audio processing pipeline")
            print("[ACTION] Compare our audio processing with working implementations")
        else:
            print("[CONCLUSION] OpenWakeWord has fundamental issues even with WAV files")
            print("[NEXT STEP] Check OpenWakeWord installation and compatibility")
            print("[ACTION] Try different OpenWakeWord versions or models")
    
    except Exception as e:
        print(f"[ERROR] Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()