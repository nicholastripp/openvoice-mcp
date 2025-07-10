#!/usr/bin/env python3
"""
Test OpenWakeWord directly without our processing pipeline
"""
import numpy as np
import time
from openwakeword import Model

def test_raw_openwakeword():
    print("Testing OpenWakeWord directly...")
    
    # Create model
    print("Loading alexa model...")
    model = Model(wakeword_models=["alexa_v0.1"])
    print(f"Model loaded. Available models: {list(model.models.keys())}")
    
    # Test with different audio patterns
    print("\nTesting with different audio patterns:")
    
    # Test 1: Silence
    print("\n1. Testing with silence (zeros):")
    silence = np.zeros(1280, dtype=np.float32)
    result = model.predict(silence)
    print(f"   Result: {result}")
    
    # Test 2: White noise
    print("\n2. Testing with white noise:")
    noise = np.random.normal(0, 0.1, 1280).astype(np.float32)
    noise = np.clip(noise, -1.0, 1.0)
    result = model.predict(noise)
    print(f"   Result: {result}")
    
    # Test 3: Sine wave (simulating speech tone)
    print("\n3. Testing with sine wave (440Hz tone):")
    t = np.linspace(0, 0.08, 1280)  # 80ms at 16kHz
    sine = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    result = model.predict(sine)
    print(f"   Result: {result}")
    
    # Test 4: Multiple predictions to check for stuck state
    print("\n4. Testing multiple predictions for stuck state:")
    predictions = []
    for i in range(10):
        # Vary the input slightly each time
        noise = np.random.normal(0, 0.05 + i*0.01, 1280).astype(np.float32)
        noise = np.clip(noise, -1.0, 1.0)
        result = model.predict(noise)
        predictions.append(result['alexa_v0.1'])
        print(f"   Prediction {i+1}: {result}")
    
    # Check if stuck
    unique_values = len(set(predictions))
    print(f"\nUnique prediction values: {unique_values}/10")
    if unique_values == 1:
        print("WARNING: Model appears stuck - all predictions identical!")
    
    # Test 5: Check audio format requirements
    print("\n5. Testing different audio formats:")
    
    # Float64 (wrong dtype)
    print("   Testing float64 (should fail or convert):")
    float64_audio = np.random.normal(0, 0.1, 1280).astype(np.float64)
    try:
        result = model.predict(float64_audio)
        print(f"   Result: {result}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Wrong size
    print("   Testing wrong chunk size (should fail):")
    wrong_size = np.random.normal(0, 0.1, 1000).astype(np.float32)
    try:
        result = model.predict(wrong_size)
        print(f"   Result: {result}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 6: Warm up the model
    print("\n6. Model warmup test:")
    print("   Warming up with 20 predictions...")
    for i in range(20):
        noise = np.random.normal(0, 0.1, 1280).astype(np.float32)
        result = model.predict(noise)
        if i % 5 == 0:
            print(f"   Warmup {i}: {result}")
    
    print("\n   Testing after warmup:")
    for i in range(5):
        noise = np.random.normal(0, 0.1, 1280).astype(np.float32)
        result = model.predict(noise)
        print(f"   Post-warmup {i}: {result}")
    
    print("\nTest complete!")

if __name__ == "__main__":
    test_raw_openwakeword()