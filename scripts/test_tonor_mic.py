#!/usr/bin/env python3
"""
Test TONOR G11 USB Microphone audio levels
"""
import pyaudio
import numpy as np
import time

def test_microphone():
    print("TONOR G11 USB Microphone Test")
    print("=" * 50)
    
    pa = pyaudio.PyAudio()
    
    # Find TONOR microphone
    tonor_index = None
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if 'TONOR' in info['name'] and info['maxInputChannels'] > 0:
            tonor_index = i
            print(f"Found TONOR microphone:")
            print(f"  Index: {i}")
            print(f"  Name: {info['name']}")
            print(f"  Channels: {info['maxInputChannels']}")
            print(f"  Sample Rate: {int(info['defaultSampleRate'])}")
            break
    
    if tonor_index is None:
        print("ERROR: TONOR microphone not found!")
        return
    
    # Test at different sample rates
    for sample_rate in [16000, 48000]:
        print(f"\nTesting at {sample_rate}Hz...")
        
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                input=True,
                input_device_index=tonor_index,
                frames_per_buffer=1024
            )
            
            print("Recording for 5 seconds... Speak into the microphone!")
            
            max_levels = []
            rms_levels = []
            
            start_time = time.time()
            while time.time() - start_time < 5:
                data = stream.read(1024, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                max_level = np.max(np.abs(audio_data))
                rms_level = np.sqrt(np.mean(audio_data.astype(float)**2))
                
                max_levels.append(max_level)
                rms_levels.append(rms_level)
                
                # Print live levels
                bar_length = int(max_level / 32768 * 50)
                bar = '=' * bar_length
                print(f"  Level: {max_level:5d} |{bar:<50}|", end='\r')
            
            stream.stop_stream()
            stream.close()
            
            print(f"\n  Statistics:")
            print(f"    Peak level: {max(max_levels)}")
            print(f"    Average max: {np.mean(max_levels):.0f}")
            print(f"    Average RMS: {np.mean(rms_levels):.0f}")
            
        except Exception as e:
            print(f"  Error at {sample_rate}Hz: {e}")
    
    pa.terminate()
    
    print("\n" + "=" * 50)
    print("Microphone Quality Assessment:")
    peak = max(max_levels) if max_levels else 0
    if peak > 20000:
        print("EXCELLENT: High gain microphone, perfect for wake word detection")
    elif peak > 10000:
        print("GOOD: Adequate gain for wake word detection")
    elif peak > 5000:
        print("FAIR: May need audio gain boost")
    else:
        print("POOR: Microphone gain too low, needs significant amplification")

if __name__ == "__main__":
    test_microphone()