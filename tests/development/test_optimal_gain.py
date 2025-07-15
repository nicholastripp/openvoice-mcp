#!/usr/bin/env python3
"""
Test to find optimal audio gain without clipping
"""
import asyncio
import sys
import time
import numpy as np
sys.path.insert(0, '/home/ansible/ha-realtime-assist/src')

from config import load_config
from wake_word import create_wake_word_detector

# Track detections and audio stats
detections = []
audio_stats = {
    'peaks': [],
    'gain': 0,
    'clipped': 0,
    'total': 0
}

def on_detection(keyword, confidence):
    detections.append((keyword, time.time()))
    print(f"\n*** WAKE WORD DETECTED: '{keyword}' ***\n")

async def test_gain_level(gain_value):
    """Test a specific gain level"""
    print(f"\nTesting gain level: {gain_value}")
    print("-" * 40)
    
    # Reset stats
    detections.clear()
    audio_stats['peaks'].clear()
    audio_stats['gain'] = gain_value
    audio_stats['clipped'] = 0
    audio_stats['total'] = 0
    
    # Load config and override gain
    config = load_config('/home/ansible/ha-realtime-assist/config/config.yaml')
    config.wake_word.audio_gain = gain_value
    
    print(f"Configuration:")
    print(f"  Wake word: {config.wake_word.model}")
    print(f"  Sensitivity: {config.wake_word.sensitivity}")
    print(f"  Audio gain: {gain_value}")
    
    # Create detector
    detector = create_wake_word_detector(config.wake_word)
    detector.add_detection_callback(on_detection)
    
    # Monitor audio levels
    original_process = detector.process_audio
    
    def process_with_monitoring(audio_data, sample_rate):
        # Get pre-gain stats
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        max_level = np.max(np.abs(audio_array))
        
        # Calculate post-gain level
        post_gain = max_level * gain_value
        audio_stats['peaks'].append(post_gain)
        audio_stats['total'] += 1
        
        if post_gain > 32767:
            audio_stats['clipped'] += 1
        
        # Call original
        return original_process(audio_data, sample_rate)
    
    detector.process_audio = process_with_monitoring
    
    print("\nStarting detector...")
    await detector.start()
    
    print("Recording for 20 seconds... Speak normally and say 'picovoice'\n")
    
    # Audio input
    import pyaudio
    pa = pyaudio.PyAudio()
    
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=48000,
        input=True,
        frames_per_buffer=600
    )
    
    start_time = time.time()
    
    try:
        while time.time() - start_time < 20:
            audio_data = stream.read(600, exception_on_overflow=False)
            detector.process_audio(audio_data, 48000)
            
            # Status update
            if audio_stats['total'] % 50 == 0 and audio_stats['peaks']:
                current_peak = max(audio_stats['peaks'][-50:]) if len(audio_stats['peaks']) >= 50 else max(audio_stats['peaks'])
                clip_pct = (audio_stats['clipped'] / audio_stats['total'] * 100) if audio_stats['total'] > 0 else 0
                print(f"Peak: {int(current_peak):5d} | Clipping: {clip_pct:4.1f}% | Detections: {len(detections)}", end='\r')
            
            await asyncio.sleep(0.001)
            
    except KeyboardInterrupt:
        pass
    
    # Cleanup
    stream.stop_stream()
    stream.close()
    pa.terminate()
    await detector.stop()
    
    # Results
    print(f"\n\nResults for gain {gain_value}:")
    if audio_stats['peaks']:
        avg_peak = np.mean(audio_stats['peaks'])
        max_peak = max(audio_stats['peaks'])
        clip_pct = (audio_stats['clipped'] / audio_stats['total'] * 100) if audio_stats['total'] > 0 else 0
        
        print(f"  Average peak: {int(avg_peak)}")
        print(f"  Maximum peak: {int(max_peak)}")
        print(f"  Clipping: {clip_pct:.1f}% ({audio_stats['clipped']}/{audio_stats['total']} frames)")
        print(f"  Detections: {len(detections)}")
        
        # Assessment
        if clip_pct > 5:
            print("  Assessment: TOO HIGH - Excessive clipping")
        elif clip_pct > 1:
            print("  Assessment: HIGH - Some clipping")
        elif max_peak < 10000:
            print("  Assessment: TOO LOW - May miss detections")
        elif max_peak > 25000:
            print("  Assessment: GOOD - High levels without excessive clipping")
        else:
            print("  Assessment: ACCEPTABLE - Moderate levels")
    
    return {
        'gain': gain_value,
        'avg_peak': int(avg_peak) if audio_stats['peaks'] else 0,
        'max_peak': int(max_peak) if audio_stats['peaks'] else 0,
        'clip_pct': clip_pct if audio_stats['peaks'] else 0,
        'detections': len(detections)
    }

async def main():
    print("Optimal Audio Gain Test")
    print("=" * 50)
    print("This test will help find the best audio gain setting")
    print("Please speak normally and say 'picovoice' several times\n")
    
    # Test different gain levels
    results = []
    for gain in [2.0, 2.5, 3.0, 3.5, 4.0]:
        result = await test_gain_level(gain)
        results.append(result)
        await asyncio.sleep(2)  # Brief pause between tests
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"{'Gain':>4} | {'Avg Peak':>8} | {'Max Peak':>8} | {'Clip %':>6} | {'Detections':>10}")
    print("-" * 50)
    
    for r in results:
        print(f"{r['gain']:>4.1f} | {r['avg_peak']:>8d} | {r['max_peak']:>8d} | {r['clip_pct']:>6.1f} | {r['detections']:>10d}")
    
    # Recommendation
    print("\nRecommendation:")
    # Find best gain: highest without excessive clipping
    good_gains = [r for r in results if r['clip_pct'] < 2.0]
    if good_gains:
        best = max(good_gains, key=lambda x: x['avg_peak'])
        print(f"Optimal gain: {best['gain']} (avg peak: {best['avg_peak']}, clipping: {best['clip_pct']:.1f}%)")
    else:
        print("All gains caused excessive clipping - microphone may be too loud")

if __name__ == "__main__":
    asyncio.run(main())