#!/usr/bin/env python3
"""
Autonomous wake word testing using macOS audio playback
"""
import subprocess
import time
import asyncio
import os

class AutonomousWakeTest:
    def __init__(self):
        self.audio_files = {
            'picovoice_normal': 'picovoice_normal.aiff',
            'picovoice_slow': 'picovoice_slow.aiff',
            'picovoice_clear': 'picovoice_clear.aiff',
            'alexa': 'alexa_normal.aiff',
            'hey_jarvis': 'hey_jarvis.aiff',
            'test_phrase': 'test_phrase.aiff'
        }
        
    def play_audio(self, filename, volume=50):
        """Play audio file on macOS"""
        try:
            # Set system volume
            subprocess.run(['osascript', '-e', f'set volume output volume {volume}'], check=True)
            # Play audio file
            subprocess.run(['afplay', filename], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error playing audio: {e}")
            return False
    
    async def start_pi_test(self, test_duration=30, gain=3.0):
        """Start wake word test on Raspberry Pi"""
        print(f"\nStarting Pi test with gain={gain}")
        
        # SSH command to start test on Pi
        cmd = f"""ssh ansible@williams "cd ~/ha-realtime-assist && source venv/bin/activate && source .env && timeout {test_duration} python scripts/test_wake_word_quick.py > /tmp/wake_test_{gain}.log 2>&1" &"""
        
        subprocess.Popen(cmd, shell=True)
        await asyncio.sleep(2)  # Let Pi test initialize
        
    def stop_pi_test(self):
        """Stop any running tests on Pi"""
        subprocess.run(['ssh', 'ansible@williams', 'pkill -f test_wake_word_quick.py'], stderr=subprocess.DEVNULL)
    
    async def test_wake_word(self, wake_word='picovoice_normal', gain=3.0, volume=50):
        """Test a specific wake word"""
        print(f"\n{'='*60}")
        print(f"Testing: {wake_word} | Gain: {gain} | Volume: {volume}%")
        print(f"{'='*60}")
        
        # Start Pi test
        await self.start_pi_test(test_duration=20, gain=gain)
        
        # Wait for initialization
        await asyncio.sleep(5)
        
        # Play wake word multiple times
        audio_file = self.audio_files.get(wake_word)
        if audio_file and os.path.exists(audio_file):
            print(f"Playing '{wake_word}' 3 times...")
            for i in range(3):
                print(f"  Attempt {i+1}/3...", end='', flush=True)
                if self.play_audio(audio_file, volume):
                    print(" ✓")
                else:
                    print(" ✗")
                await asyncio.sleep(3)
        
        # Wait for test to complete
        await asyncio.sleep(5)
        
        # Get results from Pi
        result = subprocess.run(
            ['ssh', 'ansible@williams', f'grep -c "DETECTED" /tmp/wake_test_{gain}.log 2>/dev/null || echo 0'],
            capture_output=True, text=True
        )
        detections = int(result.stdout.strip())
        
        # Check for clipping
        result = subprocess.run(
            ['ssh', 'ansible@williams', f'grep "Clipping:" /tmp/wake_test_{gain}.log | tail -1'],
            capture_output=True, text=True
        )
        clipping_info = result.stdout.strip()
        
        print(f"\nResults:")
        print(f"  Detections: {detections}")
        print(f"  Clipping: {clipping_info if clipping_info else 'No clipping data'}")
        
        return {'wake_word': wake_word, 'gain': gain, 'volume': volume, 'detections': detections, 'clipping': clipping_info}
    
    async def test_gain_levels(self):
        """Test different gain levels to find optimal setting"""
        print("Autonomous Wake Word Testing")
        print("Testing different gain levels with real speech\n")
        
        results = []
        
        # First test current gain (3.0) to see clipping
        print("Phase 1: Testing current gain (3.0)")
        result = await self.test_wake_word('picovoice_normal', gain=3.0, volume=30)
        results.append(result)
        
        # Test lower gains
        print("\nPhase 2: Testing lower gain levels")
        for gain in [1.0, 1.5, 2.0]:
            for volume in [30, 50]:
                result = await self.test_wake_word('picovoice_normal', gain=gain, volume=volume)
                results.append(result)
                await asyncio.sleep(2)
        
        # Test different pronunciations with best gain
        print("\nPhase 3: Testing pronunciations")
        best_gain = 1.5  # Adjust based on results
        for wake_word in ['picovoice_slow', 'picovoice_clear']:
            result = await self.test_wake_word(wake_word, gain=best_gain, volume=40)
            results.append(result)
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"{'Wake Word':<20} {'Gain':<6} {'Vol':<5} {'Detections':<12} {'Clipping'}")
        print("-"*60)
        
        for r in results:
            clipping = r['clipping'].split('Clipping: ')[-1] if 'Clipping:' in r['clipping'] else 'N/A'
            print(f"{r['wake_word']:<20} {r['gain']:<6.1f} {r['volume']:<5}% {r['detections']:<12} {clipping}")
        
        # Find best configuration
        successful = [r for r in results if r['detections'] > 0]
        if successful:
            best = max(successful, key=lambda x: x['detections'])
            print(f"\nBest configuration: Gain={best['gain']}, Volume={best['volume']}%")
        else:
            print("\nNo successful detections - check audio setup")

async def main():
    tester = AutonomousWakeTest()
    
    # Make sure no tests are running
    tester.stop_pi_test()
    
    try:
        await tester.test_gain_levels()
    finally:
        # Cleanup
        tester.stop_pi_test()

if __name__ == "__main__":
    asyncio.run(main())