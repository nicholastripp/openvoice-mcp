#!/usr/bin/env python3
"""
Test wake word detection in test mode (without OpenAI)
"""
import asyncio
import subprocess
import time

def main():
    print("Starting wake word test mode...")
    print("This will run the assistant with wake word detection only (no OpenAI connection)")
    print("")
    print("Configuration:")
    print("- Wake word model: alexa")
    print("- Sensitivity: 0.000001 (ultra low for testing)")
    print("- Test mode: enabled (no OpenAI connection)")
    print("")
    print("When you say the wake word, you should:")
    print("1. See detection in the logs")
    print("2. Hear a beep confirmation sound")
    print("")
    print("Press Ctrl+C to stop")
    print("")
    
    try:
        # Run the main application with test mode enabled
        subprocess.run([
            "python", "src/main.py",
            "--config", "config/config.yaml",
            "--log-level", "INFO"
        ])
    except KeyboardInterrupt:
        print("\nTest stopped by user")

if __name__ == "__main__":
    main()