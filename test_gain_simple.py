#!/usr/bin/env python3
"""
Simple gain calculation test (no dependencies required)
"""


def test_gain_calculations():
    """Test different gain calculations without any dependencies"""
    
    # Test configurations
    test_configs = [
        {"name": "Fixed 2.0x (old)", "fixed_gain": 2.0, "use_fixed": True},
        {"name": "Fixed 3.5x (new)", "fixed_gain": 3.5, "use_fixed": True},
        {"name": "Fixed 4.0x", "fixed_gain": 4.0, "use_fixed": True},
        {"name": "Dynamic 2-5x", "use_fixed": False, "min": 2.0, "max": 5.0, "target_rms": 0.04},
        {"name": "Dynamic 3-6x", "use_fixed": False, "min": 3.0, "max": 6.0, "target_rms": 0.04},
    ]
    
    print("Wake Word Gain Calculation Testing")
    print("=" * 60)
    print("Testing gain calculations for different input RMS levels")
    print("Note: This tests gain calculations only, not actual model predictions")
    print("=" * 60)
    
    # Test with different audio RMS levels
    test_rms_levels = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]
    
    for test_config in test_configs:
        print(f"\n{test_config['name']}:")
        print("-" * 40)
        
        for rms_level in test_rms_levels:
            if test_config["use_fixed"]:
                applied_gain = test_config["fixed_gain"]
                gain_type = "fixed"
            else:
                # Calculate bounded dynamic gain
                target_rms = test_config["target_rms"]
                raw_gain = target_rms / rms_level
                applied_gain = max(test_config["min"], min(test_config["max"], raw_gain))  # clip equivalent
                gain_type = "dynamic"
            
            # Calculate final RMS after gain
            final_rms = rms_level * applied_gain
            
            # Analysis
            status = ""
            if final_rms < 0.05:
                status = " [LOW - may not detect]"
            elif final_rms > 0.2:
                status = " [HIGH - may clip/distort]"
            elif 0.1 <= final_rms <= 0.2:
                status = " [GOOD - optimal range]"
            
            print(f"  RMS={rms_level:.3f} -> Gain={applied_gain:.1f}x ({gain_type}) -> Final RMS={final_rms:.3f}{status}")
    
    print("\n" + "=" * 60)
    print("Analysis:")
    print("- Fixed gain provides consistent amplification regardless of input level")
    print("- Dynamic gain adapts to input level but is bounded to prevent extremes") 
    print("- Target final RMS should be in 0.1-0.2 range for best wake word detection")
    print("- Values below 0.05 may be too quiet for reliable detection")
    print("- Values above 0.2 may cause clipping or distortion")
    
    print("\nRecommendations based on log analysis:")
    print("- Previous 2.0x gain was insufficient (max confidence only 0.010612)")
    print("- New 3.5x gain should improve detection significantly") 
    print("- If 3.5x still insufficient, try 4.0x before going higher")
    print("- Monitor for false positives with higher gains")


if __name__ == "__main__":
    test_gain_calculations()