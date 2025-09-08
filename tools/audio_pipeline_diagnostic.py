#!/usr/bin/env python3
"""
Audio Pipeline Diagnostic Tool for HA Realtime Voice Assistant

This tool captures and analyzes audio at each transformation stage in the pipeline
to identify where distortion occurs and provide optimization recommendations.

Usage:
    python tools/audio_pipeline_diagnostic.py --mode realtime
    python tools/audio_pipeline_diagnostic.py --mode record --duration 10
    python tools/audio_pipeline_diagnostic.py --mode analyze --input recording.npz
    python tools/audio_pipeline_diagnostic.py --mode compare --config1 config1.yaml --config2 config2.yaml
"""

import sys
import os
import argparse
import asyncio
import numpy as np
import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import audio pipeline components
from audio.capture import AudioCapture
from audio.agc import AutomaticGainControl
from config import load_config, AudioConfig
from utils.logger import setup_logging, get_logger

# Import scipy for signal processing
try:
    from scipy import signal
    from scipy.fft import fft, fftfreq
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available - some features will be limited")

# Import sounddevice for audio I/O
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    warnings.warn("sounddevice not available - audio capture will be limited")

# Import matplotlib for visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib not available - visualization features disabled")

# Import our analysis modules
from audio_analysis.metrics import AudioMetrics
from audio_analysis.visualization import AudioVisualizer
from audio_analysis.stage_capture import PipelineStageCapture


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage"""
    stage_name: str
    rms: float
    peak: float
    peak_to_average: float
    thd: float
    snr: float
    clipping_count: int
    clipping_ratio: float
    dc_offset: float
    frequency_response: Dict[str, float]
    timestamp: float


@dataclass
class DiagnosticReport:
    """Complete diagnostic report for all pipeline stages"""
    device_info: Dict[str, Any]
    configuration: Dict[str, Any]
    stages: List[StageMetrics]
    recommendations: List[str]
    overall_assessment: str
    timestamp: str
    duration: float


class AudioPipelineDiagnostic:
    """
    Comprehensive audio pipeline diagnostic tool
    
    Captures and analyzes audio at each transformation stage to identify
    distortion sources and provide optimization recommendations.
    """
    
    # Pipeline stage definitions
    PIPELINE_STAGES = [
        "raw_input",           # Stage 1: Raw microphone input
        "volume_adjusted",     # Stage 2: After input volume multiplication
        "agc_processed",       # Stage 3: After AGC processing
        "resampled_24k",      # Stage 4: After resampling to 24kHz
        "pcm16_converted",    # Stage 5: After PCM16 conversion
        "wake_word_gain",     # Stage 6: After wake word gain application
        "highpass_filtered"   # Stage 7: After high-pass filter
    ]
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the diagnostic tool
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = get_logger("AudioPipelineDiagnostic")
        self.config = load_config(config_path)
        
        # Initialize components
        self.metrics = AudioMetrics()
        self.visualizer = AudioVisualizer() if MATPLOTLIB_AVAILABLE else None
        self.stage_capture = PipelineStageCapture(self.config)
        
        # Storage for captured audio
        self.stage_audio: Dict[str, np.ndarray] = {}
        self.stage_metrics: List[StageMetrics] = []
        
        # Device information
        self.device_info = self._get_device_info()
        
        # Timing
        self.start_time = None
        self.duration = 0
        
        self.logger.info(f"Audio Pipeline Diagnostic initialized")
        self.logger.info(f"Configuration loaded from: {config_path}")
        
    def _get_device_info(self) -> Dict[str, Any]:
        """Get audio device information"""
        info = {
            "input_device": self.config.audio.input_device,
            "sample_rate": self.config.audio.sample_rate,
            "channels": self.config.audio.channels,
            "chunk_size": self.config.audio.chunk_size
        }
        
        if SOUNDDEVICE_AVAILABLE:
            try:
                if self.config.audio.input_device == "default":
                    device_id = sd.default.device[0]
                else:
                    device_id = int(self.config.audio.input_device)
                
                device = sd.query_devices(device_id)
                info.update({
                    "device_name": device['name'],
                    "max_input_channels": device['max_input_channels'],
                    "default_samplerate": device['default_samplerate']
                })
            except Exception as e:
                self.logger.warning(f"Could not query device info: {e}")
        
        return info
    
    async def capture_stage(self, stage_name: str, duration: float = 10.0) -> np.ndarray:
        """
        Capture audio at a specific pipeline stage
        
        Args:
            stage_name: Name of the pipeline stage
            duration: Capture duration in seconds
            
        Returns:
            Captured audio data
        """
        self.logger.info(f"Capturing stage: {stage_name} for {duration}s")
        
        # Use our stage capture module to hook into the pipeline
        audio_data = await self.stage_capture.capture(stage_name, duration)
        
        # Store the captured audio
        self.stage_audio[stage_name] = audio_data
        
        return audio_data
    
    def calculate_metrics(self, audio_data: np.ndarray, sample_rate: int, stage_name: str) -> StageMetrics:
        """
        Calculate comprehensive metrics for audio data
        
        Args:
            audio_data: Audio data to analyze
            sample_rate: Sample rate of the audio
            stage_name: Name of the pipeline stage
            
        Returns:
            Calculated metrics
        """
        # Basic metrics
        rms = self.metrics.calculate_rms(audio_data)
        peak = self.metrics.calculate_peak(audio_data)
        peak_to_average = self.metrics.calculate_peak_to_average(audio_data)
        
        # Distortion metrics
        thd = self.metrics.calculate_thd(audio_data, sample_rate)
        snr = self.metrics.calculate_snr(audio_data)
        
        # Clipping analysis
        clipping_count, clipping_ratio = self.metrics.analyze_clipping(audio_data)
        
        # DC offset
        dc_offset = self.metrics.calculate_dc_offset(audio_data)
        
        # Frequency response
        freq_response = self.metrics.calculate_frequency_response(audio_data, sample_rate)
        
        return StageMetrics(
            stage_name=stage_name,
            rms=rms,
            peak=peak,
            peak_to_average=peak_to_average,
            thd=thd,
            snr=snr,
            clipping_count=clipping_count,
            clipping_ratio=clipping_ratio,
            dc_offset=dc_offset,
            frequency_response=freq_response,
            timestamp=time.time()
        )
    
    def visualize_results(self, output_dir: str = "reports/figures") -> None:
        """
        Generate visualization plots for all captured stages
        
        Args:
            output_dir: Directory to save plots
        """
        if not self.visualizer:
            self.logger.warning("Visualization not available - matplotlib not installed")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate individual stage plots
        for stage_name, audio_data in self.stage_audio.items():
            self.logger.info(f"Generating visualizations for {stage_name}")
            
            # Waveform plot
            fig = self.visualizer.plot_waveform(
                audio_data, 
                self.config.audio.sample_rate,
                title=f"Waveform - {stage_name}"
            )
            fig.savefig(f"{output_dir}/waveform_{stage_name}_{timestamp}.png", dpi=150)
            plt.close(fig)
            
            # Spectrogram
            fig = self.visualizer.plot_spectrogram(
                audio_data,
                self.config.audio.sample_rate,
                title=f"Spectrogram - {stage_name}"
            )
            fig.savefig(f"{output_dir}/spectrogram_{stage_name}_{timestamp}.png", dpi=150)
            plt.close(fig)
            
            # Frequency response
            fig = self.visualizer.plot_frequency_response(
                audio_data,
                self.config.audio.sample_rate,
                title=f"Frequency Response - {stage_name}"
            )
            fig.savefig(f"{output_dir}/freq_response_{stage_name}_{timestamp}.png", dpi=150)
            plt.close(fig)
        
        # Generate comparison plots
        if len(self.stage_audio) > 1:
            self.logger.info("Generating stage comparison plots")
            
            # Gain cascade visualization
            fig = self.visualizer.plot_gain_cascade(self.stage_metrics)
            fig.savefig(f"{output_dir}/gain_cascade_{timestamp}.png", dpi=150)
            plt.close(fig)
            
            # Metrics comparison
            fig = self.visualizer.plot_metrics_comparison(self.stage_metrics)
            fig.savefig(f"{output_dir}/metrics_comparison_{timestamp}.png", dpi=150)
            plt.close(fig)
        
        self.logger.info(f"Visualizations saved to {output_dir}")
    
    def generate_report(self) -> DiagnosticReport:
        """
        Generate comprehensive diagnostic report
        
        Returns:
            Complete diagnostic report
        """
        # Analyze metrics and generate recommendations
        recommendations = self._generate_recommendations()
        overall_assessment = self._generate_overall_assessment()
        
        report = DiagnosticReport(
            device_info=self.device_info,
            configuration={
                "input_volume": self.config.audio.input_volume,
                "agc_enabled": self.config.audio.agc_enabled,
                "sample_rate": self.config.audio.sample_rate,
                "wake_word_gain": self.config.wake_word.audio_gain if hasattr(self.config, 'wake_word') else 1.0
            },
            stages=self.stage_metrics,
            recommendations=recommendations,
            overall_assessment=overall_assessment,
            timestamp=datetime.now().isoformat(),
            duration=self.duration
        )
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate specific recommendations based on analysis"""
        recommendations = []
        
        if not self.stage_metrics:
            return ["No data collected - run diagnostic first"]
        
        # Check for clipping at each stage
        for metrics in self.stage_metrics:
            if metrics.clipping_ratio > 0.01:  # More than 1% clipping
                recommendations.append(
                    f"High clipping detected at {metrics.stage_name}: {metrics.clipping_ratio:.1%}. "
                    f"Reduce gain before this stage."
                )
        
        # Check for cumulative gain
        if len(self.stage_metrics) > 1:
            first_rms = self.stage_metrics[0].rms
            last_rms = self.stage_metrics[-1].rms
            total_gain = last_rms / max(first_rms, 0.001)
            
            if total_gain > 10:
                recommendations.append(
                    f"Excessive cumulative gain detected: {total_gain:.1f}x. "
                    f"Reduce input_volume and wake_word gain settings."
                )
        
        # Check for DC offset
        for metrics in self.stage_metrics:
            if abs(metrics.dc_offset) > 0.01:
                recommendations.append(
                    f"DC offset detected at {metrics.stage_name}: {metrics.dc_offset:.4f}. "
                    f"Enable high-pass filtering or check hardware."
                )
        
        # Check THD levels
        for metrics in self.stage_metrics:
            if metrics.thd > 5.0:  # THD > 5%
                recommendations.append(
                    f"High distortion at {metrics.stage_name}: THD={metrics.thd:.1f}%. "
                    f"Check resampling quality or reduce gain."
                )
        
        # Specific stage recommendations
        if "resampled_24k" in [m.stage_name for m in self.stage_metrics]:
            resampled_metrics = next(m for m in self.stage_metrics if m.stage_name == "resampled_24k")
            if resampled_metrics.thd > 3.0:
                recommendations.append(
                    "Resampling introduces distortion. Consider using higher quality resampling "
                    "(e.g., scipy.signal.resample_poly instead of resample)"
                )
        
        # Wake word specific
        if any(m.peak < 0.01 for m in self.stage_metrics):
            recommendations.append(
                "Audio levels too low for reliable wake word detection. "
                "Increase microphone gain in ALSA mixer (alsamixer)."
            )
        
        return recommendations if recommendations else ["Audio pipeline appears healthy"]
    
    def _generate_overall_assessment(self) -> str:
        """Generate overall assessment of audio pipeline health"""
        if not self.stage_metrics:
            return "No assessment available - no data collected"
        
        # Count issues
        issues = {
            "critical": 0,
            "warning": 0,
            "info": 0
        }
        
        for metrics in self.stage_metrics:
            if metrics.clipping_ratio > 0.05:
                issues["critical"] += 1
            elif metrics.clipping_ratio > 0.01:
                issues["warning"] += 1
            
            if metrics.thd > 10.0:
                issues["critical"] += 1
            elif metrics.thd > 5.0:
                issues["warning"] += 1
            
            if metrics.peak > 0.99:
                issues["warning"] += 1
            elif metrics.peak < 0.01:
                issues["warning"] += 1
        
        # Generate assessment
        if issues["critical"] > 0:
            return (f"CRITICAL: {issues['critical']} critical issues found. "
                   f"Audio quality severely degraded. Immediate action required.")
        elif issues["warning"] > 0:
            return (f"WARNING: {issues['warning']} warnings found. "
                   f"Audio quality suboptimal. Adjustments recommended.")
        else:
            return "GOOD: Audio pipeline healthy. No significant issues detected."
    
    async def run_diagnostic(self, mode: str = "realtime", duration: float = 10.0) -> DiagnosticReport:
        """
        Run complete diagnostic analysis
        
        Args:
            mode: Diagnostic mode ("realtime", "record", "analyze")
            duration: Duration for capture modes
            
        Returns:
            Diagnostic report
        """
        self.start_time = time.time()
        self.duration = duration
        
        self.logger.info(f"Starting diagnostic in {mode} mode for {duration}s")
        
        if mode == "realtime":
            await self._run_realtime_diagnostic(duration)
        elif mode == "record":
            await self._run_recording_diagnostic(duration)
        elif mode == "analyze":
            self._run_analysis_diagnostic()
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        # Generate and return report
        report = self.generate_report()
        
        # Save report
        self._save_report(report)
        
        return report
    
    async def _run_realtime_diagnostic(self, duration: float) -> None:
        """Run real-time diagnostic with live monitoring"""
        print("\n" + "="*70)
        print("REAL-TIME AUDIO PIPELINE DIAGNOSTIC")
        print("="*70)
        print(f"Duration: {duration} seconds")
        print("Monitoring all pipeline stages...\n")
        
        # Capture each stage sequentially with progress
        for i, stage in enumerate(self.PIPELINE_STAGES, 1):
            print(f"[{i}/{len(self.PIPELINE_STAGES)}] Capturing {stage}...")
            
            # Capture audio at this stage
            audio_data = await self.capture_stage(stage, duration=2.0)  # Short capture for each stage
            
            # Calculate metrics
            metrics = self.calculate_metrics(
                audio_data,
                self.config.audio.sample_rate,
                stage
            )
            self.stage_metrics.append(metrics)
            
            # Display real-time metrics
            self._display_stage_metrics(metrics)
        
        print("\n" + "="*70)
        print("Diagnostic complete. Generating visualizations...")
        
        # Generate visualizations
        if self.visualizer:
            self.visualize_results()
    
    async def _run_recording_diagnostic(self, duration: float) -> None:
        """Run recording diagnostic - capture all stages for analysis"""
        print(f"\nRecording {duration}s of audio through all pipeline stages...")
        print("Please speak normally during the recording.")
        
        # TODO: Implement parallel capture of all stages
        # For now, capture sequentially
        for stage in self.PIPELINE_STAGES:
            audio_data = await self.capture_stage(stage, duration)
            metrics = self.calculate_metrics(
                audio_data,
                self.config.audio.sample_rate,
                stage
            )
            self.stage_metrics.append(metrics)
        
        # Save recording for later analysis
        self._save_recording()
    
    def _run_analysis_diagnostic(self) -> None:
        """Analyze previously recorded audio data"""
        # TODO: Load and analyze saved recording
        pass
    
    def _display_stage_metrics(self, metrics: StageMetrics) -> None:
        """Display metrics for a single stage in real-time"""
        print(f"\n  Stage: {metrics.stage_name}")
        print(f"    RMS: {metrics.rms:.6f} | Peak: {metrics.peak:.6f}")
        print(f"    THD: {metrics.thd:.2f}% | SNR: {metrics.snr:.1f} dB")
        print(f"    Clipping: {metrics.clipping_ratio:.2%} ({metrics.clipping_count} samples)")
        print(f"    DC Offset: {metrics.dc_offset:.6f}")
        
        # Visual indicator
        level_bar = int(metrics.peak * 50)
        if metrics.clipping_ratio > 0.01:
            indicator = "!" * level_bar + "." * (50 - level_bar) + " [CLIPPING]"
        else:
            indicator = "#" * level_bar + "." * (50 - level_bar)
        print(f"    Level: |{indicator}|")
    
    def _save_report(self, report: DiagnosticReport) -> None:
        """Save diagnostic report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        json_path = f"reports/diagnostic_report_{timestamp}.json"
        with open(json_path, 'w') as f:
            # Convert report to dict for JSON serialization
            report_dict = asdict(report)
            json.dump(report_dict, f, indent=2, default=str)
        self.logger.info(f"Report saved to {json_path}")
        
        # Save CSV metrics
        csv_path = f"reports/metrics_{timestamp}.csv"
        with open(csv_path, 'w', newline='') as f:
            if report.stages:
                fieldnames = [
                    'stage_name', 'rms', 'peak', 'peak_to_average',
                    'thd', 'snr', 'clipping_ratio', 'dc_offset'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for stage in report.stages:
                    writer.writerow({
                        'stage_name': stage.stage_name,
                        'rms': stage.rms,
                        'peak': stage.peak,
                        'peak_to_average': stage.peak_to_average,
                        'thd': stage.thd,
                        'snr': stage.snr,
                        'clipping_ratio': stage.clipping_ratio,
                        'dc_offset': stage.dc_offset
                    })
        self.logger.info(f"Metrics saved to {csv_path}")
    
    def _save_recording(self) -> None:
        """Save captured audio data for later analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        npz_path = f"reports/recording_{timestamp}.npz"
        
        # Save all captured audio stages
        np.savez_compressed(
            npz_path,
            **self.stage_audio,
            sample_rate=self.config.audio.sample_rate,
            config=json.dumps(asdict(self.config.audio))
        )
        
        self.logger.info(f"Recording saved to {npz_path}")


def main():
    """Main entry point for the diagnostic tool"""
    parser = argparse.ArgumentParser(
        description="Audio Pipeline Diagnostic Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--mode",
        choices=["realtime", "record", "analyze", "compare"],
        default="realtime",
        help="Diagnostic mode"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Capture duration in seconds"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--input",
        help="Input file for analyze mode"
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Output directory for reports"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, console=True)
    
    # Create diagnostic tool
    diagnostic = AudioPipelineDiagnostic(args.config)
    
    # Run diagnostic
    try:
        report = asyncio.run(
            diagnostic.run_diagnostic(
                mode=args.mode,
                duration=args.duration
            )
        )
        
        # Display summary
        print("\n" + "="*70)
        print("DIAGNOSTIC SUMMARY")
        print("="*70)
        print(f"Assessment: {report.overall_assessment}")
        print(f"\nRecommendations:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")
        
        print(f"\nFull report saved to reports/")
        
    except KeyboardInterrupt:
        print("\nDiagnostic interrupted by user")
        return 1
    except Exception as e:
        print(f"Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())