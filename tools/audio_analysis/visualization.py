"""
Audio visualization module

Provides comprehensive visualization functions for audio analysis including:
- Waveform plots
- Spectrograms
- Frequency response plots
- Gain cascade visualization
- Metrics comparison charts
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple
import warnings

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.gridspec import GridSpec
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib not available - visualization disabled")
    Figure = None

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class AudioVisualizer:
    """Generate comprehensive audio visualizations"""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize visualizer
        
        Args:
            style: Matplotlib style to use
        """
        if MATPLOTLIB_AVAILABLE:
            try:
                plt.style.use(style)
            except:
                # Fallback if style not available
                plt.style.use('default')
        
        self.colormap = 'viridis'
    
    def plot_waveform(self, audio_data: np.ndarray, sample_rate: int,
                     title: str = "Audio Waveform") -> Optional[Figure]:
        """
        Plot audio waveform with amplitude envelope
        
        Args:
            audio_data: Audio samples
            sample_rate: Sample rate in Hz
            title: Plot title
            
        Returns:
            Matplotlib figure or None if not available
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        # Create time axis
        time_axis = np.arange(len(audio_data)) / sample_rate
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Main waveform
        ax1.plot(time_axis, audio_data, linewidth=0.5, alpha=0.7, color='blue')
        ax1.fill_between(time_axis, audio_data, 0, alpha=0.3, color='blue')
        ax1.set_ylabel('Amplitude')
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        
        # Add clipping indicators
        clipping_threshold = 0.99
        clipped_samples = np.abs(audio_data) > clipping_threshold
        if np.any(clipped_samples):
            clipped_times = time_axis[clipped_samples]
            ax1.scatter(clipped_times, audio_data[clipped_samples], 
                       color='red', s=10, alpha=0.5, label='Clipping')
            ax1.legend()
        
        # RMS envelope (with moving window)
        window_size = int(sample_rate * 0.02)  # 20ms window
        if len(audio_data) > window_size:
            rms_envelope = np.array([
                np.sqrt(np.mean(audio_data[i:i+window_size]**2))
                for i in range(0, len(audio_data) - window_size, window_size // 2)
            ])
            rms_time = np.arange(len(rms_envelope)) * (window_size / 2) / sample_rate
            
            ax2.plot(rms_time, rms_envelope, linewidth=2, color='green', label='RMS')
            ax2.fill_between(rms_time, rms_envelope, 0, alpha=0.3, color='green')
            ax2.set_ylabel('RMS Level')
            ax2.set_xlabel('Time (s)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_spectrogram(self, audio_data: np.ndarray, sample_rate: int,
                        title: str = "Spectrogram") -> Optional[Figure]:
        """
        Plot audio spectrogram
        
        Args:
            audio_data: Audio samples
            sample_rate: Sample rate in Hz
            title: Plot title
            
        Returns:
            Matplotlib figure or None
        """
        if not MATPLOTLIB_AVAILABLE or not SCIPY_AVAILABLE:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Compute spectrogram
        nperseg = min(512, len(audio_data) // 8)
        noverlap = nperseg // 2
        
        frequencies, times, Sxx = signal.spectrogram(
            audio_data,
            sample_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            window='hann'
        )
        
        # Convert to dB
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        # Plot spectrogram
        im = ax.pcolormesh(times, frequencies, Sxx_db,
                          shading='gouraud',
                          cmap=self.colormap,
                          vmin=np.percentile(Sxx_db, 10),
                          vmax=np.percentile(Sxx_db, 99))
        
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')
        ax.set_title(title)
        ax.set_ylim(0, min(sample_rate // 2, 12000))  # Cap at 12kHz for visibility
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power (dB)', rotation=270, labelpad=20)
        
        plt.tight_layout()
        return fig
    
    def plot_frequency_response(self, audio_data: np.ndarray, sample_rate: int,
                               title: str = "Frequency Response") -> Optional[Figure]:
        """
        Plot frequency response
        
        Args:
            audio_data: Audio samples
            sample_rate: Sample rate in Hz
            title: Plot title
            
        Returns:
            Matplotlib figure or None
        """
        if not MATPLOTLIB_AVAILABLE or not SCIPY_AVAILABLE:
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Compute FFT
        fft_size = min(8192, len(audio_data))
        window = signal.windows.blackman(fft_size)
        
        if len(audio_data) >= fft_size:
            windowed = audio_data[:fft_size] * window
        else:
            windowed = np.pad(audio_data, (0, fft_size - len(audio_data))) * window
        
        fft_data = np.fft.rfft(windowed)
        frequencies = np.fft.rfftfreq(fft_size, 1/sample_rate)
        
        # Magnitude response
        magnitude_db = 20 * np.log10(np.abs(fft_data) + 1e-10)
        
        ax1.semilogx(frequencies[1:], magnitude_db[1:], linewidth=1)
        ax1.set_ylabel('Magnitude (dB)')
        ax1.set_title(title)
        ax1.grid(True, which='both', alpha=0.3)
        ax1.set_xlim(20, sample_rate // 2)
        
        # Add frequency bands
        bands = [
            (20, 60, 'Sub-bass'),
            (60, 250, 'Bass'),
            (250, 500, 'Low-mid'),
            (500, 2000, 'Mid'),
            (2000, 4000, 'Upper-mid'),
            (4000, 6000, 'Presence'),
            (6000, 12000, 'Brilliance')
        ]
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(bands)))
        for (low, high, name), color in zip(bands, colors):
            mask = (frequencies >= low) & (frequencies <= high)
            if np.any(mask):
                ax1.axvspan(low, high, alpha=0.1, color=color, label=name)
        
        ax1.legend(loc='upper right', fontsize=8)
        
        # Phase response
        phase = np.angle(fft_data)
        ax2.semilogx(frequencies[1:], np.degrees(phase[1:]), linewidth=1, color='orange')
        ax2.set_ylabel('Phase (degrees)')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.grid(True, which='both', alpha=0.3)
        ax2.set_xlim(20, sample_rate // 2)
        
        plt.tight_layout()
        return fig
    
    def plot_gain_cascade(self, stage_metrics: List[Any]) -> Optional[Figure]:
        """
        Plot gain cascade through pipeline stages
        
        Args:
            stage_metrics: List of StageMetrics objects
            
        Returns:
            Matplotlib figure or None
        """
        if not MATPLOTLIB_AVAILABLE or not stage_metrics:
            return None
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Extract data
        stages = [m.stage_name for m in stage_metrics]
        rms_values = [m.rms for m in stage_metrics]
        peak_values = [m.peak for m in stage_metrics]
        clipping_ratios = [m.clipping_ratio * 100 for m in stage_metrics]
        thd_values = [m.thd for m in stage_metrics]
        
        x_pos = np.arange(len(stages))
        
        # RMS and Peak levels
        ax1.plot(x_pos, rms_values, 'o-', label='RMS', linewidth=2, markersize=8)
        ax1.plot(x_pos, peak_values, 's-', label='Peak', linewidth=2, markersize=8)
        ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Clipping threshold')
        ax1.set_ylabel('Level')
        ax1.set_title('Gain Cascade Through Pipeline Stages')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, max(1.2, max(peak_values) * 1.1) if peak_values else 1.2)
        
        # Clipping percentage
        colors = ['red' if c > 1 else 'orange' if c > 0.1 else 'green' for c in clipping_ratios]
        bars = ax2.bar(x_pos, clipping_ratios, color=colors, alpha=0.7)
        ax2.set_ylabel('Clipping (%)')
        ax2.axhline(y=1, color='orange', linestyle='--', alpha=0.5, label='Warning threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, clipping_ratios):
            if value > 0.01:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{value:.2f}%', ha='center', va='bottom', fontsize=8)
        
        # THD
        ax3.plot(x_pos, thd_values, 'd-', color='purple', linewidth=2, markersize=8)
        ax3.set_ylabel('THD (%)')
        ax3.set_xlabel('Pipeline Stage')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(stages, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='5% THD')
        ax3.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_metrics_comparison(self, stage_metrics: List[Any]) -> Optional[Figure]:
        """
        Plot comprehensive metrics comparison across stages
        
        Args:
            stage_metrics: List of StageMetrics objects
            
        Returns:
            Matplotlib figure or None
        """
        if not MATPLOTLIB_AVAILABLE or not stage_metrics:
            return None
        
        # Create figure with grid layout
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        stages = [m.stage_name for m in stage_metrics]
        x_pos = np.arange(len(stages))
        
        # SNR comparison
        ax1 = fig.add_subplot(gs[0, 0])
        snr_values = [m.snr for m in stage_metrics]
        ax1.bar(x_pos, snr_values, color='blue', alpha=0.7)
        ax1.set_ylabel('SNR (dB)')
        ax1.set_title('Signal-to-Noise Ratio')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(stages, rotation=45, ha='right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # DC Offset
        ax2 = fig.add_subplot(gs[0, 1])
        dc_values = [abs(m.dc_offset) for m in stage_metrics]
        ax2.bar(x_pos, dc_values, color='orange', alpha=0.7)
        ax2.set_ylabel('|DC Offset|')
        ax2.set_title('DC Offset (Absolute)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(stages, rotation=45, ha='right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Peak-to-Average Ratio
        ax3 = fig.add_subplot(gs[1, 0])
        par_values = [m.peak_to_average for m in stage_metrics]
        ax3.plot(x_pos, par_values, 'go-', linewidth=2, markersize=8)
        ax3.set_ylabel('Peak-to-Average (dB)')
        ax3.set_title('Crest Factor')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(stages, rotation=45, ha='right', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Frequency response heatmap
        ax4 = fig.add_subplot(gs[1, 1])
        if stage_metrics[0].frequency_response:
            freq_bands = list(stage_metrics[0].frequency_response.keys())
            freq_matrix = np.array([
                [m.frequency_response.get(band, -100) for band in freq_bands]
                for m in stage_metrics
            ])
            
            im = ax4.imshow(freq_matrix.T, aspect='auto', cmap='RdYlBu_r',
                           vmin=-60, vmax=0)
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(stages, rotation=45, ha='right', fontsize=8)
            ax4.set_yticks(range(len(freq_bands)))
            ax4.set_yticklabels(freq_bands, fontsize=8)
            ax4.set_title('Frequency Response (dB)')
            plt.colorbar(im, ax=ax4)
        
        # Combined metrics radar chart
        ax5 = fig.add_subplot(gs[2, :], projection='polar')
        
        # Select key metrics for radar chart
        metrics_names = ['RMS', 'Peak', 'THD', 'SNR/100', 'Clipping%']
        
        # Normalize metrics to 0-1 range for visualization
        for i, metrics in enumerate(stage_metrics[:3]):  # Show first 3 stages
            values = [
                metrics.rms,
                metrics.peak,
                min(metrics.thd / 10, 1),  # Normalize THD
                min(metrics.snr / 100, 1),  # Normalize SNR
                min(metrics.clipping_ratio * 10, 1)  # Scale clipping
            ]
            
            angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False)
            values = np.array(values)
            values = np.concatenate((values, [values[0]]))  # Close the polygon
            angles = np.concatenate((angles, [angles[0]]))
            
            ax5.plot(angles, values, 'o-', linewidth=2, label=metrics.stage_name[:15])
            ax5.fill(angles, values, alpha=0.25)
        
        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(metrics_names, fontsize=8)
        ax5.set_title('Multi-Metric Comparison', pad=20)
        ax5.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax5.grid(True)
        
        plt.suptitle('Audio Pipeline Metrics Analysis', fontsize=14, fontweight='bold')
        return fig
    
    def plot_waterfall_spectrum(self, audio_data: np.ndarray, sample_rate: int,
                               title: str = "Waterfall Spectrum") -> Optional[Figure]:
        """
        Plot 3D waterfall spectrum over time
        
        Args:
            audio_data: Audio samples
            sample_rate: Sample rate in Hz
            title: Plot title
            
        Returns:
            Matplotlib figure or None
        """
        if not MATPLOTLIB_AVAILABLE or not SCIPY_AVAILABLE:
            return None
        
        try:
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            return None
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Compute short-time Fourier transform
        nperseg = 256
        hop_length = nperseg // 4
        
        frequencies, times, Sxx = signal.spectrogram(
            audio_data,
            sample_rate,
            nperseg=nperseg,
            noverlap=nperseg - hop_length,
            window='hann'
        )
        
        # Limit frequency range for visibility
        max_freq_idx = np.where(frequencies <= 8000)[0][-1]
        frequencies = frequencies[:max_freq_idx]
        Sxx = Sxx[:max_freq_idx, :]
        
        # Convert to dB
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        # Create mesh
        T, F = np.meshgrid(times, frequencies)
        
        # Plot waterfall
        surf = ax.plot_surface(T, F, Sxx_db, cmap=self.colormap,
                              linewidth=0, antialiased=False, alpha=0.8)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_zlabel('Power (dB)')
        ax.set_title(title)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, pad=0.1)
        
        return fig