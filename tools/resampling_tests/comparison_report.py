"""
Comparison report generator for resampling analysis

Generates comprehensive reports comparing resampling methods including:
- Quality vs Performance tradeoff matrices
- Visualization plots (frequency response, phase, etc.)
- Recommendations based on use case
- Configuration files for optimal settings
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import seaborn as sns
    sns.set_style('whitegrid')
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn("matplotlib/seaborn not available - plots will be skipped")


@dataclass
class MethodComparison:
    """Comparison data for a single method"""
    name: str
    display_name: str
    quality_score: float
    performance_score: float
    overall_score: float
    thd_percent: float
    snr_db: float
    aliasing_db: float
    latency_ms: float
    cpu_percent: float
    realtime_capable: bool
    pros: List[str]
    cons: List[str]
    use_cases: List[str]


class ComparisonReportGenerator:
    """Generate comprehensive comparison reports for resampling methods"""
    
    def __init__(self, output_dir: Path = Path('reports/resampling')):
        """Initialize report generator"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define scoring weights
        self.quality_weights = {
            'thd': 0.25,
            'snr': 0.25,
            'aliasing': 0.30,
            'frequency_response': 0.10,
            'phase_coherence': 0.10
        }
        
        self.performance_weights = {
            'latency': 0.40,
            'cpu': 0.30,
            'memory': 0.10,
            'jitter': 0.20
        }
        
        # Use case profiles
        self.use_case_profiles = {
            'realtime_voice': {
                'quality_weight': 0.6,
                'performance_weight': 0.4,
                'requirements': {
                    'max_latency_ms': 10,
                    'max_cpu_percent': 50,
                    'min_snr_db': 40,
                    'max_thd_percent': 5
                }
            },
            'wake_word': {
                'quality_weight': 0.7,
                'performance_weight': 0.3,
                'requirements': {
                    'max_latency_ms': 20,
                    'max_cpu_percent': 30,
                    'min_snr_db': 45,
                    'max_thd_percent': 3
                }
            },
            'high_quality': {
                'quality_weight': 0.9,
                'performance_weight': 0.1,
                'requirements': {
                    'max_latency_ms': 50,
                    'max_cpu_percent': 80,
                    'min_snr_db': 60,
                    'max_thd_percent': 1
                }
            },
            'low_power': {
                'quality_weight': 0.4,
                'performance_weight': 0.6,
                'requirements': {
                    'max_latency_ms': 5,
                    'max_cpu_percent': 20,
                    'min_snr_db': 35,
                    'max_thd_percent': 10
                }
            }
        }
    
    def calculate_scores(self, quality_results: List[Dict], 
                        performance_results: List[Dict]) -> Dict[str, MethodComparison]:
        """
        Calculate comprehensive scores for each method
        
        Args:
            quality_results: List of quality measurement dicts
            performance_results: List of performance measurement dicts
            
        Returns:
            Dictionary of method comparisons
        """
        methods = {}
        
        # Group results by method
        quality_by_method = {}
        perf_by_method = {}
        
        for result in quality_results:
            method = result['method']
            if method not in quality_by_method:
                quality_by_method[method] = []
            quality_by_method[method].append(result)
        
        for result in performance_results:
            method = result['method']
            if method not in perf_by_method:
                perf_by_method[method] = []
            perf_by_method[method].append(result)
        
        # Calculate scores for each method
        for method in quality_by_method.keys():
            if method not in perf_by_method:
                continue
            
            # Average quality metrics
            q_results = quality_by_method[method]
            avg_thd = np.mean([r.get('thd', 0) for r in q_results])
            avg_snr = np.mean([r.get('snr', 0) for r in q_results])
            avg_aliasing = np.mean([r.get('aliasing_rejection', 60) for r in q_results])
            
            # Average performance metrics
            p_results = perf_by_method[method]
            avg_latency = np.mean([r.get('processing_time_ms', 0) for r in p_results])
            avg_cpu = np.mean([r.get('cpu_usage_percent', 0) for r in p_results])
            
            # Calculate quality score (0-100)
            quality_score = self._calculate_quality_score(
                avg_thd, avg_snr, avg_aliasing
            )
            
            # Calculate performance score (0-100)
            performance_score = self._calculate_performance_score(
                avg_latency, avg_cpu
            )
            
            # Overall score (weighted average)
            overall_score = quality_score * 0.7 + performance_score * 0.3
            
            # Determine real-time capability
            realtime_capable = (avg_latency < 10 and avg_cpu < 50)
            
            # Generate pros and cons
            pros, cons = self._analyze_pros_cons(
                avg_thd, avg_snr, avg_aliasing, avg_latency, avg_cpu
            )
            
            # Determine best use cases
            use_cases = self._determine_use_cases(
                quality_score, performance_score, realtime_capable
            )
            
            methods[method] = MethodComparison(
                name=method,
                display_name=self._get_display_name(method),
                quality_score=quality_score,
                performance_score=performance_score,
                overall_score=overall_score,
                thd_percent=avg_thd,
                snr_db=avg_snr,
                aliasing_db=avg_aliasing,
                latency_ms=avg_latency,
                cpu_percent=avg_cpu,
                realtime_capable=realtime_capable,
                pros=pros,
                cons=cons,
                use_cases=use_cases
            )
        
        return methods
    
    def _calculate_quality_score(self, thd: float, snr: float, aliasing: float) -> float:
        """Calculate quality score from metrics"""
        # THD score (lower is better, 0% = 100 points, 10% = 0 points)
        thd_score = max(0, 100 - thd * 10)
        
        # SNR score (higher is better, 60dB = 100 points, 20dB = 0 points)
        snr_score = max(0, min(100, (snr - 20) * 2.5))
        
        # Aliasing score (higher rejection is better, 80dB = 100 points, 40dB = 0 points)
        aliasing_score = max(0, min(100, (aliasing - 40) * 2.5))
        
        # Weighted average
        score = (
            thd_score * 0.3 +
            snr_score * 0.3 +
            aliasing_score * 0.4
        )
        
        return float(score)
    
    def _calculate_performance_score(self, latency: float, cpu: float) -> float:
        """Calculate performance score from metrics"""
        # Latency score (lower is better, 1ms = 100 points, 20ms = 0 points)
        latency_score = max(0, 100 - latency * 5)
        
        # CPU score (lower is better, 10% = 100 points, 100% = 0 points)
        cpu_score = max(0, 100 - cpu)
        
        # Weighted average
        score = latency_score * 0.6 + cpu_score * 0.4
        
        return float(score)
    
    def _analyze_pros_cons(self, thd: float, snr: float, aliasing: float,
                          latency: float, cpu: float) -> Tuple[List[str], List[str]]:
        """Analyze pros and cons of method based on metrics"""
        pros = []
        cons = []
        
        # Quality analysis
        if thd < 1:
            pros.append("Excellent harmonic distortion (<1% THD)")
        elif thd > 5:
            cons.append(f"High harmonic distortion ({thd:.1f}% THD)")
        
        if snr > 60:
            pros.append(f"Excellent signal-to-noise ratio ({snr:.0f}dB)")
        elif snr < 40:
            cons.append(f"Poor signal-to-noise ratio ({snr:.0f}dB)")
        
        if aliasing > 70:
            pros.append(f"Superior aliasing rejection ({aliasing:.0f}dB)")
        elif aliasing < 50:
            cons.append(f"Weak aliasing rejection ({aliasing:.0f}dB)")
        
        # Performance analysis
        if latency < 5:
            pros.append(f"Very low latency ({latency:.1f}ms)")
        elif latency > 15:
            cons.append(f"High latency ({latency:.1f}ms)")
        
        if cpu < 20:
            pros.append(f"Low CPU usage ({cpu:.0f}%)")
        elif cpu > 60:
            cons.append(f"High CPU usage ({cpu:.0f}%)")
        
        # Ensure at least one pro and con
        if not pros:
            pros.append("Stable and reliable")
        if not cons:
            cons.append("No significant drawbacks")
        
        return pros, cons
    
    def _determine_use_cases(self, quality_score: float, performance_score: float,
                           realtime: bool) -> List[str]:
        """Determine best use cases for method"""
        use_cases = []
        
        if realtime and performance_score > 70:
            use_cases.append("Real-time voice communication")
        
        if quality_score > 80:
            use_cases.append("High-fidelity audio processing")
        
        if quality_score > 70 and performance_score > 60:
            use_cases.append("Wake word detection")
        
        if performance_score > 80:
            use_cases.append("Low-power embedded systems")
        
        if quality_score > 60 and performance_score > 50:
            use_cases.append("General voice assistant applications")
        
        if not use_cases:
            use_cases.append("Non-critical audio processing")
        
        return use_cases
    
    def _get_display_name(self, method: str) -> str:
        """Get display name for method"""
        display_names = {
            'scipy_fft': 'SciPy FFT',
            'scipy_poly': 'SciPy Polyphase',
            'librosa': 'Librosa Kaiser',
            'soxr': 'SoX Resampler',
            'resampy': 'Resampy Sinc'
        }
        return display_names.get(method, method)
    
    def generate_comparison_matrix(self, methods: Dict[str, MethodComparison]) -> pd.DataFrame:
        """Generate comparison matrix DataFrame"""
        data = []
        
        for method_name, comparison in methods.items():
            data.append({
                'Method': comparison.display_name,
                'Overall Score': f"{comparison.overall_score:.1f}",
                'Quality Score': f"{comparison.quality_score:.1f}",
                'Performance Score': f"{comparison.performance_score:.1f}",
                'THD (%)': f"{comparison.thd_percent:.2f}",
                'SNR (dB)': f"{comparison.snr_db:.1f}",
                'Aliasing (dB)': f"{comparison.aliasing_db:.1f}",
                'Latency (ms)': f"{comparison.latency_ms:.2f}",
                'CPU (%)': f"{comparison.cpu_percent:.1f}",
                'Real-time': '✓' if comparison.realtime_capable else '✗'
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Overall Score', ascending=False)
        
        return df
    
    def plot_quality_performance_scatter(self, methods: Dict[str, MethodComparison],
                                        save_path: Optional[Path] = None):
        """Create quality vs performance scatter plot"""
        if not PLOTTING_AVAILABLE:
            print("Plotting not available - matplotlib not installed")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract data
        names = []
        quality_scores = []
        performance_scores = []
        colors = []
        
        for method_name, comparison in methods.items():
            names.append(comparison.display_name)
            quality_scores.append(comparison.quality_score)
            performance_scores.append(comparison.performance_score)
            colors.append('green' if comparison.realtime_capable else 'orange')
        
        # Create scatter plot
        scatter = ax.scatter(performance_scores, quality_scores, 
                           c=colors, s=200, alpha=0.6, edgecolors='black')
        
        # Add labels
        for i, name in enumerate(names):
            ax.annotate(name, (performance_scores[i], quality_scores[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        # Add quadrant lines
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
        
        # Labels and title
        ax.set_xlabel('Performance Score (0-100)', fontsize=12)
        ax.set_ylabel('Quality Score (0-100)', fontsize=12)
        ax.set_title('Resampling Methods: Quality vs Performance', fontsize=14, fontweight='bold')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.6, label='Real-time Capable'),
            Patch(facecolor='orange', alpha=0.6, label='Not Real-time')
        ]
        ax.legend(handles=legend_elements, loc='lower left')
        
        # Set limits
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add quadrant labels
        ax.text(25, 75, 'High Quality\nLow Performance', 
               ha='center', va='center', fontsize=9, alpha=0.5)
        ax.text(75, 75, 'High Quality\nHigh Performance', 
               ha='center', va='center', fontsize=9, alpha=0.5)
        ax.text(25, 25, 'Low Quality\nLow Performance', 
               ha='center', va='center', fontsize=9, alpha=0.5)
        ax.text(75, 25, 'Low Quality\nHigh Performance', 
               ha='center', va='center', fontsize=9, alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'quality_vs_performance.png', dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def plot_metric_comparison(self, methods: Dict[str, MethodComparison],
                              save_path: Optional[Path] = None):
        """Create bar chart comparing key metrics"""
        if not PLOTTING_AVAILABLE:
            print("Plotting not available - matplotlib not installed")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Extract data
        method_names = [m.display_name for m in methods.values()]
        
        metrics = [
            ('THD (%)', [m.thd_percent for m in methods.values()], 'lower'),
            ('SNR (dB)', [m.snr_db for m in methods.values()], 'higher'),
            ('Aliasing Rejection (dB)', [m.aliasing_db for m in methods.values()], 'higher'),
            ('Latency (ms)', [m.latency_ms for m in methods.values()], 'lower'),
            ('CPU Usage (%)', [m.cpu_percent for m in methods.values()], 'lower'),
            ('Overall Score', [m.overall_score for m in methods.values()], 'higher')
        ]
        
        for ax, (metric_name, values, better) in zip(axes, metrics):
            # Determine colors based on whether higher or lower is better
            if better == 'lower':
                colors = ['green' if v == min(values) else 'orange' if v == max(values) else 'skyblue' 
                         for v in values]
            else:
                colors = ['green' if v == max(values) else 'orange' if v == min(values) else 'skyblue' 
                         for v in values]
            
            bars = ax.bar(method_names, values, color=colors, edgecolor='black', alpha=0.7)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.1f}', ha='center', va='bottom', fontsize=9)
            
            ax.set_title(metric_name, fontsize=12, fontweight='bold')
            ax.set_ylabel('Value', fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Resampling Methods Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def generate_use_case_recommendations(self, methods: Dict[str, MethodComparison]) -> Dict[str, str]:
        """Generate recommendations for each use case"""
        recommendations = {}
        
        for use_case, profile in self.use_case_profiles.items():
            best_method = None
            best_score = -1
            
            for method_name, comparison in methods.items():
                # Check if method meets requirements
                reqs = profile['requirements']
                meets_requirements = (
                    comparison.latency_ms <= reqs['max_latency_ms'] and
                    comparison.cpu_percent <= reqs['max_cpu_percent'] and
                    comparison.snr_db >= reqs['min_snr_db'] and
                    comparison.thd_percent <= reqs['max_thd_percent']
                )
                
                if meets_requirements:
                    # Calculate weighted score for this use case
                    score = (
                        comparison.quality_score * profile['quality_weight'] +
                        comparison.performance_score * profile['performance_weight']
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_method = method_name
            
            if best_method:
                recommendations[use_case] = best_method
            else:
                recommendations[use_case] = 'none_suitable'
        
        return recommendations
    
    def generate_html_report(self, methods: Dict[str, MethodComparison],
                            matrix_df: pd.DataFrame,
                            recommendations: Dict[str, str]):
        """Generate HTML report"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Audio Resampling Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .best {{ background-color: #d4edda; }}
        .worst {{ background-color: #f8d7da; }}
        .pros {{ color: green; }}
        .cons {{ color: red; }}
        .recommendation {{ background-color: #d1ecf1; padding: 10px; margin: 10px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Audio Resampling Quality Analysis Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Executive Summary</h2>
    <p>This report compares {len(methods)} audio resampling methods for use in the HA Realtime Voice Assistant pipeline.</p>
    
    <h2>Comparison Matrix</h2>
    {matrix_df.to_html(index=False, classes='comparison-table')}
    
    <h2>Method Details</h2>
"""
        
        for method_name, comparison in sorted(methods.items(), 
                                             key=lambda x: x[1].overall_score, 
                                             reverse=True):
            html_content += f"""
    <h3>{comparison.display_name}</h3>
    <p><strong>Overall Score:</strong> {comparison.overall_score:.1f}/100</p>
    
    <h4>Metrics:</h4>
    <ul>
        <li>THD: {comparison.thd_percent:.2f}%</li>
        <li>SNR: {comparison.snr_db:.1f} dB</li>
        <li>Aliasing Rejection: {comparison.aliasing_db:.1f} dB</li>
        <li>Latency: {comparison.latency_ms:.2f} ms</li>
        <li>CPU Usage: {comparison.cpu_percent:.1f}%</li>
        <li>Real-time Capable: {'Yes' if comparison.realtime_capable else 'No'}</li>
    </ul>
    
    <h4>Pros:</h4>
    <ul class="pros">
        {''.join(f'<li>{pro}</li>' for pro in comparison.pros)}
    </ul>
    
    <h4>Cons:</h4>
    <ul class="cons">
        {''.join(f'<li>{con}</li>' for con in comparison.cons)}
    </ul>
    
    <h4>Best Use Cases:</h4>
    <ul>
        {''.join(f'<li>{use_case}</li>' for use_case in comparison.use_cases)}
    </ul>
    <hr>
"""
        
        html_content += """
    <h2>Use Case Recommendations</h2>
"""
        
        for use_case, method in recommendations.items():
            if method != 'none_suitable':
                html_content += f"""
    <div class="recommendation">
        <strong>{use_case.replace('_', ' ').title()}:</strong> {methods[method].display_name}
    </div>
"""
            else:
                html_content += f"""
    <div class="recommendation" style="background-color: #f8d7da;">
        <strong>{use_case.replace('_', ' ').title()}:</strong> No suitable method found
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        # Save HTML report
        with open(self.output_dir / 'report.html', 'w') as f:
            f.write(html_content)
        
        print(f"HTML report saved to {self.output_dir / 'report.html'}")
    
    def generate_config_yaml(self, best_method: str, methods: Dict[str, MethodComparison]):
        """Generate YAML configuration for optimal resampling"""
        import yaml
        
        comparison = methods[best_method]
        
        config = {
            'resampling': {
                'method': best_method,
                'display_name': comparison.display_name,
                'parameters': self._get_method_parameters(best_method),
                'quality_metrics': {
                    'thd_percent': float(comparison.thd_percent),
                    'snr_db': float(comparison.snr_db),
                    'aliasing_rejection_db': float(comparison.aliasing_db)
                },
                'performance_metrics': {
                    'latency_ms': float(comparison.latency_ms),
                    'cpu_percent': float(comparison.cpu_percent),
                    'realtime_capable': comparison.realtime_capable
                },
                'scores': {
                    'quality': float(comparison.quality_score),
                    'performance': float(comparison.performance_score),
                    'overall': float(comparison.overall_score)
                },
                'implementation_notes': self._get_implementation_notes(best_method)
            }
        }
        
        # Save configuration
        with open(self.output_dir / 'optimal_resampling.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"Configuration saved to {self.output_dir / 'optimal_resampling.yaml'}")
        
        return config
    
    def _get_method_parameters(self, method: str) -> Dict[str, Any]:
        """Get implementation parameters for method"""
        params = {
            'scipy_fft': {},
            'scipy_poly': {
                'window': 'hamming',
                'padtype': 'constant'
            },
            'librosa': {
                'res_type': 'kaiser_best',
                'fix': True,
                'scale': False
            },
            'soxr': {
                'quality': 'HQ',
                'flags': None
            },
            'resampy': {
                'filter': 'kaiser_best',
                'parallel': False
            }
        }
        return params.get(method, {})
    
    def _get_implementation_notes(self, method: str) -> List[str]:
        """Get implementation notes for method"""
        notes = {
            'scipy_fft': [
                'Current implementation - no changes needed',
                'Consider adding window function for better frequency response'
            ],
            'scipy_poly': [
                'Replace signal.resample with signal.resample_poly',
                'Calculate GCD for up/down factors',
                'May need to handle non-integer ratios specially'
            ],
            'librosa': [
                'Add librosa dependency to requirements.txt',
                'Import: from librosa import resample',
                'Use res_type="kaiser_best" for quality'
            ],
            'soxr': [
                'Install soxr-python: pip install soxr',
                'Import: import soxr',
                'Best quality/performance balance'
            ],
            'resampy': [
                'Install resampy: pip install resampy',
                'Import: import resampy',
                'Good for band-limited signals'
            ]
        }
        return notes.get(method, ['No specific implementation notes'])
    
    def generate_full_report(self, quality_results: List[Dict],
                            performance_results: List[Dict]):
        """Generate complete comparison report with all outputs"""
        print("\nGenerating Comparison Report...")
        
        # Calculate scores
        methods = self.calculate_scores(quality_results, performance_results)
        
        if not methods:
            print("No methods to compare - insufficient data")
            return
        
        # Generate comparison matrix
        matrix_df = self.generate_comparison_matrix(methods)
        print("\nComparison Matrix:")
        print(matrix_df.to_string())
        
        # Save CSV
        matrix_df.to_csv(self.output_dir / 'comparison_matrix.csv', index=False)
        
        # Generate plots
        if PLOTTING_AVAILABLE:
            print("\nGenerating visualization plots...")
            self.plot_quality_performance_scatter(methods)
            self.plot_metric_comparison(methods)
        
        # Generate use case recommendations
        recommendations = self.generate_use_case_recommendations(methods)
        print("\nUse Case Recommendations:")
        for use_case, method in recommendations.items():
            if method != 'none_suitable':
                print(f"  {use_case}: {methods[method].display_name}")
            else:
                print(f"  {use_case}: No suitable method")
        
        # Find best overall method
        best_method = max(methods.keys(), key=lambda m: methods[m].overall_score)
        print(f"\nBest Overall Method: {methods[best_method].display_name}")
        print(f"  Overall Score: {methods[best_method].overall_score:.1f}/100")
        
        # Generate configuration
        config = self.generate_config_yaml(best_method, methods)
        
        # Generate HTML report
        self.generate_html_report(methods, matrix_df, recommendations)
        
        # Save summary JSON
        summary = {
            'timestamp': datetime.now().isoformat(),
            'best_method': best_method,
            'methods': {k: asdict(v) for k, v in methods.items()},
            'recommendations': recommendations
        }
        
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nAll reports saved to {self.output_dir}")
        
        return methods, recommendations