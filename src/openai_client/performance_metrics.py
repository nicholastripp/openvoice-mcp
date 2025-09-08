"""
Performance metrics tracking for OpenAI Realtime API.
Monitors latency, costs, accuracy, and model performance comparisons.
"""
import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path


@dataclass
class SessionMetrics:
    """Metrics for a single session"""
    session_id: str
    model: str
    start_time: float
    end_time: Optional[float] = None
    
    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    
    # Timing metrics (in milliseconds)
    connection_latency: Optional[float] = None
    first_response_latency: Optional[float] = None
    average_response_latency: float = 0
    total_processing_time: float = 0
    
    # Function calling metrics
    functions_called: int = 0
    function_success_rate: float = 0.0
    function_latencies: List[float] = field(default_factory=list)
    
    # Audio metrics
    audio_segments_sent: int = 0
    audio_segments_received: int = 0
    audio_processing_time: float = 0
    
    # Quality metrics
    instruction_following_score: Optional[float] = None
    response_quality_score: Optional[float] = None
    wake_word_detections: int = 0
    false_positives: int = 0
    
    # Cost tracking
    estimated_cost: float = 0.0
    
    # Errors and retries
    errors: List[str] = field(default_factory=list)
    retries: int = 0
    fallback_used: bool = False
    
    def calculate_duration(self) -> float:
        """Calculate session duration in seconds"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


@dataclass
class ModelComparison:
    """Comparison metrics between models"""
    timestamp: float
    old_model: str
    new_model: str
    
    # Performance improvements
    latency_improvement: float  # Percentage
    cost_reduction: float  # Percentage
    function_accuracy_improvement: float  # Percentage
    instruction_following_improvement: float  # Percentage
    
    # Feature differences
    new_features_used: List[str]
    unavailable_features: List[str]
    
    # User experience
    user_satisfaction_delta: Optional[float] = None
    error_rate_delta: float = 0.0


class PerformanceMetrics:
    """Tracks and analyzes performance metrics for OpenAI Realtime API"""
    
    def __init__(self, config: Any, logger: Optional[logging.Logger] = None,
                 metrics_dir: str = "metrics"):
        """
        Initialize performance metrics tracker.
        
        Args:
            config: Application configuration
            logger: Optional logger instance
            metrics_dir: Directory to store metrics files
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(exist_ok=True)
        
        # Active session
        self.current_session: Optional[SessionMetrics] = None
        
        # Historical data (keep last 100 sessions in memory)
        self.session_history: deque = deque(maxlen=100)
        
        # Aggregated metrics
        self.total_sessions = 0
        self.total_tokens = {"input": 0, "output": 0}
        self.total_cost = 0.0
        self.model_usage = {}
        
        # Performance baselines for comparison
        self.performance_baselines = {
            "gpt-4o-realtime-preview": {
                "avg_latency": 150,  # ms
                "function_accuracy": 0.497,
                "instruction_following": 0.206,
                "cost_per_1k_tokens": {"input": 0.04, "output": 0.08}
            },
            "gpt-realtime": {
                "avg_latency": 120,  # ms (expected improvement)
                "function_accuracy": 0.665,
                "instruction_following": 0.305,
                "cost_per_1k_tokens": {"input": 0.032, "output": 0.064}
            }
        }
        
        # Load historical metrics if available
        self._load_metrics()
    
    def start_session(self, session_id: str, model: str) -> SessionMetrics:
        """
        Start tracking a new session.
        
        Args:
            session_id: Unique session identifier
            model: Model being used
            
        Returns:
            New SessionMetrics instance
        """
        self.current_session = SessionMetrics(
            session_id=session_id,
            model=model,
            start_time=time.time()
        )
        
        self.logger.info(f"Started metrics tracking for session {session_id} using {model}")
        return self.current_session
    
    def end_session(self) -> Optional[SessionMetrics]:
        """
        End current session and save metrics.
        
        Returns:
            Completed SessionMetrics or None
        """
        if not self.current_session:
            return None
        
        self.current_session.end_time = time.time()
        
        # Calculate final metrics
        self._finalize_session_metrics()
        
        # Add to history
        self.session_history.append(self.current_session)
        self.total_sessions += 1
        
        # Update aggregated metrics
        self._update_aggregated_metrics(self.current_session)
        
        # Save to disk
        self._save_session_metrics(self.current_session)
        
        self.logger.info(
            f"Session {self.current_session.session_id} ended. "
            f"Duration: {self.current_session.calculate_duration():.2f}s, "
            f"Cost: ${self.current_session.estimated_cost:.4f}"
        )
        
        session = self.current_session
        self.current_session = None
        return session
    
    def record_connection(self, latency: float):
        """Record connection establishment latency"""
        if self.current_session:
            self.current_session.connection_latency = latency * 1000  # Convert to ms
    
    def record_response(self, latency: float, is_first: bool = False):
        """Record response latency"""
        if not self.current_session:
            return
        
        latency_ms = latency * 1000
        
        if is_first:
            self.current_session.first_response_latency = latency_ms
        
        # Update average
        responses = len(self.current_session.function_latencies) + 1
        current_avg = self.current_session.average_response_latency
        self.current_session.average_response_latency = (
            (current_avg * (responses - 1) + latency_ms) / responses
        )
    
    def record_tokens(self, input_tokens: int, output_tokens: int):
        """Record token usage"""
        if self.current_session:
            self.current_session.input_tokens += input_tokens
            self.current_session.output_tokens += output_tokens
            
            # Update cost estimate
            self.current_session.estimated_cost = self._calculate_cost(
                self.current_session.input_tokens,
                self.current_session.output_tokens,
                self.current_session.model
            )
    
    def record_function_call(self, success: bool, latency: float):
        """Record function call metrics"""
        if not self.current_session:
            return
        
        self.current_session.functions_called += 1
        self.current_session.function_latencies.append(latency * 1000)
        
        # Update success rate
        if success:
            current_rate = self.current_session.function_success_rate
            total = self.current_session.functions_called
            self.current_session.function_success_rate = (
                (current_rate * (total - 1) + 1) / total
            )
    
    def record_audio_segment(self, sent: bool = True):
        """Record audio segment processing"""
        if self.current_session:
            if sent:
                self.current_session.audio_segments_sent += 1
            else:
                self.current_session.audio_segments_received += 1
    
    def record_error(self, error: str):
        """Record an error occurrence"""
        if self.current_session:
            self.current_session.errors.append(error)
            self.logger.error(f"Session error recorded: {error}")
    
    def record_retry(self):
        """Record a retry attempt"""
        if self.current_session:
            self.current_session.retries += 1
    
    def record_fallback(self):
        """Record that fallback model was used"""
        if self.current_session:
            self.current_session.fallback_used = True
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int, 
                       model: str) -> float:
        """
        Calculate estimated cost based on token usage.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name
            
        Returns:
            Estimated cost in dollars
        """
        # Get pricing for model
        if model == "gpt-realtime":
            input_rate = 32.0 / 1_000_000  # $32 per 1M tokens
            output_rate = 64.0 / 1_000_000  # $64 per 1M tokens
        elif model in ["gpt-4o-realtime-preview", "gpt-4o-mini-realtime-preview"]:
            if "mini" in model:
                input_rate = 10.0 / 1_000_000
                output_rate = 20.0 / 1_000_000
            else:
                input_rate = 40.0 / 1_000_000
                output_rate = 80.0 / 1_000_000
        else:
            # Default/unknown model
            input_rate = 40.0 / 1_000_000
            output_rate = 80.0 / 1_000_000
        
        return (input_tokens * input_rate) + (output_tokens * output_rate)
    
    def _finalize_session_metrics(self):
        """Finalize metrics for current session"""
        if not self.current_session:
            return
        
        # Calculate total processing time
        duration = self.current_session.calculate_duration()
        self.current_session.total_processing_time = duration * 1000  # Convert to ms
        
        # Estimate quality scores based on errors and retries
        if self.current_session.errors:
            error_penalty = min(len(self.current_session.errors) * 0.1, 0.5)
            self.current_session.response_quality_score = max(0.5, 1.0 - error_penalty)
        else:
            self.current_session.response_quality_score = 1.0
        
        # Estimate instruction following based on retries
        if self.current_session.retries > 0:
            retry_penalty = min(self.current_session.retries * 0.15, 0.6)
            self.current_session.instruction_following_score = max(0.4, 1.0 - retry_penalty)
        else:
            self.current_session.instruction_following_score = 1.0
    
    def _update_aggregated_metrics(self, session: SessionMetrics):
        """Update aggregated metrics with session data"""
        # Update token totals
        self.total_tokens["input"] += session.input_tokens
        self.total_tokens["output"] += session.output_tokens
        
        # Update cost total
        self.total_cost += session.estimated_cost
        
        # Update model usage
        if session.model not in self.model_usage:
            self.model_usage[session.model] = {
                "sessions": 0,
                "total_duration": 0,
                "total_cost": 0,
                "errors": 0
            }
        
        self.model_usage[session.model]["sessions"] += 1
        self.model_usage[session.model]["total_duration"] += session.calculate_duration()
        self.model_usage[session.model]["total_cost"] += session.estimated_cost
        self.model_usage[session.model]["errors"] += len(session.errors)
    
    def _save_session_metrics(self, session: SessionMetrics):
        """Save session metrics to disk"""
        try:
            # Create filename with timestamp
            timestamp = datetime.fromtimestamp(session.start_time)
            filename = f"session_{timestamp.strftime('%Y%m%d_%H%M%S')}_{session.session_id}.json"
            filepath = self.metrics_dir / filename
            
            # Convert to dictionary and save
            metrics_dict = asdict(session)
            with open(filepath, 'w') as f:
                json.dump(metrics_dict, f, indent=2)
            
            self.logger.debug(f"Saved session metrics to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save session metrics: {e}")
    
    def _load_metrics(self):
        """Load historical metrics from disk"""
        try:
            # Load recent metrics files
            metrics_files = sorted(self.metrics_dir.glob("session_*.json"))
            
            for filepath in metrics_files[-100:]:  # Load last 100 sessions
                with open(filepath, 'r') as f:
                    metrics_dict = json.load(f)
                    session = SessionMetrics(**metrics_dict)
                    self.session_history.append(session)
                    self._update_aggregated_metrics(session)
            
            self.logger.info(f"Loaded {len(self.session_history)} historical sessions")
        except Exception as e:
            self.logger.error(f"Failed to load historical metrics: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            "total_sessions": self.total_sessions,
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 4),
            "model_usage": self.model_usage,
            "current_session": None
        }
        
        if self.current_session:
            stats["current_session"] = {
                "id": self.current_session.session_id,
                "model": self.current_session.model,
                "duration": self.current_session.calculate_duration(),
                "cost": self.current_session.estimated_cost
            }
        
        # Calculate averages from history
        if self.session_history:
            recent_sessions = list(self.session_history)[-10:]
            
            avg_duration = sum(s.calculate_duration() for s in recent_sessions) / len(recent_sessions)
            avg_cost = sum(s.estimated_cost for s in recent_sessions) / len(recent_sessions)
            avg_latency = sum(s.average_response_latency for s in recent_sessions 
                            if s.average_response_latency) / len(recent_sessions)
            
            stats["recent_averages"] = {
                "duration": round(avg_duration, 2),
                "cost": round(avg_cost, 4),
                "latency_ms": round(avg_latency, 2)
            }
        
        return stats
    
    def compare_models(self, old_model: str, new_model: str) -> ModelComparison:
        """
        Compare performance between two models.
        
        Args:
            old_model: Previous model name
            new_model: New model name
            
        Returns:
            ModelComparison instance
        """
        # Get sessions for each model
        old_sessions = [s for s in self.session_history if s.model == old_model]
        new_sessions = [s for s in self.session_history if s.model == new_model]
        
        comparison = ModelComparison(
            timestamp=time.time(),
            old_model=old_model,
            new_model=new_model,
            latency_improvement=0,
            cost_reduction=0,
            function_accuracy_improvement=0,
            instruction_following_improvement=0,
            new_features_used=[],
            unavailable_features=[]
        )
        
        # Calculate improvements if we have data
        if old_sessions and new_sessions:
            # Latency improvement
            old_latency = sum(s.average_response_latency for s in old_sessions[-5:] 
                            if s.average_response_latency) / min(5, len(old_sessions))
            new_latency = sum(s.average_response_latency for s in new_sessions[-5:]
                            if s.average_response_latency) / min(5, len(new_sessions))
            
            if old_latency > 0:
                comparison.latency_improvement = ((old_latency - new_latency) / old_latency) * 100
            
            # Cost reduction
            old_cost = sum(s.estimated_cost for s in old_sessions[-5:]) / min(5, len(old_sessions))
            new_cost = sum(s.estimated_cost for s in new_sessions[-5:]) / min(5, len(new_sessions))
            
            if old_cost > 0:
                comparison.cost_reduction = ((old_cost - new_cost) / old_cost) * 100
            
            # Function accuracy
            old_accuracy = sum(s.function_success_rate for s in old_sessions[-5:]
                             if s.functions_called > 0) / min(5, len(old_sessions))
            new_accuracy = sum(s.function_success_rate for s in new_sessions[-5:]
                             if s.functions_called > 0) / min(5, len(new_sessions))
            
            if old_accuracy > 0:
                comparison.function_accuracy_improvement = (
                    ((new_accuracy - old_accuracy) / old_accuracy) * 100
                )
        
        # Use baseline data if available
        else:
            old_baseline = self.performance_baselines.get(old_model, {})
            new_baseline = self.performance_baselines.get(new_model, {})
            
            if old_baseline and new_baseline:
                # Calculate expected improvements
                old_lat = old_baseline.get("avg_latency", 150)
                new_lat = new_baseline.get("avg_latency", 120)
                comparison.latency_improvement = ((old_lat - new_lat) / old_lat) * 100
                
                # Cost reduction based on pricing
                old_input = old_baseline.get("cost_per_1k_tokens", {}).get("input", 0.04)
                new_input = new_baseline.get("cost_per_1k_tokens", {}).get("input", 0.032)
                comparison.cost_reduction = ((old_input - new_input) / old_input) * 100
                
                # Function accuracy improvement
                old_acc = old_baseline.get("function_accuracy", 0.497)
                new_acc = new_baseline.get("function_accuracy", 0.665)
                comparison.function_accuracy_improvement = ((new_acc - old_acc) / old_acc) * 100
                
                # Instruction following improvement
                old_inst = old_baseline.get("instruction_following", 0.206)
                new_inst = new_baseline.get("instruction_following", 0.305)
                comparison.instruction_following_improvement = ((new_inst - old_inst) / old_inst) * 100
        
        # Feature differences
        if new_model == "gpt-realtime":
            comparison.new_features_used = [
                "Native MCP support",
                "Image input capability",
                "Asynchronous function calling",
                "Cedar voice",
                "Marin voice"
            ]
        
        return comparison
    
    def export_metrics(self, format: str = "json") -> str:
        """
        Export metrics in specified format.
        
        Args:
            format: Export format (json, csv)
            
        Returns:
            Exported data as string
        """
        if format == "json":
            data = {
                "statistics": self.get_statistics(),
                "sessions": [asdict(s) for s in self.session_history],
                "timestamp": time.time()
            }
            return json.dumps(data, indent=2)
        
        elif format == "csv":
            # Simple CSV export
            lines = ["session_id,model,duration,cost,tokens_in,tokens_out,errors"]
            
            for session in self.session_history:
                lines.append(
                    f"{session.session_id},{session.model},"
                    f"{session.calculate_duration():.2f},"
                    f"{session.estimated_cost:.4f},"
                    f"{session.input_tokens},{session.output_tokens},"
                    f"{len(session.errors)}"
                )
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_cost_projection(self, hours: int = 24) -> Dict[str, float]:
        """
        Project costs based on recent usage.
        
        Args:
            hours: Number of hours to project
            
        Returns:
            Cost projections
        """
        if not self.session_history:
            return {"message": "Insufficient data for projection"}
        
        # Calculate recent rate
        recent_sessions = list(self.session_history)[-20:]
        if len(recent_sessions) < 2:
            return {"message": "Insufficient data for projection"}
        
        # Time span of recent sessions
        time_span = recent_sessions[-1].start_time - recent_sessions[0].start_time
        if time_span <= 0:
            return {"message": "Insufficient time span for projection"}
        
        # Calculate rate per hour
        total_cost = sum(s.estimated_cost for s in recent_sessions)
        cost_per_second = total_cost / time_span
        cost_per_hour = cost_per_second * 3600
        
        return {
            "projected_cost": round(cost_per_hour * hours, 2),
            "hours": hours,
            "based_on_sessions": len(recent_sessions),
            "current_rate_per_hour": round(cost_per_hour, 4)
        }