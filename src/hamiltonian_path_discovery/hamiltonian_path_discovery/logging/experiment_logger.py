"""
Experiment logging system for tracking iterative improvements in the Hamiltonian Path Discovery project.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import json
from pathlib import Path
from .structured_logger import StructuredLogger
from .metrics_tracker import MetricsTracker, PerformanceMetrics

@dataclass
class IterationResult:
    """Data from a single iteration of the algorithm."""
    iteration: int
    prompt: str
    generated_code: str
    verification_result: bool
    energy_usage: float
    execution_time: float
    error_trace: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            "iteration": self.iteration,
            "timestamp": datetime.utcnow().isoformat(),
            "prompt": self.prompt,
            "code_length": len(self.generated_code),
            "verification": self.verification_result,
            "energy": self.energy_usage,
            "time": self.execution_time,
            "error": self.error_trace
        }

class ExperimentLogger:
    """
    Tracks and logs experimental results throughout the research process.
    """
    
    def __init__(self, experiment_dir: str = "experiments"):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.structured_logger = StructuredLogger()
        self.metrics_tracker = MetricsTracker()
        self.current_experiment_id: Optional[str] = None
        self.iterations: List[IterationResult] = []
    
    def start_experiment(self, description: str, config: Dict) -> str:
        """
        Start a new experiment.
        
        Args:
            description: Brief description of the experiment
            config: Configuration parameters
            
        Returns:
            experiment_id: Unique identifier for the experiment
        """
        # Create experiment directory
        self.current_experiment_id = self.structured_logger.log_experiment(
            phase="initialization",
            metrics={},
            iterations=[]
        )
        
        experiment_path = self._get_experiment_path(self.current_experiment_id)
        experiment_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_file = experiment_path / "config.json"
        with open(config_file, "w") as f:
            json.dump({
                "description": description,
                "config": config,
                "timestamp": datetime.utcnow().isoformat()
            }, f, indent=2)
        
        # Initialize metrics
        self.metrics_tracker = MetricsTracker()
        self.iterations = []
        
        return self.current_experiment_id
    
    def log_iteration(self, result: IterationResult) -> None:
        """Log a single iteration result."""
        if not self.current_experiment_id:
            # Auto-start an experiment if none is active
            self.start_experiment(
                description="Auto-started experiment",
                config={"auto_started": True}
            )
        
        # Update metrics
        self.metrics_tracker.start_iteration()  # Start tracking before processing
        self.metrics_tracker.end_iteration(
            path_found=bool(result.generated_code),
            is_valid=result.verification_result
        )
        
        # Save iteration result
        self.iterations.append(result)
        
        # Log to structured logger
        self.structured_logger.log_metrics(
            category="iteration",
            values=result.to_dict()
        )
    
    def log_code_generation(self, prompt: str, code: str, is_valid: bool) -> None:
        """Log code generation results."""
        self.metrics_tracker.log_api_call(
            success=bool(code),
            is_valid_response=is_valid
        )
    
    def end_experiment(self) -> Optional[Dict]:
        """
        End the current experiment and return summary metrics.
        
        Returns:
            Dict containing experiment summary and metrics, or None if no active experiment
        """
        if not self.current_experiment_id:
            return None
        
        # Get final metrics
        final_metrics = self.metrics_tracker.get_current_metrics()
        
        # Create summary
        summary = {
            "experiment_id": self.current_experiment_id,
            "total_iterations": len(self.iterations),
            "metrics": final_metrics.to_dict(),
            "targets_met": final_metrics.meets_targets(),
            "total_time": sum(iter.execution_time for iter in self.iterations),
            "total_energy": sum(iter.energy_usage for iter in self.iterations)
        }
        
        # Save summary
        experiment_path = self._get_experiment_path(self.current_experiment_id)
        summary_file = experiment_path / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Log completion
        self.structured_logger.log_research_milestone(
            milestone=f"Experiment {self.current_experiment_id} completed",
            target_achieved=final_metrics.meets_targets(),
            improvement=self.metrics_tracker.get_validity_rate(),
            notes=f"Total iterations: {len(self.iterations)}"
        )
        
        self.current_experiment_id = None
        return summary
    
    def _get_experiment_path(self, experiment_id: str) -> Path:
        """Get path for experiment files."""
        return self.experiment_dir / f"experiment_{experiment_id}"
    
    def save_generated_code(self, iteration: int, code: str) -> Path:
        """Save generated code to file."""
        if not self.current_experiment_id:
            raise ValueError("No active experiment")
            
        experiment_path = self._get_experiment_path(self.current_experiment_id)
        code_file = experiment_path / f"iteration_{iteration}_code.py"
        
        with open(code_file, "w") as f:
            f.write(code)
        
        return code_file
