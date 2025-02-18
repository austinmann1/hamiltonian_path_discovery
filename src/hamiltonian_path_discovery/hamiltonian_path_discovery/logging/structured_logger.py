"""
Structured logging system for the Hamiltonian Path Discovery project.
Handles experiment tracking, performance metrics, and research progress.
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

class StructuredLogger:
    """
    Structured logging system that maintains consistent JSON format for all logs.
    Supports experiment tracking, performance metrics, and research milestones.
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create separate log files for different purposes
        self.experiment_log = self.log_dir / "experiments.jsonl"
        self.metrics_log = self.log_dir / "metrics.jsonl"
        self.research_log = self.log_dir / "research.jsonl"
        
        # Set up Python logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _write_log(self, log_file: Path, data: Dict[str, Any]) -> None:
        """Write a log entry to the specified log file."""
        try:
            with open(log_file, "a") as f:
                json.dump(data, f)
                f.write("\n")
        except Exception as e:
            self.logger.error(f"Error writing to log file {log_file}: {str(e)}")
    
    def log_experiment(
        self,
        phase: str,
        metrics: Dict[str, float],
        iterations: List[Dict[str, Any]]
    ) -> str:
        """
        Log an experiment run with its metrics and iterations.
        
        Args:
            phase: Current phase (generation|verification|optimization)
            metrics: Performance metrics
            iterations: List of iteration data
            
        Returns:
            experiment_id: Unique identifier for the experiment
        """
        experiment_id = str(uuid.uuid4())
        data = {
            "experiment_id": experiment_id,
            "timestamp": datetime.utcnow().isoformat(),
            "phase": phase,
            "metrics": metrics,
            "iterations": iterations
        }
        self._write_log(self.experiment_log, data)
        return experiment_id
    
    def log_metrics(
        self,
        category: str,
        values: Dict[str, Union[float, int]]
    ) -> str:
        """
        Log performance metrics.
        
        Args:
            category: Metric category (api|verification|energy)
            values: Dictionary of metric values
            
        Returns:
            metric_id: Unique identifier for the metric entry
        """
        metric_id = str(uuid.uuid4())
        data = {
            "metric_id": metric_id,
            "timestamp": datetime.utcnow().isoformat(),
            "category": category,
            "values": values
        }
        self._write_log(self.metrics_log, data)
        return metric_id
    
    def log_research_milestone(
        self,
        milestone: str,
        target_achieved: bool,
        improvement: Optional[float] = None,
        notes: Optional[str] = None
    ) -> str:
        """
        Log research milestones and progress.
        
        Args:
            milestone: Description of the research milestone
            target_achieved: Whether the target was met
            improvement: Optional improvement metric
            notes: Optional research notes
            
        Returns:
            research_id: Unique identifier for the research entry
        """
        research_id = str(uuid.uuid4())
        data = {
            "research_id": research_id,
            "timestamp": datetime.utcnow().isoformat(),
            "milestone": milestone,
            "metrics": {
                "target_achieved": target_achieved,
                "improvement": improvement,
                "notes": notes
            }
        }
        self._write_log(self.research_log, data)
        return research_id
    
    def get_experiment_metrics(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve metrics for a specific experiment."""
        try:
            with open(self.experiment_log, "r") as f:
                for line in f:
                    data = json.loads(line)
                    if data.get("experiment_id") == experiment_id:
                        return data["metrics"]
        except Exception as e:
            self.logger.error(f"Error reading experiment metrics: {str(e)}")
        return None
    
    def get_research_progress(self) -> List[Dict[str, Any]]:
        """Get all research milestones in chronological order."""
        milestones = []
        try:
            with open(self.research_log, "r") as f:
                for line in f:
                    milestones.append(json.loads(line))
            return sorted(milestones, key=lambda x: x["timestamp"])
        except Exception as e:
            self.logger.error(f"Error reading research progress: {str(e)}")
            return []
