"""
Experiment tracking and management for Hamiltonian path discovery.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time
import numpy as np

@dataclass
class Experiment:
    """Tracks a single experiment run."""
    
    id: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    iterations: List[Dict] = field(default_factory=list)
    
    def add_iteration(self, path: List[int], success: bool, energy: float,
                     conflicts: List[Dict] = None) -> None:
        """Add an iteration to the experiment."""
        self.iterations.append({
            "path": path,
            "success": success,
            "energy": energy,
            "conflicts": conflicts or [],
            "timestamp": time.time()
        })
    
    def complete(self, success: bool = True) -> None:
        """Mark the experiment as complete."""
        self.end_time = time.time()
        self.metrics["duration"] = self.end_time - self.start_time
        self.metrics["success"] = float(success)
        self.metrics["num_iterations"] = len(self.iterations)
        
        if self.iterations:
            self.metrics["total_energy"] = sum(it["energy"] for it in self.iterations)
            self.metrics["avg_energy"] = self.metrics["total_energy"] / len(self.iterations)
            self.metrics["success_rate"] = sum(1 for it in self.iterations if it["success"]) / len(self.iterations)
    
    def to_dict(self) -> Dict:
        """Convert experiment to dictionary format."""
        return {
            "id": self.id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "metrics": self.metrics,
            "iterations": self.iterations
        }
