"""
Performance metrics tracking system for the Hamiltonian Path Discovery project.
Tracks and analyzes key metrics defined in the research objectives.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional
import psutil
import numpy as np
from .structured_logger import StructuredLogger

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    path_validity_rate: float = 0.0
    complexity_ratio: float = 0.0
    energy_per_path: float = 0.0
    hallucination_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "path_validity_rate": self.path_validity_rate,
            "complexity_ratio": self.complexity_ratio,
            "energy_per_path": self.energy_per_path,
            "hallucination_rate": self.hallucination_rate
        }
    
    def meets_targets(self) -> bool:
        """Check if metrics meet research targets."""
        return (
            self.path_validity_rate >= 0.90 and
            self.complexity_ratio <= 1.5 and
            self.energy_per_path <= 50.0 and
            self.hallucination_rate <= 0.05
        )

class MetricsTracker:
    """
    Tracks and analyzes performance metrics throughout the research project.
    """
    
    def __init__(self):
        self.logger = StructuredLogger()
        self.start_time = time.time()
        self.iteration_start = self.start_time
        self.initial_energy = 0.0
        self.energy_readings: List[float] = []
        self.valid_paths = 0
        self.total_paths = 0
        self.api_calls = 0
        self.invalid_api_calls = 0
        
    def start_iteration(self) -> None:
        """Start tracking a new iteration."""
        self.iteration_start = time.time()
        self.initial_energy = self._get_energy_usage()
    
    def end_iteration(self, path_found: bool, is_valid: bool) -> None:
        """
        End iteration and update metrics.
        
        Args:
            path_found: Whether a path was found
            is_valid: Whether the path is valid
        """
        duration = time.time() - self.iteration_start
        current_energy = self._get_energy_usage()
        energy_used = max(0.0, current_energy - self.initial_energy)  # Ensure non-negative
        
        self.energy_readings.append(energy_used)
        if path_found:
            self.total_paths += 1
            if is_valid:
                self.valid_paths += 1
        
        # Log metrics
        self.logger.log_metrics("iteration", {
            "duration": duration,
            "energy_used": energy_used,
            "path_found": path_found,
            "path_valid": is_valid
        })
    
    def log_api_call(self, success: bool, is_valid_response: bool) -> None:
        """Log API call results."""
        self.api_calls += 1
        if not is_valid_response:
            self.invalid_api_calls += 1
        
        self.logger.log_metrics("api", {
            "success": success,
            "valid_response": is_valid_response,
            "hallucination_rate": self.get_hallucination_rate()
        })
    
    def _get_energy_usage(self) -> float:
        """Get current energy usage in Joules."""
        # Using psutil as a proxy for energy usage
        cpu_percent = psutil.cpu_percent()
        memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        # Rough estimation: CPU% * time + memory footprint
        return (cpu_percent * 0.1) + (memory * 0.05)
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return PerformanceMetrics(
            path_validity_rate=self.get_validity_rate(),
            complexity_ratio=self.get_complexity_ratio(),
            energy_per_path=self.get_energy_per_path(),
            hallucination_rate=self.get_hallucination_rate()
        )
    
    def get_validity_rate(self) -> float:
        """Calculate path validity rate."""
        return self.valid_paths / max(1, self.total_paths)
    
    def get_complexity_ratio(self) -> float:
        """Calculate complexity ratio compared to optimal."""
        # TODO: Implement comparison with optimal solution
        return 1.0
    
    def get_energy_per_path(self) -> float:
        """Calculate average energy usage per valid path."""
        if not self.valid_paths:
            return float('inf')
        return sum(self.energy_readings) / self.valid_paths
    
    def get_hallucination_rate(self) -> float:
        """Calculate API hallucination rate."""
        if not self.api_calls:
            return 0.0
        return self.invalid_api_calls / self.api_calls
    
    def log_research_progress(self, milestone: str) -> None:
        """Log research milestone with current metrics."""
        metrics = self.get_current_metrics()
        self.logger.log_research_milestone(
            milestone=milestone,
            target_achieved=metrics.meets_targets(),
            improvement=self.get_validity_rate(),
            notes=f"Energy per path: {self.get_energy_per_path():.2f}J"
        )
