"""
Performance tracking system for Hamiltonian path discovery.
Tracks and analyzes solution performance across different graphs and strategies.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from datetime import datetime, timedelta
import json
import os

@dataclass
class SolutionMetrics:
    """Metrics for a single solution attempt."""
    graph_id: str
    path: List[int]
    computation_time: timedelta
    energy_usage: float
    strategy_used: str
    constraints_used: List[str]
    patterns_used: List[str]
    theorems_used: List[str]
    timestamp: datetime

@dataclass
class GraphMetrics:
    """Metrics for a specific graph."""
    graph_id: str
    graph_size: int
    is_planar: bool
    connectivity: int
    best_solution: Optional[SolutionMetrics] = None
    all_solutions: List[SolutionMetrics] = None
    
    def __post_init__(self):
        if self.all_solutions is None:
            self.all_solutions = []

class PerformanceTracker:
    """
    Tracks and analyzes performance metrics for Hamiltonian path discovery.
    Maintains records of best solutions and successful strategies.
    """
    
    def __init__(self, metrics_dir: str = "metrics"):
        """
        Initialize the performance tracker.
        
        Args:
            metrics_dir: Directory to store performance metrics
        """
        self.metrics_dir = metrics_dir
        self.graphs: Dict[str, GraphMetrics] = {}
        self.best_strategies: Dict[str, List[Tuple[str, float]]] = {}
        os.makedirs(metrics_dir, exist_ok=True)
    
    def register_graph(
        self,
        graph_id: str,
        adj_matrix: np.ndarray,
        is_planar: bool = False,
        connectivity: int = 0
    ):
        """
        Register a new graph for tracking.
        
        Args:
            graph_id: Unique identifier for the graph
            adj_matrix: The graph's adjacency matrix
            is_planar: Whether the graph is planar
            connectivity: The graph's vertex connectivity
        """
        if graph_id not in self.graphs:
            self.graphs[graph_id] = GraphMetrics(
                graph_id=graph_id,
                graph_size=len(adj_matrix),
                is_planar=is_planar,
                connectivity=connectivity
            )
    
    def record_solution(
        self,
        graph_id: str,
        path: List[int],
        computation_time: timedelta,
        energy_usage: float,
        strategy_used: str,
        constraints_used: List[str] = None,
        patterns_used: List[str] = None,
        theorems_used: List[str] = None
    ) -> bool:
        """
        Record a solution attempt.
        
        Args:
            graph_id: The graph's identifier
            path: The found Hamiltonian path
            computation_time: Time taken to find the solution
            energy_usage: Energy consumed during computation
            strategy_used: Description of the strategy used
            constraints_used: List of constraints used in the solution
            patterns_used: List of patterns used in the solution
            theorems_used: List of theorems used in the solution
            
        Returns:
            True if this is the best solution for this graph
        """
        if graph_id not in self.graphs:
            raise ValueError(f"Graph {graph_id} not registered")
            
        metrics = SolutionMetrics(
            graph_id=graph_id,
            path=path,
            computation_time=computation_time,
            energy_usage=energy_usage,
            strategy_used=strategy_used,
            constraints_used=constraints_used or [],
            patterns_used=patterns_used or [],
            theorems_used=theorems_used or [],
            timestamp=datetime.now()
        )
        
        graph_metrics = self.graphs[graph_id]
        graph_metrics.all_solutions.append(metrics)
        
        is_best = False
        if (graph_metrics.best_solution is None or
            metrics.computation_time < graph_metrics.best_solution.computation_time):
            graph_metrics.best_solution = metrics
            is_best = True
            
            # Update strategy rankings
            if strategy_used not in self.best_strategies:
                self.best_strategies[strategy_used] = []
            self.best_strategies[strategy_used].append(
                (graph_id, metrics.computation_time.total_seconds())
            )
            
        self._save_metrics(graph_id)
        return is_best
    
    def get_best_solution(self, graph_id: str) -> Optional[SolutionMetrics]:
        """Get the best solution for a specific graph."""
        return self.graphs.get(graph_id, None)?.best_solution
    
    def get_best_strategies(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Get the top performing strategies.
        
        Args:
            top_n: Number of top strategies to return
            
        Returns:
            List of (strategy, avg_time) tuples
        """
        strategy_metrics = []
        for strategy, results in self.best_strategies.items():
            avg_time = sum(time for _, time in results) / len(results)
            strategy_metrics.append((strategy, avg_time))
        
        return sorted(strategy_metrics, key=lambda x: x[1])[:top_n]
    
    def get_performance_summary(self) -> Dict:
        """Get a summary of overall performance metrics."""
        total_graphs = len(self.graphs)
        solved_graphs = sum(1 for g in self.graphs.values() 
                          if g.best_solution is not None)
        
        total_time = timedelta()
        total_energy = 0.0
        for graph in self.graphs.values():
            if graph.best_solution:
                total_time += graph.best_solution.computation_time
                total_energy += graph.best_solution.energy_usage
        
        return {
            "total_graphs": total_graphs,
            "solved_graphs": solved_graphs,
            "success_rate": solved_graphs / total_graphs if total_graphs > 0 else 0,
            "avg_time": (total_time / solved_graphs if solved_graphs > 0 
                        else timedelta()),
            "avg_energy": total_energy / solved_graphs if solved_graphs > 0 else 0,
            "best_strategies": self.get_best_strategies()
        }
    
    def _save_metrics(self, graph_id: str):
        """Save metrics for a specific graph to disk."""
        graph_metrics = self.graphs[graph_id]
        metrics_file = os.path.join(self.metrics_dir, f"{graph_id}.json")
        
        def serialize_metrics(metrics: SolutionMetrics) -> Dict:
            return {
                "path": metrics.path,
                "computation_time": metrics.computation_time.total_seconds(),
                "energy_usage": metrics.energy_usage,
                "strategy_used": metrics.strategy_used,
                "constraints_used": metrics.constraints_used,
                "patterns_used": metrics.patterns_used,
                "theorems_used": metrics.theorems_used,
                "timestamp": metrics.timestamp.isoformat()
            }
        
        data = {
            "graph_id": graph_metrics.graph_id,
            "graph_size": graph_metrics.graph_size,
            "is_planar": graph_metrics.is_planar,
            "connectivity": graph_metrics.connectivity,
            "best_solution": (serialize_metrics(graph_metrics.best_solution)
                            if graph_metrics.best_solution else None),
            "all_solutions": [serialize_metrics(s) for s in graph_metrics.all_solutions]
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(data, f, indent=2)
