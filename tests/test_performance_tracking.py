"""
Tests for the performance tracking system.
"""

import numpy as np
import pytest
from datetime import datetime, timedelta
import os
import tempfile
import json

from hamiltonian_path_discovery.src.performance_tracking.performance_tracker import (
    PerformanceTracker,
    SolutionMetrics,
    GraphMetrics
)

@pytest.fixture
def temp_metrics_dir():
    """Create a temporary directory for metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def tracker(temp_metrics_dir):
    """Create a PerformanceTracker instance."""
    return PerformanceTracker(metrics_dir=temp_metrics_dir)

@pytest.fixture
def sample_graph():
    """Create a sample graph adjacency matrix."""
    return np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ])

def test_register_graph(tracker, sample_graph):
    """Test registering a new graph."""
    tracker.register_graph("graph1", sample_graph, is_planar=True, connectivity=2)
    
    assert "graph1" in tracker.graphs
    graph_metrics = tracker.graphs["graph1"]
    assert graph_metrics.graph_size == 4
    assert graph_metrics.is_planar
    assert graph_metrics.connectivity == 2

def test_record_solution(tracker, sample_graph):
    """Test recording a solution."""
    tracker.register_graph("graph1", sample_graph)
    
    # Record first solution
    is_best = tracker.record_solution(
        graph_id="graph1",
        path=[0, 1, 2, 3],
        computation_time=timedelta(seconds=1.5),
        energy_usage=100.0,
        strategy_used="backtracking",
        constraints_used=["avoid_cycles"],
        patterns_used=["start_high_degree"],
        theorems_used=["planarity"]
    )
    
    assert is_best  # First solution should be best
    assert len(tracker.graphs["graph1"].all_solutions) == 1
    
    # Record faster solution
    is_best = tracker.record_solution(
        graph_id="graph1",
        path=[0, 3, 2, 1],
        computation_time=timedelta(seconds=1.0),
        energy_usage=90.0,
        strategy_used="pattern_based"
    )
    
    assert is_best  # Faster solution should be best
    assert len(tracker.graphs["graph1"].all_solutions) == 2
    assert tracker.graphs["graph1"].best_solution.computation_time == timedelta(seconds=1.0)

def test_get_best_solution(tracker, sample_graph):
    """Test retrieving the best solution."""
    tracker.register_graph("graph1", sample_graph)
    
    # Record two solutions
    tracker.record_solution(
        graph_id="graph1",
        path=[0, 1, 2, 3],
        computation_time=timedelta(seconds=1.5),
        energy_usage=100.0,
        strategy_used="backtracking"
    )
    
    tracker.record_solution(
        graph_id="graph1",
        path=[0, 3, 2, 1],
        computation_time=timedelta(seconds=1.0),
        energy_usage=90.0,
        strategy_used="pattern_based"
    )
    
    best = tracker.get_best_solution("graph1")
    assert best is not None
    assert best.path == [0, 3, 2, 1]
    assert best.computation_time == timedelta(seconds=1.0)

def test_get_best_strategies(tracker, sample_graph):
    """Test retrieving best performing strategies."""
    tracker.register_graph("graph1", sample_graph)
    tracker.register_graph("graph2", sample_graph)
    
    # Record solutions with different strategies
    tracker.record_solution(
        graph_id="graph1",
        path=[0, 1, 2, 3],
        computation_time=timedelta(seconds=1.5),
        energy_usage=100.0,
        strategy_used="backtracking"
    )
    
    tracker.record_solution(
        graph_id="graph2",
        path=[0, 3, 2, 1],
        computation_time=timedelta(seconds=1.0),
        energy_usage=90.0,
        strategy_used="pattern_based"
    )
    
    best_strategies = tracker.get_best_strategies()
    assert len(best_strategies) == 2
    assert best_strategies[0][0] == "pattern_based"  # Faster strategy should be first

def test_get_performance_summary(tracker, sample_graph):
    """Test getting overall performance summary."""
    tracker.register_graph("graph1", sample_graph)
    tracker.register_graph("graph2", sample_graph)
    
    # Record some solutions
    tracker.record_solution(
        graph_id="graph1",
        path=[0, 1, 2, 3],
        computation_time=timedelta(seconds=1.0),
        energy_usage=100.0,
        strategy_used="backtracking"
    )
    
    tracker.record_solution(
        graph_id="graph2",
        path=[0, 3, 2, 1],
        computation_time=timedelta(seconds=2.0),
        energy_usage=200.0,
        strategy_used="pattern_based"
    )
    
    summary = tracker.get_performance_summary()
    assert summary["total_graphs"] == 2
    assert summary["solved_graphs"] == 2
    assert summary["success_rate"] == 1.0
    assert summary["avg_time"] == timedelta(seconds=1.5)
    assert summary["avg_energy"] == 150.0

def test_metrics_persistence(tracker, sample_graph, temp_metrics_dir):
    """Test that metrics are properly saved to disk."""
    tracker.register_graph("graph1", sample_graph)
    
    tracker.record_solution(
        graph_id="graph1",
        path=[0, 1, 2, 3],
        computation_time=timedelta(seconds=1.0),
        energy_usage=100.0,
        strategy_used="backtracking"
    )
    
    # Check that metrics file exists
    metrics_file = os.path.join(temp_metrics_dir, "graph1.json")
    assert os.path.exists(metrics_file)
    
    # Verify file contents
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    assert data["graph_id"] == "graph1"
    assert data["graph_size"] == 4
    assert len(data["all_solutions"]) == 1
    assert data["best_solution"]["path"] == [0, 1, 2, 3]
