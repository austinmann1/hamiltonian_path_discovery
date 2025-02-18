"""Tests for the conflict learning system."""

import pytest
import numpy as np
from hamiltonian_path_discovery.conflict_learning.conflict_tracker import ConflictTracker, PathConflict

@pytest.fixture
def simple_graph():
    """Create a simple test graph."""
    return np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0]
    ])

@pytest.fixture
def conflict_tracker():
    """Create a ConflictTracker instance."""
    return ConflictTracker()

def test_invalid_edge_detection(conflict_tracker, simple_graph):
    """Test detection of invalid edges in path."""
    path = [0, 2]  # Edge 0->2 doesn't exist
    conflict = conflict_tracker.analyze_path_failure(path, simple_graph, 1)
    
    assert conflict is not None
    assert conflict.conflict_type == 'invalid_edge'
    assert conflict.nodes == [0, 2]
    assert 'No edge exists' in conflict.description
    assert (0, 2) in conflict_tracker.invalid_edges

def test_cycle_detection(conflict_tracker, simple_graph):
    """Test detection of cycles (repeated vertices)."""
    path = [0, 1, 2, 1]  # Vertex 1 appears twice
    conflict = conflict_tracker.analyze_path_failure(path, simple_graph, 3)
    
    assert conflict is not None
    assert conflict.conflict_type == 'cycle'
    assert 1 in conflict.nodes
    assert 'Cycle detected' in conflict.description
    assert any(1 in cycle for cycle in conflict_tracker.cycles)

def test_dead_end_detection(conflict_tracker, simple_graph):
    """Test detection of paths that lead to dead ends."""
    path = [0, 1, 2, 3]  # At vertex 3, we can't go anywhere
    conflict = conflict_tracker.analyze_path_failure(path, simple_graph, 3)
    
    assert conflict is not None
    assert conflict.conflict_type == 'dead_end'
    assert 3 in conflict.nodes
    assert 'Dead end' in conflict.description
    assert 3 in conflict_tracker.dead_ends

def test_prompt_formatting(conflict_tracker, simple_graph):
    """Test formatting of conflicts for LLM prompt."""
    # Add some conflicts
    conflict_tracker.analyze_path_failure([0, 2], simple_graph, 1)  # Invalid edge
    conflict_tracker.analyze_path_failure([0, 1, 2, 1], simple_graph, 3)  # Cycle
    conflict_tracker.analyze_path_failure([0, 1, 2, 3], simple_graph, 3)  # Dead end
    
    prompt_text = conflict_tracker.format_for_prompt()
    
    assert 'Learned Constraints:' in prompt_text
    assert 'Avoid invalid edges:' in prompt_text
    assert 'Dead end nodes to avoid:' in prompt_text
    assert 'Cycle patterns to avoid:' in prompt_text

def test_clear_conflicts(conflict_tracker, simple_graph):
    """Test clearing of all conflicts."""
    # Add some conflicts
    conflict_tracker.analyze_path_failure([0, 2], simple_graph, 1)
    conflict_tracker.analyze_path_failure([0, 1, 2, 1], simple_graph, 3)
    
    assert len(conflict_tracker.conflicts) > 0
    assert len(conflict_tracker.invalid_edges) > 0
    assert len(conflict_tracker.cycles) > 0
    
    conflict_tracker.clear()
    
    assert len(conflict_tracker.conflicts) == 0
    assert len(conflict_tracker.invalid_edges) == 0
    assert len(conflict_tracker.cycles) == 0
    assert len(conflict_tracker.dead_ends) == 0
