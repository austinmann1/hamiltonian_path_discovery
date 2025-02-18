"""
Tests for the prompt integration system.
"""

import numpy as np
import pytest

from hamiltonian_path_discovery.src.prompting.prompt_integration import PromptIntegration

@pytest.fixture
def integration():
    """Create a PromptIntegration instance for testing."""
    return PromptIntegration()

@pytest.fixture
def sample_graph():
    """Create a sample graph adjacency matrix for testing."""
    return np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ])

def test_start_new_graph(integration, sample_graph):
    """Test starting work on a new graph."""
    integration.start_new_graph(sample_graph)
    assert integration.current_graph is not None
    assert np.array_equal(integration.current_graph, sample_graph)

def test_record_path_failure_invalid_edge(integration, sample_graph):
    """Test recording a path with an invalid edge."""
    integration.start_new_graph(sample_graph)
    
    # Try path with invalid edge 0->2
    path = [0, 2, 1, 3]
    conflict = integration.record_path_failure(path, 1)
    
    assert conflict is not None
    assert conflict.conflict_type == "invalid_edge"
    assert conflict.nodes == [0, 2]
    
    # Check that the conflict appears in the next prompt
    prompt = integration.get_next_prompt()
    assert "0->2" in prompt
    assert "invalid edge" in prompt.lower()

def test_record_path_failure_cycle(integration, sample_graph):
    """Test recording a path with a cycle."""
    integration.start_new_graph(sample_graph)
    
    # Try path with cycle 0->1->2->1
    path = [0, 1, 2, 1]
    conflict = integration.record_path_failure(path, 3)
    
    assert conflict is not None
    assert conflict.conflict_type == "cycle"
    assert 1 in conflict.nodes  # Node 1 appears twice in cycle
    
    # Check that the conflict appears in the next prompt
    prompt = integration.get_next_prompt()
    assert "cycle" in prompt.lower()
    assert "1 -> 2 -> 1" in prompt

def test_record_path_failure_dead_end(integration, sample_graph):
    """Test recording a path that leads to a dead end."""
    integration.start_new_graph(sample_graph)
    
    # Try path leading to dead end
    path = [0, 1, 2]
    conflict = integration.record_path_failure(path, 2)
    
    assert conflict is not None
    assert conflict.conflict_type == "dead_end"
    assert 2 in conflict.nodes
    
    # Check that the conflict appears in the next prompt
    prompt = integration.get_next_prompt()
    assert "dead end" in prompt.lower()
    assert "2" in prompt

def test_get_next_prompt_with_patterns(integration, sample_graph):
    """Test prompt generation with patterns."""
    integration.start_new_graph(sample_graph)
    
    patterns = "Start with highest degree vertex"
    prompt = integration.get_next_prompt(known_patterns=patterns)
    
    assert "highest degree vertex" in prompt
    assert "patterns" in prompt.lower()

def test_get_next_prompt_with_theorems(integration, sample_graph):
    """Test prompt generation with theoretical insights."""
    integration.start_new_graph(sample_graph)
    
    insights = "Graph is 4-vertex-connected"
    prompt = integration.get_next_prompt(theorem_insights=insights)
    
    assert "4-vertex-connected" in prompt
    assert "theoretical" in prompt.lower()

def test_get_next_prompt_no_current_graph(integration):
    """Test error handling when no graph is set."""
    with pytest.raises(ValueError):
        integration.get_next_prompt()

def test_clear_state(integration, sample_graph):
    """Test clearing all state."""
    integration.start_new_graph(sample_graph)
    path = [0, 2, 1, 3]  # Invalid path
    integration.record_path_failure(path, 1)
    
    integration.clear_state()
    assert integration.current_graph is None
    
    # Next prompt should raise error
    with pytest.raises(ValueError):
        integration.get_next_prompt()

def test_integration_workflow(integration, sample_graph):
    """Test a complete workflow with multiple failures."""
    integration.start_new_graph(sample_graph)
    
    # First attempt: invalid edge
    path1 = [0, 2, 1, 3]
    conflict1 = integration.record_path_failure(path1, 1)
    assert conflict1.conflict_type == "invalid_edge"
    
    # Second attempt: cycle
    path2 = [0, 1, 2, 1]
    conflict2 = integration.record_path_failure(path2, 3)
    assert conflict2.conflict_type == "cycle"
    
    # Get prompt with both learnings
    prompt = integration.get_next_prompt()
    assert "0->2" in prompt  # From invalid edge
    assert "1 -> 2 -> 1" in prompt  # From cycle
    
    # Add some patterns and theoretical insights
    prompt = integration.get_next_prompt(
        known_patterns="Start with vertex 0",
        theorem_insights="Graph is planar"
    )
    assert "Start with vertex 0" in prompt
    assert "Graph is planar" in prompt
