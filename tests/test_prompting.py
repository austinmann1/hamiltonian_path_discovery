"""
Tests for the prompting system components.
"""

import numpy as np
import pytest

from hamiltonian_path_discovery.src.prompting.prompt_manager import PromptManager

@pytest.fixture
def prompt_manager():
    """Create a PromptManager instance for testing."""
    return PromptManager()

@pytest.fixture
def sample_graph():
    """Create a sample graph adjacency matrix for testing."""
    return np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ])

def test_format_adjacency_matrix(prompt_manager, sample_graph):
    """Test adjacency matrix formatting."""
    formatted = prompt_manager.format_adjacency_matrix(sample_graph)
    assert isinstance(formatted, str)
    assert '0' in formatted and '1' in formatted
    assert '[' in formatted and ']' in formatted

def test_generate_base_prompt(prompt_manager, sample_graph):
    """Test base prompt generation."""
    prompt = prompt_manager.generate_base_prompt(sample_graph)
    assert isinstance(prompt, str)
    assert 'vertices' in prompt
    assert str(len(sample_graph)) in prompt
    assert 'Hamiltonian path' in prompt

def test_generate_conflict_aware_prompt(prompt_manager, sample_graph):
    """Test conflict-aware prompt generation."""
    constraints = "1. Avoid edge (0,2)\n2. Node 3 is a dead end"
    prompt = prompt_manager.generate_conflict_aware_prompt(sample_graph, constraints)
    assert isinstance(prompt, str)
    assert 'conflicts' in prompt
    assert 'Avoid edge (0,2)' in prompt
    assert 'dead end' in prompt

def test_generate_pattern_based_prompt(prompt_manager, sample_graph):
    """Test pattern-based prompt generation."""
    patterns = "1. Start with highest degree vertex\n2. Prefer edge (0,1)"
    prompt = prompt_manager.generate_pattern_based_prompt(sample_graph, patterns)
    assert isinstance(prompt, str)
    assert 'patterns' in prompt
    assert 'highest degree vertex' in prompt
    assert 'Prefer edge' in prompt

def test_generate_theorem_guided_prompt(prompt_manager, sample_graph):
    """Test theorem-guided prompt generation."""
    insights = "Graph is 4-vertex-connected\nPlanar embedding exists"
    prompt = prompt_manager.generate_theorem_guided_prompt(sample_graph, insights)
    assert isinstance(prompt, str)
    assert 'theoretical' in prompt
    assert '4-vertex-connected' in prompt
    assert 'Planar embedding' in prompt

def test_select_best_prompt_type(prompt_manager, sample_graph):
    """Test prompt type selection logic."""
    # Test with all information available
    prompt_type = prompt_manager.select_best_prompt_type(
        sample_graph,
        has_conflicts=True,
        has_patterns=True,
        has_theorems=True
    )
    assert prompt_type == 'theorem_guided'
    
    # Test with only conflicts
    prompt_type = prompt_manager.select_best_prompt_type(
        sample_graph,
        has_conflicts=True,
        has_patterns=False,
        has_theorems=False
    )
    assert prompt_type == 'conflict_aware'
    
    # Test with no additional information
    prompt_type = prompt_manager.select_best_prompt_type(
        sample_graph,
        has_conflicts=False,
        has_patterns=False,
        has_theorems=False
    )
    assert prompt_type == 'base'

def test_generate_prompt_integration(prompt_manager, sample_graph):
    """Test the main generate_prompt method with various inputs."""
    # Test with all information
    prompt = prompt_manager.generate_prompt(
        sample_graph,
        learned_constraints="Avoid edge (0,2)",
        known_patterns="Start with highest degree vertex",
        theorem_insights="Graph is 4-vertex-connected"
    )
    assert isinstance(prompt, str)
    assert 'theoretical' in prompt
    assert '4-vertex-connected' in prompt
    
    # Test with only conflicts
    prompt = prompt_manager.generate_prompt(
        sample_graph,
        learned_constraints="Avoid edge (0,2)"
    )
    assert isinstance(prompt, str)
    assert 'conflicts' in prompt
    assert 'Avoid edge' in prompt
    
    # Test with no additional information
    prompt = prompt_manager.generate_prompt(sample_graph)
    assert isinstance(prompt, str)
    assert 'vertices' in prompt
    assert 'Hamiltonian path' in prompt
