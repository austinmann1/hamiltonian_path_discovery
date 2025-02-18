"""Tests for the pattern mining system."""

import pytest
import numpy as np
from src.pattern_mining.pattern_analyzer import (
    PatternAnalyzer,
    PathPattern
)

@pytest.fixture
def pattern_analyzer():
    """Create a pattern analyzer instance for testing."""
    return PatternAnalyzer()

@pytest.fixture
def simple_graph():
    """Create a simple path graph for testing."""
    return np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0]
    ])

@pytest.fixture
def dense_graph():
    """Create a dense graph for testing."""
    return np.array([
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0]
    ])

def test_analyze_solution(pattern_analyzer, simple_graph):
    """Test analysis of a successful solution."""
    path = [0, 1, 2, 3]  # Valid Hamiltonian path
    pattern_analyzer.analyze_solution(
        path=path,
        adj_matrix=simple_graph,
        computation_time=1.0,
        success=True
    )
    
    # Check that patterns were identified
    assert len(pattern_analyzer.patterns) > 0
    
    # Check subpath patterns
    subpath_patterns = [p for p in pattern_analyzer.patterns.values() 
                       if p.pattern_type == "subpath"]
    assert len(subpath_patterns) > 0
    
    # Check degree sequence patterns
    degree_patterns = [p for p in pattern_analyzer.patterns.values() 
                      if p.pattern_type == "degree_sequence"]
    assert len(degree_patterns) > 0
    
    # Check vertex patterns
    vertex_patterns = [p for p in pattern_analyzer.patterns.values() 
                      if p.pattern_type == "vertex"]
    assert len(vertex_patterns) > 0

def test_get_best_patterns(pattern_analyzer, simple_graph):
    """Test retrieving best patterns."""
    # Add multiple solutions
    paths = [
        ([0, 1, 2, 3], 1.0, True),   # Success
        ([3, 2, 1, 0], 1.2, True),   # Success
        ([0, 2, 1, 3], 1.5, False),  # Failure
    ]
    
    for path, time, success in paths:
        pattern_analyzer.analyze_solution(path, simple_graph, time, success)
    
    best_patterns = pattern_analyzer.get_best_patterns(min_frequency=1)
    assert len(best_patterns) > 0
    
    # Check pattern properties
    pattern = best_patterns[0]
    assert isinstance(pattern, PathPattern)
    assert pattern.frequency >= 1
    assert pattern.success_rate >= 0.0
    assert pattern.avg_computation_time > 0.0

def test_get_vertex_recommendations(pattern_analyzer, simple_graph):
    """Test getting vertex recommendations."""
    # Add some successful solutions
    pattern_analyzer.analyze_solution([0, 1, 2, 3], simple_graph, 1.0, True)
    pattern_analyzer.analyze_solution([0, 1, 3, 2], simple_graph, 1.2, True)
    
    recommendations = pattern_analyzer.get_vertex_recommendations(simple_graph)
    
    assert "start_vertices" in recommendations
    assert "end_vertices" in recommendations
    assert len(recommendations["start_vertices"]) > 0
    
    # Vertex 0 should be recommended as a start vertex
    assert 0 in recommendations["start_vertices"]

def test_format_for_prompt(pattern_analyzer, simple_graph):
    """Test formatting patterns for prompts."""
    # Add some successful solutions
    pattern_analyzer.analyze_solution([0, 1, 2, 3], simple_graph, 1.0, True)
    pattern_analyzer.analyze_solution([3, 2, 1, 0], simple_graph, 1.2, True)
    
    prompt_text = pattern_analyzer.format_for_prompt(simple_graph)
    
    assert isinstance(prompt_text, str)
    assert "Pattern Mining Insights" in prompt_text
    assert "Successful Patterns" in prompt_text
    assert "Vertex Recommendations" in prompt_text

def test_clear_patterns(pattern_analyzer, simple_graph):
    """Test clearing all patterns."""
    # Add a solution
    pattern_analyzer.analyze_solution([0, 1, 2, 3], simple_graph, 1.0, True)
    
    # Verify patterns exist
    assert len(pattern_analyzer.patterns) > 0
    assert len(pattern_analyzer.pattern_attempts) > 0
    assert len(pattern_analyzer.pattern_successes) > 0
    
    # Clear patterns
    pattern_analyzer.clear_patterns()
    
    # Verify everything is cleared
    assert len(pattern_analyzer.patterns) == 0
    assert len(pattern_analyzer.pattern_attempts) == 0
    assert len(pattern_analyzer.pattern_successes) == 0
    assert len(pattern_analyzer.pattern_times) == 0

def test_multiple_graph_analysis(pattern_analyzer, simple_graph, dense_graph):
    """Test pattern analysis across different graphs."""
    # Add solutions for both graphs
    pattern_analyzer.analyze_solution([0, 1, 2, 3], simple_graph, 1.0, True)
    pattern_analyzer.analyze_solution([0, 2, 1, 3], dense_graph, 0.8, True)
    
    # Get patterns
    patterns = pattern_analyzer.get_best_patterns(min_frequency=1)
    
    # Should find some common patterns (e.g., starting with vertex 0)
    assert len(patterns) > 0
    
    # Verify vertex 0 patterns
    start_patterns = [p for p in patterns 
                     if p.pattern_type == "vertex" and 
                     "start" in p.description and 
                     p.pattern[0] == 0]
    assert len(start_patterns) > 0
