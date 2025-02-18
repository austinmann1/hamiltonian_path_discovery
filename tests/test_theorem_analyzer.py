"""Tests for the theoretical insights system."""

import pytest
import numpy as np
import networkx as nx
from src.theoretical_insights.theorem_analyzer import (
    TheoremAnalyzer,
    TheoremInsight
)

@pytest.fixture
def theorem_analyzer():
    """Create a theorem analyzer instance for testing."""
    return TheoremAnalyzer()

@pytest.fixture
def complete_graph():
    """Create a complete graph that satisfies many theorems."""
    n = 5
    return np.ones((n, n)) - np.eye(n)

@pytest.fixture
def cycle_graph():
    """Create a cycle graph that satisfies some theorems."""
    n = 5
    adj_matrix = np.zeros((n, n))
    for i in range(n):
        adj_matrix[i, (i+1)%n] = 1
        adj_matrix[(i+1)%n, i] = 1
    return adj_matrix

@pytest.fixture
def path_graph():
    """Create a path graph that satisfies fewer theorems."""
    n = 5
    adj_matrix = np.zeros((n, n))
    for i in range(n-1):
        adj_matrix[i, i+1] = 1
        adj_matrix[i+1, i] = 1
    return adj_matrix

def test_analyze_complete_graph(theorem_analyzer, complete_graph):
    """Test analysis of a complete graph."""
    insights = theorem_analyzer.analyze_graph(complete_graph)
    
    # Complete graph should satisfy multiple theorems
    theorem_names = [insight.theorem_name for insight in insights]
    assert "Dirac's Theorem" in theorem_names
    assert "Ore's Theorem" in theorem_names
    
    # All insights should be applicable
    assert all(insight.applies for insight in insights)
    
    # Check specific conditions
    dirac_insight = next(i for i in insights if i.theorem_name == "Dirac's Theorem")
    assert any("minimum degree" in cond.lower() for cond in dirac_insight.conditions)

def test_analyze_cycle_graph(theorem_analyzer, cycle_graph):
    """Test analysis of a cycle graph."""
    insights = theorem_analyzer.analyze_graph(cycle_graph)
    
    # Cycle graph should be planar but not satisfy Dirac's theorem
    theorem_names = [insight.theorem_name for insight in insights]
    assert "Grinberg's Theorem" in theorem_names
    assert "Dirac's Theorem" not in theorem_names
    
    # Should be biconnected
    assert any(i.theorem_name == "Tutte's Theorem" for i in insights)

def test_analyze_path_graph(theorem_analyzer, path_graph, complete_graph):
    """Test analysis of a path graph."""
    insights = theorem_analyzer.analyze_graph(path_graph)
    complete_insights = theorem_analyzer.analyze_graph(complete_graph)
    
    # Path graph satisfies fewer theorems
    assert len(insights) < len(complete_insights)
    
    # Should still be planar
    assert any(i.theorem_name == "Grinberg's Theorem" for i in insights)
    
    # Should not satisfy stronger theorems
    theorem_names = [insight.theorem_name for insight in insights]
    assert "Dirac's Theorem" not in theorem_names
    assert "Ore's Theorem" not in theorem_names

def test_get_recommendations(theorem_analyzer, complete_graph):
    """Test getting recommendations from theoretical insights."""
    recommendations = theorem_analyzer.get_recommendations(complete_graph)
    
    assert "starting_vertices" in recommendations
    assert "path_constraints" in recommendations
    assert "search_strategy" in recommendations
    
    # Complete graph should have permissive recommendations
    strategies = recommendations["search_strategy"]
    assert any("any vertex" in s.lower() for s in strategies)

def test_format_for_prompt(theorem_analyzer, cycle_graph):
    """Test formatting insights for prompts."""
    prompt_text = theorem_analyzer.format_for_prompt(cycle_graph)
    
    assert isinstance(prompt_text, str)
    assert "Theoretical Insights:" in prompt_text
    assert "Conditions met:" in prompt_text
    assert "Implications:" in prompt_text
    assert "Recommendations:" in prompt_text

def test_empty_graph(theorem_analyzer):
    """Test handling of empty graph."""
    empty_graph = np.zeros((5, 5))
    insights = theorem_analyzer.analyze_graph(empty_graph)
    
    # Should return empty list for invalid graph
    assert len(insights) == 0
    
    # Should still format without error
    prompt_text = theorem_analyzer.format_for_prompt(empty_graph)
    assert isinstance(prompt_text, str)

def test_small_graph(theorem_analyzer):
    """Test handling of small graph."""
    # 2-vertex graph
    small_graph = np.array([[0, 1], [1, 0]])
    insights = theorem_analyzer.analyze_graph(small_graph)
    
    # Should handle small graphs gracefully
    assert len(insights) > 0
    
    # Recommendations should still work
    recommendations = theorem_analyzer.get_recommendations(small_graph)
    assert all(isinstance(v, list) for v in recommendations.values())
