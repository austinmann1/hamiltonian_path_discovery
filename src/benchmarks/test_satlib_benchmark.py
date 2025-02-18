"""Tests for SATLIB benchmark functionality."""

import pytest
import numpy as np
from pathlib import Path
import os

from benchmarks.satlib_benchmark import SATLIBBenchmark
from benchmarks.benchmark_generator import BenchmarkGenerator
from solution_validator import validate_hamiltonian_path, analyze_graph_properties

class TestSATLIBBenchmark:
    """Test suite for SATLIB benchmark functionality."""
    
    @pytest.fixture
    def benchmark(self):
        """Create benchmark instance for testing."""
        return SATLIBBenchmark()
    
    @pytest.fixture
    def generator(self):
        """Create benchmark generator for testing."""
        return BenchmarkGenerator()
        
    def test_load_satlib_instance(self, benchmark, generator):
        """Test loading and parsing SATLIB format files."""
        # Generate a test instance
        files = generator.generate_benchmark_set(
            num_instances=1,
            min_vertices=5,
            max_vertices=5
        )
        
        # Load and parse the instance
        adj_matrix, metadata = benchmark.load_satlib_instance(files[0])
        
        # Verify basic properties
        assert isinstance(adj_matrix, np.ndarray)
        assert adj_matrix.shape == (5, 5)
        assert metadata['num_vertices'] == 5
        assert 'known_solution' in metadata
        
        # Verify graph properties
        properties = analyze_graph_properties(adj_matrix)
        assert properties['size'] == 5
        assert 0 <= properties['density'] <= 1
        assert properties['is_connected']  # Generator should create connected graphs
        
    def test_validate_hamiltonian_path(self, benchmark, generator):
        """Test path validation with different scenarios."""
        # Generate a small test instance
        files = generator.generate_benchmark_set(
            num_instances=1,
            min_vertices=4,
            max_vertices=4
        )
        
        adj_matrix, metadata = benchmark.load_satlib_instance(files[0])
        known_solution = metadata['known_solution']
        
        # Test valid path
        is_valid, failure_info = validate_hamiltonian_path(known_solution, adj_matrix)
        assert is_valid
        assert failure_info is None
        
        # Test invalid path - duplicate vertex
        invalid_path = [0, 1, 1, 2]
        is_valid, failure_info = validate_hamiltonian_path(invalid_path, adj_matrix)
        assert not is_valid
        assert failure_info['reason'].startswith('Duplicate vertex')
        assert failure_info['duplicate_vertex'] == 1
        
        # Test invalid path - disconnected vertices
        invalid_path = [0, 2, 1, 3]  # Assuming not all vertices are connected
        is_valid, failure_info = validate_hamiltonian_path(invalid_path, adj_matrix)
        if not is_valid:  # Only check if path is invalid (might be valid in some graphs)
            assert failure_info['reason'].startswith('No edge between vertices')
            assert 'invalid_edge' in failure_info
            
    def test_pattern_analysis(self, benchmark, generator):
        """Test pattern mining and analysis."""
        # Generate test instances
        files = generator.generate_benchmark_set(
            num_instances=2,
            min_vertices=4,
            max_vertices=4
        )
        
        # Create a mock successful solution
        adj_matrix, _ = benchmark.load_satlib_instance(files[0])
        mock_path = [0, 1, 2, 3]  # Simple path for testing
        mock_code = """
def find_hamiltonian_path(adj_matrix):
    return [0, 1, 2, 3]
"""
        
        # Analyze the solution
        benchmark.pattern_analyzer.analyze_solution(
            path=mock_path,
            adj_matrix=adj_matrix,
            computation_time=0.1,
            success=True,
            code=mock_code
        )
        
        # Verify pattern storage
        patterns = benchmark.pattern_analyzer.patterns
        assert any(p.pattern_type == "code" for p in patterns.values())
        assert any(p.pattern_type == "vertex" for p in patterns.values())
        assert any(p.pattern_type == "subpath" for p in patterns.values())
        
        # Verify pattern metrics
        for pattern in patterns.values():
            assert pattern.frequency > 0
            assert 0 <= pattern.success_rate <= 1
            assert pattern.avg_computation_time >= 0
            
        # Test pattern formatting
        insights = benchmark.pattern_analyzer.format_for_prompt(adj_matrix)
        assert "Graph Properties:" in insights
        assert "Successful Implementations:" in insights
        
    def test_failure_tracking(self, benchmark, generator):
        """Test tracking and analysis of failed attempts."""
        files = generator.generate_benchmark_set(
            num_instances=1,
            min_vertices=4,
            max_vertices=4
        )
        
        adj_matrix, _ = benchmark.load_satlib_instance(files[0])
        
        # Test invalid path
        invalid_path = [0, 1, 1, 2]  # Contains duplicate
        failure_info = {
            'reason': 'Duplicate vertex found',
            'failure_point': 2,
            'duplicate_vertex': 1
        }
        
        # Analyze failed solution
        benchmark.pattern_analyzer.analyze_solution(
            path=invalid_path,
            adj_matrix=adj_matrix,
            computation_time=0.1,
            success=False,
            code="def find_path(): return [0,1,1,2]",
            failure_info=failure_info
        )
        
        # Verify failure patterns
        patterns = benchmark.pattern_analyzer.patterns
        failure_patterns = [p for p in patterns.values() if p.pattern_type == "failure"]
        assert len(failure_patterns) > 0
        
        # Check failure pattern details
        failure = failure_patterns[0]
        assert failure.pattern == invalid_path
        assert failure.metadata['failure_point'] == 2
        assert 'duplicate_vertex' in failure.metadata
        
        # Verify failures appear in prompt
        insights = benchmark.pattern_analyzer.format_for_prompt(adj_matrix)
        assert "Recent Failures:" in insights
