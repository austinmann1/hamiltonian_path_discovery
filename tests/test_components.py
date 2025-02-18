"""
Test suite for Hamiltonian Path Discovery components.
"""
import pytest
import numpy as np
from src.graph_generator import GraphGenerator
from src.verification import VerificationOracle
from src.energy_monitor import EnergyMonitor

def test_graph_generator():
    generator = GraphGenerator()
    
    # Simple SAT instance: (x1 OR x2) AND (NOT x1 OR x2)
    cnf = [[1, 2], [-1, 2]]
    graph = generator.sat_to_hamiltonian(cnf)
    
    # Basic checks
    assert graph is not None
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0
    
    # Check start and end nodes exist
    assert 'start' in graph.nodes
    assert 'end' in graph.nodes

def test_verification_oracle():
    oracle = VerificationOracle()
    
    # Simple path test
    adj_matrix = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ])
    
    # Valid path should exist
    assert oracle.verify_hamiltonian_path(adj_matrix)
    
    # Test specific path
    path = [0, 1, 2]
    assert oracle.verify_hamiltonian_path(adj_matrix, path)
    
    # Invalid path should fail
    invalid_path = [0, 2, 1]
    assert not oracle.verify_hamiltonian_path(adj_matrix, invalid_path)

def test_energy_monitor():
    monitor = EnergyMonitor()
    
    with monitor.track("test_task"):
        # Simulate some work
        [i * i for i in range(1000)]
    
    # Basic system info check
    system_info = monitor.get_system_info()
    assert "cpu_count" in system_info
    assert "total_memory_gb" in system_info

def test_end_to_end():
    # Create components
    generator = GraphGenerator()
    oracle = VerificationOracle()
    monitor = EnergyMonitor()
    
    # Simple SAT instance
    cnf = [[1], [-2]]
    
    with monitor.track("end_to_end_test"):
        # Generate graph
        graph = generator.sat_to_hamiltonian(cnf)
        
        # Get adjacency matrix
        adj_matrix = generator.get_adjacency_matrix()
        
        # Verify Hamiltonian path exists
        assert oracle.verify_hamiltonian_path(adj_matrix)
        
        # Extract and verify path
        path = oracle.extract_path(adj_matrix)
        assert path is not None
        assert oracle.verify_hamiltonian_path(adj_matrix, path)

if __name__ == "__main__":
    pytest.main([__file__])
