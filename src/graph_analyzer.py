"""Graph analysis utilities for Hamiltonian path discovery."""

from typing import Dict
import numpy as np

def analyze_graph_properties(adj_matrix: np.ndarray) -> Dict:
    """
    Analyze graph properties relevant to Hamiltonian path existence.
    
    Args:
        adj_matrix: Adjacency matrix of the graph
        
    Returns:
        Dictionary of graph properties
    """
    n = len(adj_matrix)
    degrees = np.sum(adj_matrix, axis=0)
    
    return {
        'size': n,
        'density': float(np.sum(adj_matrix)) / (n * (n - 1)),
        'min_degree': int(min(degrees)),
        'max_degree': int(max(degrees)),
        'avg_degree': float(np.mean(degrees)),
        'is_connected': is_connected(adj_matrix),
        'dirac_condition_met': min(degrees) >= n/2
    }

def is_connected(adj_matrix: np.ndarray) -> bool:
    """Check if graph is connected using DFS."""
    n = len(adj_matrix)
    visited = [False] * n
    
    def dfs(v: int) -> None:
        visited[v] = True
        for u in range(n):
            if adj_matrix[v][u] and not visited[u]:
                dfs(u)
    
    dfs(0)  # Start from vertex 0
    return all(visited)
