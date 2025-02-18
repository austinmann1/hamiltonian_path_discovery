"""Validation utilities for Hamiltonian path solutions."""

from typing import List, Tuple, Dict, Optional
import numpy as np

def is_trivial_solution(path: List[int]) -> bool:
    """Check if a solution is trivial (sequential range)."""
    if not path:
        return False
    return path == list(range(len(path)))

def validate_hamiltonian_path(path: List[int], adj_matrix: np.ndarray) -> Tuple[bool, Optional[Dict]]:
    """
    Validate a Hamiltonian path solution with detailed failure tracking.
    
    Args:
        path: List of vertices representing the path
        adj_matrix: Adjacency matrix of the graph
        
    Returns:
        Tuple of (is_valid, failure_info)
        failure_info is None if path is valid, otherwise contains failure details
    """
    n = len(adj_matrix)
    
    # Check if path is None
    if path is None:
        # Check if a path should exist (basic necessary conditions)
        min_degree = min(np.sum(adj_matrix, axis=0))
        if min_degree >= n/2:  # Dirac's theorem condition
            return False, {
                'reason': 'Path reported as None but graph likely has Hamiltonian path (Dirac\'s theorem)',
                'min_degree': int(min_degree),
                'graph_size': n,
                'failure_type': 'theoretical_violation'
            }
        return True, None  # None is acceptable if path might not exist
    
    # Check for trivial solution
    if is_trivial_solution(path):
        # Verify if the trivial solution is actually valid
        for i in range(len(path)-1):
            if not adj_matrix[i][i+1]:
                return False, {
                    'reason': 'Trivial sequential solution invalid due to missing edges',
                    'failure_point': i,
                    'invalid_edge': (i, i+1),
                    'failure_type': 'trivial_invalid'
                }
    
    # Check path length
    if len(path) != n:
        return False, {
            'reason': f'Path length {len(path)} does not match graph size {n}',
            'failure_point': len(path),
            'expected_length': n,
            'failure_type': 'length_mismatch'
        }
    
    # Check for duplicate vertices
    seen = set()
    for i, v in enumerate(path):
        if v in seen:
            return False, {
                'reason': f'Duplicate vertex {v} found in path',
                'failure_point': i,
                'duplicate_vertex': v,
                'previous_occurrence': path.index(v),
                'failure_type': 'duplicate_vertex'
            }
        seen.add(v)
    
    # Check vertex range
    if any(v < 0 or v >= n for v in path):
        invalid_v = next(v for v in path if v < 0 or v >= n)
        return False, {
            'reason': f'Invalid vertex {invalid_v} outside range [0, {n-1}]',
            'failure_point': path.index(invalid_v),
            'invalid_vertex': invalid_v,
            'valid_range': (0, n-1),
            'failure_type': 'out_of_range'
        }
    
    # Check edge connectivity with detailed tracking
    for i in range(len(path) - 1):
        v1, v2 = path[i], path[i + 1]
        if not adj_matrix[v1][v2]:
            # Get vertex degrees for context
            v1_degree = np.sum(adj_matrix[v1])
            v2_degree = np.sum(adj_matrix[v2])
            return False, {
                'reason': f'No edge between vertices {v1} and {v2}',
                'failure_point': i,
                'invalid_edge': (v1, v2),
                'vertex_degrees': {v1: int(v1_degree), v2: int(v2_degree)},
                'subpath_context': path[max(0, i-1):min(len(path), i+3)],
                'failure_type': 'missing_edge'
            }
    
    return True, None

def analyze_graph_properties(adj_matrix: np.ndarray) -> Dict:
    """
    Analyze graph properties relevant to Hamiltonian path existence.
    
    Args:
        adj_matrix: Adjacency matrix of the graph
        
    Returns:
        Dictionary of graph properties and theoretical insights
    """
    n = len(adj_matrix)
    degrees = np.sum(adj_matrix, axis=0)
    min_degree = int(min(degrees))
    
    # Calculate theoretical properties
    properties = {
        'size': n,
        'density': float(np.sum(adj_matrix)) / (n * (n - 1)),
        'min_degree': min_degree,
        'max_degree': int(max(degrees)),
        'avg_degree': float(np.mean(degrees)),
        'is_connected': is_connected(adj_matrix),
        'theoretical_insights': []
    }
    
    # Check Dirac's theorem condition
    if min_degree >= n/2:
        properties['theoretical_insights'].append({
            'name': "Dirac's Theorem",
            'condition': 'satisfied',
            'implication': 'Graph must contain a Hamiltonian path'
        })
    
    # Check Ore's theorem condition
    ore_condition = True
    for i in range(n):
        for j in range(i+1, n):
            if not adj_matrix[i][j] and degrees[i] + degrees[j] < n:
                ore_condition = False
                break
        if not ore_condition:
            break
    
    if ore_condition:
        properties['theoretical_insights'].append({
            'name': "Ore's Theorem",
            'condition': 'satisfied',
            'implication': 'Graph must contain a Hamiltonian cycle'
        })
    
    return properties

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
