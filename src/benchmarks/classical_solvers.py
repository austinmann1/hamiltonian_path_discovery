"""Classical solvers for Hamiltonian path problems.

This module provides implementations of classical algorithms for finding
Hamiltonian paths, used for benchmarking our approach.
"""

import networkx as nx
import numpy as np
from typing import List, Optional, Tuple
import time

def networkx_hamiltonian_path(adj_matrix: np.ndarray) -> Tuple[Optional[List[int]], float]:
    """Find Hamiltonian path using NetworkX's implementation.
    
    Args:
        adj_matrix: Adjacency matrix of the graph
        
    Returns:
        Tuple of (path or None, computation time)
    """
    # Convert adjacency matrix to NetworkX graph
    G = nx.from_numpy_array(adj_matrix)
    
    start_time = time.time()
    try:
        # Try to find a Hamiltonian path
        for v in G.nodes():
            try:
                # Try different algorithms
                try:
                    # Try hamiltonian_path
                    path = list(nx.hamiltonian_path(G))
                    computation_time = time.time() - start_time
                    return path, computation_time
                except (nx.NetworkXError, AttributeError):
                    # Try hamiltonian_path_iter
                    try:
                        path = next(nx.hamiltonian_path_iter(G))
                        computation_time = time.time() - start_time
                        return list(path), computation_time
                    except (StopIteration, AttributeError):
                        continue
            except nx.NetworkXError:
                continue
        return None, time.time() - start_time
    except Exception as e:
        print(f"NetworkX solver error: {str(e)}")
        return None, time.time() - start_time

def backtracking_solver(adj_matrix: np.ndarray) -> Tuple[Optional[List[int]], float]:
    """Find Hamiltonian path using backtracking.
    
    Args:
        adj_matrix: Adjacency matrix of the graph
        
    Returns:
        Tuple of (path or None, computation time)
    """
    n = len(adj_matrix)
    path = []
    used = [False] * n
    
    def dfs(v: int) -> bool:
        path.append(v)
        used[v] = True
        if len(path) == n:
            return True
        for u in range(n):
            if not used[u] and adj_matrix[v][u]:
                if dfs(u):
                    return True
        path.pop()
        used[v] = False
        return False
    
    start_time = time.time()
    for start in range(n):
        path = []
        used = [False] * n
        if dfs(start):
            return path, time.time() - start_time
    return None, time.time() - start_time

def compare_with_classical(adj_matrix: np.ndarray) -> dict:
    """Compare different solvers on the same graph.
    
    Args:
        adj_matrix: Adjacency matrix of the graph
        
    Returns:
        Dictionary containing comparison results
    """
    results = {
        "networkx": {"path": None, "time": 0.0, "success": False},
        "backtracking": {"path": None, "time": 0.0, "success": False}
    }
    
    # NetworkX solver
    path, time_taken = networkx_hamiltonian_path(adj_matrix)
    results["networkx"].update({
        "path": path,
        "time": time_taken,
        "success": path is not None
    })
    
    # Backtracking solver
    path, time_taken = backtracking_solver(adj_matrix)
    results["backtracking"].update({
        "path": path,
        "time": time_taken,
        "success": path is not None
    })
    
    return results
