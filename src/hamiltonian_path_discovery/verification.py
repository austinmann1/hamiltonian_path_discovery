"""
Verification oracle for Hamiltonian path solutions.
"""
from typing import List, Optional
import numpy as np
import networkx as nx
from z3 import *

class VerificationOracle:
    def __init__(self, timeout: int = 5000):
        """
        Initialize verification oracle.
        
        Args:
            timeout: Z3 solver timeout in milliseconds
        """
        self.timeout = timeout
    
    def verify_hamiltonian_path(self, 
                               adj_matrix: np.ndarray,
                               path: Optional[List[int]] = None) -> bool:
        """
        Verify if a Hamiltonian path exists or if a given path is valid.
        
        Args:
            adj_matrix: Adjacency matrix of the graph
            path: Optional path to verify. If None, checks existence.
            
        Returns:
            True if path exists/is valid, False otherwise
        """
        n = len(adj_matrix)
        
        # If path is provided, just verify it directly
        if path is not None:
            if len(path) != n:
                return False
            # Check if all vertices are visited exactly once
            if len(set(path)) != n:
                return False
            # Check if edges exist between consecutive vertices
            for i in range(n-1):
                if not adj_matrix[path[i]][path[i+1]]:
                    return False
            return True
        
        # For small graphs (n ≤ 10), use simple backtracking
        if n <= 10:
            return self._find_path_backtracking(adj_matrix)
        
        # For larger graphs, use Z3
        s = Solver()
        s.set("timeout", self.timeout)
        
        # Variables for vertex positions
        pos = [[Bool(f"pos_{i}_{j}") for j in range(n)] for i in range(n)]
        
        # Each vertex must appear exactly once
        for i in range(n):
            s.add(Sum([If(pos[i][j], 1, 0) for j in range(n)]) == 1)
        
        # Each position must have exactly one vertex
        for j in range(n):
            s.add(Sum([If(pos[i][j], 1, 0) for i in range(n)]) == 1)
        
        # Adjacent vertices must have an edge
        for i in range(n):
            for j in range(n):
                for k in range(n-1):
                    if not adj_matrix[i][j]:
                        s.add(Not(And(pos[i][k], pos[j][k+1])))
        
        result = s.check()
        return result == sat
    
    def _find_path_backtracking(self, adj_matrix: np.ndarray) -> bool:
        """Simple backtracking for small graphs."""
        n = len(adj_matrix)
        visited = [False] * n
        
        def backtrack(vertex: int, count: int) -> bool:
            if count == n:
                return True
                
            visited[vertex] = True
            for next_vertex in range(n):
                if (not visited[next_vertex] and 
                    adj_matrix[vertex][next_vertex]):
                    if backtrack(next_vertex, count + 1):
                        return True
            visited[vertex] = False
            return False
        
        # Try starting from each vertex
        for start in range(n):
            visited = [False] * n
            if backtrack(start, 1):
                return True
        return False
    
    def extract_path(self, adj_matrix: np.ndarray) -> Optional[List[int]]:
        """
        Extract a Hamiltonian path if one exists.
        
        Args:
            adj_matrix: Adjacency matrix of the graph
            
        Returns:
            List of vertices in the path order, or None if no path exists
        """
        n = len(adj_matrix)
        
        # For small graphs, use backtracking
        if n <= 10:
            return self._extract_path_backtracking(adj_matrix)
        
        # For larger graphs, use Z3
        s = Solver()
        s.set("timeout", self.timeout)
        
        # Variables for vertex positions
        pos = [[Bool(f"pos_{i}_{j}") for j in range(n)] for i in range(n)]
        
        # Each vertex must appear exactly once
        for i in range(n):
            s.add(Sum([If(pos[i][j], 1, 0) for j in range(n)]) == 1)
        
        # Each position must have exactly one vertex
        for j in range(n):
            s.add(Sum([If(pos[i][j], 1, 0) for i in range(n)]) == 1)
        
        # Adjacent vertices must have an edge
        for i in range(n):
            for j in range(n):
                for k in range(n-1):
                    if not adj_matrix[i][j]:
                        s.add(Not(And(pos[i][k], pos[j][k+1])))
        
        if s.check() == sat:
            model = s.model()
            path = [0] * n
            for i in range(n):
                for j in range(n):
                    if model.evaluate(pos[i][j]):
                        path[j] = i
            return path
        return None
    
    def _extract_path_backtracking(self, adj_matrix: np.ndarray) -> Optional[List[int]]:
        """Extract path using backtracking for small graphs."""
        n = len(adj_matrix)
        visited = [False] * n
        path = []
        
        def backtrack(vertex: int, count: int) -> bool:
            path.append(vertex)
            visited[vertex] = True
            
            if count == n:
                return True
                
            for next_vertex in range(n):
                if (not visited[next_vertex] and 
                    adj_matrix[vertex][next_vertex]):
                    if backtrack(next_vertex, count + 1):
                        return True
            
            path.pop()
            visited[vertex] = False
            return False
        
        # Try starting from each vertex
        for start in range(n):
            visited = [False] * n
            path = []
            if backtrack(start, 1):
                return path
        return None
    
    def verify_with_z3(self, graph: nx.DiGraph, path: List[str]) -> bool:
        """
        Verify a path in a NetworkX graph using Z3.
        
        Args:
            graph: NetworkX directed graph
            path: List of node names representing the proposed path
            
        Returns:
            bool: True if the path is a valid Hamiltonian path
        """
        # Convert graph to adjacency matrix
        adj_matrix = nx.to_numpy_array(graph)
        
        # Convert node names to indices
        node_to_idx = {node: i for i, node in enumerate(graph.nodes())}
        path_indices = [node_to_idx[node] for node in path]
        
        return self.verify_hamiltonian_path(adj_matrix, path_indices)
