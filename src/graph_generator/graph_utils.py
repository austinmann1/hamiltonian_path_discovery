"""
Utility functions for graph manipulation and test case generation.
"""

import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Optional
import random

class GraphUtils:
    """
    Utilities for graph manipulation and test case generation.
    """
    
    def __init__(self):
        pass
    
    def generate_random_graph(
        self,
        size: int,
        edge_probability: float = 0.3,
        ensure_hamiltonian: bool = True
    ) -> Tuple[np.ndarray, bool]:
        """
        Generate a random directed graph.
        
        Args:
            size: Number of nodes
            edge_probability: Probability of edge between any two nodes
            ensure_hamiltonian: If True, ensure graph has a Hamiltonian path
            
        Returns:
            Tuple of (adjacency matrix, has_solution)
        """
        if ensure_hamiltonian:
            # Generate a random permutation to ensure Hamiltonian path exists
            path = list(range(size))
            random.shuffle(path)
            
            # Create adjacency matrix with the path
            matrix = np.zeros((size, size), dtype=int)
            for i in range(size - 1):
                matrix[path[i], path[i + 1]] = 1
            
            # Add random edges
            for i in range(size):
                for j in range(size):
                    if i != j and matrix[i, j] == 0:
                        if random.random() < edge_probability:
                            matrix[i, j] = 1
            
            return matrix, True
        else:
            # Generate completely random graph
            matrix = np.random.choice(
                [0, 1],
                size=(size, size),
                p=[1 - edge_probability, edge_probability]
            )
            np.fill_diagonal(matrix, 0)  # No self-loops
            
            # Check if it has a Hamiltonian path
            G = nx.DiGraph(matrix)
            has_solution = self.check_hamiltonian_path_exists(G)
            
            return matrix, has_solution
    
    def check_hamiltonian_path_exists(self, G: nx.DiGraph) -> bool:
        """
        Check if a Hamiltonian path exists in the graph.
        Uses a simple DFS-based check (not efficient for large graphs).
        
        Args:
            G: NetworkX directed graph
            
        Returns:
            True if a Hamiltonian path exists
        """
        def dfs(node: int, visited: set, path: List[int]) -> bool:
            if len(path) == len(G):
                return True
            
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    if dfs(neighbor, visited, path):
                        return True
                    visited.remove(neighbor)
                    path.pop()
            return False
        
        # Try each node as starting point
        for start in G.nodes():
            visited = {start}
            path = [start]
            if dfs(start, visited, path):
                return True
        return False
    
    def apply_degree_heuristics(self, matrix: np.ndarray) -> Dict[str, bool]:
        """
        Apply degree-based heuristics to estimate likelihood of Hamiltonian path.
        
        Args:
            matrix: Adjacency matrix
            
        Returns:
            Dictionary of heuristic results
        """
        n = len(matrix)
        G = nx.DiGraph(matrix)
        
        # Calculate in and out degrees
        in_degrees = [d for _, d in G.in_degree()]
        out_degrees = [d for _, d in G.out_degree()]
        
        results = {
            # Dirac's theorem analog for directed graphs
            "dirac": all(d >= n/2 for d in in_degrees) and 
                    all(d >= n/2 for d in out_degrees),
            
            # Check for isolated vertices
            "no_isolated": all(d > 0 for d in in_degrees) and 
                         all(d > 0 for d in out_degrees),
            
            # Check degree sum condition
            "degree_sum": all(in_deg + out_deg >= n 
                            for in_deg, out_deg in zip(in_degrees, out_degrees))
        }
        
        return results
    
    def generate_test_case(
        self,
        size: int,
        ensure_solution: bool = True
    ) -> Dict[str, any]:
        """
        Generate a test case for the Hamiltonian path problem.
        
        Args:
            size: Number of nodes
            ensure_solution: If True, ensure the graph has a solution
            
        Returns:
            Dictionary containing test case data
        """
        # Generate graph
        matrix, has_solution = self.generate_random_graph(
            size=size,
            ensure_hamiltonian=ensure_solution
        )
        
        # Apply heuristics
        heuristics = self.apply_degree_heuristics(matrix)
        
        # Create test case
        test_case = {
            "size": size,
            "input": matrix.tolist(),
            "has_solution": has_solution,
            "heuristics": heuristics
        }
        
        return test_case
    
    def generate_test_suite(
        self,
        sizes: List[int],
        cases_per_size: int = 2
    ) -> List[Dict[str, any]]:
        """
        Generate a suite of test cases.
        
        Args:
            sizes: List of graph sizes to generate
            cases_per_size: Number of cases for each size
            
        Returns:
            List of test cases
        """
        test_cases = []
        
        for size in sizes:
            # Generate cases with and without solutions
            for _ in range(cases_per_size // 2):
                # Case with solution
                test_cases.append(
                    self.generate_test_case(size, ensure_solution=True)
                )
                # Case without guaranteed solution
                test_cases.append(
                    self.generate_test_case(size, ensure_solution=False)
                )
        
        return test_cases
    
    def generate_random_hamiltonian_graph(
        self,
        size: int,
        edge_probability: float = 0.3,
        max_attempts: int = 100
    ) -> np.ndarray:
        """
        Generate a random directed graph that contains at least one Hamiltonian path.
        
        Args:
            size: Number of nodes in the graph
            edge_probability: Probability of adding extra edges
            max_attempts: Maximum number of attempts to generate a valid graph
            
        Returns:
            Adjacency matrix of the generated graph
        """
        for _ in range(max_attempts):
            # Create a guaranteed Hamiltonian path from 0 to size-1
            path = [0]  # Start with node 0
            remaining = list(range(1, size-1))  # Middle nodes
            random.shuffle(remaining)
            path.extend(remaining)
            path.append(size-1)  # End with node size-1
            
            # Create adjacency matrix with the path
            adj_matrix = np.zeros((size, size), dtype=int)
            for i in range(size - 1):
                adj_matrix[path[i]][path[i+1]] = 1
            
            # Add random additional edges
            for i in range(size):
                for j in range(size):
                    if i != j and adj_matrix[i][j] == 0:
                        if random.random() < edge_probability:
                            adj_matrix[i][j] = 1
            
            return adj_matrix
        
        raise RuntimeError(f"Failed to generate valid graph after {max_attempts} attempts")
