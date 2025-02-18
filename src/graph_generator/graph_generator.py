"""
Graph generator for creating test graphs for Hamiltonian path discovery.
"""

import numpy as np
from typing import Optional

class GraphGenerator:
    """Generates various types of graphs for testing."""
    
    def generate_random_graph(self, num_vertices: int, density: float) -> np.ndarray:
        """
        Generate a random undirected graph with given number of vertices and edge density.
        
        Args:
            num_vertices: Number of vertices in the graph
            density: Probability of edge existence between any two vertices (0-1)
            
        Returns:
            Adjacency matrix of the generated graph
        """
        # Create empty adjacency matrix
        adj_matrix = np.zeros((num_vertices, num_vertices), dtype=int)
        
        # Add edges randomly based on density
        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):
                if np.random.random() < density:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1  # Make it symmetric for undirected graph
        
        return adj_matrix
    
    def generate_cycle_graph(self, num_vertices: int) -> np.ndarray:
        """Generate a cycle graph with given number of vertices."""
        adj_matrix = np.zeros((num_vertices, num_vertices), dtype=int)
        for i in range(num_vertices):
            adj_matrix[i, (i + 1) % num_vertices] = 1
            adj_matrix[(i + 1) % num_vertices, i] = 1
        return adj_matrix
    
    def generate_path_graph(self, num_vertices: int) -> np.ndarray:
        """Generate a path graph with given number of vertices."""
        adj_matrix = np.zeros((num_vertices, num_vertices), dtype=int)
        for i in range(num_vertices - 1):
            adj_matrix[i, i + 1] = 1
            adj_matrix[i + 1, i] = 1
        return adj_matrix
    
    def generate_complete_graph(self, num_vertices: int) -> np.ndarray:
        """Generate a complete graph with given number of vertices."""
        adj_matrix = np.ones((num_vertices, num_vertices), dtype=int)
        np.fill_diagonal(adj_matrix, 0)  # No self-loops
        return adj_matrix
