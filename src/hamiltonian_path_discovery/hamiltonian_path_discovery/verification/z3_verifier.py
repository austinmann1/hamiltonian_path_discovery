"""
Z3-based verification system for Hamiltonian path problems.
"""

import time
from typing import List, Tuple, Dict, Optional
import numpy as np
from z3 import *
from ..logging import StructuredLogger

class Z3HamiltonianVerifier:
    """
    Verifies Hamiltonian paths using Z3 SMT solver.
    """
    
    def __init__(self):
        self.logger = StructuredLogger()
        self.solver = Solver()
        self.last_verification_time = 0.0
        self._cache = {}  # Initialize cache
    
    def create_variables(self, n: int) -> Tuple[List[List[Bool]], List[List[Int]]]:
        """
        Create Z3 variables for the verification.
        
        Args:
            n: Number of nodes in the graph
            
        Returns:
            Tuple of (visit variables, position variables)
        """
        # visit[i][j] means node i is visited at position j
        visit = [[Bool(f"visit_{i}_{j}") for j in range(n)] for i in range(n)]
        
        # pos[i][j] means position j contains node i
        pos = [[Int(f"pos_{i}_{j}") for j in range(n)] for i in range(n)]
        
        return visit, pos
    
    def add_path_constraints(
        self,
        visit: List[List[Bool]],
        pos: List[List[Int]],
        adj_matrix: np.ndarray
    ) -> None:
        """
        Add path constraints to the solver.
        
        Args:
            visit: Visit variables
            pos: Position variables
            adj_matrix: Adjacency matrix of the graph
        """
        n = len(visit)
        
        # Each node must be visited exactly once
        for i in range(n):
            self.solver.add(Sum([If(visit[i][j], 1, 0) for j in range(n)]) == 1)
        
        # Each position must have exactly one node
        for j in range(n):
            self.solver.add(Sum([If(visit[i][j], 1, 0) for i in range(n)]) == 1)
        
        # Ensure valid edges between consecutive positions
        for j in range(n-1):
            path_constraint = False
            for i in range(n):
                for k in range(n):
                    if adj_matrix[i][k] == 1:
                        path_constraint = Or(
                            path_constraint,
                            And(visit[i][j], visit[k][j+1])
                        )
            self.solver.add(path_constraint)
    
    def verify_path(
        self,
        adj_matrix: np.ndarray,
        path: Optional[List[int]] = None,
        timeout_ms: int = 5000
    ) -> Dict:
        """
        Verify if a Hamiltonian path exists or if a given path is valid.
        
        Args:
            adj_matrix: Adjacency matrix of the graph
            path: Optional path to verify
            timeout_ms: Timeout in milliseconds
            
        Returns:
            Dictionary with verification results
        """
        n = len(adj_matrix)
        self.solver.reset()
        
        # Create variables
        visit, pos = self.create_variables(n)
        
        # Add basic path constraints
        self.add_path_constraints(visit, pos, adj_matrix)
        
        # If path is provided, add path constraints
        if path is not None:
            for i, node in enumerate(path):
                self.solver.add(visit[node][i])
        
        # Set timeout
        self.solver.set("timeout", timeout_ms)
        
        # Add memoization cache for repeated verification requests
        cache_key = (tuple(map(tuple, adj_matrix)), tuple(path))
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Start verification
        start_time = time.time()
        result = self.solver.check()
        self.last_verification_time = time.time() - start_time
        
        verification_result = {
            "is_valid": result == sat,
            "verification_time": self.last_verification_time,
            "timeout": result == unknown
        }
        
        # If SAT and no path was provided, extract the path
        if result == sat and path is None:
            model = self.solver.model()
            found_path = []
            for j in range(n):
                for i in range(n):
                    if model.evaluate(visit[i][j]):
                        found_path.append(i)
                        break
            verification_result["path"] = found_path
        
        # Log verification
        self.logger.log_metrics("verification", {
            "result": str(result),
            "time": self.last_verification_time,
            "size": n,
            "path_provided": path is not None
        })
        
        # Store result in cache
        self._cache[cache_key] = verification_result
        
        return verification_result
    
    def verify_path_properties(
        self,
        adj_matrix: np.ndarray,
        path: List[int]
    ) -> Dict:
        """
        Verify various properties of a given path.
        
        Args:
            adj_matrix: Adjacency matrix of the graph
            path: Path to verify
            
        Returns:
            Dictionary with property verification results
        """
        n = len(adj_matrix)
        properties = {
            "valid_length": len(path) == n,
            "unique_nodes": len(set(path)) == n,
            "valid_edges": True,
            "valid_range": all(0 <= node < n for node in path)
        }
        
        # Check edges
        if properties["valid_length"]:
            for i in range(n-1):
                if adj_matrix[path[i]][path[i+1]] != 1:
                    properties["valid_edges"] = False
                    break
        
        # Overall validity
        properties["is_valid"] = all(properties.values())
        
        return properties
    
    def explain_verification_failure(
        self,
        adj_matrix: np.ndarray,
        path: List[int]
    ) -> Dict:
        """
        Provide detailed explanation of why a path is invalid.
        
        Args:
            adj_matrix: Adjacency matrix of the graph
            path: Invalid path to explain
            
        Returns:
            Dictionary with failure explanation
        """
        n = len(adj_matrix)
        properties = self.verify_path_properties(adj_matrix, path)
        
        explanation = {
            "is_valid": properties["is_valid"],
            "errors": []
        }
        
        if not properties["valid_length"]:
            explanation["errors"].append(
                f"Path length {len(path)} != graph size {n}"
            )
        
        if not properties["unique_nodes"]:
            duplicates = [
                node for node in path 
                if path.count(node) > 1
            ]
            explanation["errors"].append(
                f"Duplicate nodes found: {duplicates}"
            )
        
        if not properties["valid_range"]:
            invalid_nodes = [
                node for node in path 
                if not (0 <= node < n)
            ]
            explanation["errors"].append(
                f"Invalid node indices: {invalid_nodes}"
            )
        
        if not properties["valid_edges"]:
            invalid_edges = []
            for i in range(len(path)-1):
                if adj_matrix[path[i]][path[i+1]] != 1:
                    invalid_edges.append((path[i], path[i+1]))
            explanation["errors"].append(
                f"Invalid edges: {invalid_edges}"
            )
        
        return explanation
