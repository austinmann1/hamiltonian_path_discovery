"""
Verification oracle for Hamiltonian path problems.
Combines Z3-based verification with graph theory heuristics.
"""

import time
from typing import List, Dict, Optional, Tuple
import numpy as np
import networkx as nx
from .z3_verifier import Z3HamiltonianVerifier
from ..logging import StructuredLogger, ExperimentLogger

class VerificationOracle:
    """
    High-level verification system that combines multiple verification strategies.
    """
    
    def __init__(self):
        self.z3_verifier = Z3HamiltonianVerifier()
        self.logger = StructuredLogger()
        self.experiment = ExperimentLogger()
        
        # Verification statistics
        self.total_verifications = 0
        self.successful_verifications = 0
        self.total_time = 0.0
        self.heuristic_rejections = 0
    
    def check_degree_conditions(self, adj_matrix: np.ndarray) -> Dict:
        """
        Check degree-based necessary conditions for Hamiltonian paths.
        
        Args:
            adj_matrix: Adjacency matrix of the graph
            
        Returns:
            Dictionary with heuristic results
        """
        G = nx.DiGraph(adj_matrix)
        n = len(adj_matrix)
        
        # Calculate in and out degrees
        in_degrees = [d for _, d in G.in_degree()]
        out_degrees = [d for _, d in G.out_degree()]
        
        conditions = {
            # No isolated vertices
            "no_isolated": all(d > 0 for d in in_degrees[1:]) and 
                         all(d > 0 for d in out_degrees[:-1]),
            
            # Degree sum condition for internal nodes
            "degree_sum": all(in_deg + out_deg >= 1 
                            for in_deg, out_deg in zip(in_degrees[1:-1], out_degrees[1:-1])),
            
            # Start node should have out-degree > 0
            "valid_start": out_degrees[0] > 0,
            
            # End node should have in-degree > 0
            "valid_end": in_degrees[-1] > 0,
            
            # Path connectivity
            "has_path": nx.has_path(G, 0, len(adj_matrix)-1)
        }
        
        # Overall assessment
        conditions["passes_heuristics"] = all(conditions.values())
        
        return conditions
    
    def check_path_simple(self, adj_matrix: np.ndarray, path: List[int]) -> bool:
        """
        Quick check if a path could be valid without using Z3.
        
        Args:
            adj_matrix: Adjacency matrix of the graph
            path: Path to verify
            
        Returns:
            True if path passes basic checks
        """
        n = len(adj_matrix)
        
        # Check length and uniqueness
        if len(path) != n or len(set(path)) != n:
            return False
        
        # Check node range
        if not all(0 <= node < n for node in path):
            return False
        
        # Check edges
        for i in range(n-1):
            if adj_matrix[path[i]][path[i+1]] != 1:
                return False
        
        return True
    
    def verify_with_explanation(
        self,
        adj_matrix: np.ndarray,
        path: Optional[List[int]] = None,
        timeout_ms: int = 5000
    ) -> Dict:
        """
        Verify a path or find one, with detailed explanation.
        
        Args:
            adj_matrix: Adjacency matrix of the graph
            path: Optional path to verify
            timeout_ms: Z3 timeout in milliseconds
            
        Returns:
            Dictionary with verification results and explanation
        """
        start_time = time.time()
        
        # Start verification experiment
        exp_id = self.experiment.start_experiment(
            description="Path Verification",
            config={
                "size": len(adj_matrix),
                "timeout_ms": timeout_ms,
                "path_provided": path is not None
            }
        )
        
        try:
            # Check for empty path
            if path is not None and not path:
                return {
                    "is_valid": False,
                    "method": "simple_check",
                    "time": time.time() - start_time,
                    "explanation": {
                        "is_valid": False,
                        "errors": ["Empty path provided"]
                    }
                }
            
            # Check heuristics first
            heuristics = self.check_degree_conditions(adj_matrix)
            if not heuristics["passes_heuristics"]:
                self.heuristic_rejections += 1
                return {
                    "is_valid": False,
                    "method": "heuristics",
                    "time": time.time() - start_time,
                    "heuristic_results": heuristics,
                    "explanation": {
                        "is_valid": False,
                        "errors": ["Graph fails necessary conditions for Hamiltonian path"]
                    }
                }
            
            # If path provided, do quick check first
            if path is not None:
                self.total_verifications += 1
                if self.check_path_simple(adj_matrix, path):
                    self.successful_verifications += 1
                    return {
                        "is_valid": True,
                        "method": "simple_check",
                        "time": time.time() - start_time,
                        "explanation": {
                            "is_valid": True,
                            "errors": []
                        }
                    }
                else:
                    return {
                        "is_valid": False,
                        "method": "simple_check",
                        "time": time.time() - start_time,
                        "explanation": self.z3_verifier.explain_verification_failure(
                            adj_matrix, path
                        )
                    }
            
            # Initialize result with default values
            result = {
                "is_valid": False,
                "error": "Verification not completed"
            }
            try:
                verification = self.verify_with_explanation(adj_matrix, path, timeout_ms)
                result = {
                    "is_valid": verification.get("is_valid", False),
                    "error": verification.get("error")
                }
            except Exception as e:
                result["error"] = f"Verification failed: {str(e)}"
            
            return result
            
        finally:
            # End experiment
            self.experiment.end_experiment(exp_id, result, {"verification_time": time.time() - start_time})
    
    def get_verification_stats(self) -> Dict:
        """
        Get current verification statistics.
        
        Returns:
            Dictionary with verification statistics
        """
        stats = {
            "total_verifications": self.total_verifications,
            "successful_verifications": self.successful_verifications,
            "success_rate": (self.successful_verifications / self.total_verifications 
                           if self.total_verifications > 0 else 0),
            "average_time": (self.total_time / self.total_verifications 
                           if self.total_verifications > 0 else 0),
            "heuristic_rejections": self.heuristic_rejections
        }
        
        # Log stats
        self.logger.log_metrics("verification_stats", stats)
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset verification statistics."""
        self.total_verifications = 0
        self.successful_verifications = 0
        self.total_time = 0.0
        self.heuristic_rejections = 0
