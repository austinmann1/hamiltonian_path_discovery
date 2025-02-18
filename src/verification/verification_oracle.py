"""Verification oracle for Hamiltonian path solutions."""

import numpy as np
from typing import List, Dict, Optional
from ..solution_validator import validate_hamiltonian_path

class VerificationOracle:
    """High-level verification system that combines multiple verification strategies."""
    
    def __init__(self):
        self.total_verifications = 0
        self.successful_verifications = 0
        self.total_time = 0.0
        
    def verify_solution(self, path: List[int], adj_matrix: np.ndarray) -> bool:
        """Verify if a solution is valid.
        
        Args:
            path: Proposed Hamiltonian path
            adj_matrix: Adjacency matrix of the graph
            
        Returns:
            True if solution is valid, False otherwise
        """
        self.total_verifications += 1
        is_valid = validate_hamiltonian_path(path, adj_matrix)
        if is_valid:
            self.successful_verifications += 1
        return is_valid
        
    def get_statistics(self) -> Dict:
        """Get verification statistics.
        
        Returns:
            Dictionary of verification statistics
        """
        return {
            "total_verifications": self.total_verifications,
            "successful_verifications": self.successful_verifications,
            "success_rate": (self.successful_verifications / self.total_verifications 
                           if self.total_verifications > 0 else 0.0),
            "average_time": self.total_time / self.total_verifications 
                          if self.total_verifications > 0 else 0.0
        }
