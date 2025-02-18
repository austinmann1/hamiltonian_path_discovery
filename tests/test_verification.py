"""
Tests for the verification system.
"""

import unittest
import numpy as np
from src.verification.z3_verifier import Z3HamiltonianVerifier
from src.verification.verification_oracle import VerificationOracle

class TestVerification(unittest.TestCase):
    """Test suite for verification system."""
    
    def setUp(self):
        """Set up test environment."""
        self.z3_verifier = Z3HamiltonianVerifier()
        self.oracle = VerificationOracle()
    
    def test_simple_path_verification(self):
        """Test verification of a simple valid path."""
        # Create a simple path: 0 -> 1 -> 2
        adj_matrix = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        path = [0, 1, 2]
        
        # Verify with Z3
        z3_result = self.z3_verifier.verify_path(adj_matrix, path)
        self.assertTrue(z3_result["is_valid"])
        
        # Verify with oracle
        self.oracle.reset_stats()  # Reset stats before test
        oracle_result = self.oracle.verify_with_explanation(adj_matrix, path)
        self.assertTrue(oracle_result["is_valid"])
        self.assertEqual(oracle_result["method"], "simple_check")  # Should use simple check
        
        # Check stats
        stats = self.oracle.get_verification_stats()
        self.assertEqual(stats["total_verifications"], 1)
        self.assertEqual(stats["successful_verifications"], 1)
    
    def test_invalid_path(self):
        """Test verification of an invalid path."""
        # Create a graph with no Hamiltonian path
        adj_matrix = np.array([
            [0, 1, 0],
            [0, 0, 0],
            [0, 1, 0]
        ])
        path = [0, 1, 2]
        
        # Verify with Z3
        z3_result = self.z3_verifier.verify_path(adj_matrix, path)
        self.assertFalse(z3_result["is_valid"])
        
        # Verify with oracle
        oracle_result = self.oracle.verify_with_explanation(adj_matrix, path)
        self.assertFalse(oracle_result["is_valid"])
        self.assertIn("explanation", oracle_result)
    
    def test_path_finding(self):
        """Test finding a Hamiltonian path."""
        # Create a graph with a known Hamiltonian path
        adj_matrix = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])
        
        # Try to find path with Z3
        z3_result = self.z3_verifier.verify_path(adj_matrix)
        self.assertTrue(z3_result["is_valid"])
        self.assertIn("path", z3_result)
        
        # Verify found path
        found_path = z3_result["path"]
        self.assertEqual(len(found_path), 3)
        for i in range(2):
            self.assertEqual(adj_matrix[found_path[i]][found_path[i+1]], 1)
    
    def test_degree_conditions(self):
        """Test degree-based heuristics."""
        # Create a graph that fails degree conditions
        adj_matrix = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        
        # Check with oracle
        result = self.oracle.verify_with_explanation(adj_matrix)
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["method"], "heuristics")
    
    def test_verification_stats(self):
        """Test verification statistics tracking."""
        # Create a simple graph
        adj_matrix = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        
        # Reset stats
        self.oracle.reset_stats()
        
        # Do some verifications
        result1 = self.oracle.verify_with_explanation(adj_matrix, [0, 1, 2])  # Valid path
        result2 = self.oracle.verify_with_explanation(adj_matrix, [1, 0, 2])  # Invalid path
        
        # Check results
        self.assertTrue(result1["is_valid"])
        self.assertFalse(result2["is_valid"])
        
        # Check stats
        stats = self.oracle.get_verification_stats()
        self.assertEqual(stats["total_verifications"], 2)
        self.assertEqual(stats["successful_verifications"], 1)
    
    def test_explanation_detail(self):
        """Test detailed failure explanations."""
        # Create a graph
        adj_matrix = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        
        # Test invalid path
        invalid_path = [0, 2, 1]  # No edge 0->2
        result = self.oracle.verify_with_explanation(adj_matrix, invalid_path)
        
        self.assertFalse(result["is_valid"])
        self.assertIn("explanation", result)
        self.assertIn("errors", result["explanation"])

if __name__ == '__main__':
    unittest.main()
