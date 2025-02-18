"""
Tests for the graph generator components.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import networkx as nx
from src.graph_generator.sat_converter import SATtoHamiltonianConverter
from src.graph_generator.graph_utils import GraphUtils
from src.graph_generator.test_generator import TestGenerator

class TestGraphGenerator(unittest.TestCase):
    """Test suite for graph generator components."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.test_dir) / "data"
        self.data_dir.mkdir()
        
        # Create sample CNF file
        self.cnf_dir = self.data_dir / "satlib"
        self.cnf_dir.mkdir()
        self.create_sample_cnf()
        
        # Initialize components
        self.converter = SATtoHamiltonianConverter()
        self.utils = GraphUtils()
        self.generator = TestGenerator(str(self.data_dir))
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.test_dir)
    
    def create_sample_cnf(self):
        """Create a sample CNF file for testing."""
        cnf_content = """c Sample CNF file
p cnf 3 2
1 2 0
-2 3 0
"""
        cnf_file = self.cnf_dir / "sample.cnf"
        with open(cnf_file, "w") as f:
            f.write(cnf_content)
    
    def test_sat_conversion(self):
        """Test SAT to Hamiltonian path conversion."""
        # Parse CNF file
        cnf_file = str(self.cnf_dir / "sample.cnf")
        num_vars, clauses = self.converter.parse_cnf_file(cnf_file)
        
        self.assertEqual(num_vars, 3)
        self.assertEqual(len(clauses), 2)
        
        # Convert to graph
        G = self.converter.sat_to_graph(num_vars, clauses)
        
        # Check basic properties
        self.assertGreater(len(G), num_vars)  # Should have extra nodes
        self.assertIn("start", G)
        self.assertIn("end", G)
    
    def test_random_graph_generation(self):
        """Test random graph generation."""
        size = 5
        matrix, has_solution = self.utils.generate_random_graph(
            size,
            ensure_hamiltonian=True
        )
        
        # Check properties
        self.assertEqual(matrix.shape, (size, size))
        self.assertTrue(has_solution)
        
        # Verify Hamiltonian path exists
        G = nx.DiGraph(matrix)
        self.assertTrue(self.utils.check_hamiltonian_path_exists(G))
    
    def test_degree_heuristics(self):
        """Test degree-based heuristics."""
        # Create a simple graph with Hamiltonian path
        matrix = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        
        results = self.utils.apply_degree_heuristics(matrix)
        
        self.assertIn("dirac", results)
        self.assertIn("no_isolated", results)
        self.assertIn("degree_sum", results)
    
    def test_test_suite_generation(self):
        """Test generation of test suites."""
        sizes = [3, 4]
        cases_per_size = 2
        
        test_cases = self.generator.generate_random_test_suite(
            sizes,
            cases_per_size
        )
        
        # Check test suite properties
        self.assertEqual(len(test_cases), len(sizes) * cases_per_size)
        
        for case in test_cases:
            self.assertIn("size", case)
            self.assertIn("input", case)
            self.assertIn("has_solution", case)
            self.assertIn("heuristics", case)
    
    def test_benchmark_creation(self):
        """Test benchmark suite creation."""
        benchmark = self.generator.create_benchmark_suite(
            num_random=2,
            num_satlib=1,
            random_sizes=[3]
        )
        
        # Check benchmark properties
        self.assertIn("random_cases", benchmark)
        self.assertIn("sat_cases", benchmark)
        self.assertIn("metadata", benchmark)
        
        # Check random cases
        self.assertEqual(len(benchmark["random_cases"]), 2)
        
        # Check metadata
        self.assertIn("experiment_id", benchmark["metadata"])
        self.assertIn("random_sizes", benchmark["metadata"])

if __name__ == '__main__':
    unittest.main()
