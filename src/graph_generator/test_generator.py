"""
Test case generator for the Hamiltonian Path Discovery project.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from .sat_converter import SATtoHamiltonianConverter
from .graph_utils import GraphUtils

class TestGenerator:
    """
    Generates and manages test cases for the project.
    Combines SAT-based and random graph-based test cases.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.satlib_dir = self.data_dir / "satlib"
        self.test_dir = self.data_dir / "test_cases"
        self.converter = SATtoHamiltonianConverter()
        self.graph_utils = GraphUtils()
        
    def process_satlib_instances(self) -> List[Dict]:
        """
        Process all SAT instances in the SATLIB directory.
        
        Returns:
            List of processed instance information
        """
        processed = []
        
        try:
            for cnf_file in self.satlib_dir.glob("*.cnf"):
                info = self.converter.convert_file(
                    str(cnf_file),
                    self.data_dir
                )
                processed.append(info)
                
            return processed
            
        except Exception as e:
            raise
            
    def generate_random_test_suite(
        self,
        sizes: List[int] = [5, 10, 20],
        cases_per_size: int = 2
    ) -> List[Dict]:
        """
        Generate a suite of random test cases.
        
        Args:
            sizes: List of graph sizes
            cases_per_size: Number of cases per size
            
        Returns:
            List of test cases
        """
        try:
            test_cases = self.graph_utils.generate_test_suite(
                sizes,
                cases_per_size
            )
            
            # Save test cases
            timestamp = "random"
            output_file = self.test_dir / f"random_suite_{timestamp}.json"
            
            with open(output_file, "w") as f:
                json.dump({
                    "test_cases": test_cases,
                    "metadata": {
                        "sizes": sizes,
                        "cases_per_size": cases_per_size
                    }
                }, f, indent=2)
            
            return test_cases
            
        finally:
            pass
    
    def load_test_suite(self, suite_file: str) -> List[Dict]:
        """
        Load a previously generated test suite.
        
        Args:
            suite_file: Path to test suite JSON file
            
        Returns:
            List of test cases
        """
        with open(suite_file) as f:
            data = json.load(f)
            return data["test_cases"]
    
    def create_benchmark_suite(
        self,
        num_random: int = 4,
        num_satlib: int = 4,
        random_sizes: List[int] = [5, 10]
    ) -> Dict:
        """
        Create a benchmark suite combining random and SAT-based instances.
        
        Args:
            num_random: Number of random instances per size
            num_satlib: Number of SAT-based instances
            random_sizes: Sizes for random instances
            
        Returns:
            Dictionary with benchmark information
        """
        try:
            # Generate random instances
            random_cases = self.graph_utils.generate_test_suite(
                random_sizes,
                num_random
            )
            
            # Process some SAT instances if available
            sat_cases = []
            if self.satlib_dir.exists():
                try:
                    processed = self.process_satlib_instances()
                    if processed:
                        # Take a sample of processed instances
                        selected = processed[:num_satlib]
                        for info in selected:
                            matrix = np.load(info["matrix_file"])
                            sat_cases.append({
                                "size": len(matrix),
                                "input": matrix.tolist(),
                                "source": "satlib",
                                "original_file": info["original_file"]
                            })
                except Exception as e:
                    pass
            
            # Combine cases
            benchmark_suite = {
                "random_cases": random_cases,
                "sat_cases": sat_cases,
                "metadata": {
                    "num_random": len(random_cases),
                    "num_satlib": len(sat_cases),
                    "random_sizes": random_sizes
                }
            }
            
            # Save benchmark suite
            output_file = self.test_dir / f"benchmark_suite.json"
            with open(output_file, "w") as f:
                json.dump(benchmark_suite, f, indent=2)
            
            return benchmark_suite
            
        except Exception as e:
            raise
