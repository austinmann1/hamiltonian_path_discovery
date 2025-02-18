"""
Test case generator for the Hamiltonian Path Discovery project.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from .sat_converter import SATtoHamiltonianConverter
from .graph_utils import GraphUtils
from ..logging import StructuredLogger, ExperimentLogger

class TestGenerator:
    """
    Generates and manages test cases for the project.
    Combines SAT-based and random graph-based test cases.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.satlib_dir = self.data_dir / "satlib"
        self.processed_dir = self.data_dir / "processed"
        self.test_cases_dir = self.data_dir / "test_cases"
        
        # Create directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.test_cases_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.sat_converter = SATtoHamiltonianConverter()
        self.graph_utils = GraphUtils()
        self.logger = StructuredLogger()
        self.experiment = ExperimentLogger()
    
    def process_satlib_instances(self) -> List[Dict]:
        """
        Process all SAT instances in the SATLIB directory.
        
        Returns:
            List of processed instance information
        """
        processed = []
        exp_id = None
        
        try:
            # Start experiment
            exp_id = self.experiment.start_experiment(
                description="SATLIB Processing",
                config={"source": "satlib"}
            )
            
            for cnf_file in self.satlib_dir.glob("*.cnf"):
                info = self.sat_converter.convert_file(
                    str(cnf_file),
                    self.processed_dir
                )
                processed.append(info)
                
                # Log progress
                self.logger.log_metrics("satlib_processing", {
                    "file": cnf_file.name,
                    "graph_size": info["graph_size"],
                    "success": True
                })
            
            return processed
            
        except Exception as e:
            # Log error
            self.logger.log_metrics("satlib_processing", {
                "error": str(e),
                "success": False
            })
            raise
            
        finally:
            # End experiment if it was started
            if exp_id:
                self.experiment.end_experiment()
    
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
        # Start experiment
        exp_id = self.experiment.start_experiment(
            description="Random Test Generation",
            config={"sizes": sizes, "cases_per_size": cases_per_size}
        )
        
        try:
            test_cases = self.graph_utils.generate_test_suite(
                sizes,
                cases_per_size
            )
            
            # Save test cases
            timestamp = self.experiment.current_experiment_id
            output_file = self.test_cases_dir / f"random_suite_{timestamp}.json"
            
            with open(output_file, "w") as f:
                json.dump({
                    "test_cases": test_cases,
                    "metadata": {
                        "sizes": sizes,
                        "cases_per_size": cases_per_size,
                        "experiment_id": exp_id
                    }
                }, f, indent=2)
            
            return test_cases
            
        finally:
            # End experiment
            self.experiment.end_experiment()
    
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
        exp_id = None
        
        try:
            # Start experiment
            exp_id = self.experiment.start_experiment(
                description="Benchmark Creation",
                config={
                    "num_random": num_random,
                    "num_satlib": num_satlib,
                    "random_sizes": random_sizes
                }
            )
            
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
                    # Log error but continue with random cases
                    self.logger.log_metrics("benchmark_creation", {
                        "satlib_error": str(e),
                        "continuing": True
                    })
            
            # Combine cases
            benchmark_suite = {
                "random_cases": random_cases,
                "sat_cases": sat_cases,
                "metadata": {
                    "experiment_id": exp_id,
                    "num_random": len(random_cases),
                    "num_satlib": len(sat_cases),
                    "random_sizes": random_sizes
                }
            }
            
            # Save benchmark suite
            output_file = self.test_cases_dir / f"benchmark_suite_{exp_id}.json"
            with open(output_file, "w") as f:
                json.dump(benchmark_suite, f, indent=2)
            
            return benchmark_suite
            
        except Exception as e:
            # Log error
            self.logger.log_metrics("benchmark_creation", {
                "error": str(e),
                "success": False
            })
            raise
            
        finally:
            # End experiment if it was started
            if exp_id:
                self.experiment.end_experiment()
