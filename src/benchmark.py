"""
Benchmark suite for evaluating Hamiltonian Path solutions.
"""
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Callable
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging

from .graph_generator import GraphGenerator
from .verification import VerificationOracle
from .energy_monitor import EnergyMonitor
from .llm_interface_openrouter import OpenRouterLLM

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BenchmarkSuite:
    def __init__(self, metrics_dir: str = "metrics"):
        """
        Initialize benchmark suite.
        
        Args:
            metrics_dir: Directory to store benchmark results
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initializing components...")
        self.generator = GraphGenerator()
        self.oracle = VerificationOracle()
        self.monitor = EnergyMonitor()
        
        # Store benchmark results
        self.results: List[Dict] = []
        
    def generate_test_cases(self, 
                           sizes: List[int] = [5, 10],  
                           instances_per_size: int = 2) -> List[Dict]:  
        """
        Generate test cases of varying sizes.
        
        Args:
            sizes: List of graph sizes to generate
            instances_per_size: Number of instances per size
            
        Returns:
            List of test cases with inputs and expected results
        """
        test_cases = []
        
        for size in sizes:
            logger.info(f"Generating test cases for size {size}")
            for instance in range(instances_per_size):
                logger.info(f"  Generating instance {instance + 1}/{instances_per_size}")
                
                # Generate a simple SAT instance that's guaranteed to be satisfiable
                num_vars = size
                num_clauses = size  
                cnf = []
                for i in range(num_clauses):
                    clause = [i + 1]  
                    cnf.append(clause)
                
                logger.info("  Converting to Hamiltonian path problem...")
                # Convert to Hamiltonian path problem
                graph = self.generator.sat_to_hamiltonian(cnf)
                adj_matrix = self.generator.get_adjacency_matrix()
                
                logger.info("  Verifying solution existence...")
                # Verify if solution exists
                has_path = self.oracle.verify_hamiltonian_path(adj_matrix)
                path = self.oracle.extract_path(adj_matrix) if has_path else None
                
                test_case = {
                    "size": size,
                    "input": adj_matrix.tolist(),
                    "expected": path,
                    "has_solution": has_path
                }
                test_cases.append(test_case)
                logger.info(f"  Instance {instance + 1} complete: has_solution={has_path}")
        
        logger.info(f"Generated {len(test_cases)} test cases")
        return test_cases
    
    def benchmark_solution(self,
                         solution_func: Callable,
                         test_cases: List[Dict],
                         solution_name: str) -> Dict:
        """
        Benchmark a solution function against test cases.
        
        Args:
            solution_func: Function that takes adjacency matrix and returns path
            test_cases: List of test cases
            solution_name: Name of the solution being benchmarked
            
        Returns:
            Dict containing benchmark metrics
        """
        metrics = {
            "name": solution_name,
            "timestamp": datetime.now().isoformat(),
            "test_cases": len(test_cases),
            "correct": 0,
            "execution_times": [],
            "memory_usage": [],
            "results": []
        }
        
        for i, test_case in enumerate(test_cases):
            print(f"\rBenchmarking {solution_name}: {i+1}/{len(test_cases)}", end="")
            
            with self.monitor.track(f"{solution_name}_case_{i}"):
                start_time = time.time()
                try:
                    result = solution_func(test_case["input"])
                    execution_time = time.time() - start_time
                    
                    # Verify result
                    is_correct = False
                    if test_case["has_solution"]:
                        if result is not None:
                            is_correct = self.oracle.verify_hamiltonian_path(
                                test_case["input"],
                                result
                            )
                    else:
                        is_correct = result is None
                    
                    metrics["correct"] += int(is_correct)
                    metrics["execution_times"].append(execution_time)
                    
                    case_result = {
                        "test_case_id": i,
                        "size": test_case["size"],
                        "correct": is_correct,
                        "execution_time": execution_time,
                        "has_solution": test_case["has_solution"],
                        "found_solution": result is not None
                    }
                    metrics["results"].append(case_result)
                    
                except Exception as e:
                    print(f"\nError on test case {i}: {str(e)}")
                    case_result = {
                        "test_case_id": i,
                        "size": test_case["size"],
                        "correct": False,
                        "error": str(e)
                    }
                    metrics["results"].append(case_result)
        
        print()  # New line after progress
        
        # Calculate summary statistics
        metrics["accuracy"] = metrics["correct"] / len(test_cases)
        metrics["avg_execution_time"] = sum(metrics["execution_times"]) / len(metrics["execution_times"])
        
        # Save metrics
        self.results.append(metrics)
        self._save_metrics(metrics, solution_name)
        
        return metrics
    
    def benchmark_llm_solutions(self,
                              test_cases: List[Dict],
                              models: List[str] = ["deepseek-ai/deepseek-coder-33b-instruct"]) -> List[Dict]:
        """
        Benchmark solutions generated by different LLM models.
        
        Args:
            test_cases: List of test cases
            models: List of model identifiers to test
            
        Returns:
            List of benchmark results for each model
        """
        results = []
        
        for model in models:
            print(f"\nBenchmarking model: {model}")
            llm = OpenRouterLLM(model=model)
            
            # Generate solution for a simple test case first
            simple_case = test_cases[0]
            problem_description = f"""
            Write a Python function find_hamiltonian_path(matrix) that:
            1. Takes an adjacency matrix as input
            2. Returns a list of node indices if a Hamiltonian path exists, None otherwise
            3. Uses efficient algorithms
            4. Handles edge cases

            Example input matrix (size {simple_case['size']}x{simple_case['size']}):
            {json.dumps(simple_case['input'], indent=2)}

            Has solution: {simple_case['has_solution']}
            """
            
            try:
                code = llm.generate_code(problem_description, [simple_case])
                
                # Save generated code
                code_file = self.metrics_dir / f"{model.replace('/', '_')}_solution.py"
                with open(code_file, 'w') as f:
                    f.write(code)
                
                # Create solution function
                namespace = {}
                exec(code, namespace)
                solution_func = namespace.get('find_hamiltonian_path')
                
                if solution_func:
                    # Benchmark the solution
                    metrics = self.benchmark_solution(
                        solution_func,
                        test_cases,
                        f"llm_{model.replace('/', '_')}"
                    )
                    
                    # Add LLM-specific metrics
                    metrics["model"] = model
                    metrics["code_size"] = len(code)
                    metrics["llm_metrics"] = llm.get_performance_metrics()
                    
                    results.append(metrics)
                    
            except Exception as e:
                print(f"Error benchmarking {model}: {str(e)}")
        
        return results
    
    def plot_results(self, save_dir: Optional[str] = None) -> None:
        """
        Generate plots comparing different solutions.
        
        Args:
            save_dir: Optional directory to save plots
        """
        if not self.results:
            print("No benchmark results to plot")
            return
            
        # Convert results to DataFrame for easier analysis
        df_list = []
        for metrics in self.results:
            for result in metrics["results"]:
                df_list.append({
                    "solution": metrics["name"],
                    "size": result["size"],
                    "correct": result["correct"],
                    "execution_time": result.get("execution_time", 0),
                    "has_solution": result.get("has_solution", False),
                    "found_solution": result.get("found_solution", False)
                })
        
        df = pd.DataFrame(df_list)
        
        # 1. Accuracy by graph size
        plt.figure(figsize=(10, 6))
        df_accuracy = df.groupby(['solution', 'size'])['correct'].mean().unstack()
        df_accuracy.plot(kind='bar', ylabel='Accuracy')
        plt.title('Solution Accuracy by Graph Size')
        plt.tight_layout()
        if save_dir:
            plt.savefig(Path(save_dir) / 'accuracy_by_size.png')
        
        # 2. Execution time by graph size
        plt.figure(figsize=(10, 6))
        df_time = df.groupby(['solution', 'size'])['execution_time'].mean().unstack()
        df_time.plot(kind='bar', ylabel='Average Execution Time (s)')
        plt.title('Execution Time by Graph Size')
        plt.tight_layout()
        if save_dir:
            plt.savefig(Path(save_dir) / 'time_by_size.png')
        
        # 3. Overall metrics
        plt.figure(figsize=(10, 6))
        metrics_summary = pd.DataFrame([{
            'solution': m['name'],
            'accuracy': m['accuracy'],
            'avg_time': m['avg_execution_time']
        } for m in self.results])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        metrics_summary.plot(x='solution', y='accuracy', kind='bar', ax=ax1)
        ax1.set_title('Overall Accuracy')
        metrics_summary.plot(x='solution', y='avg_time', kind='bar', ax=ax2)
        ax2.set_title('Average Execution Time')
        plt.tight_layout()
        if save_dir:
            plt.savefig(Path(save_dir) / 'overall_metrics.png')
    
    def _save_metrics(self, metrics: Dict, solution_name: str) -> None:
        """Save benchmark metrics to file."""
        metrics_file = self.metrics_dir / f"{solution_name}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

# Example usage
if __name__ == "__main__":
    suite = BenchmarkSuite()
    
    # Generate test cases
    test_cases = suite.generate_test_cases(
        sizes=[5, 10],
        instances_per_size=2
    )
    
    # Benchmark LLM solutions
    results = suite.benchmark_llm_solutions(
        test_cases,
        models=["deepseek-ai/deepseek-coder-33b-instruct"]
    )
    
    # Plot results
    suite.plot_results(save_dir="metrics")
