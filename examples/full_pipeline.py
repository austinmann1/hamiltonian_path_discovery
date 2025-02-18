"""
Example demonstrating the full Hamiltonian Path Discovery pipeline.
"""
import os
import sys
from pathlib import Path
import json

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.dataset_preparation import DatasetPreparation
from src.graph_generator import GraphGenerator
from src.verification import VerificationOracle
from src.llm_interface import LLMInterface
from src.energy_monitor import EnergyMonitor

def main():
    # Initialize components
    prep = DatasetPreparation()
    generator = GraphGenerator()
    oracle = VerificationOracle()
    llm = LLMInterface()
    monitor = EnergyMonitor(log_file="energy_logs.json")
    
    print("Starting Hamiltonian Path Discovery Pipeline...")
    
    # 1. Prepare dataset
    print("\n1. Preparing dataset...")
    with monitor.track("dataset_preparation"):
        benchmark_data = prep.prepare_benchmark_suite(
            categories=["uf20-91"],
            num_instances=2  # Using 2 instances for demonstration
        )
    print(f"Created benchmark suite with {len(benchmark_data['instances'])} instances")
    
    # 2. Process each instance
    for instance in benchmark_data["instances"]:
        print(f"\nProcessing instance: {instance['filename']}")
        
        # Read CNF formula
        cnf = []
        with open(instance["path"], 'r') as f:
            for line in f:
                if line.startswith('c') or line.startswith('p'):
                    continue
                clause = [int(x) for x in line.strip().split()[:-1]]
                if clause:
                    cnf.append(clause)
        
        # Convert to Hamiltonian path problem
        with monitor.track("graph_generation"):
            graph = generator.sat_to_hamiltonian(cnf)
            adj_matrix = generator.get_adjacency_matrix()
        print(f"Created graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        
        # Generate solution code
        print("\nGenerating solution code...")
        problem_description = f"""
        Generate a Python function that finds a Hamiltonian path in this graph:
        - Input: Adjacency matrix of size {len(adj_matrix)}x{len(adj_matrix)}
        - Output: List of node indices representing the path, or None if no path exists
        - Use dynamic programming with bitmask optimization
        - Implement pruning techniques for efficiency
        """
        
        with monitor.track("code_generation"):
            try:
                code = llm.generate_code(
                    problem_description=problem_description,
                    test_cases=[{
                        "input": adj_matrix.tolist(),
                        "expected": None  # We don't know the answer yet
                    }]
                )
                print("Successfully generated code")
                
                # Save generated code
                code_file = Path("generated_solution.py")
                with open(code_file, 'w') as f:
                    f.write(code)
                print(f"Saved generated code to {code_file}")
                
            except Exception as e:
                print(f"Error generating code: {e}")
                continue
        
        # Verify solution
        print("\nVerifying solution...")
        with monitor.track("solution_verification"):
            has_path = oracle.verify_hamiltonian_path(adj_matrix)
            if has_path:
                path = oracle.extract_path(adj_matrix)
                print(f"Found valid Hamiltonian path: {path}")
            else:
                print("No Hamiltonian path exists in this graph")
    
    # Print resource usage summary
    print("\nResource Usage Summary:")
    system_info = monitor.get_system_info()
    print(json.dumps(system_info, indent=2))

if __name__ == "__main__":
    main()
