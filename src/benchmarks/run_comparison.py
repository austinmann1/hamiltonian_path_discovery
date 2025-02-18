"""Run comparison tests between our approach and classical solvers."""

import os
from pathlib import Path
from typing import List, Dict
import json
import time
import sys
import asyncio
from datetime import datetime
from collections import defaultdict
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.benchmark_generator import BenchmarkGenerator
from benchmarks.satlib_benchmark import SATLIBBenchmark
from benchmarks.classical_solvers import compare_with_classical
from prompting.prompt_manager import PromptManager
from pattern_mining.pattern_analyzer import PatternAnalyzer
from theoretical_analyzer import TheoreticalAnalyzer
from continuous_improvement import ContinuousImprovement

def generate_benchmark_instances(num_instances: int = 5) -> List[str]:
    """Generate benchmark instances with varying complexity."""
    instances = []
    benchmark_dir = Path("benchmarks")
    benchmark_dir.mkdir(exist_ok=True)
    
    # Generate a mix of graph types
    for i in range(num_instances):
        # Use larger graphs: 15-30 vertices
        size = np.random.randint(15, 30)
        
        if i < num_instances // 3:
            # Easy instances: Higher density (0.6-0.8)
            density = np.random.uniform(0.6, 0.8)
        elif i < 2 * (num_instances // 3):
            # Medium instances: Medium density (0.3-0.5)
            density = np.random.uniform(0.3, 0.5)
        else:
            # Hard instances: Lower density (0.15-0.25)
            density = np.random.uniform(0.15, 0.25)
        
        print(f"\nGenerating instance {i+1}/{num_instances}")
        print(f"Size: {size} vertices")
        print(f"Density: {density:.2f}")
        
        # Generate adjacency matrix
        adj_matrix = np.zeros((size, size))
        for u in range(size):
            for v in range(u + 1, size):
                if np.random.random() < density:
                    adj_matrix[u][v] = adj_matrix[v][u] = 1
        
        # Ensure graph is connected
        while not is_connected(adj_matrix):
            # Add random edges until connected
            u = np.random.randint(0, size)
            v = np.random.randint(0, size)
            if u != v and adj_matrix[u][v] == 0:
                adj_matrix[u][v] = adj_matrix[v][u] = 1
        
        # Save instance
        instance_path = benchmark_dir / f"ham_path_{size}v_{i}.npy"
        np.save(instance_path, adj_matrix)
        instances.append(str(instance_path))
        
        # Print graph properties
        num_edges = np.sum(adj_matrix) // 2
        actual_density = (2 * num_edges) / (size * (size - 1))
        print(f"Actual density: {actual_density:.2f}")
        print(f"Number of edges: {num_edges}")
    
    return instances

def is_connected(adj_matrix: np.ndarray) -> bool:
    """Check if graph is connected using DFS."""
    n = adj_matrix.shape[0]
    visited = [False] * n
    
    def dfs(v):
        visited[v] = True
        for u in range(n):
            if adj_matrix[v][u] and not visited[u]:
                dfs(u)
    
    dfs(0)
    return all(visited)

def validate_solution(adj_matrix: np.ndarray, path: List[int]) -> bool:
    """Validate if a path is a valid Hamiltonian path."""
    if path is None:
        return False
        
    n = adj_matrix.shape[0]
    
    # Check length
    if len(path) != n:
        print(f"Invalid path length: {len(path)} != {n}")
        return False
    
    # Check for duplicates
    if len(set(path)) != n:
        print("Path contains duplicate vertices")
        return False
    
    # Check vertex range
    if any(v < 0 or v >= n for v in path):
        print("Path contains invalid vertices")
        return False
    
    # Check edge connectivity
    for i in range(n-1):
        if not adj_matrix[path[i]][path[i+1]]:
            print(f"No edge between vertices {path[i]} and {path[i+1]}")
            return False
    
    return True

async def run_comparison_test(benchmark_files: List[str]) -> Dict:
    """Run comparison tests on benchmark files."""
    results = defaultdict(list)
    pattern_analyzer = PatternAnalyzer()
    theoretical_analyzer = TheoreticalAnalyzer()
    prompt_manager = PromptManager()
    
    for benchmark_file in benchmark_files:
        print(f"\nTesting {benchmark_file}")
        adj_matrix = np.load(benchmark_file)
        n = adj_matrix.shape[0]
        
        # Analyze graph properties
        density = np.sum(adj_matrix) / (n * (n-1))
        min_degree = min(np.sum(adj_matrix, axis=0))
        max_degree = max(np.sum(adj_matrix, axis=0))
        avg_degree = np.mean(np.sum(adj_matrix, axis=0))
        
        print(f"\nGraph Properties:")
        print(f"- Size: {n} vertices")
        print(f"- Density: {density:.2f}")
        print(f"- Degree range: {min_degree} to {max_degree}")
        print(f"- Average degree: {avg_degree:.2f}")
        
        # Get theoretical insights
        theoretical_insights = theoretical_analyzer.analyze_graph(adj_matrix)
        print("\nTheoretical Analysis:")
        print("=" * 50)
        print(theoretical_insights)
        print("=" * 50)
        
        # Get implementation patterns
        pattern_insights = pattern_analyzer.get_pattern_insights()
        print("\nPattern Insights:")
        print("=" * 50)
        print(pattern_insights)
        print("=" * 50)
        
        # Generate prompt
        prompt = prompt_manager.generate_prompt(
            adj_matrix=adj_matrix,
            theoretical_insights=theoretical_insights,
            pattern_insights=pattern_insights
        )
        
        print("\nFull Prompt:")
        print("=" * 50)
        print(prompt)
        print("=" * 50)
        
        # Call model and get solution
        continuous_improvement = ContinuousImprovement()
        solution_code = await continuous_improvement.call_model_async(prompt)
        
        if solution_code:
            # Execute solution with timeout
            try:
                path, execution_time = continuous_improvement.execute_solution(solution_code, adj_matrix)
                
                # Validate solution
                is_valid = validate_solution(adj_matrix, path)
                if is_valid:
                    print("\nSuccessful solution found! Analyzing quality...")
                    
                    # Analyze solution quality
                    complexity_score = pattern_analyzer._calculate_performance_score(execution_time, n)
                    print(f"Performance Score: {complexity_score:.2f}")
                    
                    # Store successful pattern if non-trivial
                    if complexity_score > 0.1:  # Only store non-trivial solutions
                        pattern_analyzer.store_successful_pattern(solution_code, path, execution_time)
                        theoretical_insights['success'] = True  # Track success in results
                    else:
                        print("Warning: Trivial solution detected, not storing pattern.")
                        theoretical_insights['success'] = False
                else:
                    print("\nInvalid solution!")
                    pattern_analyzer.store_failure_pattern(solution_code, "Invalid solution")
                    theoretical_insights['success'] = False  # Track failure in results
            except Exception as e:
                print(f"\nError executing solution: {str(e)}")
                pattern_analyzer.store_failure_pattern(solution_code, str(e))
                theoretical_insights['success'] = False  # Track failure in results
        else:
            print("\nNo solution generated")
            pattern_analyzer.store_failure_pattern(None, "No solution generated")
            theoretical_insights['success'] = False  # Track failure in results
        
        # Save results
        results["theoretical_insights"].append(theoretical_insights)
        results["pattern_insights"].append(pattern_insights)
        results["graph_properties"].append({
            "size": n,
            "density": density,
            "min_degree": min_degree,
            "max_degree": max_degree,
            "avg_degree": avg_degree
        })
    
    return results

def print_results_summary(results: Dict):
    """Print a summary of the benchmark results."""
    print("\nResults Summary")
    print("=" * 50)
    
    # Overall statistics
    total_attempts = len(results['theoretical_insights'])  # Each insight is one attempt
    successful_attempts = sum(1 for insights in results['theoretical_insights'] if insights.get('success', False))
    
    print(f"\nOverall Statistics:")
    print(f"Total test cases: {len(results['theoretical_insights'])}")
    print(f"Total attempts: {total_attempts}")
    print(f"Successful attempts: {successful_attempts}")
    if total_attempts > 0:
        print(f"Success rate: {successful_attempts/total_attempts*100:.1f}%")
    else:
        print("Success rate: N/A (no attempts)")
    
    # Graph properties summary
    sizes = [props['size'] for props in results['graph_properties']]
    densities = [props['density'] for props in results['graph_properties']]
    
    print(f"\nGraph Properties:")
    print(f"Size range: {min(sizes)} to {max(sizes)} vertices")
    print(f"Density range: {min(densities):.2f} to {max(densities):.2f}")
    
    # Pattern insights
    pattern_counts = defaultdict(int)
    for insights in results['pattern_insights']:
        for pattern_type in insights:
            pattern_counts[pattern_type] += len(insights[pattern_type])
    
    print(f"\nPattern Analysis:")
    for pattern_type, count in pattern_counts.items():
        print(f"{pattern_type}: {count} patterns")

async def main():
    print("Generating benchmark instances...")
    benchmark_files = generate_benchmark_instances()
    print(f"Generated {len(benchmark_files)} benchmark instances")
    
    print("\nRunning comparison tests...")
    results = await run_comparison_test(benchmark_files)
    
    print("\nGenerating results summary...")
    print_results_summary(results)

if __name__ == "__main__":
    asyncio.run(main())
