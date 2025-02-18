"""SATLIB benchmark integration for Hamiltonian path discovery.

This module provides tools to:
1. Load and parse SATLIB format problems
2. Convert SAT instances to graph representations
3. Run benchmarks comparing our approach against known solutions
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import os
import time
from pathlib import Path
import logging
from collections import defaultdict

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solution_validator import validate_hamiltonian_path, analyze_graph_properties
from graph_analyzer import analyze_graph_properties

class SATLIBBenchmark:
    def __init__(self, benchmark_dir: str = "benchmarks", openrouter_api_key: Optional[str] = None):
        """Initialize SATLIB benchmark runner.
        
        Args:
            benchmark_dir: Directory containing SATLIB benchmark files
            openrouter_api_key: OpenRouter API key for model access
        """
        self.benchmark_dir = Path(benchmark_dir)
        self.benchmark_results = defaultdict(dict)
        self.logger = logging.getLogger(__name__)
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        
        # Initialize pattern analyzer
        from pattern_mining.pattern_analyzer import PatternAnalyzer
        self.pattern_analyzer = PatternAnalyzer()
        
        # Initialize prompt manager
        from prompting.prompt_manager import PromptManager
        self.prompt_manager = PromptManager()
        
        # Initialize theoretical analyzer
        from theoretical_analyzer import TheoreticalAnalyzer
        self.theoretical_analyzer = TheoreticalAnalyzer()
        
    def load_satlib_instance(self, filepath: str) -> Tuple[np.ndarray, Dict]:
        """Load a SATLIB instance and convert it to our graph format.
        
        The SATLIB format for Hamiltonian Path problems uses DIMACS CNF format:
        - Each vertex is represented by variables x[i,j] meaning vertex i is at position j
        - Clauses ensure:
          1. Each position has exactly one vertex
          2. Each vertex appears exactly once
          3. Adjacent vertices in path must be connected in graph
        
        Args:
            filepath: Path to the SATLIB format file
            
        Returns:
            Tuple of (adjacency_matrix, metadata)
        """
        metadata = {
            "filename": os.path.basename(filepath),
            "format": "SATLIB-HAM",
            "num_vertices": 0,
            "num_edges": 0,
            "known_solution": None
        }
        
        # Read DIMACS format
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        # Parse header
        for line in lines:
            line = line.strip()
            if line.startswith('c'):  # Comment
                if 'solution=' in line:
                    # Extract known solution if provided
                    sol_str = line.split('solution=')[1].strip()
                    try:
                        metadata['known_solution'] = [int(x) for x in sol_str.split(',')]
                    except:
                        self.logger.warning(f"Failed to parse solution: {sol_str}")
                continue
                
            if line.startswith('p'):  # Problem line
                _, _, n_vars, n_clauses = line.split()
                n = int(np.sqrt(int(n_vars)))  # Number of vertices
                metadata['num_vertices'] = n
                adj_matrix = np.zeros((n, n), dtype=int)
                break
        
        # Parse clauses to reconstruct graph
        edge_clauses = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('c') or line.startswith('p') or line.startswith('v'):
                continue
            
            clause = [int(x) for x in line.split()[:-1]]  # Skip trailing 0
            
            # Look for edge constraints
            if len(clause) == 2:
                # Convert SAT variables back to vertices
                v1 = (abs(clause[0]) - 1) // metadata['num_vertices']
                v2 = (abs(clause[1]) - 1) // metadata['num_vertices']
                if v1 != v2:
                    adj_matrix[v1][v2] = 1
                    adj_matrix[v2][v1] = 1  # Undirected graph
                    
        metadata['num_edges'] = np.sum(adj_matrix) // 2
        
        # If no solution was found in comments, try to extract from variable assignments
        if metadata['known_solution'] is None:
            try:
                # Look for a line with all positive literals (complete assignment)
                for line in lines:
                    if line.startswith('v '):  # Solution line
                        vars = [int(x) for x in line.split()[1:]]
                        if len(vars) == metadata['num_vertices']:
                            # Convert SAT variables back to vertices
                            solution = []
                            for var in vars:
                                vertex = (var - 1) // metadata['num_vertices']
                                position = (var - 1) % metadata['num_vertices']
                                solution.append((vertex, position))
                            # Sort by position to get path
                            solution.sort(key=lambda x: x[1])
                            metadata['known_solution'] = [v for v, _ in solution]
            except Exception as e:
                self.logger.warning(f"Failed to extract solution from variable assignments: {str(e)}")
        
        return adj_matrix, metadata
        
    async def _run_single_benchmark(self, instance_path: str, pattern_analyzer) -> Dict:
        """Run benchmark on a single SATLIB instance.
        
        Args:
            instance_path: Path to SATLIB instance file
            pattern_analyzer: PatternAnalyzer instance
            
        Returns:
            Dictionary containing instance results
        """
        instance_result = {}
        
        try:
            # Load instance
            adj_matrix, instance_info = self.load_satlib_instance(instance_path)
            instance_result.update(instance_info)
            
            # Analyze theoretical properties
            theoretical_insights = self.theoretical_analyzer.analyze_graph(adj_matrix)
            instance_result['theoretical_insights'] = theoretical_insights
            
            # Get pattern insights
            pattern_insights = {
                'implementations': [
                    p for p in self.pattern_analyzer.patterns.values()
                    if p.pattern_type == 'code' and not p.metadata.get('is_trivial', False)
                ],
                'failures': [
                    p for p in self.pattern_analyzer.patterns.values()
                    if p.pattern_type == 'failure'
                ],
                'subpaths': [
                    p for p in self.pattern_analyzer.patterns.values()
                    if p.pattern_type == 'subpath' and not p.metadata.get('is_sequential', False)
                ]
            }
            
            # Generate prompt using pattern insights and theoretical analysis
            prompt = self.prompt_manager.generate_pattern_based_prompt(
                adj_matrix=adj_matrix,
                pattern_insights=pattern_insights,
                theoretical_insights=self.theoretical_analyzer.format_insights_for_prompt(theoretical_insights)
            )
            
            # Print insights
            print("\nTheoretical Insights for this iteration:")
            print("=" * 50)
            print(self.theoretical_analyzer.format_insights_for_prompt(theoretical_insights))
            print("=" * 50)
            
            print("\nPattern Insights for this iteration:")
            print("=" * 50)
            print(pattern_insights)
            print("=" * 50)
            
            print("\nFull Prompt for this iteration:")
            print("=" * 50)
            print(prompt)
            print("=" * 50)
            
            # Call model to generate solution
            from continuous_improvement import ContinuousImprovement
            improver = ContinuousImprovement(openrouter_api_key=self.openrouter_api_key)
            
            start_time = time.time()
            response = await improver.call_model_async(prompt)
            
            # Extract and execute solution
            code = improver.extract_python_function(response)
            if code:
                instance_result["code"] = code
                
                # Execute the solution
                solution, execution_time = improver.execute_solution(code, adj_matrix)
                computation_time = time.time() - start_time + execution_time
                
                # Evaluate solution quality
                from solution_evaluator import SolutionEvaluator
                evaluator = SolutionEvaluator()
                quality_metrics = evaluator.evaluate_solution(
                    code=code,
                    path=solution,
                    adj_matrix=adj_matrix,
                    computation_time=computation_time
                )
                
                instance_result.update(quality_metrics)
                
                # Validate solution
                if solution is not None:
                    is_valid, failure_info = validate_hamiltonian_path(solution, adj_matrix)
                    instance_result["success"] = is_valid
                    if is_valid:
                        instance_result["solution"] = solution
                        print("\nSuccessful solution found! Analyzing quality...")
                        print(f"Complexity Score: {quality_metrics['complexity_score']:.2f}")
                        print(f"Code Features: {'recursive' if quality_metrics['has_recursion'] else 'iterative'}, "
                              f"{'with' if quality_metrics['has_pruning'] else 'without'} pruning")
                        
                        # Only store non-trivial successful solutions
                        if not quality_metrics.get('is_trivial', False):
                            self.pattern_analyzer.analyze_solution(
                                path=solution,
                                adj_matrix=adj_matrix,
                                computation_time=computation_time,
                                success=True,
                                code=code,
                                quality_metrics=quality_metrics
                            )
                        else:
                            print("Warning: Trivial solution detected, not storing pattern.")
                            
                    else:
                        print("\nSolution failed validation. Recording failure pattern...")
                        self.pattern_analyzer.analyze_solution(
                            path=solution,
                            adj_matrix=adj_matrix,
                            computation_time=computation_time,
                            success=False,
                            code=code,
                            failure_info=failure_info
                        )
                        
                        instance_result["failure_info"] = failure_info
            
            # Add graph properties and theoretical insights to result
            instance_result["graph_properties"] = theoretical_insights
            
        except Exception as e:
            self.logger.error(f"Error running benchmark on {instance_path}: {str(e)}")
            print(f"Error running benchmark on {instance_path}: {str(e)}")
            instance_result["error"] = str(e)
        
        return instance_result
        
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_numpy_types(x) for x in obj]
        return obj
    
    async def run_benchmark(self, 
                     instance_paths: List[str],
                     pattern_analyzer) -> Dict:
        """Run benchmarks on a set of SATLIB instances."""
        results = {
            "instances": [],
            "success_rate": 0.0,
            "avg_time": 0.0,
            "patterns": self.pattern_analyzer.patterns,
            "theoretical_insights": [],
            "algorithm_improvements": []
        }
        
        total_success = 0
        total_time = 0.0
        
        # Track theoretical patterns across all instances
        cumulative_insights = {
            'dirac_satisfied': 0,
            'ore_satisfied': 0,
            'claw_free': 0,
            'high_connectivity': 0,
            'density_distribution': [],
            'degree_distributions': [],
            'failure_patterns': defaultdict(int)
        }
        
        print("\nAnalyzing graph properties across instances...")
        for instance_path in instance_paths:
            instance_result = await self._run_single_benchmark(
                instance_path, 
                pattern_analyzer
            )
            
            # Update statistics
            if instance_result.get("success", False):
                total_success += 1
                total_time += instance_result.get("computation_time", 0)
            
            # Collect theoretical insights
            if 'theoretical_insights' in instance_result:
                insights = instance_result['theoretical_insights']
                
                # Track theorem satisfaction
                for insight in insights.get('theoretical_insights', []):
                    if insight['name'] == "Dirac's Theorem" and insight['condition'] == 'satisfied':
                        cumulative_insights['dirac_satisfied'] += 1
                    elif insight['name'] == "Ore's Theorem" and insight['condition'] == 'satisfied':
                        cumulative_insights['ore_satisfied'] += 1
                    elif insight['name'] == "Claw-Free Property" and insight['condition'] == 'satisfied':
                        cumulative_insights['claw_free'] += 1
                    elif insight['name'] == "High Connectivity" and insight['condition'] == 'satisfied':
                        cumulative_insights['high_connectivity'] += 1
                
                # Track graph properties
                if 'density' in insights:
                    cumulative_insights['density_distribution'].append(float(insights['density']))
                if 'min_degree' in insights and 'max_degree' in insights:
                    cumulative_insights['degree_distributions'].append({
                        'min': int(insights['min_degree']),
                        'max': int(insights['max_degree']),
                        'avg': float(insights['avg_degree'])
                    })
                
                # Track failure patterns
                for pattern in insights.get('forbidden_patterns', []):
                    pattern_type = pattern.get('type', 'unknown')
                    cumulative_insights['failure_patterns'][pattern_type] += 1
            
            # Convert numpy types to native Python types
            instance_result = self._convert_numpy_types(instance_result)
            results["instances"].append(instance_result)
        
        # Calculate overall statistics
        num_instances = len(instance_paths)
        if num_instances > 0:
            results["success_rate"] = (total_success / num_instances) * 100
            if total_success > 0:
                results["avg_time"] = total_time / total_success
        
        # Analyze patterns and suggest algorithm improvements
        results["theoretical_insights"] = self._analyze_cumulative_insights(cumulative_insights, num_instances)
        results["algorithm_improvements"] = self._suggest_algorithm_improvements(
            cumulative_insights, results["theoretical_insights"]
        )
        
        # Convert numpy types in results
        results = self._convert_numpy_types(results)
        
        return results
    
    def _analyze_cumulative_insights(self, insights: Dict, num_instances: int) -> List[Dict]:
        """Analyze cumulative theoretical insights across all instances."""
        analysis = []
        
        # Analyze theorem satisfaction rates
        dirac_rate = (insights['dirac_satisfied'] / num_instances) * 100
        ore_rate = (insights['ore_satisfied'] / num_instances) * 100
        claw_free_rate = (insights['claw_free'] / num_instances) * 100
        connectivity_rate = (insights['high_connectivity'] / num_instances) * 100
        
        analysis.append({
            'type': 'theorem_satisfaction',
            'description': 'Theorem satisfaction rates across instances',
            'details': {
                'dirac_theorem': f"{dirac_rate:.1f}%",
                'ore_theorem': f"{ore_rate:.1f}%",
                'claw_free': f"{claw_free_rate:.1f}%",
                'high_connectivity': f"{connectivity_rate:.1f}%"
            }
        })
        
        # Analyze density distribution
        if insights['density_distribution']:
            avg_density = np.mean(insights['density_distribution'])
            std_density = np.std(insights['density_distribution'])
            analysis.append({
                'type': 'density_analysis',
                'description': 'Graph density distribution',
                'details': {
                    'average': f"{avg_density:.2f}",
                    'std_dev': f"{std_density:.2f}",
                    'range': f"{min(insights['density_distribution']):.2f} - {max(insights['density_distribution']):.2f}"
                }
            })
        
        # Analyze degree distributions
        if insights['degree_distributions']:
            avg_min_degree = np.mean([d['min'] for d in insights['degree_distributions']])
            avg_max_degree = np.mean([d['max'] for d in insights['degree_distributions']])
            avg_avg_degree = np.mean([d['avg'] for d in insights['degree_distributions']])
            
            analysis.append({
                'type': 'degree_analysis',
                'description': 'Average degree characteristics',
                'details': {
                    'min_degree': f"{avg_min_degree:.1f}",
                    'max_degree': f"{avg_max_degree:.1f}",
                    'avg_degree': f"{avg_avg_degree:.1f}"
                }
            })
        
        # Analyze common failure patterns
        if insights['failure_patterns']:
            total_failures = sum(insights['failure_patterns'].values())
            failure_analysis = {
                k: f"{(v/total_failures)*100:.1f}%" 
                for k, v in insights['failure_patterns'].items()
            }
            analysis.append({
                'type': 'failure_analysis',
                'description': 'Common failure patterns',
                'details': failure_analysis
            })
        
        return analysis
    
    def _suggest_algorithm_improvements(self, insights: Dict, analysis: List[Dict]) -> List[Dict]:
        """Suggest algorithm improvements based on analysis."""
        suggestions = []
        
        # Check if we need more sophisticated degree-based approaches
        degree_analysis = next((a for a in analysis if a['type'] == 'degree_analysis'), None)
        if degree_analysis:
            details = degree_analysis['details']
            if float(details['avg_degree'].rstrip('f')) < len(insights['degree_distributions'][0]) / 2:
                suggestions.append({
                    'type': 'degree_based_improvement',
                    'description': 'Implement degree-based vertex selection',
                    'rationale': 'Low average degree suggests need for careful vertex selection',
                    'priority': 'high'
                })
        
        # Check if we need better handling of dense graphs
        density_analysis = next((a for a in analysis if a['type'] == 'density_analysis'), None)
        if density_analysis:
            avg_density = float(density_analysis['details']['average'])
            if avg_density > 0.7:
                suggestions.append({
                    'type': 'dense_graph_optimization',
                    'description': 'Optimize for dense graphs',
                    'rationale': 'High average density suggests potential for optimization',
                    'priority': 'medium'
                })
        
        # Analyze failure patterns for improvements
        failure_analysis = next((a for a in analysis if a['type'] == 'failure_analysis'), None)
        if failure_analysis:
            for pattern, rate in failure_analysis['details'].items():
                rate_value = float(rate.rstrip('%'))
                if rate_value > 20:  # If pattern accounts for >20% of failures
                    suggestions.append({
                        'type': 'failure_pattern_mitigation',
                        'description': f'Implement specific handling for {pattern}',
                        'rationale': f'Common failure pattern ({rate} of failures)',
                        'priority': 'high'
                    })
        
        # Check theorem satisfaction rates
        theorem_analysis = next((a for a in analysis if a['type'] == 'theorem_satisfaction'), None)
        if theorem_analysis:
            details = theorem_analysis['details']
            for theorem, rate in details.items():
                rate_value = float(rate.rstrip('%'))
                if rate_value > 30:  # If theorem applies to >30% of instances
                    suggestions.append({
                        'type': 'theorem_based_optimization',
                        'description': f'Add specialized handling for {theorem}',
                        'rationale': f'Applicable to {rate} of instances',
                        'priority': 'medium'
                    })
        
        return suggestions

    def compare_with_classical(self, instance_path: str) -> Dict:
        """Compare our approach with classical SAT solvers.
        
        Args:
            instance_path: Path to SATLIB instance
            
        Returns:
            Comparison metrics between our approach and classical solvers
        """
        # TODO: Implement classical solver comparison
        pass
