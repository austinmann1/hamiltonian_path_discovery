"""Solution quality evaluation for Hamiltonian path algorithms."""

from typing import Dict, List, Optional
import numpy as np
import ast
from collections import defaultdict

class SolutionEvaluator:
    """Evaluates the quality and complexity of Hamiltonian path solutions."""
    
    def __init__(self):
        self.complexity_weights = {
            'branching': 2.0,  # Weight for branching statements
            'backtracking': 2.0,  # Weight for recursive backtracking
            'pruning': 1.5,  # Weight for early pruning
            'theoretical': 1.5,  # Weight for theoretical checks
            'trivial': -3.0  # Penalty for trivial solutions
        }
    
    def evaluate_solution(self, code: str, path: Optional[List[int]], 
                         adj_matrix: np.ndarray, computation_time: float) -> Dict:
        """
        Evaluate a solution's quality and complexity.
        
        Args:
            code: The Python code implementation
            path: The solution path (if found)
            adj_matrix: The graph's adjacency matrix
            computation_time: Time taken to find solution
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Parse the code
        try:
            tree = ast.parse(code)
        except:
            return self._create_error_evaluation("Invalid Python code")
        
        # Basic metrics
        metrics = {
            'code_length': len(code),
            'computation_time': computation_time,
            'has_solution': path is not None
        }
        
        # Analyze code structure
        visitor = CodeAnalysisVisitor()
        visitor.visit(tree)
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(visitor, path, adj_matrix)
        metrics['complexity_score'] = complexity_score
        
        # Analyze solution characteristics
        if path is not None:
            path_metrics = self._analyze_path(path, adj_matrix)
            metrics.update(path_metrics)
        
        # Add code analysis metrics
        metrics.update({
            'has_recursion': visitor.has_recursion,
            'has_backtracking': visitor.has_backtracking,
            'branching_factor': visitor.branching_factor,
            'has_pruning': visitor.has_pruning,
            'theoretical_checks': visitor.theoretical_checks
        })
        
        return metrics
    
    def _calculate_complexity_score(self, visitor, path: Optional[List[int]], 
                                  adj_matrix: np.ndarray) -> float:
        """Calculate a weighted complexity score."""
        score = 0.0
        
        # Add points for sophisticated code features
        if visitor.has_recursion:
            score += self.complexity_weights['backtracking']
        if visitor.has_pruning:
            score += self.complexity_weights['pruning']
        if visitor.theoretical_checks:
            score += self.complexity_weights['theoretical']
        
        # Add points for branching
        score += min(visitor.branching_factor * self.complexity_weights['branching'], 5.0)
        
        # Penalize trivial solutions
        if path is not None and self._is_trivial_solution(path):
            score += self.complexity_weights['trivial']
        
        return max(score, 0.0)  # Don't allow negative scores
    
    def _analyze_path(self, path: List[int], adj_matrix: np.ndarray) -> Dict:
        """Analyze characteristics of the solution path."""
        n = len(adj_matrix)
        
        # Calculate path properties
        edge_weights = []
        vertex_degrees = []
        for i in range(len(path)-1):
            v1, v2 = path[i], path[i+1]
            edge_weights.append(adj_matrix[v1][v2])
            vertex_degrees.append(np.sum(adj_matrix[v1]))
        vertex_degrees.append(np.sum(adj_matrix[path[-1]]))
        
        return {
            'path_length': len(path),
            'avg_vertex_degree': np.mean(vertex_degrees),
            'degree_variance': np.var(vertex_degrees),
            'is_trivial': self._is_trivial_solution(path)
        }
    
    def _is_trivial_solution(self, path: List[int]) -> bool:
        """Check if a solution is trivial (sequential range)."""
        if not path:
            return False
        return path == list(range(len(path)))
    
    def _create_error_evaluation(self, error: str) -> Dict:
        """Create an evaluation result for error cases."""
        return {
            'error': error,
            'complexity_score': 0.0,
            'code_length': 0,
            'computation_time': 0.0,
            'has_solution': False
        }


class CodeAnalysisVisitor(ast.NodeVisitor):
    """AST visitor to analyze code complexity and features."""
    
    def __init__(self):
        self.has_recursion = False
        self.has_backtracking = False
        self.branching_factor = 0
        self.has_pruning = False
        self.theoretical_checks = False
        self.function_calls = defaultdict(int)
        
    def visit_Call(self, node):
        """Track function calls."""
        if isinstance(node.func, ast.Name):
            self.function_calls[node.func.id] += 1
        self.generic_visit(node)
        
    def visit_If(self, node):
        """Track branching."""
        self.branching_factor += 1
        
        # Check for pruning conditions
        if isinstance(node.test, ast.Compare):
            if any(isinstance(op, (ast.Lt, ast.Gt, ast.LtE, ast.GtE)) 
                  for op in node.ops):
                self.has_pruning = True
        
        self.generic_visit(node)
        
    def visit_FunctionDef(self, node):
        """Detect recursive functions."""
        for n in ast.walk(node):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                if n.func.id == node.name:
                    self.has_recursion = True
                    self.has_backtracking = True
                    break
        self.generic_visit(node)
        
    def visit_Compare(self, node):
        """Detect theoretical checks."""
        if isinstance(node.left, ast.Call):
            if isinstance(node.left.func, ast.Name):
                if node.left.func.id in ['min', 'max', 'sum']:
                    self.theoretical_checks = True
        self.generic_visit(node)
