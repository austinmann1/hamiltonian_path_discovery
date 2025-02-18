"""Pattern analysis system for discovering novel Hamiltonian path algorithms.

This module analyzes successful solutions to identify patterns that could lead to
novel algorithmic approaches for solving Hamiltonian path problems.
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from dataclasses import dataclass
import json


@dataclass
class GraphPattern:
    """Represents a pattern discovered in successful solutions."""
    pattern_id: str
    graph_properties: Dict[str, float]  # Properties like density, clustering coefficient, etc.
    solution_strategy: Dict[str, any]  # The approach that worked
    success_rate: float
    avg_path_length: float
    num_occurrences: int
    example_graphs: List[str]  # References to graphs where this pattern worked


class PatternAnalyzer:
    """Analyzes successful solutions to discover novel solution patterns."""

    def __init__(self):
        self.patterns: Dict[str, GraphPattern] = {}
        self.graph_properties_cache: Dict[str, Dict[str, float]] = {}

    def analyze_solution(
        self,
        adjacency_matrix: np.ndarray,
        solution_path: List[int],
        execution_info: Dict[str, any]
    ) -> Optional[str]:
        """Analyze a successful solution to identify any patterns.

        Args:
            adjacency_matrix: The graph's adjacency matrix
            solution_path: The successful Hamiltonian path
            execution_info: Information about how the solution was found

        Returns:
            Pattern ID if a pattern was identified, None otherwise
        """
        # Extract graph properties
        properties = self._extract_graph_properties(adjacency_matrix)
        
        # Analyze solution strategy
        strategy = self._analyze_solution_strategy(solution_path, adjacency_matrix)
        
        # Look for matching patterns or create new one
        pattern_id = self._find_or_create_pattern(properties, strategy)
        
        # Update pattern statistics
        self._update_pattern_stats(pattern_id, properties, strategy, execution_info)
        
        return pattern_id

    def _extract_graph_properties(self, adjacency_matrix: np.ndarray) -> Dict[str, float]:
        """Extract key properties from the graph that might indicate patterns."""
        n = len(adjacency_matrix)
        
        # Calculate basic properties
        edge_count = np.sum(adjacency_matrix) / 2  # Divide by 2 since matrix is symmetric
        max_edges = n * (n - 1) / 2  # Maximum possible edges in undirected graph
        density = edge_count / max_edges if max_edges > 0 else 0
        
        # Calculate degree statistics
        degrees = np.sum(adjacency_matrix, axis=1)
        min_degree = np.min(degrees)
        max_degree = np.max(degrees)
        avg_degree = np.mean(degrees)
        
        # Calculate clustering coefficient
        clustering = self._calculate_clustering_coefficient(adjacency_matrix)
        
        return {
            "size": n,
            "density": density,
            "min_degree": min_degree,
            "max_degree": max_degree,
            "avg_degree": avg_degree,
            "clustering": clustering
        }

    def _calculate_clustering_coefficient(self, adjacency_matrix: np.ndarray) -> float:
        """Calculate the global clustering coefficient of the graph."""
        n = len(adjacency_matrix)
        triangles = 0
        triplets = 0
        
        for i in range(n):
            neighbors = np.where(adjacency_matrix[i] == 1)[0]
            if len(neighbors) < 2:
                continue
                
            # Count triangles
            for j in range(len(neighbors)):
                for k in range(j + 1, len(neighbors)):
                    if adjacency_matrix[neighbors[j], neighbors[k]] == 1:
                        triangles += 1
                        
            # Count possible triplets
            triplets += len(neighbors) * (len(neighbors) - 1) / 2
            
        return triangles / triplets if triplets > 0 else 0

    def _analyze_solution_strategy(
        self,
        path: List[int],
        adjacency_matrix: np.ndarray
    ) -> Dict[str, any]:
        """Analyze the solution path to identify the strategy used."""
        n = len(adjacency_matrix)
        
        # Analyze path construction pattern
        degree_sequence = [np.sum(adjacency_matrix[node]) for node in path]
        
        # Check if path follows degree ordering
        follows_degree_order = all(
            degree_sequence[i] >= degree_sequence[i+1]
            for i in range(len(degree_sequence)-1)
        )
        
        # Check for local optimization patterns
        local_opts = self._analyze_local_decisions(path, adjacency_matrix)
        
        return {
            "follows_degree_order": follows_degree_order,
            "local_optimization_patterns": local_opts,
            "path_length": len(path),
            "start_degree": degree_sequence[0],
            "end_degree": degree_sequence[-1]
        }

    def _analyze_local_decisions(
        self,
        path: List[int],
        adjacency_matrix: np.ndarray
    ) -> List[Dict[str, any]]:
        """Analyze local decision patterns in the solution."""
        patterns = []
        
        for i in range(len(path) - 1):
            current = path[i]
            next_node = path[i + 1]
            
            # Get all possible next nodes at this point
            possible_next = np.where(adjacency_matrix[current] == 1)[0]
            
            # Analyze why this particular node was chosen
            pattern = {
                "position": i,
                "current_node": current,
                "chosen_node": next_node,
                "num_options": len(possible_next),
                "chosen_degree": np.sum(adjacency_matrix[next_node])
            }
            
            patterns.append(pattern)
            
        return patterns

    def _find_or_create_pattern(
        self,
        properties: Dict[str, float],
        strategy: Dict[str, any]
    ) -> str:
        """Find a matching pattern or create a new one."""
        # Generate pattern signature
        signature = self._generate_pattern_signature(properties, strategy)
        
        # Check if pattern exists
        if signature not in self.patterns:
            # Create new pattern
            pattern = GraphPattern(
                pattern_id=signature,
                graph_properties=properties,
                solution_strategy=strategy,
                success_rate=1.0,
                avg_path_length=strategy["path_length"],
                num_occurrences=0,
                example_graphs=[]
            )
            self.patterns[signature] = pattern
            
        return signature

    def _generate_pattern_signature(
        self,
        properties: Dict[str, float],
        strategy: Dict[str, any]
    ) -> str:
        """Generate a unique signature for a pattern based on its properties."""
        # Combine key properties into a signature
        key_props = {
            "size_range": round(properties["size"] / 5) * 5,  # Group similar sizes
            "density_range": round(properties["density"] * 4) / 4,  # Quarter intervals
            "degree_order": strategy["follows_degree_order"],
            "start_degree_ratio": round(strategy["start_degree"] / properties["size"] * 4) / 4
        }
        
        return json.dumps(key_props, sort_keys=True)

    def _update_pattern_stats(
        self,
        pattern_id: str,
        properties: Dict[str, float],
        strategy: Dict[str, any],
        execution_info: Dict[str, any]
    ) -> None:
        """Update statistics for a pattern."""
        pattern = self.patterns[pattern_id]
        
        # Update occurrence count
        pattern.num_occurrences += 1
        
        # Update success rate
        pattern.success_rate = (
            (pattern.success_rate * (pattern.num_occurrences - 1) + 1.0)
            / pattern.num_occurrences
        )
        
        # Update average path length
        pattern.avg_path_length = (
            (pattern.avg_path_length * (pattern.num_occurrences - 1) + strategy["path_length"])
            / pattern.num_occurrences
        )
        
        # Store example graph reference
        if "graph_id" in execution_info and len(pattern.example_graphs) < 5:
            pattern.example_graphs.append(execution_info["graph_id"])

    def get_pattern_insights(self) -> List[Dict[str, any]]:
        """Get insights about discovered patterns."""
        insights = []
        
        for pattern_id, pattern in self.patterns.items():
            if pattern.num_occurrences < 5:  # Need enough samples
                continue
                
            insight = {
                "pattern_id": pattern_id,
                "success_rate": pattern.success_rate,
                "num_occurrences": pattern.num_occurrences,
                "avg_path_length": pattern.avg_path_length,
                "graph_properties": pattern.graph_properties,
                "strategy": pattern.solution_strategy,
                "example_graphs": pattern.example_graphs
            }
            
            insights.append(insight)
            
        return sorted(
            insights,
            key=lambda x: (x["success_rate"], x["num_occurrences"]),
            reverse=True
        )
