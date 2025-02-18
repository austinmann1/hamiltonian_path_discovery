"""Conflict tracking and learning system for Hamiltonian path discovery.

This module implements CDCL-style (Conflict-Driven Clause Learning) conflict tracking
for Hamiltonian paths. It identifies and learns from specific path failures, such as:
- Invalid edges that don't exist in the graph
- Repeated vertices causing cycles
- Dead-end paths that can't be completed
- Suboptimal path segments that consistently lead to failures
"""

from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class PathConflict:
    """Represents a specific conflict in a path attempt."""
    conflict_type: str  # 'invalid_edge', 'cycle', 'dead_end', 'suboptimal'
    position: int  # Where in the path the conflict occurred
    vertices_involved: List[int]  # Vertices involved in the conflict
    context: Dict  # Additional context about the conflict
    frequency: int = 1  # How often this conflict pattern appears

@dataclass
class LearnedClause:
    """Represents a learned constraint from conflicts."""
    forbidden_pattern: Tuple[int, ...]  # Sequence of vertices that should be avoided
    reason: str  # Why this pattern is problematic
    confidence: float  # How confident we are this is a real issue (0.0 to 1.0)
    occurrences: int = 1  # How many times we've seen this issue

class ConflictTracker:
    def __init__(self):
        self.conflicts: List[PathConflict] = []
        self.learned_clauses: List[LearnedClause] = []
        self.edge_frequencies = defaultdict(int)  # Track problematic edges
        self.vertex_frequencies = defaultdict(int)  # Track problematic vertices
        
    def analyze_path_failure(self, 
                           path: List[int],
                           adjacency_matrix: np.ndarray,
                           failure_point: int) -> PathConflict:
        """Analyze a failed path attempt to identify the specific conflict.
        
        Args:
            path: The attempted path that failed
            adjacency_matrix: The graph's adjacency matrix
            failure_point: Index in the path where the failure occurred
            
        Returns:
            PathConflict object describing the failure
        """
        n = len(adjacency_matrix)
        seen = set()
        
        # Check for invalid edges
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if not adjacency_matrix[u][v]:
                return PathConflict(
                    conflict_type='invalid_edge',
                    position=i,
                    vertices_involved=[u, v],
                    context={'msg': f'Edge ({u}->{v}) does not exist in graph'}
                )
            seen.add(u)
        
        # Add last vertex to seen set
        if path:
            seen.add(path[-1])
        
        # Check for cycles (repeated vertices)
        seen_cycle = set()
        for i, v in enumerate(path):
            if v in seen_cycle:
                prev_pos = path.index(v)
                return PathConflict(
                    conflict_type='cycle',
                    position=i,
                    vertices_involved=[v],
                    context={
                        'msg': f'Vertex {v} repeated at positions {prev_pos} and {i}',
                        'cycle_length': i - prev_pos
                    }
                )
            seen_cycle.add(v)
        
        # Check for dead ends
        if failure_point < len(path):
            current = path[failure_point]
            # Find all unvisited neighbors
            available = [v for v in range(n) 
                        if adjacency_matrix[current][v] == 1 and v not in seen]
            
            if not available:
                return PathConflict(
                    conflict_type='dead_end',
                    position=failure_point,
                    vertices_involved=[current],
                    context={
                        'msg': f'No valid moves from vertex {current}',
                        'visited': list(seen)
                    }
                )
        
        # Default to suboptimal path
        return PathConflict(
            conflict_type='suboptimal',
            position=failure_point,
            vertices_involved=path[:failure_point+1],
            context={'msg': 'Path leads to no valid solution'}
        )
    
    def learn_from_conflict(self, conflict: PathConflict) -> Optional[LearnedClause]:
        """Learn a new clause from a conflict.
        
        Args:
            conflict: The conflict to learn from
            
        Returns:
            A new learned clause, if one can be derived
        """
        # Update frequency counters
        for v in conflict.vertices_involved:
            self.vertex_frequencies[v] += 1
        
        if conflict.conflict_type == 'invalid_edge':
            u, v = conflict.vertices_involved
            self.edge_frequencies[(u, v)] += 1
            
            # If we see this edge fail multiple times, create a learned clause
            if self.edge_frequencies[(u, v)] >= 3:
                return LearnedClause(
                    forbidden_pattern=(u, v),
                    reason=f'Edge ({u}->{v}) consistently fails',
                    confidence=0.8,
                    occurrences=self.edge_frequencies[(u, v)]
                )
                
        elif conflict.conflict_type == 'dead_end':
            # Learn to avoid paths that lead to this dead end
            if len(conflict.vertices_involved) >= 2:
                pattern = tuple(conflict.vertices_involved[-2:])
                return LearnedClause(
                    forbidden_pattern=pattern,
                    reason=f'Path segment {pattern} leads to dead end',
                    confidence=0.6,
                    occurrences=1
                )
        
        return None
    
    def get_conflict_summary(self) -> Dict:
        """Generate a summary of conflicts for the LLM.
        
        Returns:
            Dictionary containing conflict patterns and learned clauses
        """
        return {
            'forbidden_edges': [
                (edge, freq) for edge, freq in self.edge_frequencies.items()
                if freq >= 2
            ],
            'problematic_vertices': [
                (v, freq) for v, freq in self.vertex_frequencies.items()
                if freq >= 3
            ],
            'learned_clauses': [
                {
                    'pattern': clause.forbidden_pattern,
                    'reason': clause.reason,
                    'confidence': clause.confidence
                }
                for clause in self.learned_clauses
                if clause.confidence >= 0.6
            ]
        }
    
    def format_for_prompt(self) -> str:
        """Format conflict information for inclusion in LLM prompt.
        
        Returns:
            Formatted string describing learned constraints
        """
        summary = self.get_conflict_summary()
        
        lines = ["Based on previous attempts, avoid these patterns:"]
        
        if summary['forbidden_edges']:
            lines.append("\nForbidden edges (frequently failing):")
            for (u, v), freq in summary['forbidden_edges']:
                lines.append(f"- Edge ({u}->{v}) failed {freq} times")
                
        if summary['problematic_vertices']:
            lines.append("\nProblematic vertices (often lead to failures):")
            for v, freq in summary['problematic_vertices']:
                lines.append(f"- Vertex {v} involved in {freq} failures")
                
        if summary['learned_clauses']:
            lines.append("\nLearned constraints:")
            for clause in summary['learned_clauses']:
                lines.append(f"- Avoid pattern {clause['pattern']}: {clause['reason']}")
                
        return "\n".join(lines)
