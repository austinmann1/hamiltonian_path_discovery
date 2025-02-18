"""
Conflict learning system for Hamiltonian path discovery.
Tracks and analyzes path failures to improve future attempts.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple
import numpy as np

@dataclass
class PathConflict:
    """Represents a specific conflict in a path attempt."""
    conflict_type: str  # 'invalid_edge', 'cycle', 'dead_end', 'duplicate'
    nodes: List[int]
    description: str
    learned_clause: str

class ConflictTracker:
    """Tracks and analyzes conflicts in path finding attempts."""
    
    def __init__(self):
        self.conflicts: List[PathConflict] = []
        self.invalid_edges: Set[Tuple[int, int]] = set()
        self.dead_ends: Set[int] = set()
        self.cycles: List[List[int]] = []
    
    def analyze_path_failure(self, path: List[int], adj_matrix: np.ndarray,
                           failure_point: int) -> Optional[PathConflict]:
        """
        Analyze a failed path attempt to identify the conflict.
        
        Args:
            path: The attempted path that failed
            adj_matrix: The graph's adjacency matrix
            failure_point: Index in path where failure occurred
            
        Returns:
            PathConflict if conflict identified, None otherwise
        """
        n = adj_matrix.shape[0]
        print(f"Analyzing path failure: path={path}, failure_point={failure_point}, n={n}")
        print(f"Adjacency matrix:\n{adj_matrix}")
        
        # Check for invalid edges
        for i in range(len(path)-1):
            if not adj_matrix[path[i], path[i+1]]:
                conflict = PathConflict(
                    conflict_type="invalid_edge",
                    nodes=[path[i], path[i+1]],
                    description=f"No edge exists between nodes {path[i]} and {path[i+1]}",
                    learned_clause=f"Avoid direct connection {path[i]}->{path[i+1]}"
                )
                self.invalid_edges.add((path[i], path[i+1]))
                self.conflicts.append(conflict)
                return conflict
        
        # Check for cycles
        visited = set()
        for i, node in enumerate(path):
            if node in visited:
                cycle = path[path.index(node):i+1]
                conflict = PathConflict(
                    conflict_type="cycle",
                    nodes=cycle,
                    description=f"Cycle detected: {cycle}",
                    learned_clause=f"Avoid cycle pattern: {' -> '.join(map(str, cycle))}"
                )
                self.cycles.append(cycle)
                self.conflicts.append(conflict)
                return conflict
            visited.add(node)
        
        # Check for dead ends
        print(f"Checking dead end condition: failure_point={failure_point} <= len(path)-1={len(path)-1}")
        if failure_point <= len(path)-1:
            current_node = path[failure_point]
            visited_nodes = set(path)
            print(f"Checking dead end: current_node={current_node}, visited_nodes={visited_nodes}")
            
            # First check if we have any unvisited moves
            unvisited_moves = [i for i in range(n) if adj_matrix[current_node][i] and i not in visited_nodes]
            possible_moves = [i for i in range(n) if adj_matrix[current_node][i]]
            print(f"Possible moves: {possible_moves}")
            print(f"Unvisited moves: {unvisited_moves}")
            
            if not unvisited_moves:
                # If we've visited all nodes, we need to check if we can get back to the start
                if len(visited_nodes) == n:
                    # Check if we can get back to node 0 to complete the path
                    if not adj_matrix[current_node][0]:
                        conflict = PathConflict(
                            conflict_type="dead_end",
                            nodes=[current_node],
                            description=f"Dead end at node {current_node}, cannot complete Hamiltonian path back to start",
                            learned_clause=f"Node {current_node} leads to dead end - no path back to start"
                        )
                        self.dead_ends.add(current_node)
                        self.conflicts.append(conflict)
                        print(f"Found dead end conflict: {conflict}")
                        return conflict
                else:
                    # We haven't visited all nodes but have no unvisited moves
                    conflict = PathConflict(
                        conflict_type="dead_end",
                        nodes=[current_node],
                        description=f"Dead end at node {current_node}, all neighbors already visited",
                        learned_clause=f"Node {current_node} leads to dead end when neighbors {possible_moves} are already visited"
                    )
                    self.dead_ends.add(current_node)
                    self.conflicts.append(conflict)
                    print(f"Found dead end conflict: {conflict}")
                    return conflict
        else:
            print("Not checking dead end - conditions not met")
        
        return None
    
    def format_for_prompt(self) -> str:
        """Format learned conflicts as constraints for the prompt."""
        constraints = []
        
        if self.invalid_edges:
            edges_str = ", ".join(f"{a}->{b}" for a, b in self.invalid_edges)
            constraints.append(f"Avoid invalid edges: {edges_str}")
        
        if self.dead_ends:
            ends_str = ", ".join(map(str, self.dead_ends))
            constraints.append(f"Dead end nodes to avoid: {ends_str}")
        
        if self.cycles:
            cycle_patterns = [" -> ".join(map(str, cycle)) for cycle in self.cycles]
            constraints.append("Cycle patterns to avoid:")
            constraints.extend(f"  - {pattern}" for pattern in cycle_patterns)
        
        if not constraints:
            return "No learned constraints yet."
        
        return "\n".join(["Learned Constraints:"] + constraints)
    
    def clear(self):
        """Clear all tracked conflicts."""
        self.conflicts.clear()
        self.invalid_edges.clear()
        self.dead_ends.clear()
        self.cycles.clear()
