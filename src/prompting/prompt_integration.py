"""
Integration layer between the prompting system and other components.
Handles communication between PromptManager and various learning systems.
"""

from typing import List, Optional, Dict
import numpy as np

from .prompt_manager import PromptManager
from ..conflict_learning.conflict_tracker import ConflictTracker

class PromptIntegration:
    """
    Integrates the prompting system with conflict learning and pattern mining.
    Manages the flow of information between components to generate optimal prompts.
    """
    
    def __init__(self):
        """Initialize the integration components."""
        self.prompt_manager = PromptManager()
        self.conflict_tracker = ConflictTracker()
        self.current_graph: Optional[np.ndarray] = None
        
    def start_new_graph(self, adj_matrix: np.ndarray):
        """
        Start working on a new graph.
        
        Args:
            adj_matrix: The adjacency matrix of the new graph
        """
        self.current_graph = adj_matrix
        self.conflict_tracker.clear()
    
    def record_path_failure(self, path: List[int], failure_point: int):
        """
        Record a failed path attempt and analyze the failure.
        
        Args:
            path: The failed path attempt
            failure_point: Index where the path failed
            
        Returns:
            The identified conflict, if any
        """
        if self.current_graph is None:
            raise ValueError("No current graph set. Call start_new_graph first.")
            
        return self.conflict_tracker.analyze_path_failure(
            path,
            self.current_graph,
            failure_point
        )
    
    def get_next_prompt(
        self,
        known_patterns: Optional[str] = None,
        theorem_insights: Optional[str] = None
    ) -> str:
        """
        Generate the next prompt based on current state.
        
        Args:
            known_patterns: Optional string describing known successful patterns
            theorem_insights: Optional string containing theoretical insights
            
        Returns:
            The generated prompt
        """
        if self.current_graph is None:
            raise ValueError("No current graph set. Call start_new_graph first.")
            
        # Get learned constraints from conflict tracker
        learned_constraints = self.conflict_tracker.format_for_prompt()
        
        return self.prompt_manager.generate_prompt(
            self.current_graph,
            learned_constraints=learned_constraints,
            known_patterns=known_patterns,
            theorem_insights=theorem_insights
        )
    
    def clear_state(self):
        """Clear all tracked state."""
        self.conflict_tracker.clear()
        self.current_graph = None
