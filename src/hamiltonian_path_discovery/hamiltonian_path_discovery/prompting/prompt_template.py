"""Defines the PromptTemplate class for managing prompt templates."""

from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class PromptTemplate:
    """Template for generating prompts."""
    name: str
    template: str
    parameters: Dict[str, str]
    description: str
    examples: list
    metrics: Dict[str, float]
    
    def format(self, params: Dict) -> str:
        """Format the template with the given parameters."""
        return self.template.format(**params)
    
    def update_metrics(self, success: bool, energy: float):
        """Update template metrics with new result."""
        total_attempts = self.metrics.get("total_attempts", 0) + 1
        self.metrics["total_attempts"] = total_attempts
        
        if success:
            successes = self.metrics.get("successes", 0) + 1
            self.metrics["successes"] = successes
            self.metrics["success_rate"] = successes / total_attempts
        
        # Update energy metrics
        total_energy = self.metrics.get("total_energy", 0) + energy
        self.metrics["total_energy"] = total_energy
        self.metrics["average_energy"] = total_energy / total_attempts
