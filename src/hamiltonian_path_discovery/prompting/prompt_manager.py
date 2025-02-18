"""
Prompt engineering system for Hamiltonian Path Discovery.
"""

from typing import Dict, List, Optional
import os
import json
import time
from pathlib import Path
import numpy as np

from .prompt_template import PromptTemplate, PromptResult
from ..verification.verification_oracle import VerificationOracle
from ..logging.structured_logger import StructuredLogger
from ..experimentation.experiment import Experiment
from ..pattern_mining.pattern_analyzer import PatternAnalyzer
from ..conflict_learning.conflict_tracker import ConflictTracker, PathConflict

class PromptManager:
    """Manages prompt templates and execution."""

    def __init__(
        self,
        verifier: Optional[VerificationOracle] = None,
        logger: Optional[StructuredLogger] = None,
        experiment: Optional[Experiment] = None
    ):
        self.templates = {}
        self.verifier = verifier or VerificationOracle()
        self.logger = logger or StructuredLogger()
        self.experiment = experiment or Experiment()
        self.pattern_analyzer = PatternAnalyzer()
        self.conflict_tracker = ConflictTracker()

    def _load_templates(self) -> Dict[str, PromptTemplate]:
        """Load prompt templates from file."""
        if not self.templates_file.exists():
            # Create default templates
            templates = {
                "basic": PromptTemplate(
                    name="basic",
                    template="""Given a graph with {num_nodes} nodes represented by its adjacency matrix:
{adjacency_matrix}

Find a Hamiltonian path in this graph. A Hamiltonian path visits each node exactly once.
Return the path as a list of node indices, where each consecutive pair of nodes must have
an edge between them in the graph.

Requirements:
1. The path must start at node {start_node} and end at node {end_node}
2. Each node must be visited exactly once
3. There must be an edge between consecutive nodes in the path
4. The path must be valid according to the adjacency matrix

Example format:
[0, 2, 1, 3]  # This means: 0 -> 2 -> 1 -> 3""",
                    description="Basic prompt for finding Hamiltonian paths",
                    parameters={
                        "num_nodes": "Number of nodes in the graph",
                        "adjacency_matrix": "Adjacency matrix as a string",
                        "start_node": "Starting node index",
                        "end_node": "Ending node index"
                    },
                    examples=[],
                    metrics={
                        "success_rate": 0.0,
                        "average_time": 0.0,
                        "average_energy": 0.0
                    }
                ),
                "advanced": PromptTemplate(
                    name="advanced",
                    template="""You are tasked with finding a Hamiltonian path in a graph with {num_nodes} nodes.

Graph Structure:
- Adjacency Matrix:
{adjacency_matrix}
- Start Node: {start_node}
- End Node: {end_node}
- Known Properties: {properties}

A Hamiltonian path must:
1. Start at the specified start node
2. End at the specified end node
3. Visit each node exactly once
4. Only use edges that exist in the graph (1's in the adjacency matrix)

Previous attempts: {previous_attempts}

Strategy hints:
1. Check degree constraints
2. Look for forced moves
3. Consider backtracking at decision points
4. Verify edge existence at each step

Return ONLY the path as a list of integers, e.g.: [0, 2, 1, 3]""",
                    description="Advanced prompt with graph properties and strategy hints",
                    parameters={
                        "num_nodes": "Number of nodes in the graph",
                        "adjacency_matrix": "Adjacency matrix as a string",
                        "start_node": "Starting node index",
                        "end_node": "Ending node index",
                        "properties": "Known graph properties",
                        "previous_attempts": "Previous failed attempts"
                    },
                    examples=[],
                    metrics={
                        "success_rate": 0.0,
                        "average_time": 0.0,
                        "average_energy": 0.0
                    }
                )
            }
            self._save_templates(templates)
            return templates
        
        with open(self.templates_file, 'r') as f:
            data = json.load(f)
            return {
                name: PromptTemplate.from_dict(template_data)
                for name, template_data in data.items()
            }

    def _save_templates(self, templates: Dict[str, PromptTemplate]):
        """Save prompt templates to file."""
        with open(self.templates_file, 'w') as f:
            json.dump(
                {name: template.to_dict() for name, template in templates.items()},
                f,
                indent=2
            )

    def create_template(self, template: PromptTemplate) -> None:
        """
        Create a new prompt template.
        
        Args:
            template: The template to create
        """
        if template.name in self.templates:
            raise ValueError(f"Template {template.name} already exists")
        
        self.templates[template.name] = template
        self._save_templates(self.templates)
        
        # Log creation
        self.logger.log_metrics(
            f"prompt_template_created_{template.name}",
            {"version": template.version}
        )

    def update_template(self, name: str, updates: Dict) -> None:
        """
        Update an existing template.
        
        Args:
            name: Name of the template to update
            updates: Dictionary of fields to update
        """
        if name not in self.templates:
            raise ValueError(f"Template {name} not found")
        
        template = self.templates[name]
        template_dict = template.to_dict()
        template_dict.update(updates)
        template_dict["version"] += 1
        
        self.templates[name] = PromptTemplate.from_dict(template_dict)
        self._save_templates(self.templates)
        
        # Log update
        self.logger.log_metrics(
            f"prompt_template_updated_{name}",
            {"version": template_dict["version"]}
        )

    def get_template(self, name: str) -> PromptTemplate:
        """Get a prompt template by name."""
        return self.templates.get(name)

    def list_templates(self) -> List[str]:
        """Get list of available templates."""
        return list(self.templates.keys())

    def format_prompt(
        self,
        template_name: str,
        params: Dict
    ) -> str:
        """
        Format a prompt template with parameters.
        
        Args:
            template_name: Name of the template to use
            params: Parameters to fill in the template
            
        Returns:
            Formatted prompt string
        """
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template {template_name} not found")
        
        # Create a copy of params to avoid modifying the original
        format_params = params.copy()
        
        # Set default values for optional parameters
        if "previous_attempts" not in format_params:
            format_params["previous_attempts"] = "[]"
        elif isinstance(format_params["previous_attempts"], list):
            format_params["previous_attempts"] = json.dumps(format_params["previous_attempts"], indent=2)
        
        # Validate parameters
        missing = set(template.parameters.keys()) - set(format_params.keys())
        if missing:
            raise ValueError(f"Missing parameters: {missing}")
        
        return template.template.format(**format_params)

    def execute_prompt(
        self,
        template_name: str,
        params: Dict,
        llm_interface: any,
        max_attempts: int = 5,
        min_success_threshold: float = 0.7
    ) -> PromptResult:
        """
        Execute a prompt with conflict learning and iterative refinement.
        
        Args:
            template_name: Name of the template to use
            params: Parameters for the template
            llm_interface: LLM interface to use
            max_attempts: Maximum number of attempts to try (default: 5)
            min_success_threshold: Minimum success rate to consider solution acceptable (default: 0.7)
            
        Returns:
            PromptResult with execution details of best attempt
        """
        exp_id = self.experiment.start_experiment(
            description=f"Prompt Execution: {template_name}",
            config={"template": template_name, "params": params}
        )
        
        best_result = None
        attempts = []
        start_time = time.time()
        verification = None
        error = None
        path = []
        current_result = None
        
        try:
            # Add conflict learning information to params
            if hasattr(self, 'conflict_tracker'):
                params['conflict_learning'] = self.conflict_tracker.format_for_prompt()
            
            # Track the current attempt
            current_attempt = 0
            
            while current_attempt < max_attempts:
                try:
                    # Format and execute prompt
                    prompt = self.format_prompt(template_name, params)
                    
                    response = llm_interface.generate_code(prompt)
                    generated_code = response.get("code", [])
                    success = response.get("success", False)
                    error = response.get("error")
                    
                    # Parse and verify path
                    try:
                        if isinstance(generated_code, list):
                            path = generated_code
                        else:
                            # Parse string representation of path
                            clean_content = "".join(str(generated_code).split())
                            
                            if not clean_content.startswith("[") or not clean_content.endswith("]"):
                                raise ValueError("Invalid path format - must be a list enclosed in []")
                            
                            if clean_content == "[]":
                                path = []
                            else:
                                parts = clean_content[1:-1].split(",")
                                path = [int(p.strip()) for p in parts if p.strip()]
                                
                                if not all(isinstance(x, int) for x in path):
                                    raise ValueError("Path contains non-integer elements")
                        
                        # Convert adjacency matrix and verify solution
                        matrix = np.array([[int(x) for x in row.split()] for row in params["adjacency_matrix"].strip().split("\n")])
                        verification = self.verifier.verify_with_explanation(matrix, path)
                        
                        success = verification.get("is_valid", False)
                        error = None if success else verification.get("error", "Invalid path")
                        
                        # If successful, analyze the pattern
                        if success:
                            pattern_id = self.pattern_analyzer.analyze_solution(
                                matrix,
                                path,
                                {
                                    "graph_id": params.get("graph_id", f"graph_{exp_id}"),
                                    "template": template_name,
                                    "attempt": current_attempt + 1
                                }
                            )
                            if pattern_id:
                                self.logger.log_metrics(
                                    f"pattern_discovery_{template_name}",
                                    {
                                        "pattern_id": pattern_id,
                                        "graph_id": params.get("graph_id", f"graph_{exp_id}"),
                                        "attempt": current_attempt + 1
                                    }
                                )
                        
                    except Exception as e:
                        success = False
                        error = str(e)
                        verification = {"is_valid": False, "error": str(e)}
                        path = []
                    
                    # Create result for this attempt
                    current_result = PromptResult(
                        template_name=template_name,
                        input_params=params,
                        generated_code=path,  # Store the parsed path
                        verification_result=verification,
                        execution_time=time.time() - start_time,
                        success=success,
                        error=error
                    )
                    
                    # Add attempt info for feedback
                    attempt_info = {
                        "path": path,
                        "error": error,
                        "attempt": current_attempt + 1,
                        "success": success
                    }
                    attempts.append(attempt_info)
                    
                    # Update best result if this is better
                    if best_result is None or (
                        current_result.success and (
                            not best_result.success or
                            len(current_result.generated_code) < len(best_result.generated_code)
                        )
                    ):
                        best_result = current_result
                    
                    # Check stopping conditions
                    if success:
                        if verification.get("success_rate", 0) >= min_success_threshold:
                            break
                    
                    # If verification fails, analyze the conflict
                    failure_point = self._find_failure_point(path, params['adjacency_matrix'])
                    conflict = self.conflict_tracker.analyze_path_failure(
                        path=path,
                        adjacency_matrix=params['adjacency_matrix'],
                        failure_point=failure_point
                    )
                    
                    # Learn from the conflict
                    learned_clause = self.conflict_tracker.learn_from_conflict(conflict)
                    if learned_clause:
                        # Update params with new conflict information
                        params['conflict_learning'] = self.conflict_tracker.format_for_prompt()
                
                except Exception as e:
                    error = str(e)
                    attempt_info = {
                        "error": error,
                        "attempt": current_attempt + 1,
                        "success": False
                    }
                    attempts.append(attempt_info)
                
                current_attempt += 1
            
            # If we never got a successful result, use the last attempt
            if best_result is None:
                if current_result is not None:
                    best_result = current_result
                else:
                    verification = {"is_valid": False, "error": error or "No successful solution found"}
                    best_result = PromptResult(
                        template_name=template_name,
                        input_params=params,
                        generated_code=path,
                        verification_result=verification,
                        execution_time=time.time() - start_time,
                        success=False,
                        error=error or "No successful solution found"
                    )
            
            # Update template metrics with best result
            self._update_metrics(template_name, best_result)
            
            # Log final result
            self.logger.log_metrics(
                f"prompt_execution_{template_name}",
                {
                    "success": best_result.success,
                    "attempts": len(attempts),
                    "execution_time": best_result.execution_time
                }
            )
            
            return best_result
            
        except Exception as e:
            # Log error using log_metrics since log_error doesn't exist
            self.logger.log_metrics(
                f"prompt_execution_{template_name}_error",
                {"error": str(e)}
            )
            return PromptResult(
                template_name=template_name,
                input_params=params,
                generated_code=path,
                verification_result={"is_valid": False, "error": str(e)},
                execution_time=time.time() - start_time,
                success=False,
                error=str(e)
            )
        finally:
            self.experiment.end_experiment()

    def _find_failure_point(self, path: List[int], adjacency_matrix: np.ndarray) -> int:
        """Find the point in the path where verification fails."""
        n = len(adjacency_matrix)
        seen = set()
        
        for i in range(len(path)):
            # Check for repeated vertices
            if path[i] in seen:
                return i
            seen.add(path[i])
            
            # Check for invalid edges
            if i < len(path) - 1:
                if not adjacency_matrix[path[i]][path[i + 1]]:
                    return i
            
            # Check for dead ends
            if i < len(path) - 1:
                current = path[i]
                next_options = set(v for v in range(n) 
                                 if adjacency_matrix[current][v] and v not in seen)
                if not next_options:
                    return i
        
        return len(path) - 1

    def _update_metrics(self, template_name: str, result: PromptResult):
        """Update template metrics with new result."""
        template = self.templates[template_name]
        metrics = template.metrics
        
        # Update running averages
        n = len(self._load_results(template_name))
        metrics["success_rate"] = (
            (metrics["success_rate"] * (n-1) + int(result.success)) / n
            if n > 0 else int(result.success)
        )
        metrics["average_time"] = (
            (metrics["average_time"] * (n-1) + result.execution_time) / n
            if n > 0 else result.execution_time
        )
        
        self.update_template(template_name, {"metrics": metrics})

    def _save_result(self, result: PromptResult):
        """Save prompt execution result."""
        result_file = (
            self.results_dir / 
            f"{result.template_name}_{int(time.time())}.json"
        )
        with open(result_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

    def _load_results(self, template_name: str) -> List[PromptResult]:
        """Load all results for a template."""
        results = []
        for file in self.results_dir.glob(f"{template_name}_*.json"):
            with open(file, 'r') as f:
                data = json.load(f)
                results.append(PromptResult(**data))
        return results

    def get_template_performance(self, template_name: str) -> Dict:
        """
        Get detailed performance metrics for a template.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Dictionary with performance metrics
        """
        results = self._load_results(template_name)
        if not results:
            return {}
        
        success_rate = sum(1 for r in results if r.success) / len(results)
        avg_time = np.mean([r.execution_time for r in results])
        
        return {
            "total_executions": len(results),
            "success_rate": success_rate,
            "average_time": avg_time,
            "recent_success_rate": sum(
                1 for r in results[-10:] if r.success
            ) / min(10, len(results))
        }
