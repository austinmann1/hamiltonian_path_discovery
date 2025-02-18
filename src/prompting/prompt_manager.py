"""Manages prompt generation for Hamiltonian path discovery."""

from typing import Dict, List, Optional
import numpy as np

from .templates.novel_path_discovery import (
    PATTERN_BASED_PROMPT,
    CONFLICT_AWARE_PROMPT,
    OPTIMIZATION_PROMPT,
    format_graph_properties,
    format_theoretical_insights,
    format_failure_patterns,
    format_implementation_analysis,
    format_subpath_patterns
)

class PromptManager:
    """Manages the generation and selection of prompts."""
    
    def generate_pattern_based_prompt(self, adj_matrix: np.ndarray,
                                    pattern_insights: Optional[Dict] = None,
                                    theoretical_insights: Optional[str] = None) -> str:
        """
        Generate a prompt incorporating pattern insights.
        
        Args:
            adj_matrix: Graph adjacency matrix
            pattern_insights: Optional dictionary of pattern insights
            theoretical_insights: Optional string of theoretical insights
            
        Returns:
            Formatted prompt string
        """
        # Get graph properties
        from graph_analyzer import analyze_graph_properties
        properties = analyze_graph_properties(adj_matrix)
        
        # Format sections
        graph_props = format_graph_properties(properties)
        
        # If no theoretical insights provided, use basic properties
        if theoretical_insights is None:
            theoretical = format_theoretical_insights(properties)
        else:
            theoretical = theoretical_insights
        
        # Get pattern insights if available
        implementations = []
        failure_patterns = []
        subpath_patterns = []
        
        if pattern_insights:
            implementations = pattern_insights.get('implementations', [])
            failure_patterns = pattern_insights.get('failures', [])
            subpath_patterns = pattern_insights.get('subpaths', [])
        
        # Format pattern sections
        impl_analysis = format_implementation_analysis(implementations)
        failures = format_failure_patterns(failure_patterns)
        
        # Combine all sections into final prompt
        prompt = f"""You are an expert in graph theory and algorithm design. Your task is to find a Hamiltonian path in the following undirected graph.

{graph_props}

Theoretical Analysis:
{theoretical}

Implementation Guidance:
{impl_analysis}

Your goal is to write a Python function that finds a Hamiltonian path in this graph. Consider:
1. Use the theoretical properties above to guide your approach
2. Implement the suggested optimizations from the theoretical analysis
3. Learn from both successful and failed patterns
4. Use degree information for vertex selection
5. Consider the Bondy-Chvátal closure when planning paths

Requirements:
1. Function signature must be: def find_hamiltonian_path(adj_matrix)
2. Input is a numpy array representing the adjacency matrix
3. Return a list of vertices forming a valid Hamiltonian path, or None if no path exists
4. Each vertex must appear exactly once
5. Consecutive vertices must be connected by edges
6. Include necessary imports (e.g., import numpy as np)

Your output should be ONLY the raw Python code. Do not include any explanatory text, comments, or formatting - only the raw code:

import numpy as np

def find_hamiltonian_path(adj_matrix):
    # Your implementation here
    return path_or_none"""

        return prompt
    
    def generate_conflict_aware_prompt(self, adj_matrix: np.ndarray,
                                     pattern_insights: Optional[Dict] = None,
                                     theoretical_insights: Optional[str] = None) -> str:
        """
        Generate a prompt focusing on conflict avoidance.
        
        Args:
            adj_matrix: Graph adjacency matrix
            pattern_insights: Optional dictionary of pattern insights
            theoretical_insights: Optional string of theoretical insights
            
        Returns:
            Formatted prompt string
        """
        # Get graph properties
        from graph_analyzer import analyze_graph_properties
        properties = analyze_graph_properties(adj_matrix)
        
        # Format sections
        graph_props = format_graph_properties(properties)
        
        # If no theoretical insights provided, use basic properties
        if theoretical_insights is None:
            theoretical = format_theoretical_insights(properties)
        else:
            theoretical = theoretical_insights
        
        # Get pattern insights if available
        failure_patterns = []
        subpath_patterns = []
        
        if pattern_insights:
            failure_patterns = pattern_insights.get('failures', [])
            subpath_patterns = pattern_insights.get('subpaths', [])
        
        # Format pattern sections
        failures = format_failure_patterns(failure_patterns)
        subpaths = format_subpath_patterns(subpath_patterns)
        
        # Combine all sections into final prompt
        prompt = f"""You are an expert in graph theory and algorithm design. Your task is to find a Hamiltonian path in the following undirected graph.

{graph_props}

Theoretical Analysis:
{theoretical}

Implementation Guidance:
{failures}

Your goal is to write a Python function that finds a Hamiltonian path in this graph. Consider:
1. Use the theoretical properties above to guide your approach
2. Implement the suggested optimizations from the theoretical analysis
3. Learn from both successful and failed patterns
4. Use degree information for vertex selection
5. Consider the Bondy-Chvátal closure when planning paths

Requirements:
1. Function signature must be: def find_hamiltonian_path(adj_matrix)
2. Input is a numpy array representing the adjacency matrix
3. Return a list of vertices forming a valid Hamiltonian path, or None if no path exists
4. Each vertex must appear exactly once
5. Consecutive vertices must be connected by edges
6. Include necessary imports (e.g., import numpy as np)

Your output should be ONLY the raw Python code. Do not include any explanatory text, comments, or formatting - only the raw code:

import numpy as np

def find_hamiltonian_path(adj_matrix):
    # Your implementation here
    return path_or_none"""

        return prompt
    
    def generate_optimization_prompt(self, adj_matrix: np.ndarray,
                                   pattern_insights: Optional[Dict] = None,
                                   theoretical_insights: Optional[str] = None,
                                   performance_metrics: Optional[Dict] = None) -> str:
        """
        Generate a prompt focusing on optimization.
        
        Args:
            adj_matrix: Graph adjacency matrix
            pattern_insights: Optional dictionary of pattern insights
            theoretical_insights: Optional string of theoretical insights
            performance_metrics: Optional performance metrics
            
        Returns:
            Formatted prompt string
        """
        # Get graph properties
        from graph_analyzer import analyze_graph_properties
        properties = analyze_graph_properties(adj_matrix)
        
        # Format sections
        graph_props = format_graph_properties(properties)
        
        # If no theoretical insights provided, use basic properties
        if theoretical_insights is None:
            theoretical = format_theoretical_insights(properties)
        else:
            theoretical = theoretical_insights
        
        # Get pattern insights if available
        implementations = []
        if pattern_insights:
            implementations = pattern_insights.get('implementations', [])
        
        # Format sections
        impl_analysis = format_implementation_analysis(implementations)
        
        # Format performance metrics
        perf_metrics = "No performance data available."
        if performance_metrics:
            perf_lines = []
            for metric, value in performance_metrics.items():
                if isinstance(value, (int, float)):
                    perf_lines.append(f"- {metric}: {value:.2f}")
                else:
                    perf_lines.append(f"- {metric}: {value}")
            perf_metrics = "\n".join(perf_lines)
        
        # Combine all sections into final prompt
        prompt = f"""You are an expert in graph theory and algorithm design. Your task is to find a Hamiltonian path in the following undirected graph.

{graph_props}

Theoretical Analysis:
{theoretical}

Implementation Guidance:
{impl_analysis}

Performance Metrics:
{perf_metrics}

Your goal is to write a Python function that finds a Hamiltonian path in this graph. Consider:
1. Use the theoretical properties above to guide your approach
2. Implement the suggested optimizations from the theoretical analysis
3. Learn from both successful and failed patterns
4. Use degree information for vertex selection
5. Consider the Bondy-Chvátal closure when planning paths

Requirements:
1. Function signature must be: def find_hamiltonian_path(adj_matrix)
2. Input is a numpy array representing the adjacency matrix
3. Return a list of vertices forming a valid Hamiltonian path, or None if no path exists
4. Each vertex must appear exactly once
5. Consecutive vertices must be connected by edges
6. Include necessary imports (e.g., import numpy as np)

Your output should be ONLY the raw Python code. Do not include any explanatory text, comments, or formatting - only the raw code:

import numpy as np

def find_hamiltonian_path(adj_matrix):
    # Your implementation here
    return path_or_none"""

        return prompt
    
    def generate_prompt(self, adj_matrix: np.ndarray, theoretical_insights: Dict, pattern_insights: Dict) -> str:
        """Generate a prompt for the model based on graph properties and insights."""
        n = adj_matrix.shape[0]
        density = np.sum(adj_matrix) / (n * (n-1))
        min_degree = min(np.sum(adj_matrix, axis=0))
        max_degree = max(np.sum(adj_matrix, axis=0))
        avg_degree = np.mean(np.sum(adj_matrix, axis=0))
        
        # Extract key theoretical properties
        dirac_insight = next((x for x in theoretical_insights.get('theoretical_insights', []) if x['name'] == "Dirac's Theorem"), None)
        ore_insight = next((x for x in theoretical_insights.get('theoretical_insights', []) if x['name'] == "Ore's Theorem"), None)
        closure_insight = next((x for x in theoretical_insights.get('theoretical_insights', []) if x['name'] == "Bondy-Chvátal Closure"), None)
        claw_patterns = next((x for x in theoretical_insights.get('forbidden_patterns', []) if x['type'] == "claw_subgraph"), None)
        
        # Build theoretical guidance section
        theoretical_guidance = []
        
        if dirac_insight:
            if dirac_insight['condition'] == 'satisfied':
                theoretical_guidance.append(
                    "1. Dirac's Theorem is satisfied - every vertex has degree ≥ n/2. "
                    "This guarantees a Hamiltonian path exists. Focus on efficient path construction."
                )
            else:
                theoretical_guidance.append(
                    "1. Dirac's Theorem is not satisfied - some vertices have degree < n/2. "
                    "Need careful vertex selection and backtracking."
                )
        
        if ore_insight:
            if ore_insight['condition'] == 'satisfied':
                theoretical_guidance.append(
                    "2. Ore's Theorem is satisfied - for any non-adjacent vertices u,v: deg(u) + deg(v) ≥ n. "
                    "A Hamiltonian cycle exists. Consider finding a cycle and removing one edge."
                )
            else:
                theoretical_guidance.append(
                    "2. Ore's Theorem is not satisfied. Look for vertex pairs with low degree sums "
                    "as potential bottlenecks."
                )
        
        if closure_insight:
            if closure_insight['condition'] == 'complete':
                theoretical_guidance.append(
                    "3. Bondy-Chvátal closure is complete - the graph can be closed to a complete graph. "
                    "This preserves Hamiltonicity."
                )
            else:
                edges = closure_insight.get('edges', [])
                theoretical_guidance.append(
                    f"3. Bondy-Chvátal closure adds {len(edges)} edges. Consider these virtual edges "
                    "when planning path construction."
                )
        
        if claw_patterns:
            num_claws = len(claw_patterns.get('claws', []))
            if num_claws == 0:
                theoretical_guidance.append(
                    "4. Graph is claw-free (no K1,3 subgraphs). Use simpler path expansion techniques."
                )
            else:
                theoretical_guidance.append(
                    f"4. Found {num_claws} claw subgraphs. These may complicate path finding - "
                    "consider avoiding these configurations in path construction."
                )
        
        # Build implementation guidance from patterns
        implementation_guidance = []
        if pattern_insights['implementations']:
            implementation_guidance.append("Previous successful strategies:")
            for impl in pattern_insights['implementations'][:3]:  # Show top 3
                implementation_guidance.append(f"- {impl.get('description', 'No description')}")
        
        if pattern_insights['failures']:
            implementation_guidance.append("\nPatterns to avoid:")
            for fail in pattern_insights['failures'][:3]:  # Show top 3
                implementation_guidance.append(f"- {fail.get('reason', 'No reason given')}")
        
        prompt = f"""You are an expert in graph theory and algorithm design. Your task is to find a Hamiltonian path in the following undirected graph.

Graph Properties:
- Size: {n} vertices
- Density: {density:.2f}
- Degree range: {min_degree} to {max_degree}
- Average degree: {avg_degree:.2f}
- Connected: Yes

Theoretical Analysis:
{chr(10).join(theoretical_guidance)}

Implementation Guidance:
{chr(10).join(implementation_guidance)}

Your goal is to write a Python function that finds a Hamiltonian path in this graph. Consider:
1. Use the theoretical properties above to guide your approach
2. Implement the suggested optimizations from the theoretical analysis
3. Learn from both successful and failed patterns
4. Use degree information for vertex selection
5. Consider the Bondy-Chvátal closure when planning paths

Requirements:
1. Function signature must be: def find_hamiltonian_path(adj_matrix)
2. Input is a numpy array representing the adjacency matrix
3. Return a list of vertices forming a valid Hamiltonian path, or None if no path exists
4. Each vertex must appear exactly once
5. Consecutive vertices must be connected by edges
6. Include necessary imports (e.g., import numpy as np)

Your output should be ONLY the raw Python code. Do not include any explanatory text, comments, or formatting - only the raw code:

import numpy as np

def find_hamiltonian_path(adj_matrix):
    # Your implementation here
    return path_or_none"""

        return prompt
