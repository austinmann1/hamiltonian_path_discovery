"""
Templates for generating prompts for Hamiltonian path discovery.
These templates are designed for the Deepseek-coder-33b-instruct model,
optimized for larger context windows and mathematical reasoning.
"""

from typing import Dict, List

# Pattern-based prompt incorporating learned insights and theoretical guidance
PATTERN_BASED_PROMPT = '''You are an expert in graph theory and algorithm design. Your task is to find a Hamiltonian path in the following undirected graph.

{graph_properties}

Theoretical Analysis:
{theoretical_insights}

Previous Implementation Analysis:
{implementation_analysis}

Failure Patterns to Avoid:
{failure_patterns}

Your goal is to write a Python function that finds a Hamiltonian path in this graph. Consider:
1. Use the theoretical properties above to guide your approach:
   - If Dirac's theorem is satisfied, focus on constructing the path
   - If Ore's theorem is satisfied, you can find a Hamiltonian cycle
   - For claw-free graphs, use simpler path expansion
   - Use connectivity information for pruning
2. Learn from both successful and failed attempts
3. Avoid patterns that led to failures
4. Use successful subpath patterns as building blocks
5. Do NOT return a simple sequential path unless the graph is complete

Requirements:
1. Function signature must be: def find_hamiltonian_path(adj_matrix)
2. Input is a numpy array representing the adjacency matrix
3. Return a list of vertices forming a valid Hamiltonian path, or None if no path exists
4. Each vertex must appear exactly once
5. Consecutive vertices must be connected by edges
6. Include necessary imports (e.g., import numpy as np)

Your output should be ONLY the raw Python code.

import numpy as np

def find_hamiltonian_path(adj_matrix):
    # Your implementation here
    return path_or_none

Do not include any explanatory text, comments, or formatting - only the raw Python code.'''

# Conflict-aware prompt focusing on failure avoidance
CONFLICT_AWARE_PROMPT = '''You are an expert in graph theory and algorithm design. Your task is to find a Hamiltonian path while avoiding known failure patterns.

{graph_properties}

Critical Failure Patterns:
{failure_patterns}

Theoretical Constraints:
{theoretical_insights}

Successful Subpath Patterns:
{subpath_patterns}

Your goal is to write a Python function that finds a Hamiltonian path in this graph. Important considerations:
1. Use theoretical insights to guide your search:
   - Check degree conditions (Dirac/Ore theorems)
   - Look for forbidden subgraphs (claws, low-degree pairs)
   - Use connectivity information
2. Avoid the failure patterns listed above
3. Consider the graph's density and degree distribution
4. Use successful subpath patterns as building blocks
5. Do NOT return a simple sequential path unless the graph is complete

Requirements:
1. Function signature must be: def find_hamiltonian_path(adj_matrix)
2. Input is a numpy array representing the adjacency matrix
3. Return a list of vertices forming a valid Hamiltonian path, or None if no path exists
4. Each vertex must appear exactly once
5. Consecutive vertices must be connected by edges
6. Include necessary imports (e.g., import numpy as np)

Your output should be ONLY the raw Python code.

import numpy as np

def find_hamiltonian_path(adj_matrix):
    # Your implementation here
    return path_or_none

Do not include any explanatory text, comments, or formatting - only the raw Python code.'''

# Optimization-focused prompt incorporating performance metrics
OPTIMIZATION_PROMPT = '''You are an expert in graph theory and algorithm design. Your task is to find a Hamiltonian path with improved performance.

{graph_properties}

Performance Analysis:
{performance_metrics}

Theoretical Optimizations:
{theoretical_insights}

Successful Implementation Patterns:
{implementation_patterns}

Your goal is to write a Python function that efficiently finds a Hamiltonian path. Focus on:
1. Using theoretical properties for early pruning:
   - Check Dirac/Ore conditions first
   - Use connectivity information
   - Look for forbidden subgraphs
2. Applying successful patterns from previous implementations
3. Avoiding known performance bottlenecks
4. Implementing sophisticated backtracking with pruning
5. Do NOT return a simple sequential path unless the graph is complete

Requirements:
1. Function signature must be: def find_hamiltonian_path(adj_matrix)
2. Input is a numpy array representing the adjacency matrix
3. Return a list of vertices forming a valid Hamiltonian path, or None if no path exists
4. Each vertex must appear exactly once
5. Consecutive vertices must be connected by edges
6. Include necessary imports (e.g., import numpy as np)

Your output should be ONLY the raw Python code.

import numpy as np

def find_hamiltonian_path(adj_matrix):
    # Your implementation here
    return path_or_none

Do not include any explanatory text, comments, or formatting - only the raw Python code.'''

def format_graph_properties(properties: Dict) -> str:
    """Format graph properties section of the prompt."""
    lines = ["Graph Properties:"]
    lines.append(f"- Size: {properties['size']} vertices")
    lines.append(f"- Density: {properties['density']:.2f}")
    lines.append(f"- Degree range: {properties['min_degree']} to {properties['max_degree']}")
    lines.append(f"- Average degree: {properties['avg_degree']:.2f}")
    lines.append(f"- Connected: {'Yes' if properties['is_connected'] else 'No'}")
    return "\n".join(lines)

def format_theoretical_insights(properties: Dict) -> str:
    """Format theoretical insights section of the prompt."""
    lines = []
    for insight in properties.get('theoretical_insights', []):
        lines.append(f"- {insight['name']}: {insight['condition']}")
        lines.append(f"  Implication: {insight['implication']}")
    
    if not lines:
        lines = ["No specific theoretical guarantees found."]
    
    return "\n".join(lines)

def format_failure_patterns(patterns: List[Dict]) -> str:
    """Format failure patterns section of the prompt."""
    if not patterns:
        return "No specific failure patterns to avoid."
    
    lines = []
    for pattern in patterns:
        if pattern['failure_type'] == 'trivial_invalid':
            lines.append("- CRITICAL: Simple sequential paths are invalid in this graph!")
            lines.append(f"  Missing edge at position {pattern['failure_point']}")
        elif pattern['failure_type'] == 'missing_edge':
            lines.append(f"- No edge between vertices {pattern['invalid_edge']}")
            lines.append(f"  Context: {pattern['subpath_context']}")
        elif pattern['failure_type'] == 'theoretical_violation':
            lines.append(f"- {pattern['reason']}")
    
    return "\n".join(lines)

def format_implementation_analysis(implementations: List[Dict]) -> str:
    """Format implementation analysis section of the prompt."""
    if not implementations:
        return "No previous successful implementations."
    
    lines = ["Previous Implementation Analysis:"]
    for impl in implementations:
        if not impl['metadata'].get('is_trivial', False):
            lines.append(f"\nImplementation (success_rate={impl['success_rate']:.2f}%, "
                        f"avg_time={impl['avg_computation_time']:.3f}s):")
            lines.append(impl['description'])
            
            if 'complexity_score' in impl['metadata']:
                lines.append(f"Complexity Score: {impl['metadata']['complexity_score']}")
    
    return "\n".join(lines)

def format_subpath_patterns(patterns: List[Dict]) -> str:
    """Format subpath patterns section of the prompt."""
    if not patterns:
        return "No successful subpath patterns identified."
    
    lines = ["Successful Subpath Patterns:"]
    for pattern in patterns:
        if not pattern['metadata'].get('is_sequential', False):
            lines.append(f"- {pattern['description']} "
                        f"(success_rate={pattern['success_rate']:.2f}%, "
                        f"frequency={pattern['frequency']})")
            lines.append(f"  Average degree: {pattern['metadata']['avg_degree']:.1f}")
    
    return "\n".join(lines)
