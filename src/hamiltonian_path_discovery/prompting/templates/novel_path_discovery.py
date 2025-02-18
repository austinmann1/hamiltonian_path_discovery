"""Templates for novel Hamiltonian path discovery strategies."""

from typing import Dict

NOVEL_PATH_TEMPLATE = {
    "name": "novel_path_discovery",
    "template": """You are an AI researcher exploring novel approaches to solving Hamiltonian path problems. Your goal is to discover new patterns and strategies that could lead to breakthrough algorithms.

Graph Description:
- Adjacency Matrix:
{adjacency_matrix}
- Number of Nodes: {num_nodes}
- Start Node: {start_node}
- End Node: {end_node}

{conflict_learning}

Analysis Instructions:
1. Study the conflict patterns above carefully. They represent paths and choices that have been proven not to work.
2. Look for structural insights:
   - Are there vertices that frequently lead to dead ends?
   - Are there edge patterns that consistently fail?
   - What do the successful partial paths have in common?
3. Consider these strategies:
   - Avoid paths that share characteristics with known failures
   - Look for alternative routes around problematic vertices
   - Try to understand why certain edges are consistently problematic
   - Consider the graph's degree distribution and connectivity patterns

Strategy Development:
1. Based on the conflict patterns, identify:
   - Safe vertices (rarely involved in conflicts)
   - Risky vertices (frequently lead to dead ends)
   - Reliable subpaths (never appear in conflict logs)
2. Develop a novel approach that:
   - Actively avoids known problematic patterns
   - Prefers paths through safe vertices
   - Has a backup strategy when approaching risky areas
3. If you identify any new patterns or insights about why certain paths fail,
   explain them briefly before providing your solution.

Your response should include:
1. A brief explanation of any patterns or insights you've identified
2. Your strategy for finding a valid path while avoiding known conflicts
3. The actual path as a list of integers, e.g. [0, 1, 2, 3]

Remember:
- Previous conflicts are valuable learning opportunities
- The goal is to discover new algorithmic insights
- Consider both local (vertex-level) and global (path structure) patterns
- Failed attempts can reveal important structural properties of the graph""",
    
    "parameters": {
        "adjacency_matrix": "string representation of the graph's adjacency matrix",
        "num_nodes": "integer number of nodes in the graph",
        "start_node": "integer starting node for the path",
        "end_node": "integer ending node for the path",
        "conflict_learning": "string containing learned conflicts and patterns"
    },
    
    "description": """
    A template designed to encourage the discovery of novel algorithmic approaches
    to solving Hamiltonian path problems. It leverages conflict learning to help
    the LLM avoid known pitfalls and discover new strategies.
    """,
    
    "examples": [
        {
            "input": {
                "adjacency_matrix": "0 1 0 1\n1 0 1 0\n0 1 0 1\n1 0 1 0",
                "num_nodes": 4,
                "start_node": 0,
                "end_node": 3,
                "conflict_learning": """Based on previous attempts, avoid these patterns:
- Edge (0->2) failed 3 times
- Vertex 1 involved in 4 dead ends
- Avoid sequence [0,1,2]: leads to no valid solution"""
            },
            "output": """Pattern Analysis:
1. Noticed that vertex 1 often leads to dead ends when visited early
2. Edge (0->2) consistently fails, suggesting we need a different approach
3. The sequence [0,1,2] is problematic, likely due to limited options afterward

Strategy:
Will try a path that delays visiting vertex 1 until necessary, using alternative routes first.

Path: [0, 3, 2, 1]""",
            "explanation": """
            This example shows how to:
            1. Analyze conflict patterns
            2. Develop a strategy that avoids known issues
            3. Find a novel path that doesn't repeat previous mistakes
            """
        }
    ],
    
    "metrics": {
        "success_rate": 0.0,
        "average_attempts": 0.0,
        "average_energy": 0.0,
        "novel_patterns_discovered": 0.0,
        "conflicts_avoided": 0.0
    }
}

def get_novel_path_template() -> Dict:
    """Get the template for novel path discovery."""
    return NOVEL_PATH_TEMPLATE
