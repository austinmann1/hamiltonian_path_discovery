"""
Example script demonstrating the prompt engineering system.
"""

import os
from pathlib import Path
import numpy as np
from src.prompting.prompt_manager import PromptManager, PromptTemplate
from src.llm_interface_openrouter import OpenRouterLLM
from src.graph_generator.graph_utils import GraphUtils

def create_custom_template():
    """Create a custom prompt template."""
    return PromptTemplate(
        name="optimized",
        template="""You are an expert in graph theory tasked with finding a Hamiltonian path.

Graph Analysis:
- Nodes: {num_nodes}
- Adjacency Matrix:
{adjacency_matrix}
- Start Node: {start_node}
- End Node: {end_node}
- Degree Analysis: {degree_analysis}

Key Constraints:
1. Path MUST start at node {start_node}
2. Path MUST end at node {end_node}
3. Each node MUST be visited exactly once
4. Only use edges where adjacency_matrix[i][j] = 1

Strategy:
1. First, verify the path is possible:
   - Check if start/end nodes have sufficient degree
   - Ensure no node has degree < 2 (except possibly start/end)
   
2. Build the path:
   - Start at node {start_node}
   - At each step, choose the neighbor with fewest remaining options
   - Backtrack if no valid neighbors exist
   - Continue until reaching node {end_node} with all nodes visited

CRITICAL: Return ONLY a Python list of integers representing the path, e.g. [0, 2, 1, 3].
- The list MUST start with {start_node} and end with {end_node}
- Each number MUST be between 0 and {num_nodes}-1
- Do not include ANY other text or explanation
- If no valid path exists, return []
""",
        description="Optimized prompt with degree analysis and strategic guidance",
        parameters={
            "num_nodes": "Number of nodes in the graph",
            "adjacency_matrix": "Adjacency matrix as a string",
            "start_node": "Starting node index",
            "end_node": "Ending node index",
            "degree_analysis": "Analysis of node degrees"
        },
        examples=[],
        metrics={
            "success_rate": 0.0,
            "average_time": 0.0,
            "average_energy": 0.0
        }
    )

def analyze_degrees(matrix):
    """Analyze node degrees in the graph."""
    degrees = np.sum(matrix, axis=1)
    return {
        "degrees": degrees.tolist(),
        "min_degree": int(min(degrees)),
        "max_degree": int(max(degrees)),
        "isolated_nodes": [i for i, d in enumerate(degrees) if d == 0],
        "leaf_nodes": [i for i, d in enumerate(degrees) if d == 1]
    }

def main():
    # Initialize components
    data_dir = Path(os.path.dirname(__file__)) / ".." / "data"
    prompt_manager = PromptManager(str(data_dir))
    llm = OpenRouterLLM(timeout=10.0)  # 10 second timeout
    graph_utils = GraphUtils()
    
    # Create a custom template
    template = create_custom_template()
    try:
        prompt_manager.create_template(template)
        print(f"Created template: {template.name}")
    except ValueError:
        print(f"Template {template.name} already exists")
    
    # List available templates
    print("\nAvailable templates:")
    for name in prompt_manager.list_templates():
        print(f"- {name}")
    
    # Generate test graphs
    print("\nTesting templates on random graphs...")
    for size in [5, 7, 10]:
        print(f"\nTesting graph with {size} nodes:")
        
        # Generate random graph
        matrix = graph_utils.generate_random_hamiltonian_graph(size)
        start_node = 0
        end_node = size - 1
        
        print(f"\nGenerated graph:")
        print(f"Adjacency matrix:\n{matrix}")
        print(f"Start node: {start_node}")
        print(f"End node: {end_node}")
        
        # Prepare parameters
        params = {
            "num_nodes": size,
            "adjacency_matrix": "\n".join(str(row) for row in matrix.tolist()),
            "start_node": start_node,
            "end_node": end_node,
            "degree_analysis": str(analyze_degrees(matrix)),
            "properties": str({
                "is_directed": False,
                "has_self_loops": False,
                "density": np.sum(matrix) / (size * size),
                "min_degree": np.min(np.sum(matrix, axis=1)),
                "max_degree": np.max(np.sum(matrix, axis=1))
            }),
            "previous_attempts": "[]"  # No previous attempts
        }
        
        # Test each template
        for template_name in prompt_manager.list_templates():
            print(f"\nTesting template: {template_name}")
            
            # Execute prompt
            try:
                result = prompt_manager.execute_prompt(
                    template_name,
                    params,
                    llm
                )
                
                # Print results
                print(f"Success: {result.success}")
                if result.success:
                    print(f"Path found: {result.generated_code}")
                else:
                    print(f"Error: {result.error}")
                
                print(f"Time: {result.execution_time:.2f}s")
                if result.energy_metrics:
                    print(
                        f"Energy: {result.energy_metrics.get('total_energy_joules', 0):.2f} joules"
                    )
            except Exception as e:
                print(f"Error executing prompt: {str(e)}")
                continue
            
            # Get template performance
            print("\nTemplate Performance:")
            for template_name in prompt_manager.list_templates():
                perf = prompt_manager.get_template_performance(template_name)
                if perf:
                    print(f"\n{template_name}:")
                    print(f"- Success Rate: {perf['success_rate']:.2%}")
                    print(f"- Average Time: {perf['average_time']:.2f}s")
                    print(f"- Average Energy: {perf['average_energy']:.2f} joules")

if __name__ == "__main__":
    main()
