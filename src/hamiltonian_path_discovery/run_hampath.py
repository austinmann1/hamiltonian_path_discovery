"""Main script to run Hamiltonian path discovery with conflict learning."""

import sys
from pathlib import Path
import numpy as np

# Add src directory to Python path
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

from prompting.prompt_manager import PromptManager
from llm_interface_openrouter import OpenRouterLLM
from graph_generator import GraphGenerator

def main():
    # Initialize components
    prompt_manager = PromptManager()
    llm = OpenRouterLLM()
    graph_gen = GraphGenerator()
    
    # Generate a test graph
    n_vertices = 6  # Start with a small graph
    density = 0.4   # Moderate density
    graph = graph_gen.generate_graph(n_vertices, density)
    
    # Set up parameters for the prompt
    params = {
        "adjacency_matrix": str(graph.tolist()),
        "num_nodes": n_vertices,
        "start_node": 0,
        "end_node": n_vertices - 1
    }
    
    print("Starting Hamiltonian path search...")
    print(f"Graph size: {n_vertices} vertices")
    print(f"Looking for path from node 0 to node {n_vertices - 1}")
    
    # Try to find a path with multiple attempts
    result = prompt_manager.execute_prompt(
        template_name="novel_path_discovery",
        params=params,
        llm_interface=llm,
        max_attempts=5
    )
    
    if result.success:
        print("\nSuccess! Found Hamiltonian path:")
        print(f"Path: {result.path}")
        print(f"Energy used: {result.energy_used:.2f} joules")
    else:
        print("\nFailed to find Hamiltonian path")
        print(f"Error: {result.error}")
        print(f"Energy used: {result.energy_used:.2f} joules")
    
    # Print learned patterns
    print("\nLearned Patterns:")
    print(prompt_manager.conflict_tracker.format_for_prompt())

if __name__ == "__main__":
    main()
