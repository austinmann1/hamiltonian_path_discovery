"""
Converts SAT instances to Hamiltonian path problems.
Based on the reduction from SAT to Hamiltonian Path.
"""

import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Set
from pathlib import Path
import json
from ..logging import StructuredLogger

class SATtoHamiltonianConverter:
    """
    Converts SAT instances to Hamiltonian path problems using the
    polynomial-time reduction from SAT to Hamiltonian Path.
    """
    
    def __init__(self):
        self.logger = StructuredLogger()
    
    def parse_cnf_file(self, file_path: str) -> Tuple[int, List[List[int]]]:
        """
        Parse a DIMACS CNF file.
        
        Args:
            file_path: Path to the CNF file
            
        Returns:
            Tuple of (num_variables, clauses)
        """
        num_vars = 0
        clauses = []
        
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('c'):  # Comment
                    continue
                if line.startswith('p'):  # Problem line
                    _, _, num_vars_str, _ = line.split()
                    num_vars = int(num_vars_str)
                    continue
                
                # Parse clause
                clause = [int(x) for x in line.split()[:-1]]  # Remove trailing 0
                if clause:  # Skip empty lines
                    clauses.append(clause)
        
        return num_vars, clauses
    
    def create_variable_gadget(self, G: nx.DiGraph, var_idx: int, pos_x: int) -> Tuple[int, int]:
        """
        Create a variable gadget in the graph.
        
        Args:
            G: NetworkX directed graph
            var_idx: Variable index
            pos_x: X position for visualization
            
        Returns:
            Tuple of (true_node, false_node)
        """
        # Create variable choice nodes
        true_node = f"var_{var_idx}_true"
        false_node = f"var_{var_idx}_false"
        
        # Add nodes with positions for visualization
        G.add_node(true_node, pos=(pos_x, 1))
        G.add_node(false_node, pos=(pos_x, -1))
        
        # Add edges to force choice
        G.add_edge(true_node, false_node)
        G.add_edge(false_node, true_node)
        
        return true_node, false_node
    
    def create_clause_gadget(
        self, 
        G: nx.DiGraph, 
        clause_idx: int,
        clause: List[int],
        var_nodes: Dict[int, Tuple[str, str]],
        pos_x: int
    ) -> str:
        """
        Create a clause gadget in the graph.
        
        Args:
            G: NetworkX directed graph
            clause_idx: Clause index
            clause: List of literals in the clause
            var_nodes: Dictionary mapping variable indices to their nodes
            pos_x: X position for visualization
            
        Returns:
            Name of the clause verification node
        """
        clause_node = f"clause_{clause_idx}"
        G.add_node(clause_node, pos=(pos_x, 0))
        
        # Connect to satisfying variable assignments
        for literal in clause:
            var_idx = abs(literal)
            true_node, false_node = var_nodes[var_idx]
            
            # Connect to appropriate node based on whether literal is negated
            if literal > 0:
                G.add_edge(true_node, clause_node)
            else:
                G.add_edge(false_node, clause_node)
        
        return clause_node
    
    def sat_to_graph(self, num_vars: int, clauses: List[List[int]]) -> nx.DiGraph:
        """
        Convert SAT instance to a directed graph for Hamiltonian path.
        
        Args:
            num_vars: Number of variables
            clauses: List of clauses (each clause is a list of literals)
            
        Returns:
            NetworkX directed graph
        """
        G = nx.DiGraph()
        
        # Create start and end nodes
        G.add_node("start", pos=(0, 0))
        G.add_node("end", pos=(num_vars + len(clauses) + 1, 0))
        
        # Create variable gadgets
        var_nodes = {}
        for i in range(1, num_vars + 1):
            var_nodes[i] = self.create_variable_gadget(G, i, i)
        
        # Create clause gadgets
        clause_nodes = []
        for i, clause in enumerate(clauses):
            clause_node = self.create_clause_gadget(
                G, i, clause, var_nodes,
                num_vars + i + 1
            )
            clause_nodes.append(clause_node)
        
        # Connect start to first variable layer
        for i in range(1, num_vars + 1):
            true_node, false_node = var_nodes[i]
            G.add_edge("start", true_node)
            G.add_edge("start", false_node)
        
        # Connect last clause to end
        for node in clause_nodes:
            G.add_edge(node, "end")
        
        return G
    
    def graph_to_matrix(self, G: nx.DiGraph) -> Tuple[np.ndarray, Dict[int, str]]:
        """
        Convert NetworkX graph to adjacency matrix format.
        
        Args:
            G: NetworkX directed graph
            
        Returns:
            Tuple of (adjacency matrix, node mapping)
        """
        # Create node mapping
        nodes = list(G.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        idx_to_node = {i: node for node, i in node_to_idx.items()}
        
        # Create adjacency matrix
        n = len(nodes)
        matrix = np.zeros((n, n), dtype=int)
        
        for u, v in G.edges():
            i, j = node_to_idx[u], node_to_idx[v]
            matrix[i, j] = 1
        
        return matrix, idx_to_node
    
    def convert_file(self, cnf_file: str, output_dir: Path) -> Dict:
        """
        Convert a CNF file to a Hamiltonian path instance.
        
        Args:
            cnf_file: Path to CNF file
            output_dir: Directory to save output files
            
        Returns:
            Dictionary with problem information
        """
        # Parse CNF file
        num_vars, clauses = self.parse_cnf_file(cnf_file)
        
        # Create graph
        G = self.sat_to_graph(num_vars, clauses)
        
        # Convert to matrix format
        matrix, node_mapping = self.graph_to_matrix(G)
        
        # Save files
        base_name = Path(cnf_file).stem
        matrix_file = output_dir / f"{base_name}_matrix.npy"
        mapping_file = output_dir / f"{base_name}_mapping.json"
        graph_file = output_dir / f"{base_name}_graph.gexf"
        
        # Save files
        np.save(matrix_file, matrix)
        nx.write_gexf(G, graph_file)
        with open(mapping_file, 'w') as f:
            json.dump(node_mapping, f, indent=2)
        
        # Log the conversion
        self.logger.log_metrics("conversion", {
            "num_variables": num_vars,
            "num_clauses": len(clauses),
            "graph_nodes": len(G),
            "graph_edges": len(G.edges())
        })
        
        return {
            "original_file": cnf_file,
            "matrix_file": str(matrix_file),
            "mapping_file": str(mapping_file),
            "graph_file": str(graph_file),
            "num_variables": num_vars,
            "num_clauses": len(clauses),
            "graph_size": len(G)
        }
