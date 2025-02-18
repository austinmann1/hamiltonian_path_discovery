"""
Graph Generator module for converting SAT instances to Hamiltonian Path problems.
"""
import networkx as nx
import numpy as np
from typing import List, Tuple, Dict

class GraphGenerator:
    def __init__(self):
        self.graph = nx.DiGraph()
        
    def sat_to_hamiltonian(self, cnf: List[List[int]]) -> nx.DiGraph:
        """
        Convert a SAT formula in CNF to a directed graph for Hamiltonian path finding.
        Based on the reduction described in the Cook-Levin theorem.
        
        Args:
            cnf: List of clauses, where each clause is a list of integers.
                 Positive integers represent positive literals, negative integers
                 represent negated literals.
                 
        Returns:
            nx.DiGraph: A directed graph where a Hamiltonian path exists iff the
                       original SAT formula is satisfiable.
        """
        # Reset graph
        self.graph.clear()
        
        # Get number of variables and clauses
        variables = set(abs(lit) for clause in cnf for lit in clause)
        num_vars = len(variables)
        num_clauses = len(cnf)
        
        # Create variable selection gadgets
        prev_var = None
        for var in sorted(variables):
            self._create_variable_gadget(var)
            if prev_var:
                # Connect previous variable's exit to current variable's entry
                self.graph.add_edge(f'var_{prev_var}_exit', f'var_{var}_entry')
            prev_var = var
            
        # Create clause satisfaction verification gadgets
        for i, clause in enumerate(cnf):
            self._create_clause_gadget(clause, i)
            if i > 0:
                # Connect previous clause's exit to current clause's entry
                self.graph.add_edge(f'clause_{i-1}_exit', f'clause_{i}_entry')
                
        # Add start and end nodes
        self.graph.add_node('start')
        self.graph.add_node('end')
        
        # Connect start to first variable gadget
        min_var = min(variables)
        self.graph.add_edge('start', f'var_{min_var}_entry')
        
        # Connect last variable to first clause
        if num_clauses > 0:
            self.graph.add_edge(f'var_{max(variables)}_exit', 'clause_0_entry')
            # Connect last clause to end
            self.graph.add_edge(f'clause_{num_clauses-1}_exit', 'end')
        else:
            # If no clauses, connect last variable directly to end
            self.graph.add_edge(f'var_{max(variables)}_exit', 'end')
        
        return self.graph
    
    def _create_variable_gadget(self, var: int) -> None:
        """Create a variable gadget that forces choosing true/false assignment."""
        # Create entry and exit nodes
        self.graph.add_node(f'var_{var}_entry')
        self.graph.add_node(f'var_{var}_exit')
        
        # Create true/false path nodes
        self.graph.add_node(f'var_{var}_true')
        self.graph.add_node(f'var_{var}_false')
        
        # Add edges to form the gadget
        self.graph.add_edge(f'var_{var}_entry', f'var_{var}_true')
        self.graph.add_edge(f'var_{var}_entry', f'var_{var}_false')
        self.graph.add_edge(f'var_{var}_true', f'var_{var}_exit')
        self.graph.add_edge(f'var_{var}_false', f'var_{var}_exit')
        
    def _create_clause_gadget(self, clause: List[int], clause_idx: int) -> None:
        """Create a clause satisfaction verification gadget."""
        # Create entry and exit nodes for the clause
        self.graph.add_node(f'clause_{clause_idx}_entry')
        self.graph.add_node(f'clause_{clause_idx}_exit')
        
        # Create intermediate nodes for each literal
        for i, lit in enumerate(clause):
            var = abs(lit)
            # Create intermediate node for this literal
            node_name = f'clause_{clause_idx}_lit_{i}'
            self.graph.add_node(node_name)
            
            # Connect to entry
            self.graph.add_edge(f'clause_{clause_idx}_entry', node_name)
            
            # Connect to corresponding variable node
            target = f'var_{var}_{"true" if lit > 0 else "false"}'
            self.graph.add_edge(node_name, target)
            self.graph.add_edge(target, f'clause_{clause_idx}_exit')
            
    def get_adjacency_matrix(self) -> np.ndarray:
        """Convert the graph to an adjacency matrix representation."""
        return nx.to_numpy_array(self.graph)
    
    def verify_hamiltonian_path(self, path: List[str]) -> bool:
        """
        Verify if a given path is a valid Hamiltonian path in the graph.
        
        Args:
            path: List of node names representing the proposed path
            
        Returns:
            bool: True if path is a valid Hamiltonian path, False otherwise
        """
        if len(path) != len(self.graph):
            return False
            
        # Check if all nodes are unique
        if len(set(path)) != len(path):
            return False
            
        # Check if all edges exist
        for i in range(len(path)-1):
            if not self.graph.has_edge(path[i], path[i+1]):
                return False
                
        return True

def load_satlib_instance(filepath: str) -> List[List[int]]:
    """
    Load a SATLIB instance in DIMACS format.
    
    Args:
        filepath: Path to the DIMACS format CNF file
        
    Returns:
        List[List[int]]: CNF formula as a list of clauses
    """
    cnf = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('c') or line.startswith('p'):
                continue
            clause = [int(x) for x in line.strip().split()[:-1]]
            if clause:  # Skip empty lines
                cnf.append(clause)
    return cnf
