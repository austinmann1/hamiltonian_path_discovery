"""Generate SATLIB benchmark instances for Hamiltonian path problems.

This module creates DIMACS CNF format files representing Hamiltonian path problems
with known solutions for benchmarking purposes.
"""

import numpy as np
from pathlib import Path
import random
from typing import List, Tuple, Optional

class BenchmarkGenerator:
    def __init__(self, output_dir: str = "benchmarks"):
        """Initialize benchmark generator.
        
        Args:
            output_dir: Directory to store generated benchmark files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_benchmark_set(self,
                             num_instances: int = 10,
                             min_vertices: int = 4,
                             max_vertices: int = 20) -> List[str]:
        """Generate a set of benchmark instances.
        
        Args:
            num_instances: Number of instances to generate
            min_vertices: Minimum number of vertices per instance
            max_vertices: Maximum number of vertices per instance
            
        Returns:
            List of paths to generated benchmark files
        """
        benchmark_files = []
        
        for i in range(num_instances):
            n = random.randint(min_vertices, max_vertices)
            path = self._generate_instance(n, i)
            benchmark_files.append(str(path))
            
        return benchmark_files
        
    def _generate_instance(self, num_vertices: int, instance_id: int) -> Path:
        """Generate a single benchmark instance.
        
        Args:
            num_vertices: Number of vertices in the graph
            instance_id: Unique identifier for this instance
            
        Returns:
            Path to generated benchmark file
        """
        # Generate a random Hamiltonian path
        vertices = list(range(num_vertices))
        random.shuffle(vertices)
        path = vertices  # This is our known solution
        
        # Create adjacency matrix ensuring path is valid
        adj_matrix = np.zeros((num_vertices, num_vertices), dtype=int)
        for i in range(len(path) - 1):
            v1, v2 = path[i], path[i + 1]
            adj_matrix[v1][v2] = 1
            adj_matrix[v2][v1] = 1
        
        # Add some random edges (30% density)
        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):
                if random.random() < 0.3 and adj_matrix[i][j] == 0:
                    adj_matrix[i][j] = 1
                    adj_matrix[j][i] = 1
        
        # Convert to DIMACS CNF format
        clauses = []
        
        # 1. Each position must have exactly one vertex
        for pos in range(num_vertices):
            # At least one vertex in each position
            clause = [self._var(i, pos, num_vertices) for i in range(num_vertices)]
            clauses.append(clause)
            
            # No two vertices in same position
            for i in range(num_vertices):
                for j in range(i + 1, num_vertices):
                    clauses.append([-self._var(i, pos, num_vertices), -self._var(j, pos, num_vertices)])
        
        # 2. Each vertex must appear exactly once
        for v in range(num_vertices):
            # At least one position for each vertex
            clause = [self._var(v, i, num_vertices) for i in range(num_vertices)]
            clauses.append(clause)
            
            # No vertex in two positions
            for i in range(num_vertices):
                for j in range(i + 1, num_vertices):
                    clauses.append([-self._var(v, i, num_vertices), -self._var(v, j, num_vertices)])
        
        # 3. Adjacent vertices must be connected
        for i in range(num_vertices - 1):
            for v1 in range(num_vertices):
                for v2 in range(num_vertices):
                    if adj_matrix[v1][v2] == 0:
                        clauses.append([-self._var(v1, i, num_vertices), -self._var(v2, i + 1, num_vertices)])
        
        # Write to file
        filename = f"ham_path_{num_vertices}v_{instance_id}.cnf"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            # Header
            f.write(f"c SATLIB Hamiltonian Path instance\n")
            f.write(f"c Number of vertices: {num_vertices}\n")
            f.write(f"c Known solution={','.join(map(str, path))}\n")
            f.write(f"p cnf {num_vertices * num_vertices} {len(clauses)}\n")
            
            # Clauses
            for clause in clauses:
                f.write(' '.join(map(str, clause)) + ' 0\n')
                
            # Solution (as variable assignments)
            solution_vars = []
            for pos, vertex in enumerate(path):
                var = self._var(vertex, pos, num_vertices)
                solution_vars.append(str(var))
            f.write('v ' + ' '.join(solution_vars) + '\n')
        
        return filepath
        
    def _var(self, vertex: int, position: int, num_vertices: int) -> int:
        """Convert vertex and position to CNF variable number.
        
        Args:
            vertex: Vertex number (0-based)
            position: Position in path (0-based)
            num_vertices: Total number of vertices
            
        Returns:
            CNF variable number (1-based)
        """
        return vertex * num_vertices + position + 1
