"""Theoretical analysis for Hamiltonian path discovery."""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from collections import defaultdict

class TheoreticalAnalyzer:
    """Analyzes graph properties using theoretical insights."""
    
    def analyze_graph(self, adj_matrix: np.ndarray) -> Dict:
        """Analyze graph properties and theoretical insights."""
        n = adj_matrix.shape[0]
        degrees = np.sum(adj_matrix, axis=1)
        min_degree = np.min(degrees)
        max_degree = np.max(degrees)
        avg_degree = np.mean(degrees)
        density = np.sum(adj_matrix) / (n * (n-1))
        
        # Basic properties
        properties = {
            'size': n,
            'min_degree': min_degree,
            'max_degree': max_degree,
            'avg_degree': avg_degree,
            'density': density,
            'theoretical_insights': []
        }
        
        # Check Dirac's Theorem
        dirac_satisfied = min_degree >= n/2
        properties['theoretical_insights'].append({
            'name': "Dirac's Theorem",
            'condition': 'satisfied' if dirac_satisfied else 'not satisfied',
            'implication': 'Graph must contain a Hamiltonian path' if dirac_satisfied else 'Cannot guarantee Hamiltonian path',
            'confidence': 1.0 if dirac_satisfied else 0.0
        })
        
        # Check Ore's Theorem
        ore_satisfied = True
        for i in range(n):
            for j in range(i+1, n):
                if adj_matrix[i][j] == 0:  # Non-adjacent vertices
                    if degrees[i] + degrees[j] < n:
                        ore_satisfied = False
                        break
            if not ore_satisfied:
                break
        
        properties['theoretical_insights'].append({
            'name': "Ore's Theorem",
            'condition': 'satisfied' if ore_satisfied else 'not satisfied',
            'implication': 'Graph must contain a Hamiltonian cycle' if ore_satisfied else 'Cannot guarantee Hamiltonian cycle',
            'confidence': 1.0 if ore_satisfied else 0.0
        })
        
        # Check Bondy-Chvátal Closure
        closure_matrix = adj_matrix.copy()
        edges_added = []
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i+1, n):
                    if closure_matrix[i][j] == 0:
                        if degrees[i] + degrees[j] >= n:
                            closure_matrix[i][j] = closure_matrix[j][i] = 1
                            edges_added.append((i, j))
                            changed = True
        
        properties['theoretical_insights'].append({
            'name': 'Bondy-Chvátal Closure',
            'condition': 'partial' if edges_added else 'complete',
            'implication': f'Can add {len(edges_added)} edges via closure',
            'edges': edges_added,
            'confidence': 0.8 if edges_added else 1.0
        })
        
        # Check for claw-free property
        claws = []
        for v in range(n):
            neighbors = np.where(adj_matrix[v] == 1)[0]
            if len(neighbors) >= 3:
                for i in range(len(neighbors)-2):
                    for j in range(i+1, len(neighbors)-1):
                        for k in range(j+1, len(neighbors)):
                            n1, n2, n3 = neighbors[i], neighbors[j], neighbors[k]
                            if (adj_matrix[n1][n2] == 0 and 
                                adj_matrix[n2][n3] == 0 and 
                                adj_matrix[n1][n3] == 0):
                                claws.append({
                                    'center': v,
                                    'leaves': [int(n1), int(n2), int(n3)]
                                })
        
        properties['forbidden_patterns'] = [{
            'type': 'claw_subgraph',
            'description': f'Found {len(claws)} claw subgraphs',
            'claws': claws,
            'theorem': 'Claw-Free Property'
        }]
        
        # Set guaranteed properties based on theoretical results
        properties['guaranteed_properties'] = []
        if dirac_satisfied or ore_satisfied:
            properties['guaranteed_properties'].extend(['hamiltonian', 'hamiltonian_cycle'])
        
        # Check connectivity (basic version)
        visited = np.zeros(n, dtype=bool)
        def dfs(v):
            visited[v] = True
            for u in range(n):
                if adj_matrix[v][u] and not visited[u]:
                    dfs(u)
        dfs(0)
        properties['connectivity'] = 1 if np.all(visited) else 0
        
        return properties
    
    def format_insights_for_prompt(self, insights: Dict) -> str:
        """Format theoretical insights for inclusion in prompts."""
        lines = ["Theoretical Analysis:"]
        
        # Add guaranteed properties
        if insights.get('guaranteed_properties'):
            lines.append("\nGuaranteed Properties:")
            for prop in insights['guaranteed_properties']:
                lines.append(f"- {prop}")
        
        # Add theoretical insights
        if insights.get('theoretical_insights'):
            lines.append("\nTheoretical Insights:")
            for insight in insights['theoretical_insights']:
                lines.append(f"- {insight['name']}: {insight['condition']}")
                lines.append(f"  {insight['implication']}")
        
        # Add forbidden patterns
        if insights.get('forbidden_patterns'):
            lines.append("\nForbidden Patterns:")
            for pattern in insights['forbidden_patterns']:
                lines.append(f"- {pattern['type']}: {pattern['description']}")
                if 'theorem' in pattern:
                    lines.append(f"  (Based on {pattern['theorem']})")
        
        # Add connectivity information
        if 'connectivity' in insights:
            lines.append(f"\nConnectivity: {insights['connectivity']}-connected")
        
        return "\n".join(lines)
