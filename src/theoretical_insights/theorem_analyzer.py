"""
Theoretical insights system for Hamiltonian path discovery.
Analyzes graph properties and applies theoretical results to guide path finding.
"""

from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import networkx as nx

@dataclass
class TheoremInsight:
    """Represents a theoretical insight about a graph."""
    theorem_name: str
    applies: bool
    conditions: List[str]
    implications: List[str]
    guidance: str

class TheoremAnalyzer:
    """
    Analyzes graphs using theoretical results to guide path finding.
    Implements checks for various theorems and conditions.
    """
    
    def __init__(self):
        """Initialize the theorem analyzer."""
        self.insights: Dict[str, List[TheoremInsight]] = {}
    
    def analyze_graph(self, adj_matrix: np.ndarray) -> List[TheoremInsight]:
        """
        Analyze a graph using various theoretical results.
        
        Args:
            adj_matrix: The graph's adjacency matrix
            
        Returns:
            List of applicable theoretical insights
        """
        insights = []
        
        # Handle empty or invalid graphs
        if adj_matrix.size == 0 or not np.any(adj_matrix):
            return []
            
        # Convert to NetworkX graph
        G = nx.Graph(adj_matrix)
        n = len(adj_matrix)
        
        # Check Dirac's theorem conditions
        min_degree = min(dict(G.degree()).values())
        if min_degree >= n/2:
            insights.append(TheoremInsight(
                theorem_name="Dirac's Theorem",
                applies=True,
                conditions=[f"Minimum degree ({min_degree}) ≥ n/2 ({n/2})"],
                implications=["Graph contains a Hamiltonian cycle"],
                guidance="Start with any vertex - degree conditions guarantee success"
            ))
        
        # Check Ore's theorem conditions
        ore_condition = True
        for u in G.nodes():
            for v in G.nodes():
                if u != v and not G.has_edge(u, v):
                    if G.degree(u) + G.degree(v) < n:
                        ore_condition = False
                        break
            if not ore_condition:
                break
        
        if ore_condition:
            insights.append(TheoremInsight(
                theorem_name="Ore's Theorem",
                applies=True,
                conditions=["For all non-adjacent vertices u,v: deg(u) + deg(v) ≥ n"],
                implications=["Graph contains a Hamiltonian cycle"],
                guidance="Any vertex ordering avoiding non-edges will succeed"
            ))
        
        # Check connectivity conditions
        if nx.is_connected(G):
            try:
                connectivity = nx.node_connectivity(G)
                if connectivity >= n/2:
                    insights.append(TheoremInsight(
                        theorem_name="Chvátal-Erdős Theorem",
                        applies=True,
                        conditions=[
                            f"Graph is {connectivity}-connected",
                            f"Connectivity ({connectivity}) ≥ n/2 ({n/2})"
                        ],
                        implications=["Graph contains a Hamiltonian cycle"],
                        guidance="High connectivity ensures multiple valid paths"
                    ))
            except nx.NetworkXError:
                pass  # Skip if connectivity calculation fails
        
        # Check planarity and face structure for Grinberg's theorem
        try:
            if nx.check_planarity(G)[0]:
                insights.append(TheoremInsight(
                    theorem_name="Grinberg's Theorem",
                    applies=True,
                    conditions=["Graph is planar"],
                    implications=["Face sizes constrain possible Hamiltonian cycles"],
                    guidance="Consider face structure when selecting path"
                ))
        except nx.NetworkXError:
            pass  # Skip if planarity check fails
        
        # Check biconnectivity for Tutte's theorem application
        try:
            if nx.is_biconnected(G):
                insights.append(TheoremInsight(
                    theorem_name="Tutte's Theorem",
                    applies=True,
                    conditions=["Graph is biconnected"],
                    implications=["4-connected planar graphs are Hamiltonian"],
                    guidance="Use biconnected structure to guide path finding"
                ))
        except nx.NetworkXError:
            pass  # Skip if biconnectivity check fails
        
        return insights
    
    def get_recommendations(self, adj_matrix: np.ndarray) -> Dict[str, List[str]]:
        """
        Get recommendations based on theoretical insights.
        
        Args:
            adj_matrix: The graph's adjacency matrix
            
        Returns:
            Dictionary of recommendations by category
        """
        insights = self.analyze_graph(adj_matrix)
        recommendations = {
            "starting_vertices": [],
            "path_constraints": [],
            "search_strategy": []
        }
        
        for insight in insights:
            if insight.theorem_name == "Dirac's Theorem":
                recommendations["search_strategy"].append(
                    "Any vertex ordering that respects edge constraints will work"
                )
            elif insight.theorem_name == "Ore's Theorem":
                recommendations["path_constraints"].append(
                    "Avoid paths between non-adjacent vertices with low degree sum"
                )
            elif insight.theorem_name == "Chvátal-Erdős Theorem":
                recommendations["search_strategy"].append(
                    "Multiple path options exist - use backtracking freely"
                )
            elif insight.theorem_name == "Grinberg's Theorem":
                recommendations["path_constraints"].append(
                    "Consider face sizes when selecting next vertex"
                )
            elif insight.theorem_name == "Tutte's Theorem":
                recommendations["search_strategy"].append(
                    "Use biconnected components to structure search"
                )
        
        return recommendations
    
    def format_for_prompt(self, adj_matrix: np.ndarray) -> str:
        """
        Format theoretical insights for use in prompts.
        
        Args:
            adj_matrix: The graph's adjacency matrix
            
        Returns:
            Formatted string of theoretical insights and recommendations
        """
        insights = self.analyze_graph(adj_matrix)
        recommendations = self.get_recommendations(adj_matrix)
        
        sections = ["Theoretical Insights:"]
        
        if insights:
            for insight in insights:
                sections.append(f"\n{insight.theorem_name}:")
                sections.append("Conditions met:")
                sections.extend(f"- {cond}" for cond in insight.conditions)
                sections.append("Implications:")
                sections.extend(f"- {imp}" for imp in insight.implications)
                sections.append(f"Guidance: {insight.guidance}")
        
        sections.append("\nRecommendations:")
        for category, recs in recommendations.items():
            if recs:
                sections.append(f"\n{category.replace('_', ' ').title()}:")
                sections.extend(f"- {rec}" for rec in recs)
        
        return "\n".join(sections)
