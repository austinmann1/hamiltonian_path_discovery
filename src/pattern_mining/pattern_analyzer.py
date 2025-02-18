"""
Pattern mining system for Hamiltonian path discovery.
Analyzes successful solutions to identify effective patterns and strategies.
"""

from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from dataclasses import dataclass, asdict, field
import json
import os
from datetime import datetime
import time
from collections import defaultdict

@dataclass
class PathPattern:
    """Represents a pattern found in successful paths."""
    pattern_type: str  # "code", "vertex", "subpath", "failure"
    pattern: List[int]
    frequency: int
    success_rate: float
    avg_computation_time: float
    description: str
    metadata: Dict = field(default_factory=dict)  # Additional pattern-specific data
    
    def to_dict(self):
        """Convert pattern to dictionary for serialization."""
        data = asdict(self)
        # Convert numpy types to Python native types
        if isinstance(data['pattern'], np.ndarray):
            data['pattern'] = data['pattern'].tolist()
        data['pattern'] = [int(x) if isinstance(x, np.integer) else x for x in data['pattern']]
        data['frequency'] = int(data['frequency']) if isinstance(data['frequency'], np.integer) else data['frequency']
        data['success_rate'] = float(data['success_rate']) if isinstance(data['success_rate'], np.floating) else data['success_rate']
        data['avg_computation_time'] = float(data['avg_computation_time']) if isinstance(data['avg_computation_time'], np.floating) else data['avg_computation_time']
        return data
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create pattern from dictionary."""
        return cls(**data)
        
    def format_metrics(self) -> str:
        """Format pattern metrics for prompt inclusion."""
        return (f"success_rate={self.success_rate:.1%}, "
                f"frequency={self.frequency}, "
                f"avg_time={self.avg_computation_time:.3f}s")

class PatternAnalyzer:
    """
    Analyzes successful Hamiltonian paths to identify effective patterns.
    Tracks pattern frequency, success rates, and computation times.
    """
    
    def __init__(self):
        """Initialize the pattern analyzer."""
        self.patterns = defaultdict(dict)
        self.successful_patterns = []
        self.failure_patterns = []
        self.subpath_patterns = []
        self.pattern_attempts: Dict[str, int] = {}
        self.pattern_successes: Dict[str, int] = {}
        self.pattern_times: Dict[str, List[float]] = {}
        self.stats = {
            'total_attempts': 0,
            'successful_attempts': 0,
            'start_time': datetime.now().isoformat()
        }
        
        # Create results directory if it doesn't exist
        self.results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'results'
        )
        os.makedirs(self.results_dir, exist_ok=True)
    
    def save_state(self, filename: Optional[str] = None) -> None:
        """Save the current state to a file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"pattern_state_{timestamp}.json"
            
        state = {
            "patterns": {k: v.to_dict() for k, v in self.patterns.items()},
            "pattern_attempts": self.pattern_attempts,
            "pattern_successes": self.pattern_successes,
            "pattern_times": self.pattern_times,
            "stats": self.stats
        }
        
        # Create results directory if it doesn't exist
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save to timestamped file
        filepath = os.path.join(results_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
            
        # Also save to latest.json
        latest_path = os.path.join(results_dir, "pattern_state_latest.json")
        with open(latest_path, 'w') as f:
            json.dump(state, f, indent=2)
            
        print(f"\nSaved pattern state to: {filepath}")
        
    def serialize_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return obj
    
    @classmethod
    def load_state(cls, filename: str) -> 'PatternAnalyzer':
        """Load state from file."""
        analyzer = cls()
        
        with open(filename, 'r') as f:
            state = json.load(f)
        
        analyzer.patterns = {
            k: PathPattern.from_dict(v) for k, v in state['patterns'].items()
        }
        analyzer.pattern_attempts = state['pattern_attempts']
        analyzer.pattern_successes = state['pattern_successes']
        analyzer.pattern_times = state['pattern_times']
        analyzer.stats = state['stats']
        
        return analyzer
    
    @classmethod
    def load_latest(cls) -> 'PatternAnalyzer':
        """Load the latest saved state."""
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'results'
        )
        latest_file = os.path.join(results_dir, 'pattern_state_latest.json')
        
        if os.path.exists(latest_file):
            return cls.load_state(latest_file)
        return cls()

    def analyze_solution(self, path: List[int], adj_matrix: np.ndarray, 
                        computation_time: float, success: bool, code: Optional[str] = None,
                        failure_info: Optional[Dict] = None):
        """
        Analyze a solution attempt to identify patterns.
        
        Args:
            path: Attempted Hamiltonian path
            adj_matrix: Graph adjacency matrix
            computation_time: Time taken for attempt
            success: Whether attempt was successful
            code: The Python code implementation if successful
            failure_info: Dictionary containing failure details if unsuccessful
        """
        # Track overall stats
        self.stats['total_attempts'] += 1
        if success:
            self.stats['successful_attempts'] += 1
        
        # Check for trivial solution
        is_trivial = False
        if path is not None:
            from solution_validator import is_trivial_solution
            is_trivial = is_trivial_solution(path)
            if is_trivial:
                self.stats['trivial_attempts'] = self.stats.get('trivial_attempts', 0) + 1
        
        # Store failure patterns for learning
        if not success and failure_info:
            failure_pattern_id = f"failure_{len(self.failure_patterns)}"
            failure_pattern = PathPattern(
                pattern_type="failure",
                pattern=[],
                frequency=1,
                success_rate=0.0,
                avg_computation_time=computation_time,
                description=str(failure_info.get('reason', 'Unknown failure')),
                metadata={
                    'code': code,
                    'graph_size': len(adj_matrix),
                    'graph_density': np.sum(adj_matrix) / (len(adj_matrix) * (len(adj_matrix) - 1)),
                    'error_type': failure_info.get('type', 'unknown'),
                    'error_location': failure_info.get('location', 'unknown')
                }
            )
            self.failure_patterns.append(failure_pattern)
            
            # Update failure stats
            self.stats['failure_patterns'] = self.stats.get('failure_patterns', 0) + 1
            failure_type = failure_info.get('type', 'unknown')
            self.stats[f'failures_{failure_type}'] = self.stats.get(f'failures_{failure_type}', 0) + 1
        
        # If successful and not trivial, store the code pattern
        if success and code and not is_trivial:
            code_pattern_id = f"code_{len(self.patterns)}"
            if code_pattern_id not in self.patterns:
                self.patterns[code_pattern_id] = PathPattern(
                    pattern_type="code",
                    pattern=[],  # Empty for code patterns
                    frequency=0,
                    success_rate=0.0,
                    avg_computation_time=0.0,
                    description=code,  # Store the actual code
                    metadata={
                        "graph_size": len(adj_matrix),
                        "graph_density": np.sum(adj_matrix) / (len(adj_matrix) * (len(adj_matrix) - 1)),
                        "is_trivial": is_trivial,
                        "path_length": len(path) if path else 0,
                        "performance_score": self._calculate_performance_score(computation_time, len(adj_matrix))
                    }
                )
                self.pattern_attempts[code_pattern_id] = 0
                self.pattern_successes[code_pattern_id] = 0
                self.pattern_times[code_pattern_id] = []
            
            # Update code pattern stats
            self.patterns[code_pattern_id].frequency += 1
            self.pattern_attempts[code_pattern_id] += 1
            self.pattern_successes[code_pattern_id] += 1
            self.pattern_times[code_pattern_id].append(computation_time)
        
        # Extract and analyze subpaths only if path is valid and non-trivial
        if success and not is_trivial:
            self._analyze_subpaths(path, adj_matrix, computation_time)
            self._analyze_vertex_patterns(path, adj_matrix, computation_time)
        
        # Update success rates and computation times
        self._update_pattern_metrics()
    
    def _calculate_performance_score(self, computation_time: float, graph_size: int) -> float:
        """Calculate a performance score based on computation time and graph size."""
        # Normalize computation time relative to graph size
        # Lower times get higher scores
        base_score = 1.0 / (1.0 + computation_time)
        
        # Adjust for graph size (larger graphs get more lenient scoring)
        size_factor = np.log2(graph_size + 1) / 10
        
        return base_score * (1 + size_factor)
    
    def _analyze_subpaths(self, path: List[int], adj_matrix: np.ndarray, computation_time: float):
        """Analyze subpath patterns from a successful solution."""
        for i in range(len(path)-2):
            subpath = path[i:i+3]
            subpath_id = f"subpath_{'_'.join(map(str, subpath))}"
            
            # Check if this subpath is valid (vertices are connected)
            is_valid = True
            for j in range(len(subpath)-1):
                if not adj_matrix[subpath[j]][subpath[j+1]]:
                    is_valid = False
                    break
            
            if is_valid:
                if subpath_id not in self.patterns:
                    self.patterns[subpath_id] = PathPattern(
                        pattern_type="subpath",
                        pattern=subpath,
                        frequency=0,
                        success_rate=0.0,
                        avg_computation_time=0.0,
                        description=f"subpath pattern: {subpath}",
                        metadata={
                            "avg_degree": sum(np.sum(adj_matrix[v]) for v in subpath) / 3,
                            "is_sequential": subpath == list(range(subpath[0], subpath[0]+3))
                        }
                    )
                    self.pattern_attempts[subpath_id] = 0
                    self.pattern_successes[subpath_id] = 0
                    self.pattern_times[subpath_id] = []
                
                self.patterns[subpath_id].frequency += 1
                self.pattern_attempts[subpath_id] += 1
                self.pattern_successes[subpath_id] += 1
                self.pattern_times[subpath_id].append(computation_time)
    
    def _analyze_vertex_patterns(self, path: List[int], adj_matrix: np.ndarray, computation_time: float):
        """Analyze vertex patterns from a successful solution."""
        if len(path) >= 2:
            start_vertex = path[0]
            end_vertex = path[-1]
            
            # Create/update start vertex pattern
            start_pattern_id = f"start_{start_vertex}"
            if start_pattern_id not in self.patterns:
                self.patterns[start_pattern_id] = PathPattern(
                    pattern_type="vertex",
                    pattern=[start_vertex],
                    frequency=0,
                    success_rate=0.0,
                    avg_computation_time=0.0,
                    description=f"start vertex: {start_vertex}",
                    metadata={
                        "vertex_degree": int(np.sum(adj_matrix[start_vertex])),
                        "position": "start"
                    }
                )
                self.pattern_attempts[start_pattern_id] = 0
                self.pattern_successes[start_pattern_id] = 0
                self.pattern_times[start_pattern_id] = []
            
            self.patterns[start_pattern_id].frequency += 1
            self.pattern_attempts[start_pattern_id] += 1
            self.pattern_successes[start_pattern_id] += 1
            self.pattern_times[start_pattern_id].append(computation_time)
            
            # Create/update end vertex pattern
            end_pattern_id = f"end_{end_vertex}"
            if end_pattern_id not in self.patterns:
                self.patterns[end_pattern_id] = PathPattern(
                    pattern_type="vertex",
                    pattern=[end_vertex],
                    frequency=0,
                    success_rate=0.0,
                    avg_computation_time=0.0,
                    description=f"end vertex: {end_vertex}",
                    metadata={
                        "vertex_degree": int(np.sum(adj_matrix[end_vertex])),
                        "position": "end"
                    }
                )
                self.pattern_attempts[end_pattern_id] = 0
                self.pattern_successes[end_pattern_id] = 0
                self.pattern_times[end_pattern_id] = []
            
            self.patterns[end_pattern_id].frequency += 1
            self.pattern_attempts[end_pattern_id] += 1
            self.pattern_successes[end_pattern_id] += 1
            self.pattern_times[end_pattern_id].append(computation_time)
    
    def _update_pattern_metrics(self):
        """Update success rates and computation times for all patterns."""
        for pattern_id in self.patterns:
            attempts = self.pattern_attempts.get(pattern_id, 0)
            successes = self.pattern_successes.get(pattern_id, 0)
            times = self.pattern_times.get(pattern_id, [])
            
            if attempts > 0:
                self.patterns[pattern_id].success_rate = successes / attempts
                self.patterns[pattern_id].avg_computation_time = sum(times) / len(times)
    
    def get_best_patterns(self, min_frequency: int = 5, min_success_rate: float = 0.7) -> List[PathPattern]:
        """
        Get the best patterns based on frequency and success rate.
        
        Args:
            min_frequency: Minimum frequency threshold
            min_success_rate: Minimum success rate threshold
            
        Returns:
            List of best patterns
        """
        # Update success rates and computation times for all patterns
        self._update_pattern_metrics()
        
        # Get patterns meeting criteria (use more lenient criteria for vertex patterns)
        best_patterns = []
        for pattern in self.patterns.values():
            if pattern.pattern_type == "vertex":
                if pattern.success_rate > 0:  # Include all successful vertex patterns
                    best_patterns.append(pattern)
            else:
                if (pattern.frequency >= min_frequency or 
                    (pattern.success_rate >= min_success_rate and pattern.frequency > 0)):
                    best_patterns.append(pattern)
        
        # Sort by success rate and frequency
        best_patterns.sort(
            key=lambda x: (x.success_rate, x.frequency),
            reverse=True
        )
        
        return best_patterns
    
    def get_vertex_recommendations(self, adj_matrix: np.ndarray) -> Dict[str, List[int]]:
        """
        Get vertex recommendations based on pattern analysis.
        
        Args:
            adj_matrix: Graph adjacency matrix
            
        Returns:
            Dictionary with recommendations for start/end vertices
        """
        recommendations = {
            "start_vertices": [],
            "end_vertices": []
        }
        
        # Update success rates for all patterns
        self._update_pattern_metrics()
        
        # Get vertex patterns with any success
        vertex_patterns = [p for p in self.patterns.values() 
                         if p.pattern_type == "vertex" and p.success_rate > 0]
        
        # Extract vertex recommendations
        for pattern in vertex_patterns:
            if "start vertex" in pattern.description:
                if pattern.pattern[0] not in recommendations["start_vertices"]:
                    recommendations["start_vertices"].append(pattern.pattern[0])
            elif "end vertex" in pattern.description:
                if pattern.pattern[0] not in recommendations["end_vertices"]:
                    recommendations["end_vertices"].append(pattern.pattern[0])
        
        return recommendations
    
    def get_successful_code_patterns(self, min_success_rate: float = 0.7) -> List[str]:
        """Get successful code implementations for prompting.
        
        Args:
            min_success_rate: Minimum success rate threshold
            
        Returns:
            List of successful code implementations
        """
        code_patterns = []
        for pattern in self.patterns.values():
            if (pattern.pattern_type == "code" and 
                pattern.success_rate >= min_success_rate):
                code_patterns.append(pattern.description)
        return code_patterns
        
    def format_for_prompt(self, adj_matrix: np.ndarray) -> str:
        """Format pattern insights for use in prompts.
        
        Args:
            adj_matrix: Graph adjacency matrix
            
        Returns:
            Formatted string of pattern insights
        """
        insights = []
        
        # Graph properties
        n = len(adj_matrix)
        density = np.sum(adj_matrix) / (n * (n - 1))
        min_degree = min(np.sum(adj_matrix, axis=0))
        max_degree = max(np.sum(adj_matrix, axis=0))
        
        insights.append("Graph Properties:")
        insights.append(f"- Size: {n} vertices")
        insights.append(f"- Density: {density:.2f}")
        insights.append(f"- Degree range: {min_degree} to {max_degree}")
        
        # Add successful code patterns with metrics
        code_patterns = [(p, p.format_metrics()) 
                        for p in self.patterns.values() 
                        if p.pattern_type == "code" and p.success_rate > 0.5]
        
        if code_patterns:
            insights.append("\nSuccessful Implementations:")
            # Sort by success rate and time
            code_patterns.sort(key=lambda x: (x[0].success_rate, -x[0].avg_computation_time), reverse=True)
            for i, (pattern, metrics) in enumerate(code_patterns[:3], 1):
                insights.append(f"\nImplementation {i} ({metrics}):")
                insights.append(pattern.description)
        
        # Add successful subpath patterns
        subpath_patterns = [(p, p.format_metrics()) 
                          for p in self.patterns.values()
                          if p.pattern_type == "subpath" and p.success_rate > 0.6]
        
        if subpath_patterns:
            insights.append("\nSuccessful Subpaths:")
            # Sort by success rate
            subpath_patterns.sort(key=lambda x: x[0].success_rate, reverse=True)
            for pattern, metrics in subpath_patterns[:5]:
                insights.append(f"- {pattern.description} ({metrics})")
                if 'avg_degree' in pattern.metadata:
                    insights.append(f"  Average degree: {pattern.metadata['avg_degree']:.1f}")
        
        # Add vertex patterns with metrics
        vertex_patterns = [(p, p.format_metrics())
                         for p in self.patterns.values()
                         if p.pattern_type == "vertex" and p.success_rate > 0.5]
        
        if vertex_patterns:
            insights.append("\nVertex Patterns:")
            # Sort by success rate
            vertex_patterns.sort(key=lambda x: x[0].success_rate, reverse=True)
            for pattern, metrics in vertex_patterns:
                insights.append(f"- {pattern.description} ({metrics})")
                if 'vertex_degree' in pattern.metadata:
                    insights.append(f"  Degree: {pattern.metadata['vertex_degree']}")
        
        # Add recent failures
        failure_patterns = [p for p in self.patterns.values()
                          if p.pattern_type == "failure"][-5:]  # Last 5 failures
        
        if failure_patterns:
            insights.append("\nRecent Failures:")
            for pattern in failure_patterns:
                failure_point = pattern.metadata.get('failure_point', 'unknown')
                invalid_edge = pattern.metadata.get('invalid_edge', None)
                insights.append(f"- {pattern.description}")
                if invalid_edge:
                    insights.append(f"  Invalid edge: {invalid_edge}")
                insights.append(f"  Failed at vertex: {failure_point}")
        
        return "\n".join(insights)
    
    def get_pattern_insights(self) -> Dict:
        """Get insights from stored patterns."""
        return {
            'implementations': self.successful_patterns,
            'failures': self.failure_patterns,
            'subpaths': self.subpath_patterns
        }
    
    def clear_patterns(self):
        """Clear all stored patterns and statistics."""
        self.patterns.clear()
        self.pattern_attempts.clear()
        self.pattern_successes.clear()
        self.pattern_times.clear()

    def store_failure_pattern(self, code: Optional[str], reason: str):
        """Store a failed implementation pattern."""
        self.failure_patterns.append({
            'code': code,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })
