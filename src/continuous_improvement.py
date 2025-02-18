"""Continuous improvement system for Hamiltonian path discovery."""

import os
import sys
import time
import json
import ast
import asyncio
import threading
import traceback
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import numpy as np
import httpx

from pattern_mining.pattern_analyzer import PatternAnalyzer
from solution_validator import validate_hamiltonian_path

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prompting.prompt_manager import PromptManager
from src.graph_generator import GraphGenerator

@dataclass
class ExplorationBranch:
    """Represents a branch of exploration with its patterns and success metrics."""
    patterns: Dict[str, Any]
    success_rate: float
    computation_time: float
    solutions: List[List[int]]
    
class ContinuousImprovement:
    def __init__(self, 
                 max_vertices: int = 20,
                 min_vertices: int = 4,
                 model_name: str = "deepseek/deepseek-r1",
                 max_tokens: int = 4096,
                 output_tokens: int = 16384,  
                 parallel_branches: int = 3,
                 openrouter_api_key: Optional[str] = None,
                 site_url: Optional[str] = None,
                 site_name: Optional[str] = None):
        """Initialize with parallel exploration support."""
        self.max_vertices = max_vertices
        self.min_vertices = min_vertices
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.output_tokens = output_tokens
        self.parallel_branches = parallel_branches
        
        # Initialize API client
        self.client = httpx.Client(timeout=300.0)  
        self.api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        
        # Headers setup
        self.extra_headers = {
            "HTTP-Referer": site_url or "https://github.com/austinmw/hamiltonian-path-discovery",
            "X-Title": site_name or "Hamiltonian Path Discovery"
        }
        
        # Initialize components
        self.pattern_analyzer = PatternAnalyzer()
        self.prompt_manager = PromptManager()
        self.graph_generator = GraphGenerator()
        
        # Parallel exploration setup
        self.branches: List[ExplorationBranch] = []
        self.branch_lock = threading.Lock()
        self.result_queue = Queue()
        
        # Statistics tracking
        self.stats = {
            'total_attempts': 0,
            'successful_attempts': 0,
            'patterns_discovered': 0,
            'avg_computation_time': 0.0
        }
    
    def serialize_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
        
    async def _call_openrouter_api(self, prompt: str) -> Dict:
        """Call OpenRouter API with retry logic."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.extra_headers["HTTP-Referer"],
            "X-Title": self.extra_headers["X-Title"],
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert in graph theory and algorithm design. Your task is to write Python code to solve graph problems. Always respond with ONLY the raw Python code, without any markdown formatting, comments outside functions, or additional text."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self.output_tokens,
            "temperature": 0.7,
            "stream": False
        }
        
        print(f"\nMaking API request to {self.model_name}...")
        print(f"Max output tokens: {self.output_tokens}")
        
        max_retries = 3
        base_delay = 2
        timeout = 600.0  # 10 minutes timeout
        
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json=data,
                        timeout=timeout
                    )
                    
                    if response.status_code == 429:  # Rate limit
                        delay = base_delay ** attempt
                        print(f"Rate limited. Waiting {delay} seconds...")
                        await asyncio.sleep(delay)
                        continue
                        
                    response.raise_for_status()
                    response_data = response.json()
                    
                    if 'choices' not in response_data or not response_data['choices']:
                        print("No choices in response")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(base_delay ** attempt)
                            continue
                        return None
                    
                    return response_data
                    
            except Exception as e:
                print(f"API call error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    delay = base_delay ** attempt
                    print(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    print("Max retries reached")
                    return None
    
    async def call_model_async(self, prompt: str) -> str:
        """Call the model asynchronously."""
        try:
            response = await self._call_openrouter_api(prompt)
            if not response or 'choices' not in response or not response['choices']:
                print("No valid response from model")
                return None
                
            content = response['choices'][0]['message']['content']
            print("\nModel response received. Extracting code...")
            print("\nRaw response:", content)
            return content
            
        except Exception as e:
            print(f"Error calling model: {str(e)}")
            return None
    
    def extract_python_function(self, response: str) -> Optional[str]:
        """Extract Python function from model response."""
        if not response:
            print("No response to extract from")
            return None
            
        try:
            # Clean up any whitespace and remove any markdown artifacts
            code = response.strip()
            if code.startswith("```"):
                code = code[code.find("\n")+1:]
            if code.endswith("```"):
                code = code[:code.rfind("```")]
            code = code.strip()
            
            # Verify it's valid Python code
            try:
                ast.parse(code)
                return code
            except SyntaxError as e:
                print(f"Invalid Python syntax in response: {str(e)}")
                return None
                
        except Exception as e:
            print(f"Error extracting code: {str(e)}")
            return None
    
    async def explore_branch(self, graph: np.ndarray) -> None:
        """Explore one branch of solutions."""
        try:
            # Generate prompt with current patterns
            prompt = self.prompt_manager.generate_pattern_based_prompt(
                graph,
                self.pattern_analyzer.format_for_prompt(self.pattern_analyzer.get_best_patterns())
            )
            
            # Call model
            response = await self.call_model_async(prompt)
            code = self.extract_python_function(response)
            
            if code:
                # Execute solution and get execution time
                path, execution_time = self.execute_solution(code, graph)
                success = path is not None and validate_hamiltonian_path(path, graph)
                
                # Create branch record
                branch = ExplorationBranch(
                    patterns=self.pattern_analyzer.patterns.copy(),
                    success_rate=1.0 if success else 0.0,
                    computation_time=execution_time,  
                    solutions=[path] if success else []
                )
                
                # Add to branches
                with self.branch_lock:
                    self.branches.append(branch)
                        
        except Exception as e:
            print(f"Branch exploration error: {str(e)}")
            traceback.print_exc()
    
    async def run_parallel_exploration(self, graph: np.ndarray) -> None:
        """Run multiple exploration branches in parallel."""
        tasks = []
        for _ in range(self.parallel_branches):
            tasks.append(self.explore_branch(graph))
        await asyncio.gather(*tasks)
        
        # Combine results from branches
        self.combine_branches()
    
    def execute_solution(self, code: str, adj_matrix: np.ndarray) -> Tuple[Optional[List[int]], float]:
        """Execute solution code with timeout."""
        if not code:
            return None, 0.0
            
        # Create execution environment
        exec_globals = {
            'np': np,
            'sorted': sorted,  
            'list': list,
            'set': set,
            'len': len,
            'sum': sum,
            'range': range,
            'print': print,
            'any': any,
            'all': all
        }
        
        try:
            # Execute code
            exec(code, exec_globals)
            
            # Get function
            find_hamiltonian_path = exec_globals.get('find_hamiltonian_path')
            if not find_hamiltonian_path:
                print("No find_hamiltonian_path function defined")
                return None, 0.0
            
            # Time execution
            start_time = time.time()
            path = find_hamiltonian_path(adj_matrix)
            execution_time = time.time() - start_time
            
            return path, execution_time
            
        except Exception as e:
            print(f"Error executing solution: {str(e)}")
            return None, 0.0
    
    def generate_random_graph(self, min_vertices: int = 4, max_vertices: int = 20) -> np.ndarray:
        """Generate a random graph for testing."""
        num_vertices = np.random.randint(min_vertices, max_vertices)
        density = np.random.uniform(0.3, 0.7)  # Vary graph density
        return self.graph_generator.generate_random_graph(num_vertices, density)
    
    def combine_branches(self) -> None:
        """Combine successful patterns from different branches."""
        with self.branch_lock:
            if not self.branches:
                return
                
            # Sort branches by success rate
            self.branches.sort(key=lambda x: x.success_rate, reverse=True)
            
            # Take best patterns from top branches
            best_patterns = {}
            for branch in self.branches[:2]:  # Take top 2 branches
                for pattern_id, pattern in branch.patterns.items():
                    if pattern_id not in best_patterns or pattern.success_rate > best_patterns[pattern_id].success_rate:
                        best_patterns[pattern_id] = pattern
            
            # Update pattern analyzer with combined patterns
            self.pattern_analyzer.patterns = best_patterns
            
            # Clear branches for next round
            self.branches = []
    
    def run_continuous_loop(self):
        """Run the continuous improvement loop with parallel exploration."""
        print(f"Starting continuous improvement loop with {self.model_name}")
        print(f"Parallel branches: {self.parallel_branches}")
        print("Press Ctrl+C to stop the loop")
        print("-" * 80)
        
        try:
            while True:
                # Generate a new random graph
                print("\nGenerating new random graph...")
                current_graph = self.generate_random_graph(min_vertices=4, max_vertices=8)
                print(f"Generated graph with {len(current_graph)} vertices")
                
                # Run parallel exploration
                asyncio.run(self.run_parallel_exploration(current_graph))
                
                # Update statistics
                self.stats['total_attempts'] += 1
                successful_branches = [b for b in self.branches if b.success_rate > 0]
                if successful_branches:
                    self.stats['successful_attempts'] += 1
                    avg_exec_time = sum(b.computation_time for b in successful_branches) / len(successful_branches)
                    self.stats['avg_computation_time'] = (
                        (self.stats['avg_computation_time'] * (self.stats['successful_attempts'] - 1) +
                         avg_exec_time) / self.stats['successful_attempts']
                    )
                
                # Log progress
                success_rate = self.stats['successful_attempts'] / self.stats['total_attempts'] * 100
                print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Round complete")
                print(f"Graph size: {len(current_graph)}x{len(current_graph)}")
                if successful_branches:
                    print(f"Algorithm execution time: {avg_exec_time:.6f}s")  
                print(f"Success rate: {success_rate:.2f}%")
                print(f"Patterns discovered: {len(self.pattern_analyzer.patterns)}")
                print(f"Average algorithm time: {self.stats['avg_computation_time']:.6f}s")
                print("-" * 80)
                
                # Save state periodically
                if self.stats['total_attempts'] % 5 == 0:
                    self.pattern_analyzer.save_state()
                
                # Add delay between rounds
                print("Waiting 5 seconds before next round...")
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\nStopping continuous improvement loop...")
            print("Final statistics:")
            print(f"Total attempts: {self.stats['total_attempts']}")
            print(f"Successful attempts: {self.stats['successful_attempts']}")
            print(f"Success rate: {(self.stats['successful_attempts'] / max(1, self.stats['total_attempts']) * 100):.2f}%")
            print(f"Patterns discovered: {len(self.pattern_analyzer.patterns)}")
            print(f"Average algorithm time: {self.stats['avg_computation_time']:.6f}s")
            
            # Save final state
            print("Saving final state...")
            self.pattern_analyzer.save_state()

if __name__ == "__main__":
    improver = ContinuousImprovement(
        max_vertices=20,
        min_vertices=4,
        model_name="deepseek/deepseek-r1",
        max_tokens=4096,
        output_tokens=16384,
        parallel_branches=3,
        site_url=os.getenv("SITE_URL"),
        site_name=os.getenv("SITE_NAME")
    )
    improver.run_continuous_loop()
