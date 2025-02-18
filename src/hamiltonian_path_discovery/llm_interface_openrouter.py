"""
LLM Interface module using OpenRouter API for code generation.
"""
import os
from typing import Dict, List, Optional, Union, Any
import httpx
from dotenv import load_dotenv
import json
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential
import time
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import requests
from .utils.logger import StructuredLogger

logger = logging.getLogger(__name__)

class OpenRouterLLM:
    def __init__(self, 
                 model: str = "anthropic/claude-3-r1",
                 api_key: Optional[str] = None,
                 timeout: float = 10.0):  # Default 10 second timeout
        """Initialize the OpenRouter LLM interface."""
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        
        self.model = model
        self.timeout = timeout
        self.logger = StructuredLogger()
        self.conversation_history = []
        self.system_prompt = """You are a graph theory expert that finds Hamiltonian paths in directed graphs.
You MUST ONLY return a Python list of integers representing the path.
CRITICAL FORMAT RULES:
1. Return ONLY the list, no other text
2. NO spaces between numbers and commas
3. Example format: [0,1,2,3] or []
4. NO explanations or comments
5. NO newlines
If no valid path exists, return an empty list []."""
        self.base_url = "https://openrouter.ai/api/v1"
        self.client = httpx.AsyncClient(timeout=timeout)
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        # Metrics tracking
        self.metrics: Dict[str, List] = {
            "response_times": [],
            "token_counts": [],
            "model_used": [],
            "success_rate": []
        }
    
    async def _async_call_api(self, messages: List[Dict]) -> Dict:
        """Make async API call to OpenRouter."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/codeium/hamiltonian_path_discovery",
            "X-Title": "Hamiltonian Path Discovery",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 4096,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stream": False
        }
        
        logger.info(f"Making API request to {self.base_url}/chat/completions")
        
        try:
            async with self.client as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    error_msg = f"API request failed with status {response.status_code}"
                    try:
                        error_data = response.json()
                        if isinstance(error_data, dict):
                            error_msg += f": {error_data.get('error', '')}"
                    except:
                        pass
                    raise RuntimeError(error_msg)
                    
        except httpx.TimeoutException:
            raise RuntimeError(f"API request timed out after {self.timeout} seconds")
        except Exception as e:
            raise RuntimeError(f"API request failed: {str(e)}")
    
    @retry(stop=stop_after_attempt(2), 
           wait=wait_exponential(multiplier=1, min=2, max=4))
    def _call_api(self, messages: List[Dict]) -> Dict:
        """Make API call to OpenRouter."""
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:3000",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        response = requests.post(url, headers=headers, json=data)
        if response.status_code != 200:
            raise RuntimeError(f"API call failed: {response.text}")
        
        return response.json()

    def parse_response(self, response: Dict) -> Dict:
        """Parse LLM response to extract code."""
        try:
            content = response["choices"][0]["message"]["content"].strip()
            print(f"Raw LLM response: {content}")  # Debug print
            
            print(f"Initial content: {content} (type: {type(content)})")  # Debug print
            
            # Try to extract list from response
            if content.startswith("[") and content.endswith("]"):
                try:
                    # Clean up the string by removing all whitespace
                    clean_content = "".join(content.split())
                    print(f"Cleaned content: {clean_content} (type: {type(clean_content)})")  # Debug print
                    
                    # If it's an empty list, return it
                    if clean_content == "[]":
                        return {
                            "code": [],
                            "success": True,
                            "error": None
                        }
                    
                    # Split by commas and parse integers
                    parts = clean_content[1:-1].split(",")
                    print(f"Parts: {parts} (type: {type(parts)})")  # Debug print
                    path = [int(p) for p in parts if p]
                    print(f"Parsed path: {path} (type: {type(path)})")  # Debug print
                    
                    return {
                        "code": path,
                        "success": True,
                        "error": None
                    }
                except Exception as e:
                    print(f"Exception during parsing: {str(e)}")  # Debug print
                    return {
                        "code": [],
                        "success": False,
                        "error": f"Failed to parse list: {str(e)}"
                    }
            
            # If not a clean list, look for list in the content
            import re
            list_pattern = r'\[[0-9, ]+\]'
            matches = re.findall(list_pattern, content)
            if matches:
                try:
                    # Clean up the string by removing all whitespace
                    clean_content = "".join(matches[0].split())
                    print(f"Cleaned matched content: {clean_content} (type: {type(clean_content)})")  # Debug print
                    
                    # If it's an empty list, return it
                    if clean_content == "[]":
                        return {
                            "code": [],
                            "success": True,
                            "error": None
                        }
                    
                    # Split by commas and parse integers
                    parts = clean_content[1:-1].split(",")
                    print(f"Matched parts: {parts} (type: {type(parts)})")  # Debug print
                    path = [int(p) for p in parts if p]
                    print(f"Parsed matched path: {path} (type: {type(path)})")  # Debug print
                    
                    return {
                        "code": path,
                        "success": True,
                        "error": None
                    }
                except Exception as e:
                    print(f"Exception during matched parsing: {str(e)}")  # Debug print
                    return {
                        "code": [],
                        "success": False,
                        "error": f"Failed to parse extracted list: {str(e)}"
                    }
            
            # If no valid list found, return empty list
            return {
                "code": [],
                "success": False,
                "error": "No valid list found in response"
            }
            
        except Exception as e:
            self.logger.log_error(f"Failed to parse response: {str(e)}")
            return {
                "code": [],
                "success": False,
                "error": str(e)
            }

    def generate_code(self, prompt: str) -> Dict:
        """Generate code using the LLM."""
        start_time = time.time()
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Run API call in thread pool to avoid blocking
            try:
                future = self.executor.submit(self._call_api, messages)
                response = future.result(timeout=self.timeout + 2)  # Add 2s buffer
                
                # Validate response format
                if "choices" not in response or not response["choices"]:
                    return {
                        "code": [],
                        "success": False,
                        "error": "Invalid API response format"
                    }
                
                # Extract code from response
                result = self.parse_response(response)
                
                # Update metrics
                duration = time.time() - start_time
                self.metrics["response_times"].append(duration)
                self.metrics["model_used"].append(self.model)
                self.metrics["success_rate"].append(1 if result["success"] else 0)
                
                if "usage" in response:
                    self.metrics["token_counts"].append(response["usage"]["total_tokens"])
                
                return result
                
            except Exception as e:
                self.logger.log_error(f"Code generation failed: {str(e)}")
                return {
                    "code": [],
                    "success": False,
                    "error": str(e)
                }
            
        except Exception as e:
            self.logger.log_error(f"Code generation failed: {str(e)}")
            return {
                "code": [],
                "success": False,
                "error": str(e)
            }
    
    def refine_code(self, 
                    code: str, 
                    error_message: str,
                    test_case: Optional[Dict] = None) -> str:
        """
        Refine code based on error feedback.
        
        Args:
            code: The original code that failed
            error_message: Error message or test case failure details
            test_case: Optional failing test case
            
        Returns:
            str: Refined code addressing the error
        """
        messages = [{
            "role": "system",
            "content": "You are debugging and improving code for Hamiltonian path detection."
        }]
        
        debug_prompt = f"""
        The following code failed:
        
        ```python
        {code}
        ```
        
        Error:
        {error_message}
        """
        
        if test_case:
            debug_prompt += f"\nFailing test case:\n{json.dumps(test_case, indent=2)}"
        
        messages.append({
            "role": "user",
            "content": debug_prompt
        })
        
        try:
            response = self._call_api(messages)
            refined_code = self.parse_response(response)
            
            # Store the interaction
            self.conversation_history.extend([
                {"role": "user", "content": debug_prompt},
                {"role": "assistant", "content": refined_code}
            ])
            
            return refined_code
            
        except Exception as e:
            raise Exception(f"Failed to refine code: {str(e)}")
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics for model usage."""
        if not self.metrics["response_times"]:
            return {}
            
        return {
            "avg_response_time": sum(self.metrics["response_times"]) / len(self.metrics["response_times"]),
            "avg_tokens": sum(self.metrics["token_counts"]) / len(self.metrics["token_counts"]),
            "success_rate": sum(self.metrics["success_rate"]) / len(self.metrics["success_rate"]),
            "total_calls": len(self.metrics["response_times"]),
            "models_used": list(set(self.metrics["model_used"]))
        }
    
    def save_metrics(self, filepath: str) -> None:
        """Save performance metrics to a file."""
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def load_metrics(self, filepath: str) -> None:
        """Load performance metrics from a file."""
        if Path(filepath).exists():
            with open(filepath, 'r') as f:
                self.metrics = json.load(f)

    def get_energy_usage(self) -> float:
        """Get current energy usage."""
        # For now just return a random value between 0 and 100
        # In the future we could integrate with actual energy monitoring
        import random
        return random.random() * 100

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        if hasattr(self, 'client'):
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self.client.aclose())
            finally:
                loop.close()

# Example usage
if __name__ == "__main__":
    llm = OpenRouterLLM()
    
    # Example problem description
    problem = """
    Generate a Python function that finds a Hamiltonian path in a directed graph
    using dynamic programming with bitmask optimization. The function should:
    1. Take an adjacency matrix as input
    2. Return the path if it exists, None otherwise
    3. Use efficient pruning techniques
    """
    
    # Example test case
    test_case = {
        "input": [[0, 1, 0], [0, 0, 1], [0, 0, 0]],
        "expected": [0, 1, 2],
        "has_solution": True
    }
    
    try:
        code = llm.generate_code(problem)
        print("Generated code:")
        print(code)
        
        # Print performance metrics
        print("\nPerformance metrics:")
        print(json.dumps(llm.get_performance_metrics(), indent=2))
    except Exception as e:
        print(f"Error: {e}")
