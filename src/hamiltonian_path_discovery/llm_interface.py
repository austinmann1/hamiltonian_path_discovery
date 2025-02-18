"""
LLM Interface module for code generation using OpenAI's API.
"""
import os
from typing import Dict, List, Optional, Union
import openai
from dotenv import load_dotenv
import json
from pathlib import Path

class LLMInterface:
    def __init__(self, model: str = "gpt-4-turbo-preview"):
        """
        Initialize the LLM interface.
        
        Args:
            model: The OpenAI model to use for code generation
        """
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        self.conversation_history: List[Dict] = []
        
    def generate_code(self, 
                     problem_description: str,
                     test_cases: Optional[List[Dict]] = None,
                     max_attempts: int = 3) -> str:
        """
        Generate code for a given problem using the LLM.
        
        Args:
            problem_description: Description of the coding problem
            test_cases: Optional list of test cases for validation
            max_attempts: Maximum number of attempts to generate valid code
            
        Returns:
            str: Generated code that passes the test cases
        """
        system_message = {
            "role": "system",
            "content": """You are an expert algorithm designer specializing in Hamiltonian path detection.
            Given the adjacency matrix of a directed graph, generate Python code that:
            1. Uses dynamic programming (bitmask optimization)
            2. Implements branch-and-bound with conflict learning
            3. Avoids O(n!) time complexity
            Return ONLY the code with inline comments."""
        }
        
        for attempt in range(max_attempts):
            messages = [system_message]
            
            # Add problem description
            messages.append({
                "role": "user",
                "content": problem_description
            })
            
            # Add test cases if provided
            if test_cases:
                test_case_str = "Test cases:\n" + json.dumps(test_cases, indent=2)
                messages.append({
                    "role": "user",
                    "content": test_case_str
                })
            
            # Add conversation history
            messages.extend(self.conversation_history)
            
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                
                generated_code = response.choices[0].message.content
                
                # Store the interaction in conversation history
                self.conversation_history.extend([
                    {"role": "assistant", "content": generated_code}
                ])
                
                return generated_code
                
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise Exception(f"Failed to generate code after {max_attempts} attempts: {str(e)}")
                continue
    
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
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            refined_code = response.choices[0].message.content
            
            # Store the interaction in conversation history
            self.conversation_history.extend([
                {"role": "user", "content": debug_prompt},
                {"role": "assistant", "content": refined_code}
            ])
            
            return refined_code
            
        except Exception as e:
            raise Exception(f"Failed to refine code: {str(e)}")
    
    def save_conversation(self, filepath: str) -> None:
        """Save the conversation history to a file."""
        with open(filepath, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
    
    def load_conversation(self, filepath: str) -> None:
        """Load conversation history from a file."""
        if Path(filepath).exists():
            with open(filepath, 'r') as f:
                self.conversation_history = json.load(f)

# Example usage
if __name__ == "__main__":
    llm = LLMInterface()
    
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
        "expected": [0, 1, 2]
    }
    
    try:
        code = llm.generate_code(problem, [test_case])
        print("Generated code:")
        print(code)
    except Exception as e:
        print(f"Error: {e}")
