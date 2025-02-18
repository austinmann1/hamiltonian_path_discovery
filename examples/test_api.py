"""
Simple script to test OpenRouter API integration.
"""
import os
import sys
from pathlib import Path
import logging
import json
from dotenv import load_dotenv
import httpx
import time

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.llm_interface_openrouter import OpenRouterLLM

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_simple_prompt():
    """Test API with a simple prompt."""
    load_dotenv()
    
    # Test one model at a time for better debugging
    model = "anthropic/claude-2"  # This model worked in previous test
    
    logger.info(f"\nTesting model: {model}")
    llm = OpenRouterLLM(model=model)
    
    # Simple prompt
    messages = [
        {
            "role": "system",
            "content": "You are a helpful Python programming assistant. Provide ONLY the implementation code without any explanation."
        },
        {
            "role": "user",
            "content": "Write a simple function to check if a number is prime."
        }
    ]
    
    try:
        with httpx.Client(timeout=30.0) as client:  # Set 30s timeout
            llm.client = client
            # Make direct API call to see raw response
            response = llm._call_api(messages)
            logger.info(f"Full API Response:\n{json.dumps(response, indent=2)}")
            
            # Extract and test the code
            if response.get("choices") and response["choices"][0]["message"]["content"]:
                code = response["choices"][0]["message"]["content"]
                logger.info(f"Generated Code:\n{code}")
                
                # Try to execute the code
                try:
                    # Extract just the Python code between ```python and ``` markers
                    if "```python" in code:
                        code = code.split("```python")[1].split("```")[0].strip()
                    elif "```" in code:
                        code = code.split("```")[1].split("```")[0].strip()
                    
                    namespace = {}
                    exec(code, namespace)
                    if "is_prime" in namespace:
                        result = namespace["is_prime"](7)
                        logger.info(f"Test result for is_prime(7): {result}")
                except Exception as e:
                    logger.error(f"Error executing code: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error with {model}: {str(e)}")
    
    logger.info(f"Performance Metrics:\n{json.dumps(llm.get_performance_metrics(), indent=2)}")
    logger.info("-" * 80)

if __name__ == "__main__":
    test_simple_prompt()
