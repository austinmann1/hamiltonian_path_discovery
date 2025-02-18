"""
Example script demonstrating the benchmarking process.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import logging

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.benchmark import BenchmarkSuite

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Load environment variables
    load_dotenv()
    
    # Verify OpenRouter API key is set
    if not os.getenv("OPENROUTER_API_KEY"):
        logger.error("Error: OPENROUTER_API_KEY not set in .env file")
        return
    
    # Initialize benchmark suite
    logger.info("Initializing benchmark suite...")
    suite = BenchmarkSuite(metrics_dir="metrics")
    
    logger.info("Starting test case generation...")
    test_cases = suite.generate_test_cases(
        sizes=[5, 10],  # Start with very small sizes for testing
        instances_per_size=2  # Use minimal instances
    )
    logger.info(f"Generated {len(test_cases)} test cases")
    
    # List of models to benchmark
    models = [
        "deepseek/deepseek-r1",  # Base model without :free suffix
    ]
    
    logger.info("Starting benchmarks...")
    results = suite.benchmark_llm_solutions(test_cases, models)
    
    # Print summary
    logger.info("\nBenchmark Results:")
    for result in results:
        logger.info(f"\nModel: {result['model']}")
        logger.info(f"Accuracy: {result['accuracy']:.2%}")
        logger.info(f"Average execution time: {result['avg_execution_time']:.2f}s")
        logger.info(f"Code size: {result['code_size']} bytes")
        
        llm_metrics = result.get('llm_metrics', {})
        if llm_metrics:
            logger.info("\nLLM Performance:")
            logger.info(f"Average response time: {llm_metrics.get('avg_response_time', 0):.2f}s")
            logger.info(f"Success rate: {llm_metrics.get('success_rate', 0):.2%}")
    
    # Generate plots
    logger.info("\nGenerating visualization plots...")
    suite.plot_results(save_dir="metrics")
    logger.info("Plots saved in metrics directory")

if __name__ == "__main__":
    main()
