"""
Example script for creating and managing benchmark datasets.
"""

import os
from pathlib import Path
from src.benchmarks.benchmark_manager import BenchmarkManager

def main():
    # Initialize benchmark manager
    data_dir = Path(os.path.dirname(__file__)) / ".." / "data"
    manager = BenchmarkManager(str(data_dir))
    
    # Create a small benchmark dataset
    small_dataset = manager.create_benchmark_dataset(
        name="small_benchmark",
        num_satlib=5,
        num_random=5,
        random_sizes=[5, 8],
        verify=True
    )
    print("\nCreated small benchmark dataset:")
    print(f"Total instances: {small_dataset['statistics']['total_instances']}")
    print(f"Verified instances: {small_dataset['statistics']['verified_instances']}")
    print(f"Size distribution: {small_dataset['statistics']['size_distribution']}")
    
    # Create a larger benchmark dataset
    large_dataset = manager.create_benchmark_dataset(
        name="large_benchmark",
        num_satlib=10,
        num_random=10,
        random_sizes=[10, 15, 20],
        verify=True
    )
    print("\nCreated large benchmark dataset:")
    print(f"Total instances: {large_dataset['statistics']['total_instances']}")
    print(f"Verified instances: {large_dataset['statistics']['verified_instances']}")
    print(f"Size distribution: {large_dataset['statistics']['size_distribution']}")
    
    # Load and verify a specific instance
    graph, metadata = manager.load_instance("small_benchmark", 0)
    print("\nLoaded instance from small_benchmark:")
    print(f"Graph size: {len(graph)}")
    print(f"Source: {metadata['source']}")
    
    # Get global statistics
    stats = manager.get_statistics()
    print("\nGlobal benchmark statistics:")
    print(f"Total instances across all datasets: {stats['total_instances']}")
    print(f"Average instance size: {stats['average_size']:.2f}")
    print(f"Size distribution: {stats['size_distribution']}")

if __name__ == "__main__":
    main()
