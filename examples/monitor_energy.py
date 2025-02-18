"""
Example script demonstrating energy monitoring functionality.
"""

import os
import time
import numpy as np
from pathlib import Path
from src.monitoring.energy_monitor import EnergyMonitor
from src.benchmarks.benchmark_manager import BenchmarkManager
from src.verification.verification_oracle import VerificationOracle

def simulate_workload():
    """Simulate some CPU-intensive work."""
    # Matrix operations
    for _ in range(5):
        a = np.random.rand(1000, 1000)
        b = np.random.rand(1000, 1000)
        np.dot(a, b)
        time.sleep(0.1)

def main():
    # Initialize components
    data_dir = Path(os.path.dirname(__file__)) / ".." / "data"
    monitor = EnergyMonitor(str(data_dir), sampling_interval=0.5)
    benchmark_manager = BenchmarkManager(str(data_dir))
    verifier = VerificationOracle()
    
    # Start monitoring
    print("\nStarting energy monitoring...")
    monitor.start_monitoring("benchmark_verification")
    
    try:
        # Load and verify benchmark instances
        datasets = benchmark_manager.list_datasets()
        for dataset_name in datasets:
            print(f"\nProcessing dataset: {dataset_name}")
            dataset_info = benchmark_manager.get_dataset_info(dataset_name)
            
            for instance in dataset_info["instances"]:
                print(f"Verifying instance {instance['id']}...")
                graph, metadata = benchmark_manager.load_instance(dataset_name, instance["id"])
                
                # Get current energy stats
                current_stats = monitor.get_current_stats()
                print(f"Current power: {current_stats['average_power_watts']:.2f} watts")
                
                # Verify instance
                verification = verifier.verify_with_explanation(graph)
                print(f"Verification result: {verification['is_valid']}")
                
                # Simulate additional work
                simulate_workload()
        
        # Get final statistics
        stats = monitor.stop_monitoring()
        
        print("\nEnergy Monitoring Results:")
        print(f"Total Energy: {stats['total_energy_joules']:.2f} joules")
        print(f"Average Power: {stats['average_power_watts']:.2f} watts")
        print(f"Peak Power: {stats['peak_power_watts']:.2f} watts")
        print(f"Peak CPU: {stats['peak_cpu_percent']:.1f}%")
        print(f"Peak Memory: {stats['peak_memory_percent']:.1f}%")
        
        # Estimate cost
        cost_estimate = monitor.estimate_total_cost()
        print(f"\nEstimated Cost: ${cost_estimate['cost_usd']:.6f}")
        print(f"Total Energy: {cost_estimate['energy_kwh']:.6f} kWh")
    
    finally:
        # Ensure monitoring is stopped
        if monitor.is_monitoring:
            monitor.stop_monitoring()

if __name__ == "__main__":
    main()
