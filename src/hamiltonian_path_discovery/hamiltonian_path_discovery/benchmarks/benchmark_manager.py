"""
Benchmark dataset manager for the Hamiltonian Path Discovery project.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from ..graph_generator.test_generator import TestGenerator
from ..graph_generator.graph_utils import GraphUtils
from ..verification.verification_oracle import VerificationOracle
from ..logging import StructuredLogger, ExperimentLogger

class BenchmarkManager:
    """
    Manages benchmark datasets for evaluating the Hamiltonian Path Discovery framework.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.benchmarks_dir = self.data_dir / "benchmarks"
        self.metadata_file = self.benchmarks_dir / "metadata.json"
        
        # Create directories
        self.benchmarks_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.test_generator = TestGenerator(data_dir)
        self.graph_utils = GraphUtils()
        self.verifier = VerificationOracle()
        self.logger = StructuredLogger()
        self.experiment = ExperimentLogger()
        
        # Load or create metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load benchmark metadata or create if not exists."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            "datasets": {},
            "statistics": {
                "total_instances": 0,
                "verified_instances": 0,
                "average_size": 0.0,
                "size_distribution": {}
            }
        }
    
    def _save_metadata(self):
        """Save benchmark metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def create_benchmark_dataset(
        self,
        name: str,
        num_satlib: int = 10,
        num_random: int = 10,
        random_sizes: List[int] = [5, 10, 15, 20],
        verify: bool = True
    ) -> Dict:
        """
        Create a new benchmark dataset combining SATLIB and random instances.
        
        Args:
            name: Name of the benchmark dataset
            num_satlib: Number of SATLIB instances to include
            num_random: Number of random instances to include
            random_sizes: List of sizes for random instances
            verify: Whether to verify instances during creation
            
        Returns:
            Dictionary with dataset information
        """
        exp_id = self.experiment.start_experiment(
            description=f"Creating benchmark dataset: {name}",
            config={
                "num_satlib": num_satlib,
                "num_random": num_random,
                "random_sizes": random_sizes,
                "verify": verify
            }
        )
        
        try:
            dataset_dir = self.benchmarks_dir / name
            dataset_dir.mkdir(exist_ok=True)
            
            # Process SATLIB instances
            satlib_instances = self.test_generator.process_satlib_instances()
            selected_satlib = random.sample(satlib_instances, min(num_satlib, len(satlib_instances)))
            
            # Generate random instances
            random_instances = []
            for size in random_sizes:
                for _ in range(num_random // len(random_sizes)):
                    graph = self.graph_utils.generate_random_hamiltonian_graph(size)
                    random_instances.append({
                        "graph": graph,
                        "size": size,
                        "source": "random"
                    })
            
            # Combine and verify instances
            all_instances = []
            verified_count = 0
            
            for instance in selected_satlib + random_instances:
                instance_id = len(all_instances)
                instance_path = dataset_dir / f"instance_{instance_id}.npz"
                
                # Save instance
                np.savez(
                    instance_path,
                    graph=instance["graph"],
                    size=instance["size"],
                    source=instance["source"]
                )
                
                # Verify if requested
                if verify:
                    verification = self.verifier.verify_with_explanation(instance["graph"])
                    if verification["is_valid"]:
                        verified_count += 1
                
                all_instances.append({
                    "id": instance_id,
                    "path": str(instance_path),
                    "size": instance["size"],
                    "source": instance["source"],
                    "verified": verify and verification["is_valid"] if verify else None
                })
            
            # Update metadata
            dataset_info = {
                "name": name,
                "instances": all_instances,
                "statistics": {
                    "total_instances": len(all_instances),
                    "verified_instances": verified_count if verify else None,
                    "average_size": sum(i["size"] for i in all_instances) / len(all_instances),
                    "size_distribution": {
                        str(size): len([i for i in all_instances if i["size"] == size])
                        for size in set(i["size"] for i in all_instances)
                    }
                }
            }
            
            self.metadata["datasets"][name] = dataset_info
            self._update_global_statistics()
            self._save_metadata()
            
            # Log success
            self.logger.log_metrics(f"benchmark_creation_{name}", dataset_info["statistics"])
            
            return dataset_info
            
        finally:
            self.experiment.end_experiment()
    
    def _update_global_statistics(self):
        """Update global benchmark statistics."""
        stats = self.metadata["statistics"]
        datasets = self.metadata["datasets"].values()
        
        if not datasets:
            return
        
        stats["total_instances"] = sum(d["statistics"]["total_instances"] for d in datasets)
        stats["verified_instances"] = sum(
            d["statistics"]["verified_instances"] or 0 
            for d in datasets
        )
        
        total_size = sum(
            d["statistics"]["average_size"] * d["statistics"]["total_instances"]
            for d in datasets
        )
        stats["average_size"] = total_size / stats["total_instances"]
        
        # Update size distribution
        size_dist = {}
        for dataset in datasets:
            for size, count in dataset["statistics"]["size_distribution"].items():
                size_dist[size] = size_dist.get(size, 0) + count
        stats["size_distribution"] = size_dist
    
    def load_instance(self, dataset_name: str, instance_id: int) -> Tuple[np.ndarray, Dict]:
        """
        Load a specific instance from a dataset.
        
        Args:
            dataset_name: Name of the dataset
            instance_id: ID of the instance
            
        Returns:
            Tuple of (graph adjacency matrix, instance metadata)
        """
        dataset = self.metadata["datasets"].get(dataset_name)
        if not dataset:
            raise ValueError(f"Dataset {dataset_name} not found")
        
        instance = next((i for i in dataset["instances"] if i["id"] == instance_id), None)
        if not instance:
            raise ValueError(f"Instance {instance_id} not found in dataset {dataset_name}")
        
        data = np.load(instance["path"])
        return data["graph"], {
            "size": data["size"].item(),
            "source": data["source"].item()
        }
    
    def get_dataset_info(self, name: str) -> Optional[Dict]:
        """Get information about a specific dataset."""
        return self.metadata["datasets"].get(name)
    
    def list_datasets(self) -> List[str]:
        """Get list of available datasets."""
        return list(self.metadata["datasets"].keys())
    
    def get_statistics(self) -> Dict:
        """Get global benchmark statistics."""
        return self.metadata["statistics"]
