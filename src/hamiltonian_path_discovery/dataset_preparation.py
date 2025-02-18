"""
Dataset preparation module for converting SATLIB instances to Hamiltonian path problems.
"""
import os
import urllib.request
import tarfile
import random
from typing import List, Tuple, Dict
import json
from pathlib import Path
import shutil

class DatasetPreparation:
    def __init__(self, data_dir: str = "data"):
        """
        Initialize dataset preparation.
        
        Args:
            data_dir: Directory to store datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_satlib(self, category: str = "uf20-91") -> str:
        """
        Download SATLIB benchmark instances.
        
        Args:
            category: SATLIB benchmark category (e.g., 'uf20-91', 'uf50-218')
            
        Returns:
            str: Path to downloaded files
        """
        # SATLIB URL format
        base_url = "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/"
        filename = f"{category}.tar.gz"
        url = f"{base_url}{filename}"
        
        # Create category directory
        category_dir = self.data_dir / category
        category_dir.mkdir(exist_ok=True)
        
        # Download and extract
        tar_path = category_dir / filename
        try:
            print(f"Downloading {url}...")
            urllib.request.urlretrieve(url, tar_path)
            
            print("Extracting files...")
            with tarfile.open(tar_path) as tar:
                tar.extractall(path=category_dir)
                
            # Clean up
            tar_path.unlink()
            
            return str(category_dir)
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            if tar_path.exists():
                tar_path.unlink()
            raise
    
    def prepare_benchmark_suite(self, 
                              categories: List[str] = ["uf20-91"],
                              num_instances: int = 100) -> Dict:
        """
        Prepare a benchmark suite from SATLIB instances.
        
        Args:
            categories: List of SATLIB categories to include
            num_instances: Number of instances to include per category
            
        Returns:
            Dict containing benchmark metadata and paths
        """
        benchmark_data = {
            "metadata": {
                "categories": categories,
                "instances_per_category": num_instances,
                "total_instances": len(categories) * num_instances
            },
            "instances": []
        }
        
        for category in categories:
            try:
                category_dir = self.download_satlib(category)
                
                # Find all CNF files
                cnf_files = list(Path(category_dir).glob("*.cnf"))
                
                # Randomly select instances
                selected_files = random.sample(
                    cnf_files,
                    min(num_instances, len(cnf_files))
                )
                
                # Add to benchmark suite
                for cnf_file in selected_files:
                    instance_data = {
                        "category": category,
                        "filename": cnf_file.name,
                        "path": str(cnf_file),
                        "size": cnf_file.stat().st_size
                    }
                    benchmark_data["instances"].append(instance_data)
                    
            except Exception as e:
                print(f"Error processing category {category}: {e}")
                continue
        
        # Save benchmark metadata
        metadata_path = self.data_dir / "benchmark_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
            
        return benchmark_data
    
    def clean_data_directory(self) -> None:
        """Remove all downloaded files and reset data directory."""
        if self.data_dir.exists():
            shutil.rmtree(self.data_dir)
        self.data_dir.mkdir(parents=True)

# Example usage
if __name__ == "__main__":
    prep = DatasetPreparation()
    
    # Prepare benchmark suite with default settings
    try:
        benchmark_data = prep.prepare_benchmark_suite()
        print(f"Created benchmark suite with {len(benchmark_data['instances'])} instances")
        
        # Print some statistics
        categories = benchmark_data["metadata"]["categories"]
        for category in categories:
            instances = [i for i in benchmark_data["instances"] 
                        if i["category"] == category]
            total_size = sum(i["size"] for i in instances)
            print(f"\nCategory: {category}")
            print(f"Number of instances: {len(instances)}")
            print(f"Total size: {total_size / 1024:.2f} KB")
