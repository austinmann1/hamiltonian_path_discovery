"""
Energy Monitor module for tracking compute resource usage.
"""
import psutil
import time
from contextlib import contextmanager
from typing import Dict, Optional
import json
from pathlib import Path

class EnergyMonitor:
    def __init__(self, log_file: Optional[str] = None):
        self.start_time = None
        self.start_cpu_percent = None
        self.start_memory = None
        self.log_file = log_file
        
    @contextmanager
    def track(self, task_name: str = "unnamed_task"):
        """
        Context manager for tracking resource usage during a task.
        
        Args:
            task_name: Name of the task being monitored
        """
        try:
            self.start_monitoring()
            yield
        finally:
            stats = self.stop_monitoring()
            stats["task_name"] = task_name
            self._log_stats(stats)
            
    def start_monitoring(self) -> None:
        """Start monitoring system resources."""
        self.start_time = time.time()
        self.start_cpu_percent = psutil.cpu_percent(interval=0.1)
        self.start_memory = psutil.Process().memory_info().rss
        
    def stop_monitoring(self) -> Dict:
        """
        Stop monitoring and return resource usage statistics.
        
        Returns:
            Dict containing execution time, CPU usage, and memory consumption
        """
        if not self.start_time:
            raise RuntimeError("Monitoring was not started")
            
        end_time = time.time()
        end_cpu_percent = psutil.cpu_percent(interval=0.1)
        end_memory = psutil.Process().memory_info().rss
        
        stats = {
            "execution_time": end_time - self.start_time,
            "cpu_usage_percent": (end_cpu_percent + self.start_cpu_percent) / 2,
            "memory_usage_mb": (end_memory - self.start_memory) / (1024 * 1024),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return stats
        
    def _log_stats(self, stats: Dict) -> None:
        """Log statistics to file if log_file is specified."""
        if self.log_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Read existing logs if file exists
            existing_logs = []
            if log_path.exists():
                with open(log_path, 'r') as f:
                    existing_logs = json.load(f)
                    
            # Append new stats
            existing_logs.append(stats)
            
            # Write updated logs
            with open(log_path, 'w') as f:
                json.dump(existing_logs, f, indent=2)
                
    @staticmethod
    def get_system_info() -> Dict:
        """
        Get general system information.
        
        Returns:
            Dict containing CPU count, total memory, etc.
        """
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            "platform": psutil.Process().parent().name()
        }

# Example usage:
if __name__ == "__main__":
    monitor = EnergyMonitor(log_file="energy_logs.json")
    
    # Print system info
    print("System Information:")
    print(json.dumps(monitor.get_system_info(), indent=2))
    
    # Example monitoring
    with monitor.track("example_task"):
        # Simulate some work
        time.sleep(1)
        [i * i for i in range(1000000)]
