"""
Energy monitoring system for tracking computational resource usage.
"""

import time
import psutil
import threading
from typing import Dict, Optional, List
from dataclasses import dataclass
from pathlib import Path
import json
from ..logging import StructuredLogger, ExperimentLogger

@dataclass
class ResourceSnapshot:
    """Snapshot of system resource usage."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    power_consumption: float  # Estimated in watts
    temperature: float  # CPU temperature in Celsius
    disk_io: Dict[str, int]  # Read/write bytes
    
    def to_dict(self) -> Dict:
        """Convert snapshot to dictionary."""
        return {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "power_consumption": self.power_consumption,
            "temperature": self.temperature,
            "disk_io": self.disk_io
        }

class EnergyMonitor:
    """
    Monitors and tracks energy consumption and system resources.
    Uses psutil for system metrics and estimates power consumption
    based on CPU usage and system information.
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        sampling_interval: float = 1.0,
        tdp: float = 15.0  # Thermal Design Power in watts
    ):
        self.data_dir = Path(data_dir)
        self.monitoring_dir = self.data_dir / "monitoring"
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        
        self.sampling_interval = sampling_interval
        self.tdp = tdp
        self.is_monitoring = False
        self.current_session = None
        
        # Initialize components
        self.logger = StructuredLogger()
        self.experiment = ExperimentLogger()
        
        # Initialize metrics
        self._reset_metrics()
        
        # Get initial disk IO counters
        self.last_disk_io = psutil.disk_io_counters()
    
    def _reset_metrics(self):
        """Reset monitoring metrics."""
        self.snapshots: List[ResourceSnapshot] = []
        self.start_time: Optional[float] = None
        self.total_energy = 0.0
        self.peak_power = 0.0
        self.peak_temperature = 0.0
        self.peak_cpu = 0.0
        self.peak_memory = 0.0
    
    def _estimate_power(self, cpu_percent: float) -> float:
        """
        Estimate power consumption based on CPU usage and TDP.
        Uses a simplified model where power scales with CPU usage.
        
        Args:
            cpu_percent: CPU usage percentage
            
        Returns:
            Estimated power consumption in watts
        """
        # Base power (idle) is roughly 30% of TDP
        base_power = 0.3 * self.tdp
        
        # Dynamic power scales with CPU usage
        dynamic_power = (self.tdp - base_power) * (cpu_percent / 100.0)
        
        return base_power + dynamic_power
    
    def _get_snapshot(self) -> ResourceSnapshot:
        """
        Get current system resource snapshot.
        
        Returns:
            ResourceSnapshot with current metrics
        """
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        
        # Get CPU temperature if available
        temperature = 0.0
        try:
            temperature = psutil.sensors_temperatures()['coretemp'][0].current
        except:
            pass  # Temperature monitoring may not be available
        
        # Get disk IO
        current_disk_io = psutil.disk_io_counters()
        disk_io = {
            "read_bytes": current_disk_io.read_bytes - self.last_disk_io.read_bytes,
            "write_bytes": current_disk_io.write_bytes - self.last_disk_io.write_bytes
        }
        self.last_disk_io = current_disk_io
        
        # Estimate power consumption
        power = self._estimate_power(cpu_percent)
        
        return ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            power_consumption=power,
            temperature=temperature,
            disk_io=disk_io
        )
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.is_monitoring:
            snapshot = self._get_snapshot()
            self.snapshots.append(snapshot)
            
            # Update peak values
            self.peak_power = max(self.peak_power, snapshot.power_consumption)
            self.peak_temperature = max(self.peak_temperature, snapshot.temperature)
            self.peak_cpu = max(self.peak_cpu, snapshot.cpu_percent)
            self.peak_memory = max(self.peak_memory, snapshot.memory_percent)
            
            # Update total energy (power * time)
            if len(self.snapshots) > 1:
                time_diff = snapshot.timestamp - self.snapshots[-2].timestamp
                energy = snapshot.power_consumption * time_diff
                self.total_energy += energy
            
            time.sleep(self.sampling_interval)
    
    def start_monitoring(self, session_name: str):
        """
        Start monitoring system resources.
        
        Args:
            session_name: Name of the monitoring session
        """
        if self.is_monitoring:
            return
        
        self._reset_metrics()
        self.current_session = session_name
        self.start_time = time.time()
        self.is_monitoring = True
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Log start
        self.logger.log_metrics(
            "energy_monitoring_start",
            {"session": session_name, "sampling_interval": self.sampling_interval}
        )
    
    def stop_monitoring(self) -> Dict:
        """
        Stop monitoring and return statistics.
        
        Returns:
            Dictionary with monitoring statistics
        """
        if not self.is_monitoring:
            return {}
        
        self.is_monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        
        # Calculate statistics
        duration = time.time() - self.start_time
        stats = {
            "session": self.current_session,
            "duration": duration,
            "total_energy_joules": self.total_energy,
            "average_power_watts": self.total_energy / duration if duration > 0 else 0,
            "peak_power_watts": self.peak_power,
            "peak_temperature_celsius": self.peak_temperature,
            "peak_cpu_percent": self.peak_cpu,
            "peak_memory_percent": self.peak_memory,
            "number_of_samples": len(self.snapshots)
        }
        
        # Save session data
        session_file = self.monitoring_dir / f"{self.current_session}_{int(self.start_time)}.json"
        session_data = {
            "statistics": stats,
            "snapshots": [s.to_dict() for s in self.snapshots]
        }
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        # Log statistics
        self.logger.log_metrics(f"energy_monitoring_{self.current_session}", stats)
        
        return stats
    
    def get_current_stats(self) -> Dict:
        """
        Get current monitoring statistics without stopping.
        
        Returns:
            Dictionary with current statistics
        """
        if not self.is_monitoring:
            return {}
        
        duration = time.time() - self.start_time
        return {
            "session": self.current_session,
            "duration": duration,
            "total_energy_joules": self.total_energy,
            "average_power_watts": self.total_energy / duration if duration > 0 else 0,
            "peak_power_watts": self.peak_power,
            "peak_temperature_celsius": self.peak_temperature,
            "peak_cpu_percent": self.peak_cpu,
            "peak_memory_percent": self.peak_memory,
            "is_monitoring": True
        }
    
    def estimate_total_cost(self, kwh_rate: float = 0.12) -> Dict:
        """
        Estimate energy cost based on current usage.
        
        Args:
            kwh_rate: Cost per kilowatt-hour in dollars
            
        Returns:
            Dictionary with cost estimates
        """
        if not self.total_energy:
            return {
                "energy_kwh": 0.0,
                "cost_usd": 0.0,
                "rate_used": kwh_rate
            }
        
        # Convert joules to kilowatt-hours
        kwh = self.total_energy / 3600000
        cost = kwh * kwh_rate
        
        return {
            "energy_kwh": kwh,
            "cost_usd": cost,
            "rate_used": kwh_rate
        }
