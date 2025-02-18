"""
Tests for the logging infrastructure.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import json
from src.logging.structured_logger import StructuredLogger
from src.logging.metrics_tracker import MetricsTracker
from src.logging.experiment_logger import ExperimentLogger, IterationResult

class TestLoggingInfrastructure(unittest.TestCase):
    """Test suite for logging infrastructure."""
    
    def setUp(self):
        """Set up temporary directories for testing."""
        self.test_dir = tempfile.mkdtemp()
        self.logger = StructuredLogger(self.test_dir)
        self.metrics = MetricsTracker()
        self.experiment = ExperimentLogger(self.test_dir)
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.test_dir)
    
    def test_structured_logging(self):
        """Test structured logging functionality."""
        # Log an experiment
        exp_id = self.logger.log_experiment(
            phase="testing",
            metrics={"accuracy": 0.95},
            iterations=[{"iter": 1, "result": "success"}]
        )
        
        # Verify log file exists and contains valid JSON
        log_file = Path(self.test_dir) / "experiments.jsonl"
        self.assertTrue(log_file.exists())
        
        with open(log_file) as f:
            data = json.loads(f.readline())
            self.assertEqual(data["phase"], "testing")
            self.assertEqual(data["metrics"]["accuracy"], 0.95)
    
    def test_metrics_tracking(self):
        """Test metrics tracking functionality."""
        # Simulate some iterations
        self.metrics.start_iteration()
        self.metrics.end_iteration(path_found=True, is_valid=True)
        
        self.metrics.start_iteration()
        self.metrics.end_iteration(path_found=True, is_valid=False)
        
        # Check metrics
        current_metrics = self.metrics.get_current_metrics()
        self.assertEqual(current_metrics.path_validity_rate, 0.5)
        self.assertGreaterEqual(current_metrics.energy_per_path, 0)
    
    def test_experiment_logging(self):
        """Test experiment logging functionality."""
        # Start experiment
        exp_id = self.experiment.start_experiment(
            description="Test experiment",
            config={"model": "test", "params": {"temp": 0.5}}
        )
        
        # Log some iterations
        for i in range(3):
            result = IterationResult(
                iteration=i,
                prompt=f"Test prompt {i}",
                generated_code=f"def test{i}(): pass",
                verification_result=True,
                energy_usage=10.0,
                execution_time=1.0
            )
            self.experiment.log_iteration(result)
        
        # End experiment
        summary = self.experiment.end_experiment()
        
        # Verify summary
        self.assertEqual(summary["total_iterations"], 3)
        self.assertEqual(summary["total_energy"], 30.0)
        self.assertEqual(summary["total_time"], 3.0)

if __name__ == '__main__':
    unittest.main()
