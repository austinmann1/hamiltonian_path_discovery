"""
Tests for the prompt manager's iterative refinement functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path
from src.prompting.prompt_manager import PromptManager, PromptTemplate, PromptResult
from src.verification.verification_oracle import VerificationOracle

class MockLLMInterface:
    def __init__(self, responses):
        self.responses = responses
        self.current_response = 0
        self.prompts = []
        
    def generate_code(self, prompt):
        self.prompts.append(prompt)
        if self.current_response >= len(self.responses):
            return {"code": [], "success": False, "error": "No more responses"}
        response = self.responses[self.current_response]
        self.current_response += 1
        return response

@pytest.fixture
def test_graph():
    """Simple 4-node graph with known Hamiltonian path."""
    return {
        "adjacency_matrix": "0 1 0 1\n1 0 1 0\n0 1 0 1\n1 0 1 0",
        "num_nodes": 4,
        "start_node": 0,
        "end_node": 3,
        "valid_path": [0, 1, 2, 3]
    }

@pytest.fixture
def mock_energy_monitor():
    monitor = Mock()
    monitor.start_monitoring = Mock()
    monitor.stop_monitoring = Mock(return_value={"total_energy_joules": 10.0})
    monitor.get_current_metrics = Mock(return_value={"total_energy_joules": 2.0})
    return monitor

@pytest.fixture
def prompt_manager(tmp_path, mock_energy_monitor):
    with patch('src.prompting.prompt_manager.EnergyMonitor', return_value=mock_energy_monitor):
        # Create PromptManager first
        manager = PromptManager(data_dir=str(tmp_path))
        
        # Mock verifier directly
        def verify_side_effect(matrix, path):
            if path == [0, 1, 2, 3]:
                return {"is_valid": True, "success_rate": 1.0, "path": path}
            return {"is_valid": False, "error": "Invalid path", "path": path}
        manager.verifier.verify_with_explanation = Mock(side_effect=verify_side_effect)
        
        # Mock template
        template = PromptTemplate(
            name="basic",
            template="Test prompt for {adjacency_matrix}\nPrevious attempts: {previous_attempts}",
            parameters={"adjacency_matrix": "string", "previous_attempts": "string"},
            description="Test template for unit tests",
            examples=[
                {
                    "input": {
                        "adjacency_matrix": "0 1\n1 0",
                        "previous_attempts": "[]"
                    },
                    "output": "[0, 1]"
                }
            ],
            metrics={
                "success_rate": 1.0,
                "average_attempts": 1.0,
                "average_energy": 1.0
            }
        )
        manager.templates["basic"] = template
        
        return manager

def test_successful_first_attempt(prompt_manager, test_graph):
    """Test when LLM succeeds on first attempt."""
    llm = MockLLMInterface([
        {"code": test_graph["valid_path"], "success": True}
    ])
    
    result = prompt_manager.execute_prompt(
        "basic",
        test_graph,
        llm,
        max_attempts=3
    )
    
    assert result.success
    assert result.verification_result["is_valid"]
    assert not result.error
    assert result.generated_code == test_graph["valid_path"]

def test_success_after_refinement(prompt_manager, test_graph):
    """Test when LLM succeeds after a few attempts."""
    llm = MockLLMInterface([
        {"code": [0, 2, 1, 3], "success": True},  # Invalid path
        {"code": [0, 1, 3, 2], "success": True},  # Another invalid path
        {"code": test_graph["valid_path"], "success": True}  # Valid path
    ])
    
    result = prompt_manager.execute_prompt(
        "basic",
        test_graph,
        llm,
        max_attempts=5
    )
    
    assert result.success
    assert result.verification_result["is_valid"]
    assert result.generated_code == test_graph["valid_path"]
    assert len(llm.prompts) == 3  # Should have tried 3 times

def test_energy_threshold_stopping(prompt_manager, test_graph):
    """Test stopping when energy threshold is exceeded."""
    llm = MockLLMInterface([
        {"code": [0, 2, 1, 3], "success": True},
        {"code": [0, 1, 3, 2], "success": True}
    ])
    
    result = prompt_manager.execute_prompt(
        "basic",
        test_graph,
        llm,
        max_attempts=5,
        max_energy_threshold=5.0  # Set low threshold
    )
    
    assert not result.success
    assert "energy threshold" in result.error.lower()
    assert len(llm.prompts) <= 3  # Should stop early due to energy

def test_previous_attempts_feedback(prompt_manager, test_graph):
    """Test that previous attempts are included in subsequent prompts."""
    llm = MockLLMInterface([
        {"code": [0, 2, 1, 3], "success": True},  # First attempt
        {"code": test_graph["valid_path"], "success": True}  # Second attempt
    ])
    
    result = prompt_manager.execute_prompt(
        "basic",
        test_graph,
        llm,
        max_attempts=3
    )
    
    assert len(llm.prompts) == 2
    assert "Previous attempts: []" in llm.prompts[0]  # First attempt
    
    # Check that second prompt includes the first attempt's path
    second_prompt = llm.prompts[1]
    assert all(x in second_prompt for x in ["0", "2", "1", "3"])  # Check path numbers
    assert all(x in second_prompt for x in ["path", "error", "attempt"])  # Check fields

def test_best_result_tracking(prompt_manager, test_graph):
    """Test that the best result is tracked and returned."""
    llm = MockLLMInterface([
        {"code": [0, 2, 1, 3], "success": True},  # Invalid but shorter
        {"code": test_graph["valid_path"], "success": True},  # Valid path
        {"code": [0, 1, 2, 3, 1], "success": True}  # Invalid and longer
    ])
    
    result = prompt_manager.execute_prompt(
        "basic",
        test_graph,
        llm,
        max_attempts=3
    )
    
    assert result.success
    assert result.verification_result["is_valid"]
    assert result.generated_code == test_graph["valid_path"]

def test_error_handling(prompt_manager, test_graph):
    """Test handling of various error conditions."""
    llm = MockLLMInterface([
        {"code": "not a list", "success": True},  # Invalid format
        {"code": None, "success": False},  # None response
        {"code": [], "success": True},  # Empty path
        {"error": "API error", "success": False}  # API error
    ])
    
    result = prompt_manager.execute_prompt(
        "basic",
        test_graph,
        llm,
        max_attempts=4
    )
    
    assert not result.success
    assert result.error
    assert isinstance(result.generated_code, list)

def test_success_threshold(prompt_manager, test_graph):
    """Test stopping when success threshold is met."""
    llm = MockLLMInterface([
        {"code": test_graph["valid_path"], "success": True}
    ])
    
    result = prompt_manager.execute_prompt(
        "basic",
        test_graph,
        llm,
        max_attempts=5,
        min_success_threshold=0.8
    )
    
    assert result.success
    assert result.verification_result["success_rate"] >= 0.8
