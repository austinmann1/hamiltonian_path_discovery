"""
Test suite for LLM Interface component.
"""
import pytest
from unittest.mock import patch, MagicMock
from src.llm_interface import LLMInterface
import json
import tempfile
import os

@pytest.fixture
def llm_interface():
    return LLMInterface()

def test_generate_code_success(llm_interface):
    # Mock OpenAI API response
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="def test_function(): pass"))
    ]
    
    with patch('openai.chat.completions.create', return_value=mock_response):
        code = llm_interface.generate_code("Test problem")
        assert code == "def test_function(): pass"
        assert len(llm_interface.conversation_history) == 1

def test_generate_code_with_test_cases(llm_interface):
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="def test_function(): pass"))
    ]
    
    test_cases = [{"input": [1, 2], "expected": 3}]
    
    with patch('openai.chat.completions.create', return_value=mock_response):
        code = llm_interface.generate_code("Test problem", test_cases)
        assert code == "def test_function(): pass"

def test_refine_code(llm_interface):
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="def fixed_function(): pass"))
    ]
    
    with patch('openai.chat.completions.create', return_value=mock_response):
        refined = llm_interface.refine_code(
            "def broken_function(): pass",
            "Test error",
            {"input": [1], "expected": 1}
        )
        assert refined == "def fixed_function(): pass"
        assert len(llm_interface.conversation_history) == 2

def test_conversation_persistence(llm_interface):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        # Add some conversation history
        llm_interface.conversation_history = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "response"}
        ]
        
        # Save conversation
        llm_interface.save_conversation(tmp.name)
        
        # Clear conversation history
        llm_interface.conversation_history = []
        
        # Load conversation
        llm_interface.load_conversation(tmp.name)
        
        assert len(llm_interface.conversation_history) == 2
        assert llm_interface.conversation_history[0]["role"] == "user"
        assert llm_interface.conversation_history[1]["role"] == "assistant"
        
        # Clean up
        os.unlink(tmp.name)

def test_generate_code_failure(llm_interface):
    with patch('openai.chat.completions.create', side_effect=Exception("API Error")):
        with pytest.raises(Exception) as exc_info:
            llm_interface.generate_code("Test problem", max_attempts=1)
        assert "Failed to generate code after 1 attempts" in str(exc_info.value)

def test_refine_code_failure(llm_interface):
    with patch('openai.chat.completions.create', side_effect=Exception("API Error")):
        with pytest.raises(Exception) as exc_info:
            llm_interface.refine_code("def test(): pass", "error")
        assert "Failed to refine code" in str(exc_info.value)
