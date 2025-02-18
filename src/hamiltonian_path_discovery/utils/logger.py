"""
Logger module for structured logging.
"""
import logging
import time
from typing import Dict, Any

class StructuredLogger:
    def __init__(self):
        """Initialize structured logger."""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def log_error(self, message: str, **kwargs):
        """Log an error message with optional metadata."""
        self.logger.error(message, **kwargs)
    
    def log_info(self, message: str, **kwargs):
        """Log an info message with optional metadata."""
        self.logger.info(message, **kwargs)
    
    def log_api_request(self, url: str, method: str = "POST", **kwargs):
        """Log an API request."""
        self.log_info(f"Making API request to {url}", extra={"method": method, **kwargs})
    
    def log_api_response(self, response: Dict[str, Any], **kwargs):
        """Log an API response."""
        self.log_info("Received API response", extra={"response": response, **kwargs})
