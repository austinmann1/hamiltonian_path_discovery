"""
Template initialization for prompting system.
"""

from .novel_path_discovery import (
    PATTERN_BASED_PROMPT,
    CONFLICT_AWARE_PROMPT,
    OPTIMIZATION_PROMPT,
    format_graph_properties,
    format_theoretical_insights,
    format_failure_patterns,
    format_implementation_analysis,
    format_subpath_patterns
)

__all__ = [
    'PATTERN_BASED_PROMPT',
    'CONFLICT_AWARE_PROMPT',
    'OPTIMIZATION_PROMPT',
    'format_graph_properties',
    'format_theoretical_insights',
    'format_failure_patterns',
    'format_implementation_analysis',
    'format_subpath_patterns'
]
