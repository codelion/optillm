"""
Deep Think Plugin for OptILM

A plugin that combines SELF-DISCOVER framework with uncertainty-routed 
chain-of-thought for enhanced reasoning capabilities.
"""

from .self_discover import SelfDiscover
from .uncertainty_cot import UncertaintyRoutedCoT

__all__ = ['SelfDiscover', 'UncertaintyRoutedCoT']