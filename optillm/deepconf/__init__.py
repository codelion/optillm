"""
DeepConf plugin for OptILLM

Implements confidence-aware reasoning with early termination for local models.
Based on "Deep Think with Confidence" by Fu et al.
"""

from .deepconf import deepconf_decode

__all__ = ['deepconf_decode']