#!/usr/bin/env python3
"""
Test plugin functionality
"""

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optillm.plugins import load_plugin, is_plugin_approach
from optillm.plugins.memory_plugin import should_enable_memory


def test_plugin_loading():
    """Test loading plugins"""
    # Test loading a known plugin
    plugin = load_plugin("memory")
    assert plugin is not None
    assert hasattr(plugin, 'run')
    
    # Test loading non-existent plugin returns None
    plugin = load_plugin("nonexistent")
    assert plugin is None


def test_is_plugin_approach():
    """Test plugin approach detection"""
    # Known plugins
    assert is_plugin_approach("memory") == True
    assert is_plugin_approach("readurls") == True
    assert is_plugin_approach("privacy") == True
    
    # Non-plugins
    assert is_plugin_approach("mcts") == False
    assert is_plugin_approach("bon") == False
    assert is_plugin_approach("nonexistent") == False


def test_memory_plugin_detection():
    """Test memory plugin auto-detection"""
    # Test with context length exceeding threshold
    long_context = "x" * 500000  # 500k chars
    assert should_enable_memory(long_context) == True
    
    # Test with short context
    short_context = "Hello world"
    assert should_enable_memory(short_context) == False
    
    # Test with explicit false in config
    assert should_enable_memory(long_context, {"memory": False}) == False
    
    # Test with explicit true in config
    assert should_enable_memory(short_context, {"memory": True}) == True


def test_genselect_plugin():
    """Test genselect plugin exists"""
    plugin = load_plugin("genselect")
    assert plugin is not None
    assert hasattr(plugin, 'run')
    assert hasattr(plugin, 'DEFAULT_NUM_CANDIDATES')


def test_majority_voting_plugin():
    """Test majority voting plugin"""
    plugin = load_plugin("majority_voting")
    assert plugin is not None
    assert hasattr(plugin, 'run')
    assert hasattr(plugin, 'extract_answer')
    assert hasattr(plugin, 'normalize_answer')


if __name__ == "__main__":
    print("Running plugin tests...")
    
    try:
        test_plugin_loading()
        print("✅ Plugin loading test passed")
    except Exception as e:
        print(f"❌ Plugin loading test failed: {e}")
    
    try:
        test_is_plugin_approach()
        print("✅ Plugin approach detection test passed")
    except Exception as e:
        print(f"❌ Plugin approach detection test failed: {e}")
    
    try:
        test_memory_plugin_detection()
        print("✅ Memory plugin detection test passed")
    except Exception as e:
        print(f"❌ Memory plugin detection test failed: {e}")
    
    try:
        test_genselect_plugin()
        print("✅ GenSelect plugin test passed")
    except Exception as e:
        print(f"❌ GenSelect plugin test failed: {e}")
    
    try:
        test_majority_voting_plugin()
        print("✅ Majority voting plugin test passed")
    except Exception as e:
        print(f"❌ Majority voting plugin test failed: {e}")
    
    print("\nDone!")