#!/usr/bin/env python3
"""
Test plugin functionality
"""

import sys
import os
import importlib

# Try to import pytest, but don't fail if it's not available
try:
    import pytest
except ImportError:
    pytest = None

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optillm import plugin_approaches, load_plugins


def test_plugin_module_imports():
    """Test that plugin modules can be imported"""
    plugin_modules = [
        'optillm.plugins.memory_plugin',
        'optillm.plugins.readurls_plugin', 
        'optillm.plugins.privacy_plugin',
        'optillm.plugins.genselect_plugin',
        'optillm.plugins.majority_voting_plugin'
    ]
    
    for module_name in plugin_modules:
        try:
            module = importlib.import_module(module_name)
            assert hasattr(module, 'run'), f"{module_name} missing 'run' function"
            assert hasattr(module, 'SLUG'), f"{module_name} missing 'SLUG' attribute"
        except ImportError as e:
            if pytest:
                pytest.fail(f"Failed to import {module_name}: {e}")
            else:
                raise AssertionError(f"Failed to import {module_name}: {e}")


def test_plugin_approach_detection():
    """Test plugin approach detection after loading"""
    # Load plugins first
    load_plugins()
    
    # Check if known plugins are loaded
    expected_plugins = ["memory", "readurls", "privacy"]
    for plugin_name in expected_plugins:
        assert plugin_name in plugin_approaches, f"Plugin {plugin_name} not loaded"


def test_memory_plugin_structure():
    """Test memory plugin has required structure"""
    import optillm.plugins.memory_plugin as plugin
    assert hasattr(plugin, 'run')
    assert hasattr(plugin, 'SLUG')
    assert plugin.SLUG == "memory"
    assert hasattr(plugin, 'Memory')  # Check for Memory class


def test_genselect_plugin():
    """Test genselect plugin module"""
    import optillm.plugins.genselect_plugin as plugin
    assert hasattr(plugin, 'run')
    assert hasattr(plugin, 'SLUG')
    assert hasattr(plugin, 'DEFAULT_NUM_CANDIDATES')
    assert plugin.SLUG == "genselect"


def test_majority_voting_plugin():
    """Test majority voting plugin module"""
    import optillm.plugins.majority_voting_plugin as plugin
    assert hasattr(plugin, 'run')
    assert hasattr(plugin, 'SLUG')
    assert hasattr(plugin, 'extract_answer')
    assert hasattr(plugin, 'normalize_answer')
    assert plugin.SLUG == "majority_voting"


if __name__ == "__main__":
    print("Running plugin tests...")
    
    try:
        test_plugin_module_imports()
        print("✅ Plugin module imports test passed")
    except Exception as e:
        print(f"❌ Plugin module imports test failed: {e}")
    
    try:
        test_plugin_approach_detection()
        print("✅ Plugin approach detection test passed")
    except Exception as e:
        print(f"❌ Plugin approach detection test failed: {e}")
    
    try:
        test_memory_plugin_structure()
        print("✅ Memory plugin structure test passed")
    except Exception as e:
        print(f"❌ Memory plugin structure test failed: {e}")
    
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