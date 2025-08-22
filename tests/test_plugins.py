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
        'optillm.plugins.majority_voting_plugin',
        'optillm.plugins.web_search_plugin',
        'optillm.plugins.deep_research_plugin',
        'optillm.plugins.deepthink_plugin',
        'optillm.plugins.longcepo_plugin',
        'optillm.plugins.spl_plugin'
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
    expected_plugins = ["memory", "readurls", "privacy", "web_search", "deep_research", "deepthink", "longcepo", "spl"]
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
    assert hasattr(plugin, 'extract_final_answer')
    assert hasattr(plugin, 'normalize_response')
    assert plugin.SLUG == "majority_voting"


def test_web_search_plugin():
    """Test web search plugin module"""
    import optillm.plugins.web_search_plugin as plugin
    assert hasattr(plugin, 'run')
    assert hasattr(plugin, 'SLUG')
    assert hasattr(plugin, 'GoogleSearcher')
    assert hasattr(plugin, 'extract_search_queries')
    assert plugin.SLUG == "web_search"


def test_deep_research_plugin():
    """Test deep research plugin module"""
    import optillm.plugins.deep_research_plugin as plugin
    assert hasattr(plugin, 'run')
    assert hasattr(plugin, 'SLUG')
    assert hasattr(plugin, 'DeepResearcher')
    assert plugin.SLUG == "deep_research"


def test_deepthink_plugin_imports():
    """Test deepthink plugin and its submodules can be imported"""
    # Test main plugin
    import optillm.plugins.deepthink_plugin as plugin
    assert hasattr(plugin, 'run')
    assert hasattr(plugin, 'SLUG')
    assert plugin.SLUG == "deepthink"
    
    # Test submodules can be imported
    from optillm.plugins.deepthink import SelfDiscover, UncertaintyRoutedCoT
    assert SelfDiscover is not None
    assert UncertaintyRoutedCoT is not None


def test_longcepo_plugin():
    """Test longcepo plugin module"""
    import optillm.plugins.longcepo_plugin as plugin
    assert hasattr(plugin, 'run')
    assert hasattr(plugin, 'SLUG')
    assert plugin.SLUG == "longcepo"
    
    # Test submodule can be imported
    from optillm.plugins.longcepo import run_longcepo
    assert run_longcepo is not None


def test_spl_plugin():
    """Test spl plugin module"""
    import optillm.plugins.spl_plugin as plugin
    assert hasattr(plugin, 'run')
    assert hasattr(plugin, 'SLUG')
    assert plugin.SLUG == "spl"
    
    # Test submodule can be imported
    from optillm.plugins.spl import run_spl
    assert run_spl is not None


def test_plugin_subdirectory_imports():
    """Test all plugins with subdirectories can import their submodules"""
    # Test deep_research
    from optillm.plugins.deep_research import DeepResearcher
    assert DeepResearcher is not None
    
    # Test deepthink
    from optillm.plugins.deepthink import SelfDiscover, UncertaintyRoutedCoT
    assert SelfDiscover is not None
    assert UncertaintyRoutedCoT is not None
    
    # Test longcepo
    from optillm.plugins.longcepo import run_longcepo
    assert run_longcepo is not None
    
    # Test spl
    from optillm.plugins.spl import run_spl
    assert run_spl is not None


def test_no_relative_import_errors():
    """Test that plugins load without relative import errors"""
    import importlib
    import sys
    
    plugins_with_subdirs = [
        'optillm.plugins.deepthink_plugin',
        'optillm.plugins.deep_research_plugin',
        'optillm.plugins.longcepo_plugin',
        'optillm.plugins.spl_plugin'
    ]
    
    for plugin_name in plugins_with_subdirs:
        # Clear any previously loaded modules to test fresh import
        modules_to_clear = [k for k in sys.modules.keys() if k.startswith(plugin_name)]
        for mod in modules_to_clear:
            del sys.modules[mod]
        
        try:
            module = importlib.import_module(plugin_name)
            # Try to access the run function to ensure full initialization works
            assert hasattr(module, 'run'), f"{plugin_name} missing run function"
        except ImportError as e:
            if "attempted relative import" in str(e):
                if pytest:
                    pytest.fail(f"Relative import error in {plugin_name}: {e}")
                else:
                    raise AssertionError(f"Relative import error in {plugin_name}: {e}")
            else:
                raise


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
    
    try:
        test_web_search_plugin()
        print("✅ Web search plugin test passed")
    except Exception as e:
        print(f"❌ Web search plugin test failed: {e}")
    
    try:
        test_deep_research_plugin()
        print("✅ Deep research plugin test passed")
    except Exception as e:
        print(f"❌ Deep research plugin test failed: {e}")
    
    try:
        test_deepthink_plugin_imports()
        print("✅ Deepthink plugin imports test passed")
    except Exception as e:
        print(f"❌ Deepthink plugin imports test failed: {e}")
    
    try:
        test_longcepo_plugin()
        print("✅ LongCePO plugin test passed")
    except Exception as e:
        print(f"❌ LongCePO plugin test failed: {e}")
    
    try:
        test_spl_plugin()
        print("✅ SPL plugin test passed")
    except Exception as e:
        print(f"❌ SPL plugin test failed: {e}")
    
    try:
        test_plugin_subdirectory_imports()
        print("✅ Plugin subdirectory imports test passed")
    except Exception as e:
        print(f"❌ Plugin subdirectory imports test failed: {e}")
    
    try:
        test_no_relative_import_errors()
        print("✅ No relative import errors test passed")
    except Exception as e:
        print(f"❌ Relative import errors test failed: {e}")
    
    print("\nDone!")