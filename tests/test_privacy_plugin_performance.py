#!/usr/bin/env python3
"""
Test to ensure privacy plugin resources are properly cached and not reloaded on each request.
This test will fail if resources are being recreated on every call, preventing performance regressions.
"""

import time
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import importlib

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_privacy_plugin_resource_caching():
    """
    Test that expensive resources (AnalyzerEngine, AnonymizerEngine) are created only once
    and reused across multiple plugin invocations.
    """
    print("Testing privacy plugin resource caching...")

    # Need to reset the module state before testing
    if 'optillm.plugins.privacy_plugin' in sys.modules:
        del sys.modules['optillm.plugins.privacy_plugin']

    # Mock the expensive AnalyzerEngine and AnonymizerEngine at the module level before import
    with patch('presidio_analyzer.AnalyzerEngine') as MockAnalyzerEngine, \
         patch('presidio_anonymizer.AnonymizerEngine') as MockAnonymizerEngine, \
         patch('spacy.util.is_package', return_value=True):

        # Set up mock instances
        mock_analyzer_instance = MagicMock()
        mock_analyzer_instance.analyze.return_value = []
        MockAnalyzerEngine.return_value = mock_analyzer_instance

        mock_anonymizer_instance = MagicMock()
        mock_anonymizer_instance.anonymize.return_value = MagicMock(text="anonymized text")
        mock_anonymizer_instance.add_anonymizer = MagicMock()
        MockAnonymizerEngine.return_value = mock_anonymizer_instance

        # Import the module with mocks in place
        import optillm.plugins.privacy_plugin as privacy_plugin

        # Mock client for the run function
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="response"))]
        mock_response.usage.completion_tokens = 10
        mock_client.chat.completions.create.return_value = mock_response

        # First invocation
        print("First invocation...")
        result1, tokens1 = privacy_plugin.run("system", "query 1", mock_client, "model")

        # Check that resources were created once
        assert MockAnalyzerEngine.call_count == 1, f"AnalyzerEngine created {MockAnalyzerEngine.call_count} times, expected 1"
        assert MockAnonymizerEngine.call_count == 1, f"AnonymizerEngine created {MockAnonymizerEngine.call_count} times, expected 1"

        # Second invocation
        print("Second invocation...")
        result2, tokens2 = privacy_plugin.run("system", "query 2", mock_client, "model")

        # Check that resources were NOT created again
        assert MockAnalyzerEngine.call_count == 1, f"AnalyzerEngine created {MockAnalyzerEngine.call_count} times after 2nd call, expected 1"
        assert MockAnonymizerEngine.call_count == 1, f"AnonymizerEngine created {MockAnonymizerEngine.call_count} times after 2nd call, expected 1"

        # Third invocation to be extra sure
        print("Third invocation...")
        result3, tokens3 = privacy_plugin.run("system", "query 3", mock_client, "model")

        # Still should be 1
        assert MockAnalyzerEngine.call_count == 1, f"AnalyzerEngine created {MockAnalyzerEngine.call_count} times after 3rd call, expected 1"
        assert MockAnonymizerEngine.call_count == 1, f"AnonymizerEngine created {MockAnonymizerEngine.call_count} times after 3rd call, expected 1"

        print("✅ Privacy plugin resource caching test PASSED - Resources are properly cached!")
        return True

def test_privacy_plugin_performance():
    """
    Test that multiple invocations of the privacy plugin don't have degraded performance.
    This catches the actual performance issue even without mocking.
    """
    print("\nTesting privacy plugin performance (real execution)...")

    try:
        # Try to import the actual plugin
        import optillm.plugins.privacy_plugin as privacy_plugin

        # Check if required dependencies are available
        try:
            import spacy
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
        except ImportError as e:
            print(f"⚠️  Skipping performance test - dependencies not installed: {e}")
            return True

        # Mock client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="response"))]
        mock_response.usage.completion_tokens = 10
        mock_client.chat.completions.create.return_value = mock_response

        # Warm-up call (might include model download)
        print("Warm-up call...")
        start = time.time()
        privacy_plugin.run("system", "warm up query", mock_client, "model")
        warmup_time = time.time() - start
        print(f"Warm-up time: {warmup_time:.2f}s")

        # First real measurement
        print("First measurement call...")
        start = time.time()
        privacy_plugin.run("system", "test query 1", mock_client, "model")
        first_time = time.time() - start
        print(f"First call time: {first_time:.2f}s")

        # Second measurement - should be fast if caching works
        print("Second measurement call...")
        start = time.time()
        privacy_plugin.run("system", "test query 2", mock_client, "model")
        second_time = time.time() - start
        print(f"Second call time: {second_time:.2f}s")

        # Third measurement
        print("Third measurement call...")
        start = time.time()
        privacy_plugin.run("system", "test query 3", mock_client, "model")
        third_time = time.time() - start
        print(f"Third call time: {third_time:.2f}s")

        # Performance assertions
        # Second and third calls should be much faster than first (at least 10x faster)
        # Allow some tolerance for the first call as it might still be initializing
        max_acceptable_time = 2.0  # 2 seconds max for subsequent calls

        if second_time > max_acceptable_time:
            raise AssertionError(f"Second call took {second_time:.2f}s, expected < {max_acceptable_time}s. Resources might not be cached!")

        if third_time > max_acceptable_time:
            raise AssertionError(f"Third call took {third_time:.2f}s, expected < {max_acceptable_time}s. Resources might not be cached!")

        print(f"✅ Privacy plugin performance test PASSED - Subsequent calls are fast ({second_time:.2f}s, {third_time:.2f}s)!")
        return True

    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        raise

def test_singleton_instances_are_reused():
    """
    Direct test that singleton instances are the same object across calls.
    """
    print("\nTesting singleton instance reuse...")

    try:
        import optillm.plugins.privacy_plugin as privacy_plugin
        importlib.reload(privacy_plugin)

        # Get first instances
        analyzer1 = privacy_plugin.get_analyzer_engine()
        anonymizer1 = privacy_plugin.get_anonymizer_engine()

        # Get second instances
        analyzer2 = privacy_plugin.get_analyzer_engine()
        anonymizer2 = privacy_plugin.get_anonymizer_engine()

        # They should be the exact same object
        assert analyzer1 is analyzer2, "AnalyzerEngine instances are not the same object!"
        assert anonymizer1 is anonymizer2, "AnonymizerEngine instances are not the same object!"

        print("✅ Singleton instance test PASSED - Same objects are reused!")
        return True

    except ImportError as e:
        print(f"⚠️  Skipping singleton test - dependencies not installed: {e}")
        return True
    except Exception as e:
        print(f"❌ Singleton test failed: {e}")
        raise

def test_recognizers_not_reloaded():
    """
    Test that recognizers are not fetched/reloaded on each analyze() call.
    This prevents the performance regression where "Fetching all recognizers for language en"
    appears in logs on every request.
    """
    print("\nTesting that recognizers are not reloaded on each call...")

    # Reset module state
    if 'optillm.plugins.privacy_plugin' in sys.modules:
        del sys.modules['optillm.plugins.privacy_plugin']

    try:
        # Mock at the presidio level to track registry calls
        with patch('presidio_analyzer.AnalyzerEngine') as MockAnalyzerEngine, \
             patch('spacy.util.is_package', return_value=True):

            # Create a mock analyzer instance
            mock_analyzer_instance = MagicMock()
            mock_registry = MagicMock()

            # Track calls to get_recognizers
            mock_registry.get_recognizers = MagicMock(return_value=[])
            mock_analyzer_instance.registry = mock_registry
            mock_analyzer_instance.analyze = MagicMock(return_value=[])

            MockAnalyzerEngine.return_value = mock_analyzer_instance

            # Import module with mocks
            import optillm.plugins.privacy_plugin as privacy_plugin

            # First call to get_analyzer_engine - should create and warm up
            analyzer1 = privacy_plugin.get_analyzer_engine()
            initial_analyze_calls = mock_analyzer_instance.analyze.call_count

            print(f"Warm-up analyze calls: {initial_analyze_calls}")
            assert initial_analyze_calls == 1, f"Expected 1 warm-up analyze call, got {initial_analyze_calls}"

            # Second call - should return cached instance without additional analyze
            analyzer2 = privacy_plugin.get_analyzer_engine()
            second_analyze_calls = mock_analyzer_instance.analyze.call_count

            print(f"Total analyze calls after second get_analyzer_engine: {second_analyze_calls}")
            assert second_analyze_calls == 1, f"Analyzer should not call analyze() again on cached retrieval, got {second_analyze_calls} calls"

            # Verify it's the same instance
            assert analyzer1 is analyzer2, "Should return the same cached analyzer instance"

            print("✅ Recognizer reload test PASSED - Recognizers are pre-warmed and not reloaded!")
            return True

    except ImportError as e:
        print(f"⚠️  Skipping recognizer reload test - dependencies not installed: {e}")
        return True
    except Exception as e:
        print(f"❌ Recognizer reload test failed: {e}")
        raise

if __name__ == "__main__":
    print("=" * 60)
    print("Privacy Plugin Performance & Caching Tests")
    print("=" * 60)

    all_passed = True

    try:
        test_privacy_plugin_resource_caching()
    except Exception as e:
        all_passed = False
        print(f"❌ Resource caching test failed: {e}")

    try:
        test_singleton_instances_are_reused()
    except Exception as e:
        all_passed = False
        print(f"❌ Singleton instance test failed: {e}")

    try:
        test_recognizers_not_reloaded()
    except Exception as e:
        all_passed = False
        print(f"❌ Recognizer reload test failed: {e}")

    try:
        test_privacy_plugin_performance()
    except Exception as e:
        all_passed = False
        print(f"❌ Performance test failed: {e}")

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("Privacy plugin resources are properly cached.")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED!")
        print("Privacy plugin may have performance issues.")
        sys.exit(1)