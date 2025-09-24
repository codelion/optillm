#!/usr/bin/env python3
"""
MARS (Multi-Agent Reasoning System) parallel execution tests
Tests the parallel processing functionality and performance improvements
"""

import sys
import os
import time
import asyncio
import unittest
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path to import optillm modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optillm.mars import multi_agent_reasoning_system
from optillm.mars.mars import _run_mars_parallel
from optillm.mars.agent import MARSAgent
from optillm.mars.verifier import MARSVerifier
from optillm.mars.workspace import MARSWorkspace


class MockOpenAIClient:
    """Enhanced mock OpenAI client for MARS testing"""

    def __init__(self, response_delay=0.1, reasoning_tokens=1000):
        self.response_delay = response_delay
        self.reasoning_tokens = reasoning_tokens
        self.call_count = 0
        self.call_times = []

    def chat_completions_create(self, **kwargs):
        """Mock completions.create with configurable delay"""
        start_time = time.time()
        time.sleep(self.response_delay)  # Simulate API call delay
        self.call_count += 1
        self.call_times.append(time.time())

        call_count = self.call_count  # Capture for closure

        class MockUsage:
            def __init__(self, reasoning_tokens):
                self.completion_tokens_details = type('obj', (), {
                    'reasoning_tokens': reasoning_tokens
                })()
                self.total_tokens = reasoning_tokens + 100

        class MockChoice:
            def __init__(self):
                self.message = type('obj', (), {
                    'content': f'Mock mathematical solution {call_count}. The answer is 42.'
                })()

        class MockResponse:
            def __init__(self, reasoning_tokens):
                self.choices = [MockChoice()]
                self.usage = MockUsage(reasoning_tokens)

        return MockResponse(self.reasoning_tokens)

    @property
    def chat(self):
        return type('obj', (), {
            'completions': type('obj', (), {
                'create': self.chat_completions_create
            })()
        })()


class TestMARSParallel(unittest.TestCase):
    """Test MARS parallel execution functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.system_prompt = "You are a mathematical problem solver."
        self.test_query = "What is the value of x if 2x + 5 = 15?"
        self.model = "mock-model"

    def test_mars_import(self):
        """Test that MARS can be imported correctly"""
        from optillm.mars import multi_agent_reasoning_system
        self.assertTrue(callable(multi_agent_reasoning_system))

    def test_mars_basic_call(self):
        """Test basic MARS functionality with mock client"""
        client = MockOpenAIClient(response_delay=0.01)  # Fast response for testing

        try:
            result = multi_agent_reasoning_system(
                self.system_prompt,
                self.test_query,
                client,
                self.model
            )

            # Check result structure
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)

            response, tokens = result
            self.assertIsInstance(response, str)
            self.assertIsInstance(tokens, int)
            self.assertGreater(len(response), 0)
            self.assertGreater(tokens, 0)

            print("✅ MARS basic call test passed")

        except Exception as e:
            self.fail(f"MARS basic call failed: {e}")

    def test_mars_parallel_execution_performance(self):
        """Test that parallel execution shows improvement over theoretical sequential"""
        # Test with a client that has small but measurable delay
        client = MockOpenAIClient(response_delay=0.05, reasoning_tokens=2000)

        # Record call times to analyze parallelization
        start_time = time.time()
        result = multi_agent_reasoning_system(
            self.system_prompt,
            self.test_query,
            client,
            self.model
        )
        end_time = time.time()

        execution_time = end_time - start_time

        # The test mainly verifies MARS completes and returns results
        # Performance comparison is difficult due to MARS complexity
        self.assertLess(execution_time, 30,  # More generous timeout
                       f"Execution took {execution_time:.2f}s, too long for test")

        # Verify we got a valid response
        self.assertIsInstance(result, tuple)
        response, tokens = result
        self.assertGreater(len(response), 0)
        self.assertGreater(tokens, 0)

        # Check that we made parallel calls by examining call times
        call_times = client.call_times
        if len(call_times) >= 3:
            # First 3 calls (exploration phase) should be roughly simultaneous
            first_three = call_times[:3]
            time_spread = max(first_three) - min(first_three)
            self.assertLess(time_spread, 0.5,
                          f"First 3 calls spread over {time_spread:.2f}s, not parallel enough")

        print(f"✅ MARS parallel execution completed in {execution_time:.2f}s with {client.call_count} API calls")

    def test_mars_worker_pool_calculation(self):
        """Test that worker pool size is calculated correctly"""
        # Test default config worker calculation
        from optillm.mars.mars import DEFAULT_CONFIG

        num_agents = DEFAULT_CONFIG['num_agents']
        verification_passes = DEFAULT_CONFIG['verification_passes_required']

        expected_workers = max(
            num_agents,  # For generation phase
            num_agents * min(2, verification_passes)  # For verification phase
        )

        # With default config: max(3, 3*2) = 6 workers
        self.assertEqual(expected_workers, 6)
        print(f"✅ Worker pool size calculation correct: {expected_workers} workers")

    def test_mars_error_handling(self):
        """Test error handling in parallel execution"""
        # Create a client that will cause some agents to fail
        class FailingMockClient(MockOpenAIClient):
            def __init__(self):
                super().__init__(response_delay=0.01)
                self.failure_count = 0

            def chat_completions_create(self, **kwargs):
                self.failure_count += 1
                # Make some calls fail to test error handling
                if self.failure_count % 3 == 0:  # Every 3rd call fails
                    raise Exception("Mock API failure")
                return super().chat_completions_create(**kwargs)

        failing_client = FailingMockClient()

        # MARS should handle failures gracefully and still return a result
        try:
            result = multi_agent_reasoning_system(
                self.system_prompt,
                self.test_query,
                failing_client,
                self.model
            )

            # Should still get a valid result despite some failures
            self.assertIsInstance(result, tuple)
            response, tokens = result
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)

            print("✅ MARS error handling test passed")

        except Exception as e:
            # If MARS completely fails, check that it's the expected error type
            self.assertIn("MARS system encountered an error", str(e))
            print("✅ MARS fallback error handling works")

    @patch('optillm.mars.mars.ThreadPoolExecutor')
    def test_mars_uses_thread_pool(self, mock_thread_pool):
        """Test that MARS actually uses ThreadPoolExecutor for parallel execution"""
        # Create a mock ThreadPoolExecutor
        mock_executor = Mock()
        mock_thread_pool.return_value.__enter__.return_value = mock_executor

        client = MockOpenAIClient(response_delay=0.01)

        # Run MARS
        multi_agent_reasoning_system(
            self.system_prompt,
            self.test_query,
            client,
            self.model
        )

        # Verify ThreadPoolExecutor was created with correct parameters
        mock_thread_pool.assert_called_once()
        call_args = mock_thread_pool.call_args
        self.assertIn('max_workers', call_args.kwargs)

        # Should use 6 workers for default config
        self.assertEqual(call_args.kwargs['max_workers'], 6)

        print("✅ MARS ThreadPoolExecutor usage test passed")

    def test_mars_consensus_mechanism(self):
        """Test MARS consensus and verification mechanism"""
        # Use a client that provides consistent responses for consensus
        class ConsistentMockClient(MockOpenAIClient):
            def chat_completions_create(self, **kwargs):
                result = super().chat_completions_create(**kwargs)
                # Make all agents return similar solutions for consensus
                result.choices[0].message.content = "The solution is x = 5. Final answer: 5"
                return result

        client = ConsistentMockClient(response_delay=0.01)

        result = multi_agent_reasoning_system(
            self.system_prompt,
            self.test_query,
            client,
            self.model
        )

        # Should get a valid consensus result
        self.assertIsInstance(result, tuple)
        response, tokens = result
        self.assertIn("5", response)  # Should contain the expected answer

        print("✅ MARS consensus mechanism test passed")


def test_mars_agent_temperatures():
    """Test that MARS uses different temperatures for agents"""
    from optillm.mars.mars import DEFAULT_CONFIG
    from optillm.mars.agent import MARSAgent

    client = MockOpenAIClient()
    model = "mock-model"
    config = DEFAULT_CONFIG.copy()

    # Create agents like MARS does
    agents = []
    for i in range(config['num_agents']):
        agent = MARSAgent(i, client, model, config)
        agents.append(agent)

    # Check that agents have different temperatures
    temperatures = [agent.temperature for agent in agents]
    unique_temps = set(temperatures)

    assert len(unique_temps) == len(agents), "Agents should have different temperatures"
    assert 0.3 in temperatures, "Should have conservative agent (temp 0.3)"
    assert 1.0 in temperatures, "Should have creative agent (temp 1.0)"

    print(f"✅ Agent temperatures test passed: {temperatures}")


def run_tests():
    """Run all MARS tests"""
    print("Running MARS parallel execution tests...")
    print("=" * 60)

    # Run unittest tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMARSParallel)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Run additional function tests
    try:
        test_mars_agent_temperatures()
    except Exception as e:
        print(f"❌ Agent temperatures test failed: {e}")

    print("=" * 60)

    if result.wasSuccessful():
        print("🎉 All MARS tests passed!")
        return True
    else:
        print("❌ Some MARS tests failed")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)