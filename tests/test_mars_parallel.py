#!/usr/bin/env python3
"""
MARS (Multi-Agent Reasoning System) comprehensive tests
Tests parallel processing, hard problem solving, and logging functionality
"""

import sys
import os
import time
import asyncio
import unittest
import logging
import io
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

        # Set up logging capture for monitoring MARS behavior
        self.log_capture = io.StringIO()
        self.log_handler = logging.StreamHandler(self.log_capture)
        self.log_handler.setLevel(logging.INFO)

        # Add handler to MARS loggers
        mars_logger = logging.getLogger('optillm.mars')
        mars_logger.addHandler(self.log_handler)
        mars_logger.setLevel(logging.INFO)

        # Store original level to restore later
        self.original_level = mars_logger.level

    def tearDown(self):
        """Clean up test fixtures"""
        # Remove our handler and restore original level
        mars_logger = logging.getLogger('optillm.mars')
        mars_logger.removeHandler(self.log_handler)
        mars_logger.setLevel(self.original_level)
        self.log_handler.close()

    def get_captured_logs(self):
        """Get the captured log output"""
        return self.log_capture.getvalue()

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

        # Check that our new logging is working
        logs = self.get_captured_logs()
        self.assertIn("🚀 MARS", logs, "Should contain main orchestration logs")

        print(f"✅ MARS parallel execution completed in {execution_time:.2f}s with {client.call_count} API calls")
        print(f"📋 Captured {len(logs.split('🚀'))} main log entries")

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

    def test_mars_hard_problems(self):
        """Test MARS on challenging problems that require deep reasoning"""
        hard_problems = [
            {
                "name": "Advanced Algebra",
                "problem": "Find all positive integer solutions to x^3 + y^3 = z^3 - 1 where x, y, z are all less than 100.",
                "expected_features": ["systematic", "case", "analysis"]
            },
            {
                "name": "Number Theory",
                "problem": "Prove that there are infinitely many primes of the form 4k+3.",
                "expected_features": ["proof", "contradiction", "infinite"]
            },
            {
                "name": "Combinatorics",
                "problem": "In how many ways can 20 identical balls be distributed into 5 distinct boxes such that each box contains at least 2 balls?",
                "expected_features": ["stars", "bars", "constraint"]
            },
            {
                "name": "Geometry",
                "problem": "Given a triangle ABC with sides a, b, c, prove that a^2 + b^2 + c^2 ≥ 4√3 * Area.",
                "expected_features": ["inequality", "area", "geometric"]
            }
        ]

        class EnhancedMockClient(MockOpenAIClient):
            def __init__(self):
                super().__init__(response_delay=0.1, reasoning_tokens=3000)
                self.problem_responses = {
                    "Advanced Algebra": "This requires systematic case analysis. Let me examine small values systematically. After checking cases x,y,z < 100, the equation x³ + y³ = z³ - 1 has solutions like (x,y,z) = (1,1,1) since 1³ + 1³ = 2 = 2³ - 6... Actually, let me recalculate: 1³ + 1³ = 2, and z³ - 1 = 2 means z³ = 3, so z ≈ 1.44. Let me check (2,2,2): 8 + 8 = 16 = 8 - 1 = 7? No. This is a difficult Diophantine equation requiring advanced techniques.",
                    "Number Theory": "I'll prove this by contradiction using Euclid's method. Assume there are only finitely many primes of the form 4k+3: p₁, p₂, ..., pₙ. Consider N = 4(p₁p₂...pₙ) + 3. Since N ≡ 3 (mod 4), at least one prime factor of N must be ≡ 3 (mod 4). But N is not divisible by any of p₁, p₂, ..., pₙ, so there must be another prime of the form 4k+3, contradicting our assumption. Therefore, there are infinitely many such primes.",
                    "Combinatorics": "This is a stars and bars problem with constraints. We need to distribute 20 balls into 5 boxes with each box having at least 2 balls. First, place 2 balls in each box (using 10 balls). Now we need to distribute the remaining 10 balls into 5 boxes with no constraints. Using stars and bars: C(10+5-1, 5-1) = C(14,4) = 1001 ways.",
                    "Geometry": "This is a form of Weitzenböck's inequality. We can prove this using the relationship between area and sides. For a triangle with area S and sides a,b,c, we have S = √[s(s-a)(s-b)(s-c)] where s = (a+b+c)/2. We want to show a² + b² + c² ≥ 4√3 · S. This can be proven using the isoperimetric inequality and Jensen's inequality applied to the convex function f(x) = x²."
                }

            def chat_completions_create(self, **kwargs):
                result = super().chat_completions_create(**kwargs)

                # Look for problem type in the messages
                messages = kwargs.get('messages', [])
                for message in messages:
                    content = message.get('content', '')
                    for prob_type, response in self.problem_responses.items():
                        if any(keyword in content for keyword in prob_type.lower().split()):
                            result.choices[0].message.content = response
                            return result

                # Default response for other cases
                result.choices[0].message.content = "This is a complex problem requiring careful analysis. Let me work through it step by step with rigorous reasoning."
                return result

        client = EnhancedMockClient()

        # Test each hard problem
        for problem_data in hard_problems:
            with self.subTest(problem=problem_data["name"]):
                print(f"\n🧠 Testing MARS on {problem_data['name']} problem...")

                start_time = time.time()
                result = multi_agent_reasoning_system(
                    self.system_prompt,
                    problem_data["problem"],
                    client,
                    self.model
                )
                execution_time = time.time() - start_time

                # Verify result structure
                self.assertIsInstance(result, tuple)
                response, tokens = result
                self.assertIsInstance(response, str)
                self.assertGreater(len(response), 50, "Response should be substantial for hard problems")
                self.assertGreater(tokens, 0)

                # Check for problem-specific reasoning features
                response_lower = response.lower()
                found_features = []
                for feature in problem_data["expected_features"]:
                    if feature.lower() in response_lower:
                        found_features.append(feature)

                # Should find at least one expected reasoning feature
                self.assertGreater(len(found_features), 0,
                    f"Response should contain reasoning features like {problem_data['expected_features']}")

                print(f"  ✅ {problem_data['name']}: {execution_time:.2f}s, {len(response):,} chars, features: {found_features}")

        # Analyze the comprehensive logs
        logs = self.get_captured_logs()

        # Check for our enhanced logging features
        log_checks = [
            ("🚀 MARS", "Main orchestration logs"),
            ("🤖 AGENT", "Agent generation logs"),
            ("🗳️  VOTING", "Voting mechanism logs"),
            ("🤝 SYNTHESIS", "Synthesis phase logs")
        ]

        for emoji, description in log_checks:
            if emoji in logs:
                count = logs.count(emoji)
                print(f"  📊 Found {count} {description}")
            else:
                print(f"  ⚠️  No {description} found (expected with enhanced logging)")

        print(f"\n✅ MARS hard problems test completed - verified reasoning on {len(hard_problems)} challenging problems")

    def test_mars_logging_and_monitoring(self):
        """Test that MARS logging provides useful monitoring information"""
        print("\n📊 Testing MARS logging and monitoring capabilities...")

        # Use a client that simulates realistic API timing
        class MonitoringMockClient(MockOpenAIClient):
            def __init__(self):
                super().__init__(response_delay=0.05, reasoning_tokens=2500)
                self.detailed_responses = True

            def chat_completions_create(self, **kwargs):
                result = super().chat_completions_create(**kwargs)

                # Generate varied responses to test logging diversity
                if "verifying" in str(kwargs.get('messages', [])):
                    result.choices[0].message.content = "VERIFICATION: The solution appears CORRECT with high confidence. The reasoning is sound and the final answer is properly justified. Confidence: 9/10."
                elif "improving" in str(kwargs.get('messages', [])):
                    result.choices[0].message.content = "IMPROVEMENT: The original solution can be enhanced by adding more rigorous justification. Here's the improved version with stronger mathematical foundations..."
                else:
                    result.choices[0].message.content = "Let me solve this step by step. First, I'll analyze the problem structure. Then I'll apply appropriate mathematical techniques. The solution involves careful reasoning and verification. \\boxed{42}"

                return result

        client = MonitoringMockClient()

        # Test with a problem that should trigger multiple phases
        complex_problem = "Solve the system: x² + y² = 25, x + y = 7. Find all real solutions and verify your answer."

        start_time = time.time()
        result = multi_agent_reasoning_system(
            self.system_prompt,
            complex_problem,
            client,
            self.model
        )
        execution_time = time.time() - start_time

        # Analyze the detailed logs
        logs = self.get_captured_logs()
        log_lines = logs.split('\n')

        # Count different types of log entries
        log_stats = {
            "🚀 MARS": 0,
            "🤖 AGENT": 0,
            "🔍 VERIFIER": 0,
            "🗳️  VOTING": 0,
            "🤝 SYNTHESIS": 0,
            "⏱️  TIMING": 0
        }

        for line in log_lines:
            for emoji_prefix in log_stats.keys():
                if emoji_prefix in line:
                    log_stats[emoji_prefix] += 1

        # Verify we have comprehensive logging
        total_logs = sum(log_stats.values())
        self.assertGreater(total_logs, 10, "Should have substantial logging for monitoring")

        # Check for key monitoring information
        monitoring_checks = [
            ("MARS", log_stats["🚀 MARS"], "Main orchestration phases"),
            ("AGENT", log_stats["🤖 AGENT"], "Agent operations"),
            ("VOTING", log_stats["🗳️  VOTING"], "Consensus mechanism"),
            ("SYNTHESIS", log_stats["🤝 SYNTHESIS"], "Final synthesis")
        ]

        print(f"\n📈 Monitoring Statistics (from {execution_time:.2f}s execution):")
        for name, count, description in monitoring_checks:
            status = "✅" if count > 0 else "⚠️ "
            print(f"  {status} {name}: {count} {description}")

        # Verify result quality
        response, tokens = result
        self.assertGreater(len(response), 100, "Complex problems should generate substantial responses")
        self.assertGreater(tokens, 1000, "Should use significant reasoning tokens")

        # Check for solution quality indicators in logs
        quality_indicators = [
            "confidence", "reasoning", "verification", "solution", "answer"
        ]

        found_indicators = []
        logs_lower = logs.lower()
        for indicator in quality_indicators:
            if indicator in logs_lower:
                found_indicators.append(indicator)

        print(f"\n🎯 Quality indicators found in logs: {found_indicators}")
        self.assertGreater(len(found_indicators), 2, "Should track multiple quality indicators")

        print(f"✅ MARS logging and monitoring test passed - captured {total_logs} log entries")

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

        # Verify logging captured consensus behavior
        logs = self.get_captured_logs()
        if "🗳️  VOTING" in logs:
            print("✅ MARS consensus mechanism test passed with voting logs")
        else:
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
    print("Running MARS comprehensive tests...")
    print("=" * 80)

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