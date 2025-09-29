"""
MARS: Multi-Agent Reasoning System main orchestration with parallel execution
"""

import asyncio
import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import optillm
from optillm import conversation_logger

from .workspace import MARSWorkspace, AgentSolution
from .agent import MARSAgent
from .verifier import MARSVerifier
from .aggregator import MARSAggregator
from .strategy_network import StrategyNetwork
from .prompts import SYNTHESIS_PROMPT

logger = logging.getLogger(__name__)

# Default MARS configuration - simplified with OpenRouter effort parameter
DEFAULT_CONFIG = {
    'num_agents': 3,
    'max_iterations': 5,  # Balanced for quality vs efficiency
    'verification_passes_required': 2,  # Balanced for 5-iteration efficiency
    'consensus_threshold': 2,  # Keep at 2 for 3-agent setup
    'min_verified_solutions': 1,  # Keep minimal requirement
    'max_tokens': 64000,  # Increased default token budget for complex reasoning
    'max_verification_attempts': 3,
    'early_termination': True,
    'use_reasoning_api': True,
    # RSA-inspired aggregation parameters
    'enable_aggregation': True,  # Enable recursive self-aggregation
    'population_size': 6,  # N parameter: maintain larger population for diversity
    'aggregation_size': 3,  # K parameter: number of solutions to aggregate
    'aggregation_loops': 3,  # T parameter: number of aggregation iterations
    # Strategy Network parameters for cross-agent insight sharing
    'enable_strategy_network': True,  # Enable cross-agent strategy sharing
    'strategy_extraction_enabled': True,  # Extract reasoning strategies from solutions
    'cross_agent_enhancement': True,  # Generate enhanced solutions using peer strategies
}

def multi_agent_reasoning_system(
    system_prompt: str,
    initial_query: str,
    client,
    model: str,
    request_config: dict = None,
    request_id: str = None
) -> Tuple[str, int]:
    """
    Main MARS function implementing multi-agent reasoning with parallel execution

    Args:
        system_prompt: System-level instructions
        initial_query: The problem or task to solve
        client: OpenAI-compatible client for API calls
        model: Model identifier (should support OpenRouter reasoning API)
        request_id: Optional request ID for conversation logging

    Returns:
        Tuple of (final_solution, total_reasoning_tokens)
    """
    return asyncio.run(_run_mars_parallel(
        system_prompt, initial_query, client, model, request_config, request_id
    ))

async def _run_mars_parallel(
    system_prompt: str,
    initial_query: str,
    client,
    model: str,
    request_config: dict = None,
    request_id: str = None
) -> Tuple[str, int]:
    """Async implementation of MARS with parallel execution"""
    logger.info(f"Starting MARS with model: {model}")

    # Initialize configuration
    config = DEFAULT_CONFIG.copy()

    # Override max_tokens from request_config if provided
    if request_config and 'max_tokens' in request_config:
        config['max_tokens'] = request_config['max_tokens']
        logger.info(f"Using max_tokens from request: {config['max_tokens']}")
    else:
        logger.info(f"Using default max_tokens: {config['max_tokens']}")

    total_reasoning_tokens = 0

    # Calculate optimal worker count for parallel execution
    max_workers = max(
        config['num_agents'],  # For generation phase
        config['num_agents'] * min(2, config['verification_passes_required'])  # For verification
    )
    logger.info(f"Using {max_workers} parallel workers")

    # Initialize workspace for collaboration
    workspace = MARSWorkspace(initial_query, config)

    try:
        # Phase 1: Initialize Agents
        agents = []
        for i in range(config['num_agents']):
            agent = MARSAgent(i, client, model, config)
            agents.append(agent)

        logger.info(f"Initialized {len(agents)} agents with diverse temperatures")

        # Create thread pool executor for parallel API calls
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Phase 2: Multi-Agent Exploration (parallel)
            logger.info("Phase 1: Multi-Agent Exploration")
            exploration_tokens = await _run_exploration_phase_parallel(
                agents, workspace, request_id, executor
            )
            total_reasoning_tokens += exploration_tokens

            # Phase 2a: RSA-inspired Aggregation (if enabled)
            if config.get('enable_aggregation', True):
                logger.info("Phase 2a: RSA-inspired Solution Aggregation")
                aggregator = MARSAggregator(client, model, config)
                aggregation_tokens, aggregation_summary = await aggregator.run_aggregation_loops(
                    workspace, request_id, executor
                )
                total_reasoning_tokens += aggregation_tokens
                logger.info(f"Aggregation complete: {aggregation_summary}")

            # Phase 2b: Cross-Agent Strategy Sharing (if enabled)
            if config.get('enable_strategy_network', True):
                logger.info("Phase 2b: Cross-Agent Strategy Network")
                strategy_network = StrategyNetwork(client, model, config)

                # Extract reasoning strategies from agent solutions
                if config.get('strategy_extraction_enabled', True):
                    extracted_strategies = await strategy_network.extract_strategies_from_solutions(
                        workspace, request_id, executor
                    )

                    # Share strategies across agents and generate enhanced solutions
                    if config.get('cross_agent_enhancement', True) and extracted_strategies:
                        strategy_sharing_summary = await strategy_network.share_strategies_across_agents(
                            workspace, extracted_strategies, request_id, executor
                        )

                        strategy_insights = strategy_network.get_strategy_insights_summary()
                        logger.info(f"Strategy network complete: {strategy_insights}")

            # Phase 3: Verification System (parallel)
            logger.info("Phase 3: Verification System")
            verifier = MARSVerifier(agents, workspace, config)
            verification_summary = await verifier.verify_solutions_parallel(request_id, executor)

            # Phase 4: Iterative Improvement (if needed)
            iteration_count = 0
            while workspace.should_continue_iteration() and iteration_count < config['max_iterations']:
                iteration_count += 1
                logger.info(f"Phase 4: Iterative Improvement - Iteration {iteration_count}")

                # Improve unverified solutions (parallel)
                improvement_summary = await verifier.iterative_improvement_parallel(request_id, executor)
                total_reasoning_tokens += improvement_summary['total_reasoning_tokens']

                # Re-verify improved solutions (parallel)
                verification_summary = await verifier.verify_solutions_parallel(request_id, executor)

                # Check for early termination
                if config['early_termination'] and workspace.has_consensus():
                    logger.info("Early termination: consensus reached")
                    break

                workspace.iteration_count = iteration_count

        # Phase 5: Final Synthesis (sequential - needs all results)
        logger.info("Phase 5: Final Synthesis")
        final_solution, synthesis_tokens = _synthesize_final_solution(
            workspace, client, model, config, request_id
        )
        total_reasoning_tokens += synthesis_tokens

        # Set final solution in workspace
        workspace.set_final_solution(final_solution)

        # Log summary
        summary = workspace.get_summary()
        logger.info(f"MARS completed: {summary['verified_solutions']}/{summary['total_solutions']} solutions verified")
        logger.info(f"Total reasoning tokens: {total_reasoning_tokens}")

        return final_solution, total_reasoning_tokens

    except Exception as e:
        logger.error(f"MARS execution failed: {str(e)}")
        # Return error response
        error_response = f"MARS system encountered an error: {str(e)}\n\nAttempting direct solution approach..."

        # Fallback to single agent approach
        try:
            fallback_agent = MARSAgent(0, client, model, config)
            fallback_solution, fallback_tokens = fallback_agent.generate_solution(initial_query, request_id)
            return fallback_solution.solution, fallback_tokens
        except:
            return error_response, 0

async def _run_exploration_phase_parallel(
    agents: List[MARSAgent],
    workspace: MARSWorkspace,
    request_id: str = None,
    executor: ThreadPoolExecutor = None
) -> int:
    """Run the multi-agent exploration phase with parallel execution"""

    async def generate_solution_async(agent: MARSAgent):
        """Async wrapper for agent solution generation"""
        loop = asyncio.get_event_loop()
        try:
            solution, tokens = await loop.run_in_executor(
                executor,
                agent.generate_solution,
                workspace.problem,
                request_id
            )
            return agent.agent_id, solution, tokens, None
        except Exception as e:
            logger.error(f"Agent {agent.agent_id} failed during exploration: {str(e)}")
            return agent.agent_id, None, 0, e

    # Run all agents in parallel
    tasks = [generate_solution_async(agent) for agent in agents]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    total_tokens = 0
    successful_solutions = 0

    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Agent task failed: {str(result)}")
            continue

        agent_id, solution, tokens, error = result
        if error is None and solution is not None:
            workspace.add_solution(solution)
            total_tokens += tokens
            successful_solutions += 1

            # ENHANCED LOGGING: Log individual agent solution details
            logger.info(f"Agent {agent_id} exploration complete:")
            logger.info(f"  - Solution length: {solution.solution_length} chars")
            logger.info(f"  - Total tokens: {solution.total_tokens}")
            logger.info(f"  - Reasoning tokens: {solution.reasoning_tokens}")
            logger.info(f"  - Confidence: {solution.confidence:.2f}")
            logger.info(f"  - Solution preview: {solution.solution[:200]}...")
        else:
            logger.error(f"Agent {agent_id} generated no solution")

    logger.info(f"Exploration phase complete: {successful_solutions} solutions generated in parallel")
    return total_tokens

def _synthesize_final_solution(
    workspace: MARSWorkspace,
    client,
    model: str,
    config: Dict[str, Any],
    request_id: str = None
) -> Tuple[str, int]:
    """Synthesize the final solution from all agent outputs and verifications"""

    # Get the best verified solution
    best_solution = workspace.get_best_solution()

    if best_solution and best_solution.is_verified:
        logger.info(f"Using verified solution from agent {best_solution.agent_id}")
        return best_solution.solution, 0

    # If no verified solution, try numerical voting first
    logger.info("No verified solutions found, attempting numerical voting")

    # Try to extract numerical answers from all solutions
    import re
    from collections import Counter

    numerical_answers = []
    for solution in workspace.solutions:
        # Look for boxed answers: \boxed{123}
        boxed_match = re.search(r'\\boxed\{(\d+)\}', solution.solution)
        if boxed_match:
            try:
                answer = int(boxed_match.group(1))
                numerical_answers.append((answer, solution))
                continue
            except ValueError:
                pass

        # Look for final numerical answers at the end
        lines = solution.solution.strip().split('\n')
        for line in reversed(lines[-5:]):  # Check last 5 lines
            # Look for patterns like "answer is 123" or just "123" at the end
            number_match = re.search(r'\b(\d+)\b\s*\.?\s*$', line.strip())
            if number_match:
                try:
                    answer = int(number_match.group(1))
                    # Only accept if it's a reasonable AIME answer (1-999)
                    if 1 <= answer <= 999:
                        numerical_answers.append((answer, solution))
                        break
                except ValueError:
                    pass

    # Check for majority vote
    if len(numerical_answers) >= 2:
        answer_counts = Counter([ans for ans, _ in numerical_answers])
        most_common = answer_counts.most_common(1)[0]
        answer, count = most_common

        # If 2+ agents agree on the same number, use that
        if count >= 2:
            # Find the solution with highest confidence among those with the winning answer
            matching_solutions = [sol for ans, sol in numerical_answers if ans == answer]
            best_solution = max(matching_solutions, key=lambda s: s.confidence)

            logger.info(f"VOTING: Using majority vote answer {answer} ({count}/{len(numerical_answers)} agents agreed)")
            logger.info(f"VOTING: Selected solution from agent {best_solution.agent_id} with confidence {best_solution.confidence:.2f}")

            # Return the solution with the winning answer (no reasoning tokens since no new API call)
            return best_solution.solution, 0

    # If no consensus, fall back to synthesis
    logger.info("No numerical consensus found, attempting synthesis")

    synthesis_data = workspace.get_synthesis_input()

    # Prepare synthesis prompt
    agent_solutions_text = ""
    for i, sol_data in enumerate(synthesis_data['solutions'][:3]):  # Limit to top 3
        agent_solutions_text += f"\nAgent {sol_data['agent_id']} (confidence: {sol_data['confidence']:.2f}):\n"
        agent_solutions_text += sol_data['solution']
        agent_solutions_text += "\n" + "="*50 + "\n"

    verification_text = f"Verification Summary: {synthesis_data['verification_summary']}"

    synthesis_prompt = SYNTHESIS_PROMPT.format(
        problem=workspace.problem,
        agent_solutions=agent_solutions_text,
        verification_results=verification_text
    )

    try:
        # Use simplified synthesis with effort parameter
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a mathematical synthesis expert."},
                {"role": "user", "content": synthesis_prompt}
            ],
            max_tokens=config['max_tokens'],
            temperature=0.3,  # Lower temperature for synthesis
            timeout=300,
            extra_body={
                "reasoning": {
                    "effort": "high"  # High effort for final synthesis
                }
            }
        )

        # Log provider call if conversation logging is enabled
        if request_id:
            provider_request = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a mathematical synthesis expert."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                "max_tokens": config['max_tokens'],
                "temperature": 0.3,
                "extra_body": {
                    "reasoning": {
                        "effort": "high"
                    }
                }
            }
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
            conversation_logger.log_provider_call(request_id, provider_request, response_dict)

        final_solution = response.choices[0].message.content.strip()

        # Extract reasoning tokens from correct nested structure (matching agent.py fix)
        reasoning_tokens = 0
        total_tokens = 0
        if hasattr(response, 'usage') and response.usage:
            total_tokens = getattr(response.usage, 'total_tokens', 0)
            # Check completion_tokens_details first (OpenRouter structure)
            if hasattr(response.usage, 'completion_tokens_details') and response.usage.completion_tokens_details:
                reasoning_tokens = getattr(response.usage.completion_tokens_details, 'reasoning_tokens', 0)
            # Fallback to direct usage field (standard OpenAI structure)
            if reasoning_tokens == 0:
                reasoning_tokens = getattr(response.usage, 'reasoning_tokens', 0)

        # ENHANCED LOGGING: Log synthesis details
        logger.info(f"Synthesis complete:")
        logger.info(f"  - Synthesis solution length: {len(final_solution)} characters")
        logger.info(f"  - Reasoning tokens: {reasoning_tokens}")
        logger.info(f"  - Total tokens: {total_tokens}")
        logger.info(f"  - Final solution preview: {final_solution[:200]}...")
        return final_solution, reasoning_tokens

    except Exception as e:
        logger.error(f"Synthesis failed: {str(e)}")

        # Fallback: return the solution with highest verification score
        if workspace.solutions:
            fallback_solution = max(workspace.solutions, key=lambda s: s.verification_score)
            logger.info(f"Using fallback solution from agent {fallback_solution.agent_id}")
            return fallback_solution.solution, 0

        return "Unable to generate solution due to synthesis failure.", 0