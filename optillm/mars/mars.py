"""
MARS: Multi-Agent Reasoning System main orchestration
"""

import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime
import optillm
from optillm import conversation_logger

from .workspace import MARSWorkspace, AgentSolution
from .agent import MARSAgent
from .verifier import MARSVerifier
from .prompts import SYNTHESIS_PROMPT

logger = logging.getLogger(__name__)

# Default MARS configuration with fixed 32k token budget
DEFAULT_CONFIG = {
    'num_agents': 3,
    'max_iterations': 5,  # Balanced for quality vs efficiency
    'verification_passes_required': 2,  # Balanced for 5-iteration efficiency
    'consensus_threshold': 2,  # Keep at 2 for 3-agent setup
    'min_verified_solutions': 1,  # Keep minimal requirement
    'max_tokens': 32000,  # Fixed 32k token budget for all calls
    'max_verification_attempts': 3,
    'early_termination': True,
    'use_reasoning_api': True,
    # Fixed reasoning token allocations
    'low_effort_tokens': 8000,     # Agent 0 (temperature 0.3)
    'medium_effort_tokens': 16000, # Agent 1 (temperature 0.6)
    'high_effort_tokens': 24000,   # Agent 2 (temperature 1.0)
    'verification_tokens': 8000,   # Fixed low effort for verification consistency
    'synthesis_tokens': 24000      # Fixed high effort for final synthesis
}

def multi_agent_reasoning_system(
    system_prompt: str,
    initial_query: str,
    client,
    model: str,
    request_id: str = None
) -> Tuple[str, int]:
    """
    Main MARS function implementing multi-agent mathematical reasoning

    Args:
        system_prompt: System-level instructions
        initial_query: The mathematical problem to solve
        client: OpenAI-compatible client for API calls
        model: Model identifier (should support OpenRouter reasoning API)
        request_id: Optional request ID for conversation logging

    Returns:
        Tuple of (final_solution, total_reasoning_tokens)
    """
    logger.info(f"Starting MARS with model: {model}")

    # Initialize configuration
    config = DEFAULT_CONFIG.copy()
    total_reasoning_tokens = 0

    # Initialize workspace for collaboration
    workspace = MARSWorkspace(initial_query, config)

    try:
        # Phase 1: Initialize Agents
        agents = []
        for i in range(config['num_agents']):
            agent = MARSAgent(i, client, model, config)
            agents.append(agent)

        logger.info(f"Initialized {len(agents)} agents with diverse temperatures")

        # Phase 2: Multi-Agent Exploration
        logger.info("Phase 1: Multi-Agent Exploration")
        exploration_tokens = _run_exploration_phase(agents, workspace, request_id)
        total_reasoning_tokens += exploration_tokens

        # Phase 3: Verification System
        logger.info("Phase 2: Verification System")
        verifier = MARSVerifier(agents, workspace, config)
        verification_summary = verifier.verify_solutions(request_id)

        # Phase 4: Iterative Improvement (if needed)
        iteration_count = 0
        while workspace.should_continue_iteration() and iteration_count < config['max_iterations']:
            iteration_count += 1
            logger.info(f"Phase 3: Iterative Improvement - Iteration {iteration_count}")

            # Improve unverified solutions
            improvement_summary = verifier.iterative_improvement(request_id)
            total_reasoning_tokens += improvement_summary['total_reasoning_tokens']

            # Re-verify improved solutions
            verification_summary = verifier.verify_solutions(request_id)

            # Check for early termination
            if config['early_termination'] and workspace.has_consensus():
                logger.info("Early termination: consensus reached")
                break

            workspace.iteration_count = iteration_count

        # Phase 5: Final Synthesis
        logger.info("Phase 4: Final Synthesis")
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

def _run_exploration_phase(agents: List[MARSAgent], workspace: MARSWorkspace, request_id: str = None) -> int:
    """Run the multi-agent exploration phase"""
    total_tokens = 0

    # Generate solutions from all agents in parallel (conceptually)
    for agent in agents:
        try:
            agent_solution, reasoning_tokens = agent.generate_solution(
                workspace.problem, request_id
            )
            workspace.add_solution(agent_solution)
            total_tokens += reasoning_tokens

        except Exception as e:
            logger.error(f"Agent {agent.agent_id} failed during exploration: {str(e)}")
            continue

    logger.info(f"Exploration phase complete: {len(workspace.solutions)} solutions generated")
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

    # If no verified solution, attempt synthesis
    logger.info("No verified solutions found, attempting synthesis")

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
        # Use fixed synthesis token budgets
        synthesis_max_tokens = config['max_tokens']  # Fixed 32k
        synthesis_reasoning_tokens = config['synthesis_tokens']  # Fixed 24k

        # Use fixed reasoning effort for synthesis
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a mathematical synthesis expert."},
                {"role": "user", "content": synthesis_prompt}
            ],
            max_tokens=synthesis_max_tokens,
            temperature=0.3,  # Lower temperature for synthesis
            timeout=300,
            extra_body={
                "reasoning": {
                    "max_tokens": synthesis_reasoning_tokens,
                    "effort": "high"
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
                "max_tokens": synthesis_max_tokens,
                "temperature": 0.3,
                "extra_body": {
                    "reasoning": {
                        "max_tokens": synthesis_reasoning_tokens,
                        "effort": "high"
                    }
                }
            }
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
            conversation_logger.log_provider_call(request_id, provider_request, response_dict)

        final_solution = response.choices[0].message.content.strip()

        # Extract reasoning tokens from correct nested structure (matching agent.py fix)
        reasoning_tokens = 0
        if hasattr(response, 'usage') and response.usage:
            # Check completion_tokens_details first (OpenRouter structure)
            if hasattr(response.usage, 'completion_tokens_details') and response.usage.completion_tokens_details:
                reasoning_tokens = getattr(response.usage.completion_tokens_details, 'reasoning_tokens', 0)
            # Fallback to direct usage field (standard OpenAI structure)
            if reasoning_tokens == 0:
                reasoning_tokens = getattr(response.usage, 'reasoning_tokens', 0)

        logger.info(f"Synthesis complete with {reasoning_tokens} reasoning tokens")
        return final_solution, reasoning_tokens

    except Exception as e:
        logger.error(f"Synthesis failed: {str(e)}")

        # Fallback: return the solution with highest verification score
        if workspace.solutions:
            fallback_solution = max(workspace.solutions, key=lambda s: s.verification_score)
            logger.info(f"Using fallback solution from agent {fallback_solution.agent_id}")
            return fallback_solution.solution, 0

        return "Unable to generate solution due to synthesis failure.", 0