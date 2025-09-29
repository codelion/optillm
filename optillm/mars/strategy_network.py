"""
MARS Strategy Network: Cross-Agent Insight Sharing & Meta-Reasoning
Enables agents to share reasoning strategies and adapt approaches collaboratively
"""

import asyncio
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

from optillm import conversation_logger
from .workspace import AgentSolution, MARSWorkspace

logger = logging.getLogger(__name__)


@dataclass
class ReasoningStrategy:
    """Represents an extracted reasoning strategy from an agent solution"""
    strategy_id: str
    agent_id: str
    problem_type: str  # 'algebra', 'geometry', 'combinatorics', 'number_theory', etc.
    approach_type: str  # 'direct', 'proof_by_contradiction', 'induction', etc.
    key_insights: List[str]
    mathematical_techniques: List[str]
    solution_pattern: str
    confidence: float
    success_indicators: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StrategyEffectiveness:
    """Tracks effectiveness of strategies across different problem types"""
    strategy_id: str
    problem_type: str
    success_count: int = 0
    failure_count: int = 0
    total_uses: int = 0
    average_confidence: float = 0.0
    best_applications: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.success_count / max(self.total_uses, 1)


class StrategyNetwork:
    """
    Cross-agent strategy sharing and meta-reasoning system

    Key capabilities:
    1. Extract reasoning strategies from agent solutions
    2. Share effective strategies between agents
    3. Track strategy effectiveness across problem types
    4. Enable adaptive agent behavior based on peer insights
    """

    def __init__(self, client, model: str, config: Dict[str, Any]):
        self.client = client
        self.model = model
        self.config = config
        self.max_tokens = config.get('max_tokens', 30000)

        # Strategy storage and tracking
        self.strategies: Dict[str, ReasoningStrategy] = {}
        self.strategy_effectiveness: Dict[Tuple[str, str], StrategyEffectiveness] = {}
        self.agent_preferred_strategies: Dict[str, List[str]] = defaultdict(list)

        # Problem type classification cache
        self.problem_type_cache: Dict[str, str] = {}

        logger.info("Initialized Strategy Network for cross-agent insight sharing")

    async def extract_strategies_from_solutions(
        self,
        workspace: MARSWorkspace,
        request_id: str = None,
        executor: ThreadPoolExecutor = None
    ) -> Dict[str, ReasoningStrategy]:
        """Extract reasoning strategies from all agent solutions"""
        logger.info("Extracting strategies from agent solutions...")

        extraction_tasks = []
        for solution in workspace.solutions:
            if not solution.agent_id.startswith('agg_'):  # Skip aggregated solutions for strategy extraction
                task = self._extract_strategy_async(solution, workspace.problem, request_id, executor)
                extraction_tasks.append(task)

        # Run extractions in parallel
        results = await asyncio.gather(*extraction_tasks, return_exceptions=True)

        extracted_strategies = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Strategy extraction failed: {str(result)}")
                continue

            if result:
                strategy = result
                extracted_strategies[strategy.strategy_id] = strategy
                self.strategies[strategy.strategy_id] = strategy

                # Update agent's preferred strategies
                self.agent_preferred_strategies[strategy.agent_id].append(strategy.strategy_id)

        logger.info(f"Extracted {len(extracted_strategies)} reasoning strategies")
        return extracted_strategies

    async def _extract_strategy_async(
        self,
        solution: AgentSolution,
        problem: str,
        request_id: str = None,
        executor: ThreadPoolExecutor = None
    ) -> Optional[ReasoningStrategy]:
        """Extract strategy from a single agent solution"""
        loop = asyncio.get_event_loop()

        try:
            return await loop.run_in_executor(
                executor,
                self._extract_strategy_from_solution,
                solution,
                problem,
                request_id
            )
        except Exception as e:
            logger.error(f"Failed to extract strategy from agent {solution.agent_id}: {str(e)}")
            return None

    def _extract_strategy_from_solution(
        self,
        solution: AgentSolution,
        problem: str,
        request_id: str = None
    ) -> Optional[ReasoningStrategy]:
        """Extract reasoning strategy using LLM analysis"""

        strategy_extraction_prompt = f"""Analyze this mathematical solution and extract the key reasoning strategy:

Problem: {problem}

Agent Solution:
{solution.solution}

Extract the following strategy components:

1. PROBLEM_TYPE: Classify as one of [algebra, geometry, combinatorics, number_theory, calculus, discrete_math, probability]

2. APPROACH_TYPE: Identify the main approach [direct_computation, proof_by_contradiction, constructive_proof, case_analysis, induction, algebraic_manipulation, geometric_visualization, pattern_recognition, reduction_to_known_problem]

3. KEY_INSIGHTS: List 2-3 key mathematical insights that enabled the solution

4. MATHEMATICAL_TECHNIQUES: List specific techniques used [substitution, factorization, coordinate_geometry, symmetry, pigeonhole_principle, etc.]

5. SOLUTION_PATTERN: Describe the general pattern/template of this solution approach

6. SUCCESS_INDICATORS: What makes this approach particularly effective for this type of problem?

Format your response as:
PROBLEM_TYPE: [type]
APPROACH_TYPE: [approach]
KEY_INSIGHTS: [insight1], [insight2], [insight3]
MATHEMATICAL_TECHNIQUES: [technique1], [technique2], [technique3]
SOLUTION_PATTERN: [pattern description]
SUCCESS_INDICATORS: [indicator1], [indicator2]"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a mathematical strategy analysis expert. Extract reasoning patterns from solutions."},
                    {"role": "user", "content": strategy_extraction_prompt}
                ],
                max_tokens=self.max_tokens // 4,  # Use 1/4 of token budget for strategy extraction
                temperature=0.3,
                timeout=120,
                extra_body={
                    "reasoning": {
                        "effort": "medium"
                    }
                }
            )

            # Log provider call if conversation logging is enabled
            if request_id:
                provider_request = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a mathematical strategy analysis expert."},
                        {"role": "user", "content": strategy_extraction_prompt}
                    ],
                    "max_tokens": self.max_tokens // 4,
                    "temperature": 0.3,
                    "extra_body": {"reasoning": {"effort": "medium"}}
                }
                response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
                conversation_logger.log_provider_call(request_id, provider_request, response_dict)

            analysis = response.choices[0].message.content.strip()

            # Parse the structured response
            strategy_data = self._parse_strategy_analysis(analysis)

            if strategy_data:
                strategy_id = f"strategy_{solution.agent_id}_{datetime.now().strftime('%H%M%S')}"

                return ReasoningStrategy(
                    strategy_id=strategy_id,
                    agent_id=solution.agent_id,
                    problem_type=strategy_data.get('problem_type', 'unknown'),
                    approach_type=strategy_data.get('approach_type', 'unknown'),
                    key_insights=strategy_data.get('key_insights', []),
                    mathematical_techniques=strategy_data.get('mathematical_techniques', []),
                    solution_pattern=strategy_data.get('solution_pattern', ''),
                    confidence=solution.confidence,
                    success_indicators=strategy_data.get('success_indicators', [])
                )

        except Exception as e:
            logger.error(f"Strategy extraction failed for agent {solution.agent_id}: {str(e)}")
            return None

    def _parse_strategy_analysis(self, analysis: str) -> Optional[Dict[str, Any]]:
        """Parse structured strategy analysis response"""
        try:
            lines = analysis.split('\n')
            strategy_data = {}

            for line in lines:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()

                    if key == 'problem_type':
                        strategy_data['problem_type'] = value
                    elif key == 'approach_type':
                        strategy_data['approach_type'] = value
                    elif 'insights' in key:
                        strategy_data['key_insights'] = [insight.strip() for insight in value.split(',')]
                    elif 'techniques' in key:
                        strategy_data['mathematical_techniques'] = [tech.strip() for tech in value.split(',')]
                    elif 'pattern' in key:
                        strategy_data['solution_pattern'] = value
                    elif 'indicators' in key:
                        strategy_data['success_indicators'] = [ind.strip() for ind in value.split(',')]

            return strategy_data if strategy_data else None

        except Exception as e:
            logger.error(f"Failed to parse strategy analysis: {str(e)}")
            return None

    async def share_strategies_across_agents(
        self,
        workspace: MARSWorkspace,
        extracted_strategies: Dict[str, ReasoningStrategy],
        request_id: str = None,
        executor: ThreadPoolExecutor = None
    ) -> Dict[str, List[str]]:
        """Share effective strategies across agents and generate enhanced solutions"""
        logger.info("Sharing strategies across agents...")

        # Classify current problem type
        problem_type = await self._classify_problem_type(workspace.problem, request_id, executor)

        # Find most effective strategies for this problem type
        effective_strategies = self._get_effective_strategies_for_type(problem_type, extracted_strategies)

        # Generate strategy-enhanced solutions for each agent
        enhancement_tasks = []
        agent_strategies = {}

        for solution in workspace.solutions:
            if not solution.agent_id.startswith('agg_'):  # Only enhance original agents
                # Select strategies from other agents for this agent
                cross_agent_strategies = [
                    strategy for strategy in effective_strategies.values()
                    if strategy.agent_id != solution.agent_id
                ]

                if cross_agent_strategies:
                    agent_strategies[solution.agent_id] = [s.strategy_id for s in cross_agent_strategies]

                    task = self._generate_strategy_enhanced_solution_async(
                        solution, workspace.problem, cross_agent_strategies, request_id, executor
                    )
                    enhancement_tasks.append((solution.agent_id, task))

        # Run enhancements in parallel
        if enhancement_tasks:
            tasks = [task for _, task in enhancement_tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Add enhanced solutions to workspace
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Strategy enhancement failed: {str(result)}")
                    continue

                if result:
                    enhanced_solution = result
                    workspace.add_solution(enhanced_solution)
                    logger.info(f"Added strategy-enhanced solution from agent {enhanced_solution.agent_id}")

        logger.info(f"Strategy sharing complete: enhanced {len(enhancement_tasks)} agents")
        return agent_strategies

    async def _classify_problem_type(
        self,
        problem: str,
        request_id: str = None,
        executor: ThreadPoolExecutor = None
    ) -> str:
        """Classify the problem type for strategy matching"""
        # Check cache first
        if problem in self.problem_type_cache:
            return self.problem_type_cache[problem]

        loop = asyncio.get_event_loop()

        try:
            problem_type = await loop.run_in_executor(
                executor,
                self._classify_problem_with_llm,
                problem,
                request_id
            )

            self.problem_type_cache[problem] = problem_type
            return problem_type

        except Exception as e:
            logger.error(f"Problem classification failed: {str(e)}")
            return "unknown"

    def _classify_problem_with_llm(self, problem: str, request_id: str = None) -> str:
        """Use LLM to classify problem type"""
        classification_prompt = f"""Classify this mathematical problem into one category:

Problem: {problem}

Categories: [algebra, geometry, combinatorics, number_theory, calculus, discrete_math, probability]

Respond with just the category name."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a mathematical problem classifier."},
                    {"role": "user", "content": classification_prompt}
                ],
                max_tokens=50,
                temperature=0.1,
                timeout=60,
                extra_body={
                    "reasoning": {
                        "effort": "low"
                    }
                }
            )

            classification = response.choices[0].message.content.strip().lower()

            # Validate classification
            valid_types = ['algebra', 'geometry', 'combinatorics', 'number_theory', 'calculus', 'discrete_math', 'probability']
            if classification in valid_types:
                return classification
            else:
                return "algebra"  # Default fallback

        except Exception as e:
            logger.error(f"Problem classification failed: {str(e)}")
            return "algebra"  # Default fallback

    def _get_effective_strategies_for_type(
        self,
        problem_type: str,
        extracted_strategies: Dict[str, ReasoningStrategy]
    ) -> Dict[str, ReasoningStrategy]:
        """Get most effective strategies for the given problem type"""

        # Filter strategies by problem type and confidence
        relevant_strategies = {}
        for strategy_id, strategy in extracted_strategies.items():
            if (strategy.problem_type == problem_type or strategy.problem_type == "unknown") and strategy.confidence >= 0.6:
                relevant_strategies[strategy_id] = strategy

        # If no specific strategies found, use highest confidence strategies
        if not relevant_strategies:
            sorted_strategies = sorted(
                extracted_strategies.items(),
                key=lambda x: x[1].confidence,
                reverse=True
            )
            # Take top 2 strategies
            relevant_strategies = dict(sorted_strategies[:2])

        return relevant_strategies

    async def _generate_strategy_enhanced_solution_async(
        self,
        original_solution: AgentSolution,
        problem: str,
        peer_strategies: List[ReasoningStrategy],
        request_id: str = None,
        executor: ThreadPoolExecutor = None
    ) -> Optional[AgentSolution]:
        """Generate enhanced solution using peer strategies"""
        loop = asyncio.get_event_loop()

        try:
            return await loop.run_in_executor(
                executor,
                self._generate_strategy_enhanced_solution,
                original_solution,
                problem,
                peer_strategies,
                request_id
            )
        except Exception as e:
            logger.error(f"Strategy enhancement failed for agent {original_solution.agent_id}: {str(e)}")
            return None

    def _generate_strategy_enhanced_solution(
        self,
        original_solution: AgentSolution,
        problem: str,
        peer_strategies: List[ReasoningStrategy],
        request_id: str = None
    ) -> Optional[AgentSolution]:
        """Generate solution enhanced with peer strategies"""

        # Prepare strategy insights
        strategy_insights = ""
        for strategy in peer_strategies[:2]:  # Limit to top 2 strategies
            strategy_insights += f"\nPeer Strategy from Agent {strategy.agent_id}:\n"
            strategy_insights += f"- Approach: {strategy.approach_type}\n"
            strategy_insights += f"- Key Insights: {', '.join(strategy.key_insights[:3])}\n"
            strategy_insights += f"- Techniques: {', '.join(strategy.mathematical_techniques[:3])}\n"
            strategy_insights += f"- Success Pattern: {strategy.solution_pattern[:200]}...\n"

        enhancement_prompt = f"""You are Agent {original_solution.agent_id} collaborating with other mathematical agents.

Original Problem: {problem}

Your Current Solution:
{original_solution.solution}

Peer Agent Strategy Insights:
{strategy_insights}

Task: Enhance your solution by incorporating the most valuable insights from your peers while maintaining your unique approach. Consider:

1. Can any peer techniques strengthen your solution?
2. Do peer insights reveal gaps in your reasoning?
3. Can you combine approaches for a more robust solution?
4. What verification steps from peers could improve confidence?

Provide an enhanced solution that synthesizes the best ideas while ensuring mathematical rigor.

Enhanced Solution:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a collaborative mathematical agent learning from peer insights."},
                    {"role": "user", "content": enhancement_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=original_solution.temperature * 0.9,  # Slightly lower for focused enhancement
                timeout=300,
                extra_body={
                    "reasoning": {
                        "effort": "high"
                    }
                }
            )

            # Log provider call if conversation logging is enabled
            if request_id:
                provider_request = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a collaborative mathematical agent learning from peer insights."},
                        {"role": "user", "content": enhancement_prompt}
                    ],
                    "max_tokens": self.max_tokens,
                    "temperature": original_solution.temperature * 0.9,
                    "extra_body": {"reasoning": {"effort": "high"}}
                }
                response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
                conversation_logger.log_provider_call(request_id, provider_request, response_dict)

            enhanced_solution_text = response.choices[0].message.content.strip()

            # Extract reasoning tokens
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

            # Create enhanced solution
            enhanced_agent_solution = AgentSolution(
                agent_id=f"enhanced_{original_solution.agent_id}",
                solution=enhanced_solution_text,
                confidence=min(original_solution.confidence + 0.1, 1.0),  # Slight confidence boost
                reasoning_tokens=reasoning_tokens,
                total_tokens=total_tokens,
                solution_length=len(enhanced_solution_text),
                temperature=original_solution.temperature
            )

            logger.info(f"Generated strategy-enhanced solution for agent {original_solution.agent_id}")
            return enhanced_agent_solution

        except Exception as e:
            logger.error(f"Strategy enhancement failed for agent {original_solution.agent_id}: {str(e)}")
            return None

    def update_strategy_effectiveness(
        self,
        strategy_id: str,
        problem_type: str,
        was_successful: bool,
        confidence: float
    ):
        """Update effectiveness tracking for a strategy"""
        key = (strategy_id, problem_type)

        if key not in self.strategy_effectiveness:
            self.strategy_effectiveness[key] = StrategyEffectiveness(
                strategy_id=strategy_id,
                problem_type=problem_type
            )

        effectiveness = self.strategy_effectiveness[key]
        effectiveness.total_uses += 1

        if was_successful:
            effectiveness.success_count += 1
        else:
            effectiveness.failure_count += 1

        # Update average confidence
        effectiveness.average_confidence = (
            (effectiveness.average_confidence * (effectiveness.total_uses - 1) + confidence) /
            effectiveness.total_uses
        )

    def get_strategy_insights_summary(self) -> Dict[str, Any]:
        """Get summary of strategy network insights"""
        return {
            'total_strategies': len(self.strategies),
            'strategies_by_type': self._count_strategies_by_type(),
            'most_effective_strategies': self._get_most_effective_strategies(),
            'agent_strategy_preferences': dict(self.agent_preferred_strategies),
            'strategy_effectiveness_stats': self._get_effectiveness_stats()
        }

    def _count_strategies_by_type(self) -> Dict[str, int]:
        """Count strategies by problem type"""
        counts = defaultdict(int)
        for strategy in self.strategies.values():
            counts[strategy.problem_type] += 1
        return dict(counts)

    def _get_most_effective_strategies(self) -> List[Dict[str, Any]]:
        """Get most effective strategies across all problem types"""
        effective_strategies = []

        for effectiveness in self.strategy_effectiveness.values():
            if effectiveness.total_uses >= 2:  # Only consider strategies used multiple times
                effective_strategies.append({
                    'strategy_id': effectiveness.strategy_id,
                    'problem_type': effectiveness.problem_type,
                    'success_rate': effectiveness.success_rate,
                    'average_confidence': effectiveness.average_confidence,
                    'total_uses': effectiveness.total_uses
                })

        # Sort by success rate and confidence
        effective_strategies.sort(
            key=lambda x: (x['success_rate'], x['average_confidence']),
            reverse=True
        )

        return effective_strategies[:5]  # Top 5

    def _get_effectiveness_stats(self) -> Dict[str, float]:
        """Get overall effectiveness statistics"""
        if not self.strategy_effectiveness:
            return {}

        success_rates = [eff.success_rate for eff in self.strategy_effectiveness.values()]
        avg_confidences = [eff.average_confidence for eff in self.strategy_effectiveness.values()]

        return {
            'average_success_rate': sum(success_rates) / len(success_rates) if success_rates else 0,
            'average_confidence': sum(avg_confidences) / len(avg_confidences) if avg_confidences else 0,
            'total_strategy_applications': sum(eff.total_uses for eff in self.strategy_effectiveness.values())
        }