"""
MARS Agent implementation with OpenRouter reasoning API
"""

import logging
from typing import Dict, Any, Tuple
from datetime import datetime
import random
from .prompts import (
    MATHEMATICAL_SYSTEM_PROMPT,
    AGENT_EXPLORATION_PROMPT,
    VERIFICATION_PROMPT,
    IMPROVEMENT_PROMPT
)
from .workspace import AgentSolution, VerificationResult

logger = logging.getLogger(__name__)

class MARSAgent:
    """Individual agent for mathematical reasoning with OpenRouter reasoning API"""

    def __init__(self, agent_id: int, client, model: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.client = client
        self.model = model
        self.config = config
        self.temperature = self._assign_temperature()

    def _assign_temperature(self) -> float:
        """Assign temperature based on agent ID for diversity"""
        temperatures = [0.3, 0.5, 0.7, 0.9, 1.0]
        return temperatures[self.agent_id % len(temperatures)]

    def _get_reasoning_effort(self) -> str:
        """Get reasoning effort level based on agent temperature"""
        if self.temperature <= 0.4:
            return "low"  # 20% reasoning budget
        elif self.temperature <= 0.7:
            return "medium"  # 50% reasoning budget
        else:
            return "high"  # 80% reasoning budget

    def generate_solution(self, problem: str, request_id: str = None) -> Tuple[AgentSolution, int]:
        """Generate a solution for the given problem using reasoning API"""
        logger.info(f"Agent {self.agent_id} generating solution with temperature {self.temperature}")

        # Prepare the prompt
        exploration_prompt = AGENT_EXPLORATION_PROMPT.format(
            agent_id=self.agent_id,
            temperature=self.temperature,
            problem=problem
        )

        # Configure reasoning parameters for OpenRouter
        reasoning_config = {
            "effort": self._get_reasoning_effort()
        }

        try:
            # Make API call with reasoning via extra_body for OpenRouter compatibility
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": MATHEMATICAL_SYSTEM_PROMPT},
                    {"role": "user", "content": exploration_prompt}
                ],
                max_tokens=self.config.get('max_response_tokens', 32768),
                temperature=self.temperature,
                timeout=300,  # 5 minute timeout for complex problems
                extra_body={
                    "reasoning": reasoning_config
                }
            )

            solution_text = response.choices[0].message.content.strip()

            # Extract reasoning tokens from the correct nested structure
            reasoning_tokens = 0
            if hasattr(response, 'usage') and response.usage:
                # Check completion_tokens_details first (OpenRouter structure)
                if hasattr(response.usage, 'completion_tokens_details') and response.usage.completion_tokens_details:
                    reasoning_tokens = getattr(response.usage.completion_tokens_details, 'reasoning_tokens', 0)

                # Fallback to direct usage field (standard OpenAI structure)
                if reasoning_tokens == 0:
                    reasoning_tokens = getattr(response.usage, 'reasoning_tokens', 0)

            # Extract confidence from solution (heuristic based on response characteristics)
            confidence = self._estimate_confidence(solution_text)

            # Create agent solution object
            agent_solution = AgentSolution(
                agent_id=self.agent_id,
                temperature=self.temperature,
                solution=solution_text,
                confidence=confidence,
                reasoning_tokens=reasoning_tokens,
                timestamp=datetime.now()
            )

            logger.info(f"Agent {self.agent_id} generated solution with {reasoning_tokens} reasoning tokens")
            return agent_solution, reasoning_tokens

        except Exception as e:
            logger.error(f"Agent {self.agent_id} error generating solution: {str(e)}")
            # Return empty solution with error indication
            return AgentSolution(
                agent_id=self.agent_id,
                temperature=self.temperature,
                solution=f"Error generating solution: {str(e)}",
                confidence=0.0,
                reasoning_tokens=0,
                timestamp=datetime.now()
            ), 0

    def verify_solution(self, problem: str, solution: str, verifier_id: int, solution_agent_id: int, request_id: str = None) -> VerificationResult:
        """Verify a solution using mathematical reasoning"""
        logger.info(f"Agent {self.agent_id} verifying solution (verifier_id: {verifier_id})")

        verification_prompt = VERIFICATION_PROMPT.format(
            problem=problem,
            solution=solution
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": MATHEMATICAL_SYSTEM_PROMPT},
                    {"role": "user", "content": verification_prompt}
                ],
                max_tokens=16384,
                temperature=0.1,  # Low temperature for consistent verification
                timeout=180,
                extra_body={
                    "reasoning": {
                        "effort": "medium"
                    }
                }
            )

            verification_text = response.choices[0].message.content.strip()

            # Parse verification result
            assessment, confidence, issues, suggestions = self._parse_verification(verification_text)

            return VerificationResult(
                verifier_id=verifier_id,
                solution_id=f"agent_{solution_agent_id}_iter_0",  # Use the solution's agent_id
                assessment=assessment,
                confidence=confidence,
                issues=issues,
                suggestions=suggestions,
                detailed_report=verification_text,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Agent {self.agent_id} error in verification: {str(e)}")
            return VerificationResult(
                verifier_id=verifier_id,
                solution_id=f"agent_{solution_agent_id}_iter_0",
                assessment="INCOMPLETE",
                confidence=0.0,
                issues=[f"Verification error: {str(e)}"],
                suggestions=["Retry verification"],
                detailed_report=f"Error during verification: {str(e)}",
                timestamp=datetime.now()
            )

    def improve_solution(self, problem: str, current_solution: str, feedback: str, issues: list, request_id: str = None) -> Tuple[str, int]:
        """Improve a solution based on verification feedback"""
        logger.info(f"Agent {self.agent_id} improving solution based on feedback")

        improvement_prompt = IMPROVEMENT_PROMPT.format(
            problem=problem,
            current_solution=current_solution,
            feedback=feedback,
            issues="\n".join(f"- {issue}" for issue in issues)
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": MATHEMATICAL_SYSTEM_PROMPT},
                    {"role": "user", "content": improvement_prompt}
                ],
                max_tokens=32768,
                temperature=self.temperature * 0.8,  # Slightly lower temperature for improvement
                timeout=300,
                extra_body={
                    "reasoning": {
                        "effort": "high"
                    }
                }
            )

            improved_solution = response.choices[0].message.content.strip()
            reasoning_tokens = getattr(response.usage, 'reasoning_tokens', 0)

            logger.info(f"Agent {self.agent_id} improved solution with {reasoning_tokens} reasoning tokens")
            return improved_solution, reasoning_tokens

        except Exception as e:
            logger.error(f"Agent {self.agent_id} error improving solution: {str(e)}")
            return current_solution, 0  # Return original solution if improvement fails

    def _estimate_confidence(self, solution: str) -> float:
        """Estimate confidence based on solution characteristics"""
        confidence = 0.5  # Base confidence

        # Check for mathematical rigor indicators
        if "\\boxed{" in solution:
            confidence += 0.2
        if "therefore" in solution.lower() or "thus" in solution.lower():
            confidence += 0.1
        if "proof" in solution.lower():
            confidence += 0.1
        if len(solution.split()) > 200:  # Detailed solutions tend to be more confident
            confidence += 0.1
        if "let" in solution.lower() and "assume" in solution.lower():
            confidence += 0.1

        # Check for uncertainty indicators
        if "might" in solution.lower() or "possibly" in solution.lower():
            confidence -= 0.1
        if "unsure" in solution.lower() or "not sure" in solution.lower():
            confidence -= 0.2

        return max(0.1, min(1.0, confidence))

    def _parse_verification(self, verification_text: str) -> Tuple[str, float, list, list]:
        """Parse verification result to extract structured information"""
        assessment = "INCOMPLETE"  # Default
        confidence = 0.5
        issues = []
        suggestions = []

        text_lower = verification_text.lower()

        # Determine assessment
        if "correct" in text_lower and "incorrect" not in text_lower:
            assessment = "CORRECT"
            confidence = 0.8
        elif "incorrect" in text_lower:
            assessment = "INCORRECT"
            confidence = 0.8
        elif "incomplete" in text_lower:
            assessment = "INCOMPLETE"
            confidence = 0.6

        # Extract confidence if explicitly mentioned
        import re
        confidence_match = re.search(r'confidence.*?(\d+).*?(?:out of|/)\s*(\d+)', text_lower)
        if confidence_match:
            conf_score = float(confidence_match.group(1))
            conf_total = float(confidence_match.group(2))
            confidence = conf_score / conf_total

        # Extract issues (simple heuristic)
        lines = verification_text.split('\n')
        for line in lines:
            line_lower = line.lower()
            if any(word in line_lower for word in ['error', 'mistake', 'incorrect', 'wrong', 'issue']):
                issues.append(line.strip())

        # Extract suggestions (simple heuristic)
        for line in lines:
            line_lower = line.lower()
            if any(word in line_lower for word in ['suggest', 'recommend', 'should', 'could improve']):
                suggestions.append(line.strip())

        return assessment, confidence, issues, suggestions