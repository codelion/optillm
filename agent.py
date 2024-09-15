import logging
from typing import List, Dict
import json
import re

# Import approach modules
from mcts import chat_with_mcts
from bon import best_of_n_sampling
from moa import mixture_of_agents
from rto import round_trip_optimization
from z3_solver import Z3SolverSystem
from self_consistency import advanced_self_consistency_approach
from pvg import inference_time_pv_game
from rstar import RStar
from cot_reflection import cot_reflection
from plansearch import plansearch
from leap import leap

logger = logging.getLogger(__name__)

class Agent:
    def __init__(self, client, model: str, max_attempts: int = 3):
        self.client = client
        self.model = model
        self.max_attempts = max_attempts
        self.approaches = {
            "mcts": "Monte Carlo Tree Search - Best for decision-making tasks, strategic planning, and problems with a large search space.",
            "bon": "Best of N Sampling - Suitable for tasks where generating multiple responses and selecting the best one can improve quality.",
            "moa": "Mixture of Agents - Effective for complex tasks that benefit from diverse perspectives or expertise.",
            "rto": "Round Trip Optimization - Useful for tasks that require iterative refinement or translation between different formats.",
            "z3": "Z3 Solver - Ideal for logical reasoning, constraint satisfaction problems, and formal verification tasks.",
            "self_consistency": "Self-Consistency - Beneficial for tasks that require coherent and consistent reasoning across multiple steps.",
            "pvg": "PV Game (Prover-Verifier Game) - Suitable for tasks that involve proving statements or verifying solutions.",
            "rstar": "R* Algorithm - Effective for search problems, particularly in large state spaces with heuristics.",
            "cot_reflection": "Chain of Thought with Reflection - Useful for complex reasoning tasks that benefit from step-by-step thinking and self-correction.",
            "plansearch": "Plan Search - Ideal for tasks that require structured planning, especially in natural language problem-solving.",
            "leap": "LEAP (Learn from Examples and Principles) - Suitable for tasks that can benefit from learning task-specific principles from few-shot examples."
        }

    def determine_approaches(self, system_prompt: str, user_query: str) -> List[str]:
        prompt = f"""
        Given the following system prompt and user query, determine the most suitable approach(es) to solve the task.
        Available approaches and their descriptions:

        {json.dumps(self.approaches, indent=2)}

        System prompt: {system_prompt}
        User query: {user_query}

        Analyze the task and suggest up to 3 approaches that would be most effective. Provide a brief explanation for each suggestion, considering the specific requirements of the task and how the approach's strengths align with those requirements.

        Respond with a JSON array of objects in the following format:
        [
            {{
                "name": "approach_name",
                "explanation": "Detailed explanation of why this approach is suitable for the given task"
            }},
            ...
        ]

        Include 1 to 3 approach objects in the array. Ensure that the response is a valid JSON array.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=4096,
        )

        raw_content = response.choices[0].message.content
        # logger.info(f"Raw response for choosing approach: {raw_content}")

        # Remove any markdown code block formatting
        cleaned_content = re.sub(r'```json\s*|\s*```', '', raw_content).strip()

        try:
            response_data = json.loads(cleaned_content)
            suggested_approaches = []
            
            for approach_data in response_data:
                name = approach_data.get('name', '').lower()
                explanation = approach_data.get('explanation', 'No explanation provided')
                
                if name in self.approaches:
                    suggested_approaches.append(name)
                    logger.info(f"Suggested approach: {name}")
                    logger.info(f"Explanation: {explanation}")
                else:
                    logger.warning(f"Invalid approach suggested: {name}")

            return suggested_approaches[:3]  # Limit to top 3 approaches
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {e}")
            logger.error(f"Cleaned response content: {cleaned_content}")
            return []

    def execute_approach(self, approach: str, system_prompt: str, user_query: str) -> str:
        # Import and call the appropriate function based on the approach
        if approach == "mcts":
            final_response = chat_with_mcts(system_prompt, user_query, self.client, self.model)
        elif approach == "bon":
            final_response = best_of_n_sampling(system_prompt, user_query, self.client, self.model)
        elif approach == 'moa':
            final_response = mixture_of_agents(system_prompt, user_query,self. client, self.model)
        elif approach == 'rto':
            final_response = round_trip_optimization(system_prompt, user_query, self.client, self.model)
        elif approach == 'z3':
            z3_solver = Z3SolverSystem(system_prompt, self.client, self.model)
            final_response = z3_solver.process_query(user_query)
        elif approach == "self_consistency":
            final_response = advanced_self_consistency_approach(system_prompt, user_query, self.client, self.model)
        elif approach == "pvg":
            final_response = inference_time_pv_game(system_prompt, user_query, self.client, self.model)
        elif approach == "rstar":
            rstar = RStar(system_prompt, self.client, self.model)
            final_response = rstar.solve(user_query)
        elif approach == "cot_reflection":
            final_response = cot_reflection(system_prompt, user_query, self.client, self.model)
        elif approach == 'plansearch':
            final_response = plansearch(system_prompt, user_query, self.client, self.model)
        elif approach == 'leap':
            final_response = leap(system_prompt, user_query, self.client, self.model)
        else:
            raise ValueError(f"Unknown approach: {approach}")
        return final_response

    def reflect_on_responses(self, system_prompt: str, user_query: str, approach_responses: Dict[str, str]) -> str:
        reflection_prompt = f"""
        System prompt: {system_prompt}
        User query: {user_query}

        I have used multiple approaches to answer this query. Please analyze the responses from each approach, 
        considering their strengths and weaknesses. Then, provide a final, comprehensive response that 
        combines the best elements from each approach or selects the most appropriate response.

        Responses from different approaches:
        {json.dumps(approach_responses, indent=2)}

        Please provide:
        1. A brief analysis of each approach's response
        2. A final, comprehensive response to the user's query
        3. An explanation of how you arrived at the final response

        Format your response as a JSON object with the following structure:
        {{
            "analysis": {{
                "approach_name": "Analysis of this approach's response",
                ...
            }},
            "final_response": "The final, comprehensive response to the user's query",
            "explanation": "Explanation of how you arrived at the final response"
        }}
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": reflection_prompt}],
            temperature=0.2,
            max_tokens=8192,
        )

        try:
            reflection_data = json.loads(response.choices[0].message.content)
            logger.info("Reflection completed successfully")
            return reflection_data
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing reflection response: {e}")
            return {
                "analysis": {},
                "final_response": "Error in processing the responses.",
                "explanation": "There was an error in analyzing the responses from different approaches."
            }

    def solve(self, system_prompt: str, user_query: str) -> str:
        for attempt in range(self.max_attempts):
            logger.info(f"Attempt {attempt + 1}/{self.max_attempts}")
            
            approaches = self.determine_approaches(system_prompt, user_query)
            logger.info(f"Selected approaches: {approaches}")

            approach_responses = {}
            for approach in approaches:
                response = self.execute_approach(approach, system_prompt, user_query)
                approach_responses[approach] = response
                logger.info(f"Response from {approach}: {response[:100]}...")  # Log first 100 chars

            reflection_result = self.reflect_on_responses(system_prompt, user_query, approach_responses)
            
            final_response = reflection_result.get('final_response', '')
            if final_response:
                logger.info("Final response generated")
                logger.info(f"Analysis: {reflection_result.get('analysis', {})}")
                logger.info(f"Explanation: {reflection_result.get('explanation', '')}")
                return final_response

            logger.info("Response not satisfactory, trying again...")

        logger.warning("Maximum attempts reached without a satisfactory response")
        return "I apologize, but I couldn't generate a satisfactory response to your query after multiple attempts."

def agent_approach(system_prompt: str, user_query: str, client, model: str, max_attempts: int = 3) -> str:
    agent = Agent(client, model, max_attempts)
    return agent.solve(system_prompt, user_query)
