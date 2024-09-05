import random
import logging
import numpy as np
import networkx as nx
from typing import List, Dict

logger = logging.getLogger(__name__)

class DialogueState:
    def __init__(self, system_prompt: str, conversation_history: List[Dict[str, str]], current_query: str):
        self.system_prompt = system_prompt
        self.conversation_history = conversation_history
        self.current_query = current_query

    def __str__(self):
        return f"System: {self.system_prompt}\nHistory: {self.conversation_history}\nCurrent Query: {self.current_query}"

class MCTSNode:
    def __init__(self, state: DialogueState, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

class MCTS:
    def __init__(self, simulation_depth, exploration_weight, client, model):
        self.simulation_depth = simulation_depth
        self.exploration_weight = exploration_weight
        self.root = None
        self.graph = nx.Graph()
        self.node_labels = {}
        self.client = client
        self.model = model

    def select(self, node: MCTSNode) -> MCTSNode:
        if not node.children:
            return node
        return max(node.children, key=lambda c: c.value / (c.visits + 1e-8) + self.exploration_weight * np.sqrt(np.log(node.visits + 1) / (c.visits + 1e-8)))

    def expand(self, node: MCTSNode) -> MCTSNode:
        actions = self.generate_actions(node.state)
        for action in actions:
            new_state = self.apply_action(node.state, action)
            child = MCTSNode(new_state, parent=node)
            node.children.append(child)
            self.graph.add_edge(id(node), id(child))
            self.node_labels[id(child)] = f"Visits: {child.visits}\nValue: {child.value:.2f}"
        return random.choice(node.children)

    def simulate(self, node: MCTSNode) -> float:
        state = node.state
        for _ in range(self.simulation_depth):
            if self.is_terminal(state):
                break
            action = random.choice(self.generate_actions(state))
            state = self.apply_action(state, action)
        return self.evaluate_state(state)

    def backpropagate(self, node: MCTSNode, value: float):
        while node:
            node.visits += 1
            node.value += value
            self.node_labels[id(node)] = f"Visits: {node.visits}\nValue: {node.value:.2f}"
            node = node.parent

    def search(self, initial_state: DialogueState, num_simulations: int) -> DialogueState:
        if not self.root:
            self.root = MCTSNode(initial_state)
            self.graph.add_node(id(self.root))
            self.node_labels[id(self.root)] = f"Root\nVisits: 0\nValue: 0.00"
        
        for _ in range(num_simulations):
            node = self.select(self.root)
            if not self.is_terminal(node.state):
                node = self.expand(node)
            value = self.simulate(node)
            self.backpropagate(node, value)
            
        return max(self.root.children, key=lambda c: c.visits).state

    def generate_actions(self, state: DialogueState) -> List[str]:
        messages = [{"role": "system", "content": state.system_prompt}]
        messages.extend(state.conversation_history)
        messages.append({"role": "user", "content": state.current_query})
        # messages.append({"role": "system", "content": "Generate 3 possible responses to the user's query. Each response should be on a new line starting with 'Response:'."})
        
        completions = []
        n = 3

        response = self.client.chat.completions.create(
            model= self.model,
            messages=messages,
            max_tokens=4096,
            n=n,
            temperature=1
        )
        completions = [choice.message.content.strip() for choice in response.choices]
        # suggested_responses = response.choices[0].message.content.split("Response:")
        # return [resp.strip() for resp in suggested_responses if resp.strip()]
        return completions

    def apply_action(self, state: DialogueState, action: str) -> DialogueState:
        new_history = state.conversation_history.copy()
        new_history.append({"role": "assistant", "content": action})
        
        messages = [{"role": "system", "content": state.system_prompt}]
        messages.extend(new_history)
        messages.append({"role": "system", "content": "Based on this conversation, what might the user ask or say next? Provide a likely user query."})
        
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1024,
            n=1,
            temperature=1
        )
        
        next_query = response.choices[0].message.content
        return DialogueState(state.system_prompt, new_history, next_query)

    def is_terminal(self, state: DialogueState) -> bool:
        # Consider the state terminal if the conversation has reached a natural conclusion
        # or if it has exceeded a certain number of turns
        return len(state.conversation_history) > 10 or "goodbye" in state.current_query.lower()

    def evaluate_state(self, state: DialogueState) -> float:
        messages = [{"role": "system", "content": state.system_prompt}]
        messages.extend(state.conversation_history)
        messages.append({"role": "system", "content": "Evaluate the quality of this conversation on a scale from 0 to 1, where 0 is poor and 1 is excellent. Consider factors such as coherence, relevance, and engagement. Respond with only a number."})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=256,
            n=1,
            temperature=0.1
        )
        
        try:
            score = float(response.choices[0].message.content.strip())
            return max(0, min(score, 1))  # Ensure the score is between 0 and 1
        except ValueError:
            return 0.5  # Default to a neutral score if parsing fails

def chat_with_mcts(system_prompt: str, initial_query: str, client, model: str, num_simulations: int = 2, exploration_weight: float = 0.2, 
                   simulation_depth: int = 1) -> str:
    mcts = MCTS(simulation_depth=simulation_depth, exploration_weight=exploration_weight, client=client, model=model)
    initial_state = DialogueState(system_prompt, [], initial_query)
    final_state = mcts.search(initial_state, num_simulations)
    return final_state.conversation_history[-1]['content'] if final_state.conversation_history else ""
