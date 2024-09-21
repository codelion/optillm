import gradio as gr
import requests
import os
import json
import re

# Define the default base URL for the optillm proxy
DEFAULT_BASE_URL = "http://localhost:8000/v1"

# List of available approaches based on optillm's implementation
APPROACHES = [
    "agent", "mcts", "bon", "moa", "rto", "z3", "self_consistency",
    "pvg", "rstar", "cot_reflection", "plansearch", "leap"
]

# Add descriptions for each approach
APPROACH_DESCRIPTIONS = {
    "agent": "An agent-based approach that attempts to solve the problem multiple times.",
    "mcts": "Monte Carlo Tree Search for exploring possible solution paths.",
    "bon": "Best of N sampling, generating multiple responses and selecting the best.",
    "moa": "Mixture of Agents, combining multiple AI agents for diverse perspectives.",
    "rto": "Round Trip Optimization, refining the solution through multiple passes.",
    "z3": "Z3 Theorem Prover for logical and mathematical problem-solving.",
    "self_consistency": "Generates multiple solutions and checks for consistency among them.",
    "pvg": "Prediction vs. Generation game for improved accuracy.",
    "rstar": "R* search algorithm for efficient problem-solving in large state spaces.",
    "cot_reflection": "Chain of Thought with reflection for improved reasoning.",
    "plansearch": "Searches for the best plan to solve complex problems.",
    "leap": "Language Model Extrapolation and Planning for advanced problem-solving."
}

# List of available models
MODELS = ["gpt-3.5-turbo", "gpt-4", "gpt-4-32k", "gpt-4o-mini"]

# Mapping of parameters with their types, relevant approaches, and default values
PARAMETERS = {
    "simulations": {
        "type": "slider", "label": "MCTS Simulations", "min": 1, "max": 10, "step": 1, "default": 2,
        "approaches": ["mcts"]
    },
    "exploration": {
        "type": "slider", "label": "MCTS Exploration", "min": 0.0, "max": 1.0, "step": 0.1, "default": 0.2,
        "approaches": ["mcts"]
    },
    "depth": {
        "type": "slider", "label": "MCTS Depth", "min": 1, "max": 5, "step": 1, "default": 1,
        "approaches": ["mcts"]
    },
    "best_of_n": {
        "type": "slider", "label": "Best of N Sampling", "min": 1, "max": 10, "step": 1, "default": 3,
        "approaches": ["bon"]
    },
    "rstar_max_depth": {
        "type": "slider", "label": "R* Max Depth", "min": 1, "max": 10, "step": 1, "default": 3,
        "approaches": ["rstar"]
    },
    "rstar_num_rollouts": {
        "type": "slider", "label": "R* Number of Rollouts", "min": 1, "max": 10, "step": 1, "default": 5,
        "approaches": ["rstar"]
    },
    "rstar_c": {
        "type": "slider", "label": "R* Exploration Constant", "min": 1.0, "max": 2.0, "step": 0.1, "default": 1.4,
        "approaches": ["rstar"]
    },
    "n": {
        "type": "slider", "label": "Number of Responses", "min": 1, "max": 5, "step": 1, "default": 1,
        "approaches": APPROACHES
    },
    "return_full_response": {
        "type": "checkbox", "label": "Return Full Response", "default": False,
        "approaches": ["cot_reflection"]
    },
    "max_attempts": {
        "type": "slider", "label": "Max Attempts", "min": 1, "max": 10, "step": 1, "default": 3,
        "approaches": ["agent"]
    },
}

# Updated component_map to include all necessary component types
component_map = {
    "slider": gr.Slider,
    "checkbox": gr.Checkbox,
    "dropdown": gr.Dropdown,
    "textbox": gr.Textbox
}

# Updated parameter_components to pass only relevant arguments based on component type
parameter_components = {
    key: component_map[param["type"]](
        **{
            "minimum": param["min"],
            "maximum": param["max"],
            "step": param["step"]
        } if param["type"] == "slider" else {},
        label=param["label"],
        value=param["default"],
        visible=False
    )
    for key, param in PARAMETERS.items()
}

def send_request(message, history, selected_approach, selected_model, max_tokens, temperature, *params):
    """
    Sends a request to the optillm proxy server with the user's message,
    selected approach, and relevant parameters.
    """
    # Extract api_key and return_full_response from params
    api_key = params[-2]
    return_full_response = params[-1]
    params = params[:-2]  # Remove the last 2 items from params

    # Prepare the API endpoint
    endpoint = f"{DEFAULT_BASE_URL}/chat/completions"

    # Prepare the headers with Authorization if API key is provided
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Prepare the payload
    payload = {
        "model": f"{selected_approach}-{selected_model}",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": params[list(PARAMETERS.keys()).index('n')],
    }

    # Include additional parameters based on the selected approach
    for i, (key, param) in enumerate(PARAMETERS.items()):
        if selected_approach in param["approaches"] or selected_approach == "auto":
            payload[key] = params[i]

    # Handle 'return_full_response' if applicable
    if selected_approach == "cot_reflection" and return_full_response is not None:
        payload["return_full_response"] = return_full_response

    try:
        response = requests.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            return history + [("Error", "No response from the model.")], ""

        # Iterate through all returned choices if n > 1
        new_history = history.copy()
        for choice in choices:
            assistant_response = choice.get("message", {}).get("content", "No content.")
            new_history.append((message, assistant_response))
        
        return new_history, ""

    except requests.exceptions.RequestException as e:
        error_message = f"API request failed: {e}"
        return history + [(message, error_message)], ""

def reset_history():
    """Resets the conversation history."""
    return [], ""

def filter_parameters(selected_approach):
    """Determines which parameters should be visible based on the selected approach."""
    return {key: (selected_approach in param["approaches"] or selected_approach == "auto")
            for key, param in PARAMETERS.items()}

def load_approach_parameters(selected_approach):
    """Updates the visibility and values of parameters based on the selected approach."""
    visibility = filter_parameters(selected_approach)
    return [gr.update(visible=visibility.get(key, False), value=param["default"])
            for key, param in PARAMETERS.items()]

with gr.Blocks() as demo:
    gr.Markdown("# OptiLLM Chat Interface")
    gr.Markdown("Interact with the OptiLLM proxy to enhance your language model's performance.")

    with gr.Row():
        with gr.Column(scale=1):
            # Configuration
            with gr.Group():
                gr.Markdown("### Configuration")
                api_key = gr.Textbox(label="OpenAI API Key", type="password", placeholder="Enter your OpenAI API key here...")
                model = gr.Dropdown(choices=MODELS, label="Model", value="gpt-4o-mini")
                with gr.Row():
                    selected_approach = gr.Dropdown(
                        choices=[(f"{approach.capitalize()} - {APPROACH_DESCRIPTIONS[approach]}", approach) for approach in APPROACHES],
                        label="Select Approach",
                        value="bon",
                        info="Hover over options for descriptions",
                        allow_custom_value=False  # Ensures the value is within the choices
                    )
                    load_approach_btn = gr.Button("Load Approach")

            # Dynamic Parameter Inputs
            with gr.Group():
                gr.Markdown("### Approach Parameters")
                parameter_components = {key: component_map[param["type"]](
                    **{
                        "minimum": param["min"],
                        "maximum": param["max"],
                        "step": param["step"]
                    } if param["type"] == "slider" else {},
                    label=param["label"],
                    value=param["default"],
                    visible=False
                ) for key, param in PARAMETERS.items()}

            # General Parameters
            with gr.Group():
                gr.Markdown("### General Parameters")
                max_tokens = gr.Slider(minimum=1, maximum=4096, step=1, label="Max Tokens", value=1024)
                temperature = gr.Slider(minimum=0.0, maximum=2.0, step=0.1, label="Temperature", value=0.7)

        with gr.Column(scale=3):
            # Chat interface
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="Message", placeholder="Type your message here...")
            send_btn = gr.Button("Send")
            clear_btn = gr.Button("Clear Conversation")

    # Link the load_approach button to update parameter visibility
    load_approach_btn.click(
        fn=load_approach_parameters,
        inputs=[selected_approach],
        outputs=list(parameter_components.values())
    )

    # Link the visibility changes to the parameter components
    selected_approach.change(
        fn=load_approach_parameters,
        inputs=[selected_approach],
        outputs=list(parameter_components.values())
    )

    # Define the send button action
    send_btn.click(
        send_request,
        inputs=[
            msg, chatbot, selected_approach, model, max_tokens, temperature,
            *parameter_components.values(),
            api_key, parameter_components["return_full_response"]
        ],
        outputs=[chatbot, msg]
    )

    # Allow message submission via pressing Enter
    msg.submit(
        send_request,
        inputs=[
            msg, chatbot, selected_approach, model, max_tokens, temperature,
            *parameter_components.values(),
            api_key, parameter_components["return_full_response"]
        ],
        outputs=[chatbot, msg]
    )

    # Define the clear button action
    clear_btn.click(reset_history, inputs=None, outputs=[chatbot, msg])

if __name__ == "__main__":
    demo.launch()
