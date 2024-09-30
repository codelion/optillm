import argparse
import logging
import os
import secrets
from flask import Flask, request, jsonify
from openai import AzureOpenAI, OpenAI
from flask import Response
import json
import importlib
import glob
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import the LiteLLM wrapper
from litellm_wrapper import LiteLLMWrapper

# Import approach modules
from optillm.mcts import chat_with_mcts
from optillm.bon import best_of_n_sampling
from optillm.moa import mixture_of_agents
from optillm.rto import round_trip_optimization
from optillm.self_consistency import advanced_self_consistency_approach
from optillm.pvg import inference_time_pv_game
from optillm.z3_solver import Z3SolverSystem
from optillm.rstar import RStar
from optillm.cot_reflection import cot_reflection
from optillm.plansearch import plansearch
from optillm.leap import leap
from optillm.reread import re2_approach

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# OpenAI, Azure, or LiteLLM API configuration
if os.environ.get("OPENAI_API_KEY"):
    API_KEY = os.environ.get("OPENAI_API_KEY")
    default_client = OpenAI(api_key=API_KEY)
elif os.environ.get("AZURE_OPENAI_API_KEY"):
    API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
    API_VERSION = os.environ.get("AZURE_API_VERSION")
    AZURE_ENDPOINT = os.environ.get("AZURE_API_BASE")
    if API_KEY is not None:
        default_client = AzureOpenAI(
            api_key=API_KEY,
            api_version=API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
        )
    else:
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        azure_credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(azure_credential, "https://cognitiveservices.azure.com/.default")
        default_client = AzureOpenAI(
            api_version=API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            azure_ad_token_provider=token_provider
        )
else:
    default_client = LiteLLMWrapper()

# Server configuration
server_config = {
    'approach': 'bon',
    'mcts_simulations': 2,
    'mcts_exploration': 0.2,
    'mcts_depth': 1,
    'best_of_n': 3,
    'model': 'gpt-4o-mini',
    'rstar_max_depth': 3,
    'rstar_num_rollouts': 5,
    'rstar_c': 1.4,
    'n': 1,
    'base_url': '',
    'optillm_api_key': '',
    'return_full_response': False,
    'port': 8000,
}

# List of known approaches
known_approaches = ["mcts", "bon", "moa", "rto", "z3", "self_consistency", "pvg", "rstar",
                    "cot_reflection", "plansearch", "leap", "re2", "wim"]

plugin_approaches = {}

def load_plugins():
    plugin_dir = os.path.join(os.path.dirname(__file__), 'optillm/plugins')
    plugin_files = glob.glob(os.path.join(plugin_dir, '*.py'))

    for plugin_file in plugin_files:
        module_name = os.path.basename(plugin_file)[:-3]  # Remove .py extension
        spec = importlib.util.spec_from_file_location(module_name, plugin_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, 'SLUG') and hasattr(module, 'run'):
            plugin_approaches[module.SLUG] = module.run
            logger.info(f"Loaded plugin: {module.SLUG}")

def parse_combined_approach(model: str, known_approaches: list, plugin_approaches: dict):
    if model == 'auto':
        return 'SINGLE', ['bon'], model

    parts = model.split('-')
    approaches = []
    operation = 'SINGLE'
    model_parts = []
    parsing_approaches = True

    for part in parts:
        if parsing_approaches:
            if part in known_approaches or part in plugin_approaches:
                approaches.append(part)
            elif '&' in part:
                operation = 'AND'
                approaches.extend(part.split('&'))
            elif '|' in part:
                operation = 'OR'
                approaches.extend(part.split('|'))
            else:
                parsing_approaches = False
                model_parts.append(part)
        else:
            model_parts.append(part)

    if not approaches:
        approaches = ['bon']
        operation = 'SINGLE'

    actual_model = '-'.join(model_parts)

    return operation, approaches, actual_model
    
def execute_single_approach(approach, system_prompt, initial_query, client, model):
    if approach in known_approaches:
        # Execute known approaches
        if approach == 'mcts':
            return chat_with_mcts(system_prompt, initial_query, client, model, server_config['mcts_simulations'],
                                            server_config['mcts_exploration'], server_config['mcts_depth'])
        elif approach == 'bon':
            return  best_of_n_sampling(system_prompt, initial_query, client, model, server_config['best_of_n'])
        elif approach == 'moa':
            return mixture_of_agents(system_prompt, initial_query, client, model)
        elif approach == 'rto':
            return round_trip_optimization(system_prompt, initial_query, client, model)
        elif approach == 'z3':
            z3_solver = Z3SolverSystem(system_prompt, client, model)
            return z3_solver.process_query(initial_query)
        elif approach == "self_consistency":
            return advanced_self_consistency_approach(system_prompt, initial_query, client, model)
        elif approach == "pvg":
            return inference_time_pv_game(system_prompt, initial_query, client, model)
        elif approach == "rstar":
            rstar = RStar(system_prompt, client, model,
                          max_depth=server_config['rstar_max_depth'], num_rollouts=server_config['rstar_num_rollouts'],
                          c=server_config['rstar_c'])
            return rstar.solve(initial_query)
        elif approach == "cot_reflection":
            return cot_reflection(system_prompt, initial_query, client, model, return_full_response=server_config['return_full_response'])
        elif approach == 'plansearch':
            return plansearch(system_prompt, initial_query, client, model, n=n)
        elif approach == 'leap':
            return leap(system_prompt, initial_query, client, model)
        elif approach == 're2':
            return re2_approach(system_prompt, initial_query, client, model, n=n)
    elif approach in plugin_approaches:
        return plugin_approaches[approach](system_prompt, initial_query, client, model)
    else:
        raise ValueError(f"Unknown approach: {approach}")
    
def execute_combined_approaches(approaches, system_prompt, initial_query, client, model):
    final_response = initial_query
    total_tokens = 0
    for approach in approaches:
        response, tokens = execute_single_approach(approach, system_prompt, final_response, client, model)
        final_response = response
        total_tokens += tokens
    return final_response, total_tokens

async def execute_parallel_approaches(approaches, system_prompt, initial_query, client, model):
    async def run_approach(approach):
        return await asyncio.to_thread(execute_single_approach, approach, system_prompt, initial_query, client, model)

    tasks = [run_approach(approach) for approach in approaches]
    results = await asyncio.gather(*tasks)
    responses, tokens = zip(*results)
    return list(responses), sum(tokens)

def generate_streaming_response(final_response, model):
    # Yield the final response
    if isinstance(final_response, list):
        for index, response in enumerate(final_response):
            yield "data: " + json.dumps({
                "choices": [{"delta": {"content": response}, "index": index, "finish_reason": "stop"}],
                "model": model,
            }) + "\n\n"
    else:
        yield "data: " + json.dumps({
            "choices": [{"delta": {"content": final_response}, "index": 0, "finish_reason": "stop"}],
            "model": model,
        }) + "\n\n"

    # Yield the final message to indicate the stream has ended
    yield "data: [DONE]\n\n"

def parse_conversation(messages):
    system_prompt = ""
    conversation = []
    
    for message in messages:
        role = message['role']
        content = message['content']
        
        if role == 'system':
            system_prompt = content
        elif role in ['user', 'assistant']:
            conversation.append(f"{role.capitalize()}: {content}")
    
    initial_query = "\n".join(conversation)
    return system_prompt, initial_query

# Optional API key configuration to secure the proxy
@app.before_request
def check_api_key():
    if server_config['optillm_api_key']:
        if request.path == "/health":
            return

        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Invalid Authorization header. Expected format: 'Authorization: Bearer YOUR_API_KEY'"}), 401

        client_key = auth_header.split('Bearer ', 1)[1].strip()
        if not secrets.compare_digest(client_key, server_config['optillm_api_key']):
            return jsonify({"error": "Invalid API key"}), 401

@app.route('/v1/chat/completions', methods=['POST'])
def proxy():
    logger.info('Received request to /v1/chat/completions')
    data = request.get_json()
    logger.debug(f'Request data: {data}')

    stream = data.get('stream', False)
    messages = data.get('messages', [])
    model = data.get('model', server_config['model'])
    n = data.get('n', server_config['n'])

    system_prompt, initial_query = parse_conversation(messages)

    approach = server_config['approach']
    base_url = server_config['base_url']

    if base_url != "":
        client = OpenAI(api_key=API_KEY, base_url=base_url)
    else:
        client = default_client

    operation, approaches, model = parse_combined_approach(model, known_approaches, plugin_approaches)
    logger.info(f'Using approach(es) {approaches}, operation {operation}, with model {model}')

    try:
        if operation == 'SINGLE':
            final_response, completion_tokens = execute_single_approach(approaches[0], system_prompt, initial_query, client, model)
        elif operation == 'AND':
            final_response, completion_tokens = execute_combined_approaches(approaches, system_prompt, initial_query, client, model)
        elif operation == 'OR':
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            final_response, completion_tokens = loop.run_until_complete(execute_parallel_approaches(approaches, system_prompt, initial_query, client, model))
        else:
            raise ValueError(f"Unknown operation: {operation}")
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
    if stream:
        return Response(generate_streaming_response(final_response, model), content_type='text/event-stream')
    else:
        response_data = {
            'model': model,
            'choices': [],
            'usage': {
                'completion_tokens': completion_tokens,
            }
        }

        if isinstance(final_response, list):
            for index, response in enumerate(final_response):
                response_data['choices'].append({
                    'index': index,
                    'message': {
                        'role': 'assistant',
                        'content': response,
                    },
                    'finish_reason': 'stop'
                })
        else:
            response_data['choices'].append({
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': final_response,
                },
                'finish_reason': 'stop'
            })

    logger.debug(f'API response: {response_data}')
    return jsonify(response_data), 200

@app.route('/v1/models', methods=['GET'])
def proxy_models():
    logger.info('Received request to /v1/models')
    
    try:
        if server_config['base_url']:
            client = OpenAI(api_key=API_KEY, base_url=server_config['base_url'])
        else:
            client = default_client

        # Fetch models using the OpenAI client and return the raw response
        models_response = client.models.list()

        logger.debug('Models retrieved successfully')
        return models_response.model_dump(), 200
    except Exception as e:
        logger.error(f"Error fetching models: {str(e)}")
        return jsonify({"error": f"Error fetching models: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM inference with various approaches.")

    # Define arguments and their corresponding environment variables
    args_env = [
        ("--optillm-api-key", "OPTILLM_API_KEY", str, "", "Optional API key for client authentication to optillm"),
        ("--approach", "OPTILLM_APPROACH", str, "auto", "Inference approach to use", known_approaches),
        ("--simulations", "OPTILLM_SIMULATIONS", int, 2, "Number of MCTS simulations"),
        ("--exploration", "OPTILLM_EXPLORATION", float, 0.2, "Exploration weight for MCTS"),
        ("--depth", "OPTILLM_DEPTH", int, 1, "Simulation depth for MCTS"),
        ("--model", "OPTILLM_MODEL", str, "gpt-4o-mini", "OpenAI model to use"),
        ("--rstar-max-depth", "OPTILLM_RSTAR_MAX_DEPTH", int, 3, "Maximum depth for rStar algorithm"),
        ("--rstar-num-rollouts", "OPTILLM_RSTAR_NUM_ROLLOUTS", int, 5, "Number of rollouts for rStar algorithm"),
        ("--rstar-c", "OPTILLM_RSTAR_C", float, 1.4, "Exploration constant for rStar algorithm"),
        ("--n", "OPTILLM_N", int, 1, "Number of final responses to be returned"),
        ("--return-full-response", "OPTILLM_RETURN_FULL_RESPONSE", bool, False, "Return the full response including the CoT with <thinking> tags"),
        ("--port", "OPTILLM_PORT", int, 8000, "Specify the port to run the proxy"),
    ]

    for arg, env, type_, default, help_text, *extra in args_env:
        env_value = os.environ.get(env)
        if env_value is not None:
            if type_ == bool:
                default = env_value.lower() in ('true', '1', 'yes')
            else:
                default = type_(env_value)
        if extra and extra[0]:  # Check if there are choices for this argument
            parser.add_argument(arg, type=type_, default=default, help=help_text, choices=extra[0])
        else:
            parser.add_argument(arg, type=type_, default=default, help=help_text)

    # Special handling for best_of_n to support both formats
    best_of_n_default = int(os.environ.get("OPTILLM_BEST_OF_N", 3))
    parser.add_argument("--best-of-n", "--best_of_n", dest="best_of_n", type=int, default=best_of_n_default,
                        help="Number of samples for best_of_n approach")

    # Special handling for base_url to support both formats
    base_url_default = os.environ.get("OPTILLM_BASE_URL", "")
    parser.add_argument("--base-url", "--base_url", dest="base_url", type=str, default=base_url_default,
                        help="Base url for OpenAI compatible endpoint")

    args = parser.parse_args()

    # Convert argument names to match server_config keys
    args_dict = vars(args)
    for key in list(args_dict.keys()):
        new_key = key.replace("-", "_")
        if new_key != key:
            args_dict[new_key] = args_dict.pop(key)

    return args

def main():
    global server_config
    args = parse_args()

    # Call this function at the start of main()
    load_plugins()
    # Update server_config with all argument values
    server_config.update(vars(args))

    port = server_config['port']
    logger.info(f"Starting server with approach: {server_config['approach']}")
    server_config_clean = server_config.copy()
    if server_config_clean['optillm_api_key']:
        server_config_clean['optillm_api_key'] = '[REDACTED]'
    logger.info(f"Server configuration: {server_config_clean}")
    app.run(host='0.0.0.0', port=port)

if __name__ == "__main__":
    main()
