# Apache license 2 - modified after the fork to add the Cerebras API option and CePO as a test-time compute method
import argparse
import logging
import os
import secrets
from flask import Flask, request, jsonify
from cerebras.cloud.sdk import Cerebras
from openai import AzureOpenAI, OpenAI
from flask import Response
import json
import importlib
import glob
import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Optional, Union, Dict, Any, List
from importlib.metadata import version
from dataclasses import fields

# Import approach modules
from optillm.mcts import chat_with_mcts
from optillm.bon import best_of_n_sampling
from optillm.moa import mixture_of_agents
from optillm.rto import round_trip_optimization
from optillm.self_consistency import advanced_self_consistency_approach
from optillm.pvg import inference_time_pv_game
from optillm.z3_solver import Z3SymPySolverSystem
from optillm.rstar import RStar
from optillm.cot_reflection import cot_reflection
from optillm.plansearch import plansearch
from optillm.leap import leap
from optillm.reread import re2_approach
from optillm.cepo import cepo, CepoConfig, init_cepo_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging_levels = {
    "notset": logging.NOTSET,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

# Initialize Flask app
app = Flask(__name__)

def get_config():
    API_KEY = None
    if os.environ.get("OPTILLM_API_KEY"):
        # Use local inference engine
        from optillm.inference import create_inference_client
        API_KEY = os.environ.get("OPTILLM_API_KEY")
        default_client = create_inference_client()
    # Cerebras, OpenAI, Azure, or LiteLLM API configuration
    elif os.environ.get("CEREBRAS_API_KEY"):
        API_KEY = os.environ.get("CEREBRAS_API_KEY")
        base_url = server_config['base_url']
        if base_url != "":
            default_client = Cerebras(api_key=API_KEY, base_url=base_url)
        else:
            default_client = Cerebras(api_key=API_KEY)
    elif os.environ.get("OPENAI_API_KEY"):
        API_KEY = os.environ.get("OPENAI_API_KEY")
        base_url = server_config['base_url']
        if base_url != "":
            default_client = OpenAI(api_key=API_KEY, base_url=base_url)
        else:
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
        # Import the LiteLLM wrapper
        from optillm.litellm_wrapper import LiteLLMWrapper
        default_client = LiteLLMWrapper()
    return default_client, API_KEY

# Server configuration
server_config = {
    'approach': 'none', 
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
    'log': 'info',
}

# List of known approaches
known_approaches = ["none", "mcts", "bon", "moa", "rto", "z3", "self_consistency", 
                   "pvg", "rstar", "cot_reflection", "plansearch", "leap", "re2", "cepo"]

plugin_approaches = {}

def none_approach(
    client: Any, 
    model: str,
    original_messages: List[Dict[str, str]],
    **kwargs
) -> Dict[str, Any]:
    """
    Direct proxy approach that passes through all parameters to the underlying endpoint.
    
    Args:
        system_prompt: System prompt text (unused)
        initial_query: Initial query/conversation (unused)
        client: OpenAI client instance
        model: Model identifier
        original_messages: Original messages from the request
        **kwargs: Additional parameters to pass through
    
    Returns:
        Dict[str, Any]: Full OpenAI API response
    """
    # Strip 'none-' prefix from model if present
    if model.startswith('none-'):
        model = model[5:]
    
    try:
        # Make the direct completion call with original messages and parameters
        response = client.chat.completions.create(
            model=model,
            messages=original_messages,
            **kwargs
        )
        
        # Convert to dict if it's not already
        if hasattr(response, 'model_dump'):
            return response.model_dump()
        return response
        
    except Exception as e:
        logger.error(f"Error in none approach: {str(e)}")
        raise

def load_plugins():
   # Clear existing plugins first but modify the global dict in place
   plugin_approaches.clear()
   
   # Get installed package plugins directory
   import optillm
   package_plugin_dir = os.path.join(os.path.dirname(optillm.__file__), 'plugins')
   
   # Get local project plugins directory
   current_dir = os.getcwd()
   local_plugin_dir = os.path.join(current_dir, 'optillm', 'plugins')
   
   plugin_dirs = []
   
   # Add package plugin dir
   plugin_dirs.append((package_plugin_dir, "package"))
   
   # Add local plugin dir only if it's different from package dir
   if local_plugin_dir != package_plugin_dir:
       plugin_dirs.append((local_plugin_dir, "local"))
   
   for plugin_dir, source in plugin_dirs:
       logger.info(f"Looking for {source} plugins in: {plugin_dir}")
       
       if not os.path.exists(plugin_dir):
           logger.debug(f"{source.capitalize()} plugin directory not found: {plugin_dir}")
           continue
           
       plugin_files = glob.glob(os.path.join(plugin_dir, '*.py'))
       if not plugin_files:
           logger.debug(f"No plugin files found in {source} directory: {plugin_dir}")
           continue
           
       logger.info(f"Found {source} plugin files: {plugin_files}")
       
       for plugin_file in plugin_files:
           try:
               module_name = os.path.basename(plugin_file)[:-3]  # Remove .py extension
               spec = importlib.util.spec_from_file_location(module_name, plugin_file)
               module = importlib.util.module_from_spec(spec)
               spec.loader.exec_module(module)
               
               if hasattr(module, 'SLUG') and hasattr(module, 'run'):
                   if module.SLUG in plugin_approaches:
                       logger.info(f"Overriding {source} plugin: {module.SLUG}")
                   plugin_approaches[module.SLUG] = module.run
                   logger.info(f"Loaded {source} plugin: {module.SLUG}")
               else:
                   logger.warning(f"Plugin {module_name} from {source} missing required attributes (SLUG and run)")
           except Exception as e:
               logger.error(f"Error loading {source} plugin {plugin_file}: {str(e)}")
   
   if not plugin_approaches:
       logger.warning("No plugins loaded from any location")

def parse_combined_approach(model: str, known_approaches: list, plugin_approaches: dict):
    if model == 'auto':
        return 'SINGLE', ['none'], model

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
        approaches = ['none']
        operation = 'SINGLE'

    actual_model = '-'.join(model_parts)

    return operation, approaches, actual_model
    
def execute_single_approach(approach, system_prompt, initial_query, client, model):
    if approach in known_approaches:
        if approach == 'none':
            # Extract kwargs from the request data
            kwargs = {}
            if hasattr(request, 'json'):
                data = request.get_json()
                messages = data.get('messages', [])
                # Copy all parameters except 'model' and 'messages'
                kwargs = {k: v for k, v in data.items() 
                         if k not in ['model', 'messages', 'optillm_approach']}
            response = none_approach(original_messages=messages, client=client, model=model, **kwargs)
            
            # For none approach, we return the response and a token count of 0
            # since the full token count is already in the response
            return response, 0
        elif approach == 'mcts':
            return chat_with_mcts(system_prompt, initial_query, client, model, server_config['mcts_simulations'],
                                            server_config['mcts_exploration'], server_config['mcts_depth'])
        elif approach == 'bon':
            return  best_of_n_sampling(system_prompt, initial_query, client, model, server_config['best_of_n'])
        elif approach == 'moa':
            return mixture_of_agents(system_prompt, initial_query, client, model)
        elif approach == 'rto':
            return round_trip_optimization(system_prompt, initial_query, client, model)
        elif approach == 'z3':
            z3_solver = Z3SymPySolverSystem(system_prompt, client, model)
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
            return plansearch(system_prompt, initial_query, client, model, n=server_config['n'])
        elif approach == 'leap':
            return leap(system_prompt, initial_query, client, model)
        elif approach == 're2':
            return re2_approach(system_prompt, initial_query, client, model, n=server_config['n'])
        elif approach == 'cepo':
            logger.debug(f"Calling with {cepo_config}")
            return cepo(system_prompt, initial_query, client, model, cepo_config)
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

def execute_n_times(n: int, approaches, operation: str, system_prompt: str, initial_query: str, client: Any, model: str) -> Tuple[Union[str, List[str]], int]:
    """
    Execute the pipeline n times and return n responses.
    
    Args:
        n (int): Number of times to run the pipeline
        approaches (list): List of approaches to execute
        operation (str): Operation type ('SINGLE', 'AND', or 'OR')
        system_prompt (str): System prompt
        initial_query (str): Initial query
        client: OpenAI client instance
        model (str): Model identifier
        
    Returns:
        Tuple[Union[str, List[str]], int]: List of responses and total token count
    """
    responses = []
    total_tokens = 0
    
    for _ in range(n):
        if operation == 'SINGLE':
            response, tokens = execute_single_approach(approaches[0], system_prompt, initial_query, client, model)
        elif operation == 'AND':
            response, tokens = execute_combined_approaches(approaches, system_prompt, initial_query, client, model)
        elif operation == 'OR':
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response, tokens = loop.run_until_complete(execute_parallel_approaches(approaches, system_prompt, initial_query, client, model))
            loop.close()
        else:
            raise ValueError(f"Unknown operation: {operation}")
            
        # If response is already a list (from OR operation), extend responses
        # Otherwise append the single response
        if isinstance(response, list):
            responses.extend(response)
        else:
            responses.append(response)
        total_tokens += tokens
        
    # If n=1 and we got a single response, return it as is
    # Otherwise return the list of responses
    if n == 1 and len(responses) == 1:
        return responses[0], total_tokens
    return responses, total_tokens

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
    optillm_approach = None
    
    for message in messages:
        role = message['role']
        content = message['content']
        
        # Handle content that could be a list or string
        if isinstance(content, list):
            # Extract text content from the list
            text_content = ' '.join(
                item['text'] for item in content 
                if isinstance(item, dict) and item.get('type') == 'text'
            )
        else:
            text_content = content
        
        if role == 'system':
            system_prompt, optillm_approach = extract_optillm_approach(text_content)
        elif role == 'user':
            if not optillm_approach:
                text_content, optillm_approach = extract_optillm_approach(text_content)
            conversation.append(f"User: {text_content}")
        elif role == 'assistant':
            conversation.append(f"Assistant: {text_content}")
    
    initial_query = "\n".join(conversation)
    return system_prompt, initial_query, optillm_approach

def extract_optillm_approach(content):
    match = re.search(r'<optillm_approach>(.*?)</optillm_approach>', content)
    if match:
        approach = match.group(1)
        content = re.sub(r'<optillm_approach>.*?</optillm_approach>', '', content).strip()
        return content, approach
    return content, None

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
    auth_header = request.headers.get("Authorization")
    bearer_token = ""

    if auth_header and auth_header.startswith("Bearer "):
        bearer_token = auth_header.split("Bearer ")[1].strip()
        logger.debug(f"Intercepted Bearer Token: {bearer_token}")
    
    logger.debug(f'Request data: {data}')

    stream = data.get('stream', False)
    messages = data.get('messages', [])
    model = data.get('model', server_config['model'])
    n = data.get('n', server_config['n'])  # Get n value from request or config

    optillm_approach = data.get('optillm_approach', server_config['approach'])
    logger.debug(data)
    server_config['mcts_depth'] = data.get('mcts_depth', server_config['mcts_depth'])
    server_config['mcts_exploration'] = data.get('mcts_exploration', server_config['mcts_exploration'])
    server_config['mcts_simulations'] = data.get('mcts_simulations', server_config['mcts_simulations'])

    system_prompt, initial_query, message_optillm_approach = parse_conversation(messages)

    if message_optillm_approach:
        optillm_approach = message_optillm_approach

    if optillm_approach != "auto":
        model = f"{optillm_approach}-{model}"

    base_url = server_config['base_url']
    default_client, api_key = get_config()

    operation, approaches, model = parse_combined_approach(model, known_approaches, plugin_approaches)
    logger.info(f'Using approach(es) {approaches}, operation {operation}, with model {model}')

    if bearer_token != "" and bearer_token.startswith("sk-"):
        api_key = bearer_token
        if base_url != "":
            client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            client = OpenAI(api_key=api_key)
    else: 
        client = default_client

    try:
        # Check if any of the approaches is 'none'
        contains_none = any(approach == 'none' for approach in approaches)

        if operation == 'SINGLE' and approaches[0] == 'none':
            # For none approach with n>1, make n separate calls
            if n > 1:
                responses = []
                completion_tokens = 0
                for _ in range(n):
                    result, tokens = execute_single_approach(approaches[0], system_prompt, initial_query, client, model)
                    responses.append(result)
                    completion_tokens += tokens
                result = responses
            else:
                result, completion_tokens = execute_single_approach(approaches[0], system_prompt, initial_query, client, model)
            logger.debug(f'Direct proxy response: {result}')
            return jsonify(result), 200
            
        elif operation == 'AND' or operation == 'OR':
            if contains_none:
                raise ValueError("'none' approach cannot be combined with other approaches")

        # Handle non-none approaches with n attempts
        response, completion_tokens = execute_n_times(n, approaches, operation, system_prompt, initial_query, client, model)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
    if stream:
        return Response(generate_streaming_response(response, model), content_type='text/event-stream')
    else:
        response_data = {
            'model': model,
            'choices': [],
            'usage': {
                'completion_tokens': completion_tokens,
            }
        }

        if isinstance(response, list):
            for index, resp in enumerate(response):
                response_data['choices'].append({
                    'index': index,
                    'message': {
                        'role': 'assistant',
                        'content': resp,
                    },
                    'finish_reason': 'stop'
                })
        else:
            response_data['choices'].append({
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': response,
                },
                'finish_reason': 'stop'
            })

        logger.debug(f'API response: {response_data}')
        return jsonify(response_data), 200

@app.route('/v1/models', methods=['GET'])
def proxy_models():
    logger.info('Received request to /v1/models')
    default_client, API_KEY = get_config()
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
    
     # Add version argument using importlib.metadata
    try:
        package_version = version('optillm')
    except Exception:
        package_version = "unknown"  # Fallback if package is not installed
        
    parser.add_argument('--version', action='version', 
                       version=f'%(prog)s {package_version}',
                       help="Show program's version number and exit")

    # Define arguments and their corresponding environment variables
    args_env = [
        ("--optillm-api-key", "OPTILLM_API_KEY", str, "", "Optional API key for client authentication to optillm"),
        ("--approach", "OPTILLM_APPROACH", str, "auto", "Inference approach to use", known_approaches + list(plugin_approaches.keys())),
        ("--mcts-simulations", "OPTILLM_SIMULATIONS", int, 2, "Number of MCTS simulations"),
        ("--mcts-exploration", "OPTILLM_EXPLORATION", float, 0.2, "Exploration weight for MCTS"),
        ("--mcts-depth", "OPTILLM_DEPTH", int, 1, "Simulation depth for MCTS"),
        ("--model", "OPTILLM_MODEL", str, "gpt-4o-mini", "OpenAI model to use"),
        ("--rstar-max-depth", "OPTILLM_RSTAR_MAX_DEPTH", int, 3, "Maximum depth for rStar algorithm"),
        ("--rstar-num-rollouts", "OPTILLM_RSTAR_NUM_ROLLOUTS", int, 5, "Number of rollouts for rStar algorithm"),
        ("--rstar-c", "OPTILLM_RSTAR_C", float, 1.4, "Exploration constant for rStar algorithm"),
        ("--n", "OPTILLM_N", int, 1, "Number of final responses to be returned"),
        ("--return-full-response", "OPTILLM_RETURN_FULL_RESPONSE", bool, False, "Return the full response including the CoT with <thinking> tags"),
        ("--port", "OPTILLM_PORT", int, 8000, "Specify the port to run the proxy"),
        ("--log", "OPTILLM_LOG", str, "info", "Specify the logging level", list(logging_levels.keys()))
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

    # Special handling of all the CePO Configurations
    for field in fields(CepoConfig):
        parser.add_argument(f"--cepo_{field.name}", dest=f"cepo_{field.name}", type=field.type, default=None, help=f"CePO configuration for {field.name}")

    parser.add_argument(f"--cepo_config_file", dest=f"cepo_config_file", type=str, default="./configs/cepo_config.yaml", help="Path to CePO configuration file")

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
    global cepo_config
    # Call this function at the start of main()
    load_plugins()
    args = parse_args()

    # Update server_config with all argument values
    server_config.update(vars(args))

    port = server_config['port']

    # Set logging level from user request
    logging_level = server_config['log']
    if logging_level in logging_levels.keys():
        logger.setLevel(logging_levels[logging_level])
    
    # set and log the cepo configs
    cepo_config = init_cepo_config(server_config)
    if args.approach == 'cepo':
        logger.info(f"CePO Config: {cepo_config}")
    
    logger.info(f"Starting server with approach: {server_config['approach']}")
    server_config_clean = server_config.copy()
    if server_config_clean['optillm_api_key']:
        server_config_clean['optillm_api_key'] = '[REDACTED]'
    logger.info(f"Server configuration: {server_config_clean}")
    app.run(host='0.0.0.0', port=port)

if __name__ == "__main__":
    main()
