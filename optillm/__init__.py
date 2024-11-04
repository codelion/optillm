from importlib import util
import os

# Get the path to the root optillm.py
spec = util.spec_from_file_location(
    "optillm.root",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "optillm.py")
)
module = util.module_from_spec(spec)
spec.loader.exec_module(module)

# Export the main entry point
main = module.main

# Export the core configuration and server components
server_config = module.server_config
app = module.app
known_approaches = module.known_approaches
plugin_approaches = module.plugin_approaches

# Export utility functions
parse_combined_approach = module.parse_combined_approach
parse_conversation = module.parse_conversation
extract_optillm_approach = module.extract_optillm_approach
get_config = module.get_config
load_plugins = module.load_plugins

# Export execution functions
execute_single_approach = module.execute_single_approach
execute_combined_approaches = module.execute_combined_approaches
execute_parallel_approaches = module.execute_parallel_approaches

# Export streaming response generation
generate_streaming_response = module.generate_streaming_response

# Version information
__version__ = "0.0.8"  # Match with setup.py

# List of exported symbols
__all__ = [
    'main',
    'server_config',
    'app',
    'known_approaches',
    'plugin_approaches',
    'parse_combined_approach',
    'parse_conversation',
    'extract_optillm_approach',
    'get_config',
    'load_plugins',
    'execute_single_approach',
    'execute_combined_approaches',
    'execute_parallel_approaches',
    'generate_streaming_response',
]