import argparse
import logging
import os
from flask import Flask, request, jsonify
from openai import OpenAI

# Import approach modules
from mcts import chat_with_mcts
from bon import best_of_n_sampling
from moa import mixture_of_agents
from rto import round_trip_optimization
from z3_solver import Z3SolverSystem
from advanced_self_consistency import advanced_self_consistency
from pva import inference_time_pv_game
from rstar import RStar

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# OpenAI API configuration
API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

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
}

@app.route('/v1/chat/completions', methods=['POST'])
def proxy():
    logger.info('Received request to /v1/chat/completions')
    data = request.get_json()
    logger.info(f'Request data: {data}')
    
    messages = data.get('messages', [])
    model = data.get('model', server_config['model'])
    
    system_prompt = next((msg['content'] for msg in messages if msg['role'] == 'system'), "")
    initial_query = next((msg['content'] for msg in messages if msg['role'] == 'user'), "")
    
    approach = server_config['approach']
    
    if approach == 'auto':
        parts = model.split('-', 1)
        approach = parts[0]
        model = parts[1]

    try:
        if approach == 'mcts':
            final_response = chat_with_mcts(system_prompt, initial_query, client, model, server_config['mcts_simulations'],
                                            server_config['mcts_exploration'], server_config['mcts_depth'])
        elif approach == 'bon':
            final_response = best_of_n_sampling(system_prompt, initial_query, client, model, server_config['best_of_n'])
        elif approach == 'moa':
            final_response = mixture_of_agents(system_prompt, initial_query, client, model)
        elif approach == 'rto':
            final_response = round_trip_optimization(system_prompt, initial_query, client, model)
        elif approach == 'z3':
            z3_solver = Z3SolverSystem(system_prompt, client, model)
            final_response = z3_solver.process_query(initial_query)
        elif approach == "self_consistency":
            final_response = advanced_self_consistency(system_prompt, initial_query, client, model)
        elif approach == "pva":
            final_response = inference_time_pv_game(system_prompt, initial_query, client, model)
        elif approach == "rstar":
            rstar = RStar(system_prompt, client, model,
                          max_depth=server_config['rstar_max_depth'], num_rollouts=server_config['rstar_num_rollouts'],
                          c=server_config['rstar_c'])
            final_response = rstar.solve(initial_query)
        else:
            raise ValueError(f"Unknown approach: {approach}")
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

    response_data = {
        'model': model,
        'choices': [
            {
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': final_response,
                },
                'finish_reason': 'stop'
            }
        ],
    }
    logger.info(f'API response: {response_data}')
    return jsonify(response_data), 200

def main():
    parser = argparse.ArgumentParser(description="Run LLM inference with various approaches.")
    parser.add_argument("--approach", type=str, choices=["auto", "mcts", "bon", "moa", "rto", "z3", "self_consistency", "pva", "rstar"], default="auto", help="Inference approach to use")
    parser.add_argument("--simulations", type=int, default=2, help="Number of MCTS simulations")
    parser.add_argument("--exploration", type=float, default=0.2, help="Exploration weight for MCTS")
    parser.add_argument("--depth", type=int, default=1, help="Simulation depth for MCTS")
    parser.add_argument("--best_of_n", type=int, default=3, help="Number of samples for best_of_n approach")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--rstar-max-depth", type=int, default=3, help="Maximum depth for rStar algorithm")
    parser.add_argument("--rstar-num-rollouts", type=int, default=5, help="Number of rollouts for rStar algorithm")
    parser.add_argument("--rstar-c", type=float, default=1.4, help="Exploration constant for rStar algorithm")
    args = parser.parse_args()
    
    server_config.update({
        'model': args.model,
        'approach': args.approach,
        'mcts_simulations': args.simulations,
        'mcts_exploration': args.exploration,
        'mcts_depth': args.depth,
        'best_of_n': args.best_of_n,
        'rstar_max_depth': args.rstar_max_depth,
        'rstar_num_rollouts': args.rstar_num_rollouts,
        'rstar_c': args.rstar_c,
    })
    
    logger.info(f"Starting server with approach: {args.approach}")
    logger.info(f"Server configuration: {server_config}")
    app.run(host='0.0.0.0', port=8000)

if __name__ == "__main__":
    main()
