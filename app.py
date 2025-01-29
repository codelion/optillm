import os
import gradio as gr
from openai import OpenAI

from optillm.cot_reflection import cot_reflection
from optillm.rto import round_trip_optimization
from optillm.z3_solver import Z3SymPySolverSystem
from optillm.self_consistency import advanced_self_consistency_approach
from optillm.plansearch import plansearch
from optillm.leap import leap
from optillm.reread import re2_approach

# Check for API key
API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set. Please set it before running the application.")

def compare_responses(message, model1, approach1, model2, approach2, system_message, max_tokens, temperature, top_p):
    response1 = respond(message, [], model1, approach1, system_message, max_tokens, temperature, top_p)
    response2 = respond(message, [], model2, approach2, system_message, max_tokens, temperature, top_p)
    return response1, response2

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

def respond(message, history, model, approach, system_message, max_tokens, temperature, top_p):
    try:
        client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")
        messages = [{"role": "system", "content": system_message}]
        
        for val in history:
            if val[0]:
                messages.append({"role": "user", "content": val[0]})
            if val[1]:
                messages.append({"role": "assistant", "content": val[1]})
        
        messages.append({"role": "user", "content": message})
        
        if approach == "none":
            response = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://github.com/codelion/optillm",
                    "X-Title": "optillm"
                },
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return response.choices[0].message.content
        else:
            system_prompt, initial_query = parse_conversation(messages)
            
            if approach == 'rto':
                final_response, _ = round_trip_optimization(system_prompt, initial_query, client, model)
            elif approach == 'z3':
                z3_solver = Z3SymPySolverSystem(system_prompt, client, model)
                final_response, _ = z3_solver.process_query(initial_query)
            elif approach == "self_consistency":
                final_response, _ = advanced_self_consistency_approach(system_prompt, initial_query, client, model)
            elif approach == "cot_reflection":
                final_response, _ = cot_reflection(system_prompt, initial_query, client, model)
            elif approach == 'plansearch':
                response, _ = plansearch(system_prompt, initial_query, client, model)
                final_response = response[0]
            elif approach == 'leap':
                final_response, _ = leap(system_prompt, initial_query, client, model)
            elif approach == 're2':
                final_response, _ = re2_approach(system_prompt, initial_query, client, model)
            
            return final_response
            
    except Exception as e:
        error_message = f"Error in respond function: {str(e)}\nType: {type(e).__name__}"
        print(error_message)
        return f"An error occurred: {str(e)}"

def create_model_dropdown():
    return gr.Dropdown(
        [ "meta-llama/llama-3.1-8b-instruct:free", "nousresearch/hermes-3-llama-3.1-405b:free","meta-llama/llama-3.2-1b-instruct:free",
         "mistralai/mistral-7b-instruct:free","mistralai/pixtral-12b:free","meta-llama/llama-3.1-70b-instruct:free",
         "qwen/qwen-2-7b-instruct:free", "qwen/qwen-2-vl-7b-instruct:free", "google/gemma-2-9b-it:free", "liquid/lfm-40b:free", "meta-llama/llama-3.1-405b-instruct:free",
         "openchat/openchat-7b:free", "meta-llama/llama-3.2-90b-vision-instruct:free", "meta-llama/llama-3.2-11b-vision-instruct:free",
         "meta-llama/llama-3-8b-instruct:free", "meta-llama/llama-3.2-3b-instruct:free", "microsoft/phi-3-medium-128k-instruct:free",
         "microsoft/phi-3-mini-128k-instruct:free", "huggingfaceh4/zephyr-7b-beta:free"],
        value="meta-llama/llama-3.2-1b-instruct:free", label="Model"
    )

def create_approach_dropdown():
    return gr.Dropdown(
        ["none", "leap", "plansearch", "cot_reflection", "rto", "self_consistency", "z3", "re2"],
        value="none", label="Approach"
    )

html = """<iframe src="https://ghbtns.com/github-btn.html?user=codelion&repo=optillm&type=star&count=true&size=large" frameborder="0" scrolling="0" width="170" height="30" title="GitHub"></iframe>
"""

with gr.Blocks() as demo:
    gr.Markdown("# optillm - Optimizing LLM Inference")
    gr.HTML(html)
    
    with gr.Row():
        system_message = gr.Textbox(value="", label="System message")
        max_tokens = gr.Slider(minimum=1, maximum=4096, value=1024, step=1, label="Max new tokens")
        temperature = gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature")
        top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)")
    
    with gr.Tabs():
        with gr.TabItem("Chat"):
            model = create_model_dropdown()
            approach = create_approach_dropdown()
            chatbot = gr.Chatbot(type='messages')  # Updated to use messages type
            msg = gr.Textbox(label="Message", placeholder="Type your message here...")
            with gr.Row():
                submit = gr.Button("Submit")
                clear = gr.Button("Clear")

            def user(user_message, history):
                return "", history + [[user_message, None]]

            def bot(history, model, approach, system_message, max_tokens, temperature, top_p):
                user_message = history[-1][0]
                bot_message = respond(user_message, history[:-1], model, approach, system_message, max_tokens, temperature, top_p)
                history[-1][1] = bot_message
                return history

            msg.submit(user, [msg, chatbot], [msg, chatbot]).then(
                bot, [chatbot, model, approach, system_message, max_tokens, temperature, top_p], chatbot
            )
            submit.click(user, [msg, chatbot], [msg, chatbot]).then(
                bot, [chatbot, model, approach, system_message, max_tokens, temperature, top_p], chatbot
            )
            clear.click(lambda: None, None, chatbot, queue=False)

        with gr.TabItem("Compare"):
            with gr.Row():
                model1 = create_model_dropdown()
                approach1 = create_approach_dropdown()
                model2 = create_model_dropdown()
                approach2 = create_approach_dropdown()
            
            compare_input = gr.Textbox(label="Enter your message for comparison", placeholder="Type your message here...")
            compare_button = gr.Button("Compare")
            
            with gr.Row():
                output1 = gr.Textbox(label="Response 1")
                output2 = gr.Textbox(label="Response 2")
            
            compare_button.click(
                compare_responses,
                inputs=[compare_input, model1, approach1, model2, approach2, system_message, max_tokens, temperature, top_p],
                outputs=[output1, output2]
            )

if __name__ == "__main__":
    demo.launch(share=False) 