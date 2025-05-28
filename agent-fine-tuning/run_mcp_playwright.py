from qwen_agent.agents import Assistant

# Define LLM
llm_cfg = {
    'model': 'Qwen/Qwen3-30B-A3B-FP8',

    # Use the endpoint provided by Alibaba Model Studio:
    # 'model_type': 'qwen_dashscope',
    # 'api_key': os.getenv('DASHSCOPE_API_KEY'),

    # Use a custom endpoint compatible with OpenAI API:
    'model_server': 'https://109h3fti0xr9vz-8000.proxy.runpod.net/v1',  # api_base
    'api_key': 'EMPTY',

    # Other parameters:
    # 'generate_cfg': {
    #         # Add: When the response content is `<think>this is the thought</think>this is the answer;
    #         # Do not add: When the response has been separated by reasoning_content and content.
    #         'thought_in_content': True,
    #     },
}

# Define Tools
tools = [
    {'mcpServers': {  # You can specify the MCP configuration file
            "playwright": {"command": "npx", "args": ["@playwright/mcp@latest"]}
        }
    }
]

# Define Agent
bot = Assistant(llm=llm_cfg, function_list=tools)

# Function to print messages in a formatted way
def print_formatted_conversation(messages, response):
    print("\n" + "="*50 + "\n")
    # Print the original messages
    for message in messages:
        role = message.get('role', 'unknown')
        content = message.get('content', '')
        
        # Format based on role
        if role == 'user':
            print(f"🧑 USER:\n{content}\n")
        elif role == 'assistant':
            print(f"🤖 ASSISTANT:\n{content}\n")
        elif role == 'function':
            function_name = message.get('name', 'unknown_function')
            print(f"⚙️ FUNCTION ({function_name}):\n{content}\n")
        else:
            print(f"[{role.upper()}]:\n{content}\n")
    
    # Print the response
    print(f"🤖 ASSISTANT RESPONSE SEQUENCE:")
    if isinstance(response, list):
        # Process each item in the response list
        for i, item in enumerate(response):
            if not isinstance(item, dict):
                print(f"\n[ITEM {i+1}]:\n{item}\n")
                continue
                
            role = item.get('role', 'unknown')
            print(f"\n[{i+1}] {role.upper()}:")
            
            # Handle different content types
            if 'reasoning_content' in item and item['reasoning_content']:
                print(f"\n🧠 REASONING:\n{item['reasoning_content']}")
                
            if 'content' in item and item['content']:
                print(f"\n💬 CONTENT:\n{item['content']}")
                
            if 'function_call' in item and item['function_call']:
                fn_call = item['function_call']
                fn_name = fn_call.get('name', 'unknown')
                fn_args = fn_call.get('arguments', '{}')
                print(f"\n⚙️ FUNCTION CALL: {fn_name}\nARGUMENTS: {fn_args}")
                
            if role == 'function' and 'name' in item:
                print(f"\nFUNCTION NAME: {item['name']}")
    else:
        # For any other type
        print(f"{response}\n")
    
    print("\n" + "="*50)

# Streaming generation
messages = [{'role': 'user', 'content': 'https://qwenlm.github.io/blog/ Introduce the latest developments of Qwen'}]

# Collect all responses
for responses in bot.run(messages=messages):
    pass

# Raw response
print(f"Raw response: {responses}\n\n")

# Print the formatted conversation
print_formatted_conversation(messages, responses)