#!/usr/bin/env python3
"""
A simple Qwen-Agent implementation with detailed logging for debugging purposes.
This agent connects to a specified endpoint and executes basic tasks with verbose output.
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Any, Optional

from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
# Simple print function without typewriter effect to avoid repetition
def simple_print(text):
    """Print text without typewriter effect"""
    print(text)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("test_agent")

# Suppress verbose logging from other modules
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

# ────────────────────────────── CLI ────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(description="Simple Qwen-Agent for debugging")
    ap.add_argument("--endpoint", default="https://109h3fti0xr9vz-8000.proxy.runpod.net/v1",
                    help="API endpoint URL")
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-FP8",
                    help="Model name")
    ap.add_argument("--verbose", action="store_true",
                    help="Enable verbose logging")
    ap.add_argument("--api_key", default="EMPTY",
                    help="API key (default: EMPTY)")
    ap.add_argument("--task", default="Use your debug_logger tool to log a message, then use code_interpreter to calculate the first 10 Fibonacci numbers",
                    help="Task to perform")
    ap.add_argument("--interactive", action="store_true",
                    help="Run in interactive mode")
    return ap.parse_args()

# ────────────────────────── Custom Tool ────────────────────────────────────
@register_tool('debug_logger')
class DebugLogger(BaseTool):
    """A simple tool that logs messages for debugging purposes."""
    description = 'Log debug information for analysis. Useful for tracking agent thought process.'
    parameters = [{
        'name': 'message',
        'type': 'string',
        'description': 'The debug message to log',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        try:
            parsed_params = json.loads(params)
            message = parsed_params.get('message', 'No message provided')
            logger.info(f"AGENT DEBUG: {message}")
            return json.dumps({"status": "success", "message": f"Logged: {message}"})
        except Exception as e:
            logger.error(f"Error in debug_logger: {e}")
            return json.dumps({"status": "error", "message": str(e)})

# ────────────────────────── LLM Configuration ────────────────────────────────
def configure_llm(args) -> Dict[str, Any]:
    """Configure the LLM with the given arguments."""
    logger.info(f"Configuring LLM with model={args.model}, endpoint={args.endpoint}")
    
    return {
        'model': args.model,
        'model_server': args.endpoint,
        'api_key': args.api_key,
        'generate_cfg': {
            'thought_in_content': False,  # Separate reasoning from content
            'max_input_tokens': 8000,     # Adjust based on model capabilities
            'temperature': 0.7,
            'top_p': 0.9,
            'extra_body': {"chat_template_kwargs": {"enable_thinking": True}}
        }
    }

# ────────────────────────── Agent Runner ────────────────────────────────────
def run_agent(args):
    """Initialize and run the agent with the given configuration."""
    llm_cfg = configure_llm(args)
    
    # Log the configuration (excluding sensitive data)
    safe_cfg = llm_cfg.copy()
    safe_cfg['api_key'] = '***' if safe_cfg['api_key'] != 'EMPTY' else 'EMPTY'
    logger.info(f"Agent configuration: {json.dumps(safe_cfg, indent=2)}")
    
    # Define system instruction
    system_instruction = """You are a helpful assistant designed to demonstrate the capabilities of the Qwen-Agent framework.
Your responses should be clear, concise, and informative. 
When using tools, explain what you're doing and why.
Use the debug_logger tool to record your thought process for later analysis."""
    
    # Initialize tools
    tools = ['debug_logger', 'code_interpreter']
    
    # Create the agent
    logger.info("Initializing agent...")
    bot = Assistant(
        llm=llm_cfg,
        system_message=system_instruction,
        function_list=tools
    )
    logger.info("Agent initialized successfully")
    
    # Create initial message
    messages = [{"role": "user", "content": args.task}]
    logger.info(f"Starting task: {args.task}")
    
    # Run the agent
    logger.info("Running agent...")
    try:
        # Reset tracking sets before each run
        global seen_responses, seen_tool_calls
        seen_responses = set()
        seen_tool_calls = set()
        
        # Non-streaming mode for detailed logging
        responses = bot.run(messages=messages, stream=False)
        
        # Flatten and deduplicate responses
        flattened_responses = []
        for response in responses:
            if isinstance(response, list):
                flattened_responses.extend(response)
            else:
                flattened_responses.append(response)
        
        # Process and log unique responses
        for r in flattened_responses:
            log_response(r)
        
        # Add responses to messages for potential follow-up
        messages.extend(flattened_responses)
        logger.info("Agent execution completed successfully")
        
        # Print final conversation
        print("\n===== FINAL CONVERSATION =====")
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "").strip()
            
            if role == "user":
                print(f"\n🧑 USER: {content}")
            elif role == "assistant":
                print(f"\n🤖 ASSISTANT: {content}")
                
                # Print tool calls if any
                for tc in msg.get("tool_calls", []):
                    print(f"\n   🔧 TOOL CALL: {json.dumps(tc, indent=2)}")
            elif role == "tool":
                print(f"\n🛠️  TOOL RESPONSE: {content[:100]}{'...' if len(content) > 100 else ''}")
        
        return messages
        
    except Exception as e:
        logger.error(f"Error running agent: {e}", exc_info=args.verbose)
        return None

# Track already seen responses to avoid duplication
seen_responses = set()
seen_tool_calls = set()

def log_response(response):
    """Log a single response from the agent with deduplication."""
    role = response.get("role", "unknown")
    content = response.get("content", "").strip()
    response_id = response.get("id", "")
    
    # Create a unique identifier for this response
    content_hash = hash(f"{role}:{content[:100]}:{response_id}")
    
    # Skip if we've seen this response before
    if content_hash in seen_responses:
        return
    
    seen_responses.add(content_hash)
    
    if role == "assistant":
        logger.info(f"Assistant response: {content[:100]}{'...' if len(content) > 100 else ''}")
        
        # Log tool calls with deduplication
        for tc in response.get("tool_calls", []):
            function_name = tc.get("function", {}).get("name", "unknown")
            function_args = tc.get("function", {}).get("arguments", "{}")
            tc_id = tc.get("id", "")
            
            # Create a unique identifier for this tool call
            tc_hash = hash(f"{function_name}:{function_args[:50]}:{tc_id}")
            
            if tc_hash not in seen_tool_calls:
                seen_tool_calls.add(tc_hash)
                logger.info(f"Tool call: {function_name} with args: {function_args[:100]}{'...' if len(function_args) > 100 else ''}")
    
    elif role == "tool":
        logger.info(f"Tool response: {content[:100]}{'...' if len(content) > 100 else ''}")
    else:
        logger.info(f"Response ({role}): {content[:100]}{'...' if len(content) > 100 else ''}")

# ────────────────────────── Interactive Mode ────────────────────────────────
def interactive_mode(args):
    """Run the agent in interactive mode, allowing for conversation."""
    llm_cfg = configure_llm(args)
    
    # Define system instruction
    system_instruction = """You are a helpful assistant designed to demonstrate the capabilities of the Qwen-Agent framework.
Your responses should be clear, concise, and informative.
When using tools, explain what you're doing and why.
Use the debug_logger tool to record your thought process for later analysis."""
    
    # Initialize tools
    tools = ['debug_logger', 'code_interpreter']
    
    # Create the agent
    logger.info("Initializing agent for interactive mode...")
    bot = Assistant(
        llm=llm_cfg,
        system_message=system_instruction,
        function_list=tools
    )
    logger.info("Agent initialized successfully")
    
    # Initialize conversation history
    messages = []
    
    print("\n===== INTERACTIVE MODE =====")
    print("Type 'exit' or 'quit' to end the conversation")
    
    while True:
        # Get user input
        user_input = input("\n🧑 USER: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting interactive mode...")
            break
        
        # Add user message to history
        messages.append({"role": "user", "content": user_input})
        
        # Reset tracking sets before each run
        global seen_responses, seen_tool_calls
        seen_responses = set()
        seen_tool_calls = set()
        
        # Run the agent with streaming for better user experience
        print("\n🤖 ASSISTANT: ", end="")
        
        try:
            # Get all responses first (non-streaming)
            all_responses = bot.run(messages=messages, stream=False)
            
            # Flatten responses
            flattened_responses = []
            for response in all_responses:
                if isinstance(response, list):
                    flattened_responses.extend(response)
                else:
                    flattened_responses.append(response)
            
            # Extract assistant messages for display
            assistant_content = ""
            for r in flattened_responses:
                if r.get("role") == "assistant" and r.get("content"):
                    assistant_content = r.get("content")
                    break
            
            # Display content without typewriter effect to avoid repetition
            simple_print(assistant_content)
            
            # Log all responses for debugging
            for r in flattened_responses:
                log_response(r)
            
            # Add responses to messages for conversation history
            messages.extend(flattened_responses)
            
            # Print tool calls if any
            for r in flattened_responses:
                if r.get("role") == "assistant" and r.get("tool_calls"):
                    for tc in r.get("tool_calls"):
                        function_name = tc.get("function", {}).get("name", "unknown")
                        print(f"\n   🔧 Used tool: {function_name}")
        
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}", exc_info=args.verbose)
            print(f"\nError: {e}")

# ────────────────────────── Main Function ────────────────────────────────────
def main():
    args = parse_args()
    
    # Set logging level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Print banner
    print("\n" + "="*80)
    print(f"🤖 TEST AGENT - Qwen-Agent Framework")
    print(f"Model: {args.model}")
    print(f"Endpoint: {args.endpoint}")
    print("="*80 + "\n")
    
    # Check if interactive mode flag is set
    if args.interactive:
        print(f"Running in interactive mode. Type 'exit' to quit.")
        interactive_mode(args)
    else:
        # Run in single task mode by default
        print(f"Running in single task mode with task: '{args.task}'")
        run_agent(args)

if __name__ == "__main__":
    main()
