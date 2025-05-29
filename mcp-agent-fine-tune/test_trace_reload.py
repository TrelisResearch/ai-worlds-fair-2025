#!/usr/bin/env python3
"""
Test script to verify that tools and messages can be reloaded from traces.

This script:
1. Finds the most recent trace in the traces folder
2. Extracts all messages except the last assistant response
3. Passes those messages and tools to the API
4. Compares the new response with the original one

Usage:
  uv run test_trace_reload.py --base-url=https://0zslbmx98vpo2i-8000.proxy.runpod.net/v1 --model=Qwen/Qwen3-30B-A3B-FP8
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import openai
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

def find_latest_trace(trace_dir: str = "traces") -> Optional[Path]:
    """Find the most recent trace file in the traces directory."""
    trace_path = Path(trace_dir)
    if not trace_path.exists() or not trace_path.is_dir():
        console.print(f"[red]Error: Trace directory '{trace_dir}' does not exist[/red]")
        return None
    
    trace_files = list(trace_path.glob("*.json"))
    if not trace_files:
        console.print(f"[red]Error: No trace files found in '{trace_dir}'[/red]")
        return None
    
    # Sort by modification time, newest first
    latest_trace = max(trace_files, key=lambda p: p.stat().st_mtime)
    return latest_trace

def load_trace(trace_file: Path) -> Optional[Dict[str, Any]]:
    """Load a trace file and return its contents."""
    try:
        with open(trace_file, "r") as f:
            return json.load(f)
    except Exception as e:
        console.print(f"[red]Error loading trace file: {e}[/red]")
        return None

def prepare_messages_for_api(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Prepare messages for API call, removing the last assistant response.
    Returns the prepared messages and the removed last assistant message (if any).
    """
    # Find the index of the last assistant message
    last_assistant_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            last_assistant_idx = i
            break
    
    if last_assistant_idx is None:
        # No assistant message found
        return messages, None
    
    # Extract the last assistant message
    last_assistant_msg = messages[last_assistant_idx]
    
    # Remove the last assistant message and any subsequent messages
    prepared_messages = messages[:last_assistant_idx]
    
    return prepared_messages, last_assistant_msg

def call_api(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], model: str, base_url: str) -> Dict[str, Any]:
    """Make a call to the API with the given messages and tools."""
    client = OpenAI(api_key="dummy", base_url=base_url)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools if tools else None,
            tool_choice="auto",
        )
        return response.choices[0].message
    except Exception as e:
        console.print(f"[red]Error calling API: {e}[/red]")
        return {}

def compare_responses(original: Dict[str, Any], new: Dict[str, Any]) -> None:
    """Compare the original and new responses and print the results."""
    console.print("\n[bold blue]Comparison Results:[/bold blue]")
    
    # Check if both responses have content
    original_content = original.get("content")
    new_content = new.content if hasattr(new, "content") else None
    
    if original_content and new_content:
        content_match = original_content == new_content
        console.print(f"Content match: [{'green' if content_match else 'yellow'}]{content_match}[/{'green' if content_match else 'yellow'}]")
    
    # Check if both responses have reasoning_content
    original_reasoning = original.get("reasoning_content")
    new_reasoning = new.reasoning_content if hasattr(new, "reasoning_content") else None
    
    if original_reasoning and new_reasoning:
        # Reasoning content is likely to be different, so we just check if both exist
        console.print(f"Both responses have reasoning content: [green]True[/green]")
    
    # Check if both responses have tool_calls
    original_tool_calls = original.get("tool_calls", [])
    new_tool_calls = new.tool_calls if hasattr(new, "tool_calls") else []
    
    if original_tool_calls and new_tool_calls:
        # Compare number of tool calls
        console.print(f"Original tool calls: {len(original_tool_calls)}")
        console.print(f"New tool calls: {len(new_tool_calls)}")
        
        # Compare tool names
        original_tool_names = [tc.get("function", {}).get("name") for tc in original_tool_calls]
        new_tool_names = [tc.function.name for tc in new_tool_calls]
        
        tool_names_match = set(original_tool_names) == set(new_tool_names)
        console.print(f"Tool names match: [{'green' if tool_names_match else 'yellow'}]{tool_names_match}[/{'green' if tool_names_match else 'yellow'}]")
    
    # Display the responses
    console.print("\n[bold blue]Original Response:[/bold blue]")
    display_response(original)
    
    console.print("\n[bold blue]New Response:[/bold blue]")
    display_response(new)

def display_response(response: Dict[str, Any]) -> None:
    """Display a response in a readable format."""
    # Handle both dictionary and object formats
    if hasattr(response, "content"):
        content = response.content
        reasoning = response.reasoning_content if hasattr(response, "reasoning_content") else None
        tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
    else:
        content = response.get("content")
        reasoning = response.get("reasoning_content")
        tool_calls = response.get("tool_calls", [])
    
    if content:
        console.print(Panel(content, title="Content"))
    
    if reasoning:
        console.print(Panel(Text(reasoning[:500] + "..." if len(reasoning) > 500 else reasoning), 
                           title="Reasoning (truncated)"))
    
    if tool_calls:
        for i, tc in enumerate(tool_calls):
            # Handle both dictionary and object formats
            if hasattr(tc, "function"):
                name = tc.function.name
                args = tc.function.arguments
            else:
                name = tc.get("function", {}).get("name")
                args = tc.get("function", {}).get("arguments")
            
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except:
                    pass
            
            args_str = json.dumps(args, indent=2) if args else "{}"
            console.print(Panel(f"Name: {name}\nArguments: {args_str}", 
                               title=f"Tool Call {i+1}"))

def main():
    parser = argparse.ArgumentParser(description="Test trace reload capability")
    parser.add_argument("--trace-dir", default="traces", help="Directory containing trace files")
    parser.add_argument("--model", required=True, help="Model to use for the API call")
    parser.add_argument("--base-url", required=True, help="Base URL for the API")
    parser.add_argument("--trace-file", help="Specific trace file to use (optional)")
    
    args = parser.parse_args()
    
    console.print("[bold magenta]Trace Reload Test[/bold magenta]")
    
    # Find the latest trace file or use the specified one
    trace_file = Path(args.trace_file) if args.trace_file else find_latest_trace(args.trace_dir)
    if not trace_file:
        return
    
    console.print(f"[blue]Using trace file:[/blue] {trace_file}")
    
    # Load the trace
    trace = load_trace(trace_file)
    if not trace:
        return
    
    # Extract messages and tools
    messages = trace.get("messages", [])
    tools = trace.get("tools", [])
    
    if not messages:
        console.print("[red]Error: No messages found in trace[/red]")
        return
    
    # Prepare messages for API call
    prepared_messages, last_assistant_msg = prepare_messages_for_api(messages)
    
    if not last_assistant_msg:
        console.print("[yellow]Warning: No assistant message found to compare with[/yellow]")
    
    console.print(f"[blue]Prepared {len(prepared_messages)} messages for API call[/blue]")
    console.print(f"[blue]Using {len(tools)} tools from trace[/blue]")
    
    # Call the API
    console.print(f"\n[blue]Calling API at {args.base_url} with model {args.model}...[/blue]")
    new_response = call_api(prepared_messages, tools, args.model, args.base_url)
    
    if not new_response:
        return
    
    console.print("[green]Received response from API[/green]")
    
    # Compare responses
    if last_assistant_msg:
        compare_responses(last_assistant_msg, new_response)
    else:
        console.print("\n[bold blue]New Response:[/bold blue]")
        display_response(new_response)

if __name__ == "__main__":
    main()
