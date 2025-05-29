#!/usr/bin/env python3
"""
Push MCP Agent conversation traces to Hugging Face Hub as a dataset.

Usage:
  uv run push-to-hub.py --repo-id="owner/dataset-name" [--trace-dir="traces"]

Before running this script, make sure to log in to Hugging Face:
  huggingface-cli login
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

from datasets import Dataset
from huggingface_hub import HfApi, login
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn

console = Console()

def load_traces(trace_dir: str) -> List[Dict[str, Any]]:
    """Load all trace files from the specified directory."""
    trace_path = Path(trace_dir)
    if not trace_path.exists() or not trace_path.is_dir():
        raise ValueError(f"Trace directory '{trace_dir}' does not exist or is not a directory")
    
    traces = []
    trace_files = list(trace_path.glob("*.json"))
    
    if not trace_files:
        console.print(f"[yellow]No trace files found in '{trace_dir}'[/yellow]")
        return []
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"Loading {len(trace_files)} trace files...", total=len(trace_files))
        
        for trace_file in trace_files:
            try:
                with open(trace_file, "r") as f:
                    trace_data = json.load(f)
                    traces.append({
                        "filename": trace_file.name,
                        "trace": trace_data
                    })
                progress.update(task, advance=1)
            except Exception as e:
                console.print(f"[red]Error loading {trace_file}: {e}[/red]")
                progress.update(task, advance=1)
    
    return traces

def prepare_dataset(traces: List[Dict[str, Any]]) -> Dataset:
    """Convert traces to a Hugging Face dataset."""
    if not traces:
        raise ValueError("No valid traces to convert to dataset")
    
    # Extract relevant data from traces
    dataset_rows = []
    
    for trace_item in traces:
        trace = trace_item["trace"]
        
        # Extract conversation and tools
        messages = trace.get("messages", [])
        tools = trace.get("tools", [])
        
        if not messages:
            console.print(f"[yellow]Skipping trace {trace_item['filename']} - no messages found[/yellow]")
            continue
        
        # Format conversation for the dataset
        formatted_messages = []
        for msg in messages:
            # Create a clean message object with only the fields needed for the chat template
            formatted_msg = {"role": msg.get("role")}
            
            # Add content if present
            if "content" in msg and msg["content"] is not None:
                formatted_msg["content"] = msg["content"]
            
            # Add reasoning_content if present
            if "reasoning_content" in msg and msg["reasoning_content"]:
                formatted_msg["reasoning_content"] = msg["reasoning_content"]
            
            # Add tool_calls if present
            if "tool_calls" in msg and msg["tool_calls"]:
                formatted_msg["tool_calls"] = msg["tool_calls"]
            
            # Add tool response fields if present
            if msg.get("role") == "tool":
                if "tool_call_id" in msg:
                    formatted_msg["tool_call_id"] = msg["tool_call_id"]
                if "name" in msg:
                    formatted_msg["name"] = msg["name"]
            
            formatted_messages.append(formatted_msg)
        
        # Create dataset row
        dataset_rows.append({
            "id": trace_item["filename"],
            "timestamp": trace.get("timestamp", ""),
            "model": trace.get("model", ""),
            "messages": formatted_messages,
            "tools": tools  # Include tools in OpenAI format
        })
    
    return Dataset.from_list(dataset_rows)

def push_to_hub(dataset: Dataset, repo_id: str):
    """Push the dataset to Hugging Face Hub."""
    try:
        # Check if user is logged in
        api = HfApi()
        try:
            user_info = api.whoami()
            console.print(f"[green]Logged in as: {user_info['name']}[/green]")
        except Exception:
            console.print("[red]Not logged in to Hugging Face. Please run 'huggingface-cli login' first.[/red]")
            return False
        
        # Push to hub
        console.print(f"[bold blue]Pushing dataset to {repo_id}...[/bold blue]")
        dataset.push_to_hub(repo_id, private=True)
        console.print(f"[bold green]Successfully pushed dataset to {repo_id}![/bold green]")
        console.print(f"[bold]View your dataset at: https://huggingface.co/datasets/{repo_id}[/bold]")
        return True
    
    except Exception as e:
        console.print(f"[bold red]Error pushing to Hugging Face Hub: {e}[/bold red]")
        return False

def main():
    parser = argparse.ArgumentParser(description="Push MCP Agent traces to Hugging Face Hub")
    parser.add_argument("--repo-id", required=True, help="Hugging Face Hub repository ID (e.g., 'username/dataset-name')")
    parser.add_argument("--trace-dir", default="traces", help="Directory containing trace files (default: 'traces')")
    
    args = parser.parse_args()
    
    console.print("[bold magenta]MCP Agent Trace Uploader[/bold magenta]")
    console.print("Pushing traces to Hugging Face Hub\n")
    
    try:
        # Load traces
        console.print(f"[bold blue]Loading traces from {args.trace_dir}...[/bold blue]")
        traces = load_traces(args.trace_dir)
        
        if not traces:
            console.print("[yellow]No valid traces found. Exiting.[/yellow]")
            return
        
        console.print(f"[green]Loaded {len(traces)} trace files[/green]")
        
        # Prepare dataset
        console.print("[bold blue]Preparing dataset...[/bold blue]")
        dataset = prepare_dataset(traces)
        console.print(f"[green]Created dataset with {len(dataset)} examples[/green]")
        
        # Push to hub
        push_to_hub(dataset, args.repo_id)
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")

if __name__ == "__main__":
    main()
