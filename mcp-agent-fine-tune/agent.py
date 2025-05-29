#!/usr/bin/env python3
"""
Updated MCP agent script
------------------------------------------------
* Uses the official `openai` client.
* Converts MCP tool schemas → OpenAI function-calling schemas.
* Requires:   uv add openai click rich python-dotenv pydantic OR uv sync
"""

from __future__ import annotations

import uuid
import json
import os
import subprocess
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from rich.console import Console
from rich.prompt import Confirm

console = Console()
load_dotenv()                       # .env support, e.g. for OpenAI api key.

# -----------------------------------------------------------------------------#
#  Models                                                                      #
# -----------------------------------------------------------------------------#
class MCPToolCall(BaseModel):
    """Represents a single MCP tool call ready to send via JSON-RPC"""
    name: str
    arguments: Dict[str, Any]

# -----------------------------------------------------------------------------#
#  Agent                                                                       #
# -----------------------------------------------------------------------------#
class MCPAgent:
    """
    Bridges:
      * MCP JSON-RPC tool servers
      * OpenAI Chat Completions endpoint (function calling)

    Major responsibilities
      1. Discover tools from each MCP server listed in a config.json
      2. Translate MCP tool schemas → OpenAI function-calling schema
      3. Wrap/unwrap tool call arguments & results
      4. Keep a running conversation history
    """

    def __init__(
        self,
        *,
        config_path: str = "config.json",
        model: str = "gpt-4.1-mini",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        show_reasoning: bool = True,
        trace_dir: str = "traces",
    ):
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"),
                             base_url=base_url) if base_url else OpenAI(
                                 api_key=api_key or os.getenv("OPENAI_API_KEY"))
        if not self.client.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        self.config = self._load_config(config_path)
        self.show_reasoning = show_reasoning
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(exist_ok=True)

        self.conversation_history: List[Dict[str, Any]] = []
        self.tools: List[dict] = []          # MCP format
        self.oa_tools: List[dict] = []       # OpenAI format

        # One running process per MCP server
        self.mcp_processes: Dict[str, Tuple[subprocess.Popen, Dict[str, str]]] = {}

    # ---------------------------------------------------------------------#
    #  Utility loaders                                                     #
    # ---------------------------------------------------------------------#
    @staticmethod
    def _load_config(path: str) -> dict:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as exc:
            console.print(f"[red]Could not read config {path}: {exc}[/red]")
            return {"mcpServers": {}}

    # ---------------------------------------------------------------------#
    #  MCP server management                                               #
    # ---------------------------------------------------------------------#
    def _start_mcp_server(self, name: str) -> Tuple[subprocess.Popen, Dict[str, str]]:
        if name in self.mcp_processes:
            return self.mcp_processes[name]

        server_cfg = self.config["mcpServers"].get(name)
        if not server_cfg:
            raise ValueError(f"Unknown MCP server '{name}' (check config.json)")

        env = os.environ.copy()
        env.update(server_cfg.get("env", {}))

        proc = subprocess.Popen(
            [server_cfg["command"], *server_cfg["args"]],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        self.mcp_processes[name] = (proc, env)
        console.print(f"[green]Started MCP server:[/green] {name}")
        return proc, env

    def _list_mcp_tools(self, proc, server_name: str) -> List[dict]:
        """Call tools/list via JSON-RPC 2.0 and return raw tool objects."""
        req = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()

        line = proc.stdout.readline()
        if not line:
            return []

        rsp = json.loads(line)
        if "error" in rsp:
            console.print(f"[red]tools/list error from {server_name}: {rsp['error']}[/red]")
            return []

        tools = rsp.get("result", {}).get("tools", [])
        for t in tools:
            t["server"] = server_name  # annotate for later lookup
        return tools

    # ---------------------------------------------------------------------#
    #  Schema conversion                                                   #
    # ---------------------------------------------------------------------#
    def _normalize_root(self, schema: dict) -> dict:
        """Guarantee the root is an object so OpenAI is happy."""
        if schema.get("type") == "object":
            return schema

        # wrap non-object schemas
        return {
            "type": "object",
            "properties": {"value": schema},
            "required": ["value"],
        }

    def _strip_unsupported_formats(self, node: dict):
        """Remove JSON-Schema 'format' fields OpenAI doesn’t recognise."""
        if not isinstance(node, dict):
            return
        if node.get("type") == "string" and node.get("format") not in {"date-time", "enum"}:
            node.pop("format", None)
        if node.get("type") == "object":
            for prop in node.get("properties", {}).values():
                self._strip_unsupported_formats(prop)
        if node.get("items"):
            self._strip_unsupported_formats(node["items"])

    def _mcp_to_openai_tools(self, mcp_tools):
        oa = []
        for tool in mcp_tools:
            schema = self._normalize_root(tool.get("inputSchema", {"type": "object", "properties": {}}))
            self._strip_unsupported_formats(schema)
            oa.append({
                "type": "function",
                "function": {
                    "name": tool["name"][:64],            # OpenAI 64-char limit
                    "description": tool.get("description", "")[:1024],
                    "parameters": schema,
                },
            })
        return oa

    # ---------------------------------------------------------------------#
    #  Tool execution helpers                                              #
    # ---------------------------------------------------------------------#
    def _convert_oa_toolcall_to_mcp(self, call) -> MCPToolCall:
        raw_args = call.function.arguments
        arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        return MCPToolCall(name=call.function.name, arguments=arguments)

    def _execute_mcp_tool(self, tool: MCPToolCall) -> Dict[str, Any]:
        original = next((t for t in self.tools if t["name"] == tool.name), None)
        if original is None:
            return {"content": [{"type": "text", "text": f"Tool {tool.name} not found"}], "isError": True}

        proc, _ = self._start_mcp_server(original["server"])

        req = {
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex,                 # ① unique per request
            "method": "tools/call",
            "params": {"name": tool.name, "arguments": tool.arguments},
        }
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()

        line = proc.stdout.readline()
        if not line:
            return {"content": [{"type": "text", "text": "No response"}], "isError": True}

        rsp = json.loads(line)
        if "error" in rsp:
            return {"content": [{"type": "text", "text": str(rsp['error'])}], "isError": True}

        return rsp.get("result", {})

    @staticmethod
    def _wrap_tool_result(tool_call_id: str, result: dict) -> dict:
        """MCP result → OpenAI tool-role message (role, tool_call_id, content)."""
        if isinstance(result, dict):
            parts = result.get("content", [])
            text = " ".join(p.get("text", "") for p in parts if p.get("type") == "text")
        else:
            text = str(result)

        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": text or "(empty tool result)",
        }

    # ---------------------------------------------------------------------#
    #  Discovery                                                           #
    # ---------------------------------------------------------------------#
    def _discover_all_tools(self) -> None:
        for server_name in self.config.get("mcpServers", {}):
            try:
                proc, _ = self._start_mcp_server(server_name)
                server_tools = self._list_mcp_tools(proc, server_name)
                self.tools.extend(server_tools)
                console.print(f"[green]Discovered {len(server_tools)} tools from {server_name}.[/green]")
            except Exception as exc:
                console.print(f"[yellow]Could not list tools from {server_name}: {exc}[/yellow]")

        self.oa_tools = self._mcp_to_openai_tools(self.tools)
        console.print(f"[bold blue]Total available tools:[/bold blue] {len(self.oa_tools)}")

    # ---------------------------------------------------------------------#
    #  Chat loop                                                           #
    # ---------------------------------------------------------------------#
    def chat(self) -> None:
        console.print("[bold magenta]MCP Agent (OpenAI edition)[/bold magenta]")
        console.print("Type 'exit' to quit.\n")

        self._discover_all_tools()
        if not self.oa_tools:
            console.print("[yellow]No tools found – continuing with plain chat.[/yellow]")

        while True:
            user_msg = click.prompt("You")
            if user_msg.lower() in {"exit", "quit"}:
                # Save the conversation trace before exiting
                self._save_conversation_trace()
                break

            self.conversation_history.append({"role": "user", "content": user_msg})

            # -- 1st assistant response ----------------------------------#
            asst_msg = self._chat_once()
            
            # Create a complete assistant message for the conversation history
            asst_history_msg = {"role": "assistant"}
            
            # Add content if present
            if asst_msg.content:
                console.print(f"\n[bold green]Assistant:[/bold green] {asst_msg.content}\n")
                asst_history_msg["content"] = asst_msg.content
            
            # Add reasoning_content if present
            if hasattr(asst_msg, "reasoning_content") and asst_msg.reasoning_content:
                asst_history_msg["reasoning_content"] = asst_msg.reasoning_content
            
            # Add tool_calls if present
            tool_calls = getattr(asst_msg, "tool_calls", None)
            serialized_tool_calls = None
            if tool_calls:
                serialized_tool_calls = [self._convert_tool_call_to_dict(tc) for tc in tool_calls]
                asst_history_msg["tool_calls"] = serialized_tool_calls
            
            # Add the complete message to history
            self.conversation_history.append(asst_history_msg)

            # -- Handle function calls -----------------------------------#
            tool_calls = getattr(asst_msg, "tool_calls", None)
            while tool_calls:
                for tc in tool_calls:
                    mcp_call = self._convert_oa_toolcall_to_mcp(tc)

                    if not self._ask_permission(mcp_call):
                        # user rejected
                        self.conversation_history.append(
                            {"role": "tool", "tool_call_id": tc.id, "content": "User rejected tool call."}
                        )
                        continue

                    mcp_result = self._execute_mcp_tool(mcp_call)
                    wrapped = self._wrap_tool_result(tc.id, mcp_result)
                    self.conversation_history.append(wrapped)

                # -- follow-up after tool execution -------------------#
                follow = self._chat_once()
                
                # Create a complete assistant message for the conversation history
                follow_history_msg = {"role": "assistant"}
                
                # Add content if present
                if follow.content:
                    console.print(f"\n[bold green]Assistant:[/bold green] {follow.content}\n")
                    follow_history_msg["content"] = follow.content
                
                # Add reasoning_content if present
                if hasattr(follow, "reasoning_content") and follow.reasoning_content:
                    follow_history_msg["reasoning_content"] = follow.reasoning_content
                
                # Add tool_calls if present
                new_tool_calls = getattr(follow, "tool_calls", None)
                if new_tool_calls:
                    # Convert only the new tool calls
                    serialized_tool_calls = [self._convert_tool_call_to_dict(tc) for tc in new_tool_calls]
                    follow_history_msg["tool_calls"] = serialized_tool_calls
                
                # Add the complete message to history
                self.conversation_history.append(follow_history_msg)
                
                # Update tool_calls for the next iteration
                tool_calls = new_tool_calls

    def _prepare_messages_for_api(self, messages):
        """Prepare conversation history for API call."""
        api_messages = []
        
        for msg in messages:
            api_msg = {"role": msg["role"]}
            
            # Include content if present
            if "content" in msg and msg["content"] is not None:
                api_msg["content"] = msg["content"]
            elif "content" not in msg:
                api_msg["content"] = None
                
            # Handle tool calls - convert arguments back to JSON strings
            if "tool_calls" in msg and msg["tool_calls"]:
                api_tool_calls = []
                for tc in msg["tool_calls"]:
                    api_tc = {"id": tc["id"], "type": tc["type"]}
                    if "function" in tc:
                        func = {"name": tc["function"]["name"]}
                        # Ensure arguments is a JSON string
                        args = tc["function"]["arguments"]
                        if not isinstance(args, str):
                            func["arguments"] = json.dumps(args)
                        else:
                            func["arguments"] = args
                        api_tc["function"] = func
                    api_tool_calls.append(api_tc)
                api_msg["tool_calls"] = api_tool_calls
            
            # Handle tool responses
            if msg["role"] == "tool":
                if "tool_call_id" in msg:
                    api_msg["tool_call_id"] = msg["tool_call_id"]
                if "name" in msg:
                    api_msg["name"] = msg["name"]
            
            api_messages.append(api_msg)
            
        return api_messages
    
    def _chat_once(self):
        """Single call to OpenAI Chat Completion."""
        # Prepare messages for API call
        api_messages = self._prepare_messages_for_api(self.conversation_history)
        
        rsp = self.client.chat.completions.create(
            model=self.model,
            messages=api_messages,
            tools=self.oa_tools if self.oa_tools else None,
            tool_choice="auto",
        )
        message = rsp.choices[0].message
        
        # Extract reasoning content if available and show_reasoning is enabled
        if self.show_reasoning and hasattr(message, "reasoning_content") and message.reasoning_content:
            console.print(f"\n[bold yellow]Reasoning:[/bold yellow]\n{message.reasoning_content}\n")
            
        return message

    # ---------------------------------------------------------------------#
    #  Conversion Helpers                                                  #
    # ---------------------------------------------------------------------#
    def _convert_tool_call_to_dict(self, tool_call) -> dict:
        """Convert an OpenAI tool call object to a dictionary for storage."""
        # For trace logging, we want to store the complete data including parsed arguments
        result = {
            "id": tool_call.id,
            "type": "function"  # Required by OpenAI API
        }
        
        if hasattr(tool_call, "function"):
            function_data = {
                "name": tool_call.function.name,
            }
            
            # Handle arguments which might be a string or already parsed
            if hasattr(tool_call.function, "arguments"):
                args = tool_call.function.arguments
                if isinstance(args, str):
                    try:
                        # Store the parsed arguments for trace logging
                        function_data["arguments"] = json.loads(args)
                    except json.JSONDecodeError:
                        function_data["arguments"] = args
                else:
                    function_data["arguments"] = args
                    
            result["function"] = function_data
            
        return result
    
    # ---------------------------------------------------------------------#
    #  Logging & Tracing                                                   #
    # ---------------------------------------------------------------------#
    def _save_conversation_trace(self):
        """Save the current conversation history and tools to a trace file."""
        if not self.conversation_history:
            return
        
        # Get the first user message to use in the filename
        first_user_msg = ""
        for msg in self.conversation_history:
            if msg.get("role") == "user":
                first_user_msg = msg.get("content", "")
                break
        
        if not first_user_msg:
            return
            
        # Create a filename using the first 30 chars of the first user message and timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_msg = "".join(c if c.isalnum() else "_" for c in first_user_msg[:30]).strip("_")
        filename = f"{safe_msg}_{timestamp}.json"
        
        # Prepare the trace data
        trace_data = {
            "timestamp": timestamp,
            "model": self.model,
            "messages": self.conversation_history,
            "tools": self.oa_tools
        }
        
        # Save the trace to a file
        trace_path = self.trace_dir / filename
        with open(trace_path, "w") as f:
            json.dump(trace_data, f, indent=2)
            
        console.print(f"\n[bold blue]Conversation trace saved to:[/bold blue] {trace_path}")
    
    # ---------------------------------------------------------------------#
    #  UX helpers                                                          #
    # ---------------------------------------------------------------------#
    @staticmethod
    def _ask_permission(tool_call: MCPToolCall) -> bool:
        console.print(f"\n[yellow]Tool call requested:[/yellow] [cyan]{tool_call.name}[/cyan]")
        for k, v in tool_call.arguments.items():
            console.print(f"  • [green]{k}[/green]: {v}")
        return Confirm.ask("Run this tool?")


# -----------------------------------------------------------------------------#
#  CLI                                                                         #
# -----------------------------------------------------------------------------#
@click.command()
@click.option("--config", "-c", default="config.json", help="Path to MCP config file.")
@click.option("--model", "-m", default="gpt-4o", help="OpenAI chat model name.")
@click.option("--base-url", help="Custom OpenAI-compatible endpoint.")
@click.option("--api-key", default="EMPTY", help="Override OPENAI_API_KEY.")
@click.option("--show-reasoning", is_flag=True, help="Display model reasoning content when available")
@click.option("--trace-dir", default="traces", help="Directory to save conversation traces")
def main(config, model, base_url, api_key, show_reasoning, trace_dir):
    """Interactive agent bridging MCP tool servers with OpenAI function calling."""
    agent = MCPAgent(
        config_path=config,
        model=model,
        base_url=base_url,
        api_key=api_key,
        show_reasoning=show_reasoning,
        trace_dir=trace_dir,
    )
    try:
        agent.chat()
    finally:
        # ensure child processes die
        for proc, _ in agent.mcp_processes.values():
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                proc.kill()


if __name__ == "__main__":
    main()
