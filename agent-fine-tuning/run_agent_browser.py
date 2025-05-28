#!/usr/bin/env python3
"""
Run a Qwen-Agent equipped with a Playwright MCP server.

Features
--------
* Head-FUL browser window so you can watch each step.
* Pretty live stream of assistant + tool messages in the terminal.
*  --record : save <messages, tools, task-name> to ./traces/<ts>.json
*  --hints  : prompt for an extra system message (NOT persisted).
"""

import argparse, json, os, sys, time, uuid, pathlib, datetime as dt
from qwen_agent.agents import Assistant

# ---------- CLI -------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--start_url", required=True, help="Start URL for the browser")
ap.add_argument("--model",    default="Qwen/Qwen3-8B", help="HF / DashScope slug")
ap.add_argument("--task",     required=True, help="Human-readable task name")
ap.add_argument("--record",   action="store_true", help="Save trace to ./traces")
ap.add_argument("--hints",    action="store_true", help="Ask for extra system msg")
ap.add_argument("--endpoint", default="http://localhost:8000/v1",
                help="OpenAI-compatible endpoint that serves the model, use https://<POD-ID>-8000.proxy.runpod.net/v1 for Runpod")
args = ap.parse_args()

# ---------- LLM config ------------------------------------------------------
llm_cfg = dict(
    model=args.model,
    model_server=args.endpoint,   # vLLM / LMDeploy / DashScope etc.
    api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
    generate_cfg=dict(
        extra_body={"chat_template_kwargs": {"enable_thinking": True}}
    )
)

# ---------- Tool registry ---------------------------------------------------
# Playwright MCP – *head-ful* so we can watch the browser
playwright_mcp = {
    "playwright": {
        "command": "npx",
        "args": [
            "@playwright/mcp@latest",
            # "--headless",                   # comment in to disable pop up browser
            # "--vision"                     # to use vision, default is to use the accessibility tree
        ]
    }
}

function_list = [
    {"mcpServers": playwright_mcp}
]

# ---------- Optional system hint -------------------------------------------
messages = []
if args.hints:
    sys_hint = input("\n✍️  Enter additional SYSTEM hint (will NOT be logged):\n> ")
    messages.append({"role": "system", "content": sys_hint})

# ---------- Kick off the interaction ---------------------------------------
messages.append({
    "role": "user",
    "content": f"Open {args.start_url} and complete the task: {args.task}."
})

bot = Assistant(llm=llm_cfg, function_list=function_list)

print("\n🚀  Running agent …\n")
for chunk in bot.run(messages=messages, stream=True):
    role = chunk.get("role")
    if role == "assistant":
        print("\033[96m🤖 assistant:\033[0m", chunk["content"].strip())
        for tc in chunk.get("tool_calls", []):
            print("   ↳ tool_call:", tc)
    elif role == "tool":
        print("\033[93m🛠  tool-response:\033[0m", chunk["content"][:120], "…")
    sys.stdout.flush()

# ---------- Persist trace ---------------------------------------------------
if args.record:
    trace_dir = pathlib.Path("traces")
    trace_dir.mkdir(exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "task": args.task,
        "messages": messages + [chunk],      # final assistant turn
        "tools": bot.build_tool_schema()     # only function signatures!
    }
    out_path = trace_dir / f"{ts}_{uuid.uuid4().hex[:6]}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n💾  Saved trace → {out_path}")
