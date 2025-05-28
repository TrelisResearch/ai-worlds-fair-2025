#!/usr/bin/env python3
"""
Run a Qwen-Agent with Playwright MCP and keep the prompt within context.

* client-side truncate()  keeps individual tool messages small
"""

import argparse, json, os, sys, uuid, pathlib, datetime as dt
from qwen_agent.agents import Assistant

MAX_TOOL_CHARS = 4000          # ≈1.5 k tokens – adjust to taste

# ────────────────────────────── CLI ─────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--start_url", required=True, help="Start URL for the browser")
ap.add_argument("--model", default="Qwen/Qwen3-8B", help="Model slug")
ap.add_argument("--task", required=True, help="Free-text task name")
ap.add_argument("--record", action="store_true", help="Save trace to ./traces")
ap.add_argument("--hints",  action="store_true", help="Prompt for SYSTEM hint")
ap.add_argument("--endpoint", default="http://localhost:8000/v1",
                help="OpenAI-compatible endpoint")
args = ap.parse_args()

# ──────────────────────── LLM configuration ────────────────────────────────
llm_cfg = dict(
    model=args.model,
    model_server=args.endpoint,
    api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
    generate_cfg=dict(
        extra_body={"chat_template_kwargs": {"enable_thinking": True}}
    )
)

# ────────────────────────── Tool registry ───────────────────────────────────
playwright_mcp = {
    "playwright": {
        "command": "npx",
        "args": [
            "@playwright/mcp@latest",
            # "--vision",
            # "--headless",                 # uncomment to hide the browser
        ]
    }
}
function_list = [{"mcpServers": playwright_mcp}]

# ────────────────────── Seed conversation memory ───────────────────────────
history = []
if args.hints:
    hint = input("\n✍️  Extra SYSTEM hint (not saved):\n> ")
    history.append({"role": "system", "content": hint})

history.append({
    "role": "user",
    "content": f"Open {args.start_url} and complete the task: {args.task}."
})

# ──────────────────────────── Run agent ─────────────────────────────────────
bot = Assistant(llm=llm_cfg, function_list=function_list)

print("\n🔍  Debug: attrs containing 'tool' ↓")
print([a for a in dir(bot) if 'tool' in a.lower()])


def truncate_tool_msg(msg: str) -> str:
    if len(msg) > MAX_TOOL_CHARS:
        return msg[:MAX_TOOL_CHARS] + "\n…[truncated]"
    return msg

print("\n🚀  Running agent …\n")
streaming_msg = None           # holds the message that is still growing

for batch in bot.run(messages=history, stream=True):
    batch = batch if isinstance(batch, list) else [batch]

    # 1) everything except the last element is finalized
    *done, streaming_msg = batch

    for msg in done:
        role    = msg["role"]
        content = (msg.get("content") or "").strip()

        if role == "assistant":
            if content:
                print(f"\033[96m🤖 assistant:\033[0m {content}")
            for tc in msg.get("tool_calls", []):
                print(f"   ↳ tool_call: {tc}")
        elif role == "tool":
            head = content[:120] + ("…" if len(content) > 120 else "")
            print(f"\033[93m🛠  tool-response:\033[0m {head}")
            msg["content"] = truncate_tool_msg(content)

        history.append(msg)    # safe: these will never change
        sys.stdout.flush()

# stream ended → append the final assistant turn once
if streaming_msg:
    history.append(streaming_msg)

# ─────────────────────────── Persist trace ─────────────────────────────────
if args.record:
    trace_dir = pathlib.Path("traces"); trace_dir.mkdir(exist_ok=True)

    # Newer qwen-agent versions expose build_tool_schema(); older ones don’t.
    tool_schema = getattr(bot, "build_tool_schema", None)
    if callable(tool_schema):
        tool_schema = tool_schema()
    else:
        # fall back to the original function_list we constructed
        tool_schema = function_list

    trace = {"task": args.task,
             "messages": history,
             "tools": tool_schema}

    ts   = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = trace_dir / f"{ts}_{uuid.uuid4().hex[:6]}.json"
    path.write_text(json.dumps(trace, indent=2))
    print(f"\n💾  Saved trace → {path}")

