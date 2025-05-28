#!/usr/bin/env python3
"""
Minimal Qwen-Agent + Playwright MCP runner (non-streaming).

* opens a URL, performs a task, prints the conversation
* --record  saves {"task","messages","tools"} to ./traces
* --hints   lets you give an extra system message
"""

import argparse, json, os, pathlib, uuid, datetime as dt
from qwen_agent.agents import Assistant

MAX_TOOL_CHARS = 4_000     # trim very large snapshots

# ───────────── CLI ─────────────────────────────────────────────────────────
P = argparse.ArgumentParser()
P.add_argument("--start_url", required=True)
P.add_argument("--task",      required=True)
P.add_argument("--model",     default="Qwen/Qwen3-8B")
P.add_argument("--endpoint",  default="http://localhost:8000/v1")
P.add_argument("--record",    action="store_true")
P.add_argument("--hints",     action="store_true")
args = P.parse_args()

# ───────────── LLM config ─────────────────────────────────────────────────
llm_cfg = dict(
    model=args.model,
    model_server=args.endpoint,
    api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
    generate_cfg=dict(extra_body={"chat_template_kwargs": {"enable_thinking": True}})
)

# ───────────── Tools: Playwright MCP ───────────────────────────────────────
function_list = [{
    "mcpServers": {
        "playwright": {"command": "npx", "args": ["@playwright/mcp@latest"]}
    }
}]

# ───────────── Seed messages ───────────────────────────────────────────────
history = []
if args.hints:
    hint = input("Extra SYSTEM hint (not saved): ")
    history.append({"role": "system", "content": hint})

history.append({
    "role": "user",
    "content": f"Open {args.start_url}. {args.task}"
})

# ───────────── Run agent (single pass) ─────────────────────────────────────
bot = Assistant(
    llm=llm_cfg,
    function_list=function_list,
)

print("\n🚀  Running agent …\n")
responses = bot.run(messages=history, stream=False)   # list[dict] or list[list]

# flatten nested batches, if any
messages = [m for item in responses for m in (item if isinstance(item, list) else [item])]

def trim_tool(text: str) -> str:
    return text if len(text) <= MAX_TOOL_CHARS else text[:MAX_TOOL_CHARS] + "\n…[truncated]"

# ───────────── Print conversation ─────────────────────────────────────────
for m in messages:
    role = m["role"]
    if role == "assistant":
        fc = m.get("function_call")
        if fc:
            print(f"\033[96m🤖 assistant (call)\033[0m → {fc['name']} {fc['arguments']}")
        elif m.get("content"):
            print(f"\033[96m🤖 assistant:\033[0m {m['content']}")
    elif role == "tool":
        head = m["content"][:120] + ("…" if len(m["content"]) > 120 else "")
        print(f"\033[93m🛠  tool:\033[0m {head}")
        m["content"] = trim_tool(m["content"])

history.extend(messages)

# ───────────── Save trace ─────────────────────────────────────────────────
if args.record:
    trace_dir = pathlib.Path("traces"); trace_dir.mkdir(exist_ok=True)
    tool_schema = (
        bot.build_tool_schema() if hasattr(bot, "build_tool_schema") else function_list
    )
    trace = {"task": args.task, "messages": history, "tools": tool_schema}
    fname = f"{dt.datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}.json"
    (trace_dir / fname).write_text(json.dumps(trace, indent=2))
    print(f"\n💾  Saved trace → traces/{fname}")
