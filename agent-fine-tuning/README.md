## Quick start

Sync the requirements:
```bash
uv sync
```

```bash
# 1. run an agent and watch the browser
uv run run_agent_browser.py \
--endpoint https://109h3fti0xr9vz-8000.proxy.runpod.net/v1 \
--model Qwen/Qwen3-30B-A3B-FP8 \
--start_url https://www.trelis.com \
--task "Find the first product and read the comments on that product page" \
--record \
--hint

# 2. curate & upload the data
uv run build_dataset_hub.py \
  --traces_dir traces \
  --task "Find the first product and read the comments on that product page" \
  --repo Trelis/trelis_browser_traces
```