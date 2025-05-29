# MCP Agent Fine-tuning


## Data Collection

Start a Qwen3 server, locally or on a service like Runpod [one-click tempalte, affiliate](https://runpod.io/console/deploy?template=y3syp133lq&ref=jmfkcdio)

Run the agent with:
```bash
uv run agent.py --model Qwen/Qwen3-30B-A3B-FP8 --base_url https://0zslbmx98vpo2i-8000.proxy.runpod.net --show-reasoning
```