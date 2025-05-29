# MCP Agent Fine-tuning

An agent that bridges MCP JSON-RPC tool servers with OpenAI-compatible chat completion APIs.

Todo:
[ ] Save traces
[ ] Add custom system message that is not saved to traces

## Data Collection

Start a Qwen3 server, locally or on a service like Runpod [one-click template, affiliate](https://runpod.io/console/deploy?template=y3syp133lq&ref=jmfkcdio)

Run the agent with:
```bash
uv run agent.py --model Qwen/Qwen3-30B-A3B-FP8 --base-url https://0zslbmx98vpo2i-8000.proxy.runpod.net/v1 --show-reasoning
```

## Command-line Arguments

The agent supports the following command-line arguments:

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--config` | `-c` | `config.json` | Path to MCP config file |
| `--model` | `-m` | `gpt-4o` | Model name to use for chat completions |
| `--base-url` | | None | Custom OpenAI-compatible API endpoint |
| `--api-key` | | From env | Override OPENAI_API_KEY environment variable |
| `--show-reasoning` | | True | Display model reasoning content when available |

### Examples

Using OpenAI API:
```bash
uv run agent.py --model gpt-4o
```

Using a local server:
```bash
uv run agent.py --model any-model-name --base-url http://localhost:8000/v1
```

Using a custom API key:
```bash
uv run agent.py --api-key sk-your-api-key-here
```