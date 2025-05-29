# MCP Agent Fine-tuning

An agent that bridges MCP JSON-RPC tool servers with OpenAI-compatible chat completion APIs.

Todo:
[x] Save traces.
[ ] Allow for saving and pushing adataset to huggingface (advise in the readme for th euser to log in with huggingface-cli login)
[ ] Allow for back tracking functionality to help build higher quality traces.

## Data Collection

Start a Qwen3 server, locally or on a service like Runpod [one-click template, affiliate](https://runpod.io/console/deploy?template=y3syp133lq&ref=jmfkcdio)

Run the agent with:
```bash
uv run agent.py --model Qwen/Qwen3-30B-A3B-FP8 --base-url https://0zslbmx98vpo2i-8000.proxy.runpod.net/v1 --show-reasoning
```

The agent supports the following command-line arguments:

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--config` | `-c` | `config.json` | Path to MCP config file |
| `--model` | `-m` | `gpt-4o` | Model name to use for chat completions |
| `--base-url` | | None | Custom OpenAI-compatible API endpoint |
| `--api-key` | | From env | Override OPENAI_API_KEY environment variable |
| `--show-reasoning` | | True | Display model reasoning content when available |
| `--trace-dir` | | `traces` | Directory to save conversation traces |

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

## Trace Logging

The agent automatically logs conversation traces to the `traces` directory. Each trace is saved as a JSON file named using the first 30 characters of the user's first message plus a timestamp.

Traces include:
- Full conversation history with all messages
- Tool definitions in OpenAI format
- Model information
- Timestamp

You can specify a custom directory for traces using the `--trace-dir` option.

## Fine-tuning