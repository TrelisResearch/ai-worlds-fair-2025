# MCP Agent Fine-tuning

An agent that bridges MCP JSON-RPC tool servers with OpenAI-compatible chat completion APIs.

Todo:
[x] Save traces.
[x] Allow for saving and pushing a dataset to Hugging Face Hub.
[ ] Test that tools and messages can be reloaded from traces by making a call to the api.
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

### Trace Logging

The agent automatically logs conversation traces to the `traces` directory. Each trace is saved as a JSON file named using the first 30 characters of the user's first message plus a timestamp.

Traces include:
- Full conversation history with all messages
- Tool definitions in OpenAI format
- Model information
- Timestamp

You can specify a custom directory for traces using the `--trace-dir` option.

### Pushing Traces to Hugging Face Hub

After collecting conversation traces, you can push them to Hugging Face Hub as a dataset for fine-tuning or sharing. The repository includes a script for this purpose.

**Prerequisites**

1. Log in to Hugging Face Hub using the CLI:
   ```bash
   huggingface-cli login
   ```

2. Make sure you have collected some traces in your traces directory.

Run the script with your Hugging Face repository ID:

```bash
uv run push-to-hub.py --repo-id="Trelis/qwen-web-agent"
```

Options:
- `--repo-id` (required): The Hugging Face Hub repository ID where the dataset will be pushed
- `--trace-dir` (optional): Directory containing trace files (default: "traces")

The script will:
1. Load all JSON trace files from the specified directory
2. Convert them to a structured dataset format
3. Push the dataset to Hugging Face Hub as a private dataset

You can then use this dataset for fine-tuning models or share it with others.

## Fine-tuning