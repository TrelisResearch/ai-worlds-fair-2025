# Agent Fine-Tuning (prototyping)

In short, using Qwen bot is tricky because it doesn't show the raw tools that are prepped from mcps. Further, it doesn't easily allow for tool repsonse lengths to be truncated, which is necessary to prevent overly long responses.

They keys to being able to save traces as a dataset are:
a. Saving messages.
b. Saving tools being passed to the llm.