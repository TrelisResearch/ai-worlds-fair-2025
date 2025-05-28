#!/usr/bin/env python3
"""
Convert saved traces into a HF dataset and push to hub.
"""
import argparse, json, pathlib, getpass
from datasets import Dataset
from huggingface_hub import HfApi, create_repo

ap = argparse.ArgumentParser()
ap.add_argument("--traces_dir", default="traces", help="Folder with *.json traces")
ap.add_argument("--task", required=True, help="Only include traces with this task")
ap.add_argument("--repo", required=True, help="hub_username/dataset_name")
ap.add_argument("--private", action="store_true", help="Push as private repo")
args = ap.parse_args()

# ----- collect --------------------------------------------------------------
samples = []
for fp in pathlib.Path(args.traces_dir).glob("*.json"):
    data = json.loads(fp.read_text())
    if data.get("task") == args.task:
        samples.append({"messages": data["messages"], "tools": data["tools"]})

if not samples:
    raise SystemExit(f"No traces found for task='{args.task}' in {args.traces_dir}")

print(f"📦  Packing {len(samples)} samples …")
ds = Dataset.from_list(samples)

# ----- push -----------------------------------------------------------------
token = getpass.getpass("🔐  HF token (write scope): ")
api = HfApi(token=token)

try:
    create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True, token=token)
except Exception:
    pass  # repo already exists

ds.push_to_hub(args.repo, token=token, split="train")
print(f"✅  Uploaded to https://huggingface.co/datasets/{args.repo}")
