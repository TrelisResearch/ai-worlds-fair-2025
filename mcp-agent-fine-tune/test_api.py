#!/usr/bin/env python3
"""
Test script to check available models on an OpenAI-compatible API endpoint
"""

import requests
import json
import sys

def main():
    # Get base URL from command line or use default
    base_url = sys.argv[1] if len(sys.argv) > 1 else "https://0zslbmx98vpo2i-8000.proxy.runpod.net/v1"
    
    # Fix URL if needed
    if base_url.endswith("--show-reasoning"):
        base_url = base_url.replace("--show-reasoning", "")
    
    # Ensure URL ends with /v1
    if not base_url.endswith("/v1"):
        if "/v1" not in base_url:
            base_url = f"{base_url}/v1"
    
    print(f"Testing API endpoint: {base_url}")
    
    # Test models endpoint
    try:
        models_url = f"{base_url}/models"
        print(f"Requesting: {models_url}")
        response = requests.get(models_url)
        
        if response.status_code == 200:
            models = response.json()
            print("\nAvailable models:")
            for model in models.get("data", []):
                print(f"- {model.get('id')}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Exception: {e}")
    
    # Test a simple completion to check if the API is working
    try:
        print("\nTesting a simple completion...")
        completion_url = f"{base_url}/chat/completions"
        payload = {
            "model": "Qwen/Qwen3-30B-A3B-FP8",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 10
        }
        
        print(f"Requesting: {completion_url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(completion_url, json=payload)
        
        if response.status_code == 200:
            print("\nSuccess! Response:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    main()
