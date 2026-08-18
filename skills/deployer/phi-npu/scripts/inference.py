#!/usr/bin/env python3
"""Unified NPU inference script for phi-1, phi-1.5, phi-2.

Usage:
    python3 scripts/inference.py <model_name> [--prompt PROMPT] [--max-new-tokens N]

Examples:
    python3 scripts/inference.py phi-2
    python3 scripts/inference.py phi-1 --prompt "def fibonacci(n):" --max-new-tokens 100
"""
import argparse
import os

os.environ["TASK_QUEUE_ENABLE"] = "1"

import torch
import torch_npu
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATHS = {
    "phi-1": "/opt/atomgit/.cache/modelscope/hub/models/microsoft/phi-1",
    "phi-1.5": "/opt/atomgit/.cache/modelscope/hub/models/microsoft/phi-1_5",
    "phi-1_5": "/opt/atomgit/.cache/modelscope/hub/models/microsoft/phi-1_5",
    "phi-2": "/opt/atomgit/.cache/modelscope/hub/models/microsoft/phi-2",
}

DEFAULT_PROMPTS = {
    "phi-1": [
        "def fibonacci(n):",
        "Write a Python function to sort a list:",
        "def quicksort(arr):",
    ],
    "phi-1.5": [
        "Machine learning is a field of",
        "The meaning of life is",
        "Write a story about AI:",
    ],
    "phi-1_5": [
        "Machine learning is a field of",
        "The meaning of life is",
        "Write a story about AI:",
    ],
    "phi-2": [
        "def fibonacci(n):",
        "Write a Python function to sort a list:",
        "Explain the concept of recursion:",
    ],
}


def main():
    parser = argparse.ArgumentParser(description="Phi NPU inference")
    parser.add_argument("model", choices=list(MODEL_PATHS.keys()), help="Model name")
    parser.add_argument("--prompt", help="Single prompt (uses default prompts if omitted)")
    parser.add_argument("--max-new-tokens", type=int, default=80, help="Max new tokens")
    args = parser.parse_args()

    model_path = MODEL_PATHS[args.model]
    print(f"Loading model: {args.model} from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, trust_remote_code=True
    )
    model = model.npu()
    model.eval()
    print(f"Model loaded on {torch.npu.get_device_name(0)}\n")

    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS[args.model]

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to("npu:0")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Prompt: {prompt}")
        print(f"Output: {result}\n")


if __name__ == "__main__":
    main()
