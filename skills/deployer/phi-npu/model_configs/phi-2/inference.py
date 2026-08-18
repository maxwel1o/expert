#!/usr/bin/env python3
"""phi-2 NPU inference example."""
import os
os.environ["TASK_QUEUE_ENABLE"] = "1"

import torch
import torch_npu
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/opt/atomgit/.cache/modelscope/hub/models/microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
model = model.npu()
model.eval()

prompts = [
    "Write a quicksort implementation in Python:",
    "Explain the concept of recursion with an example:",
    "Write a SQL query to find the top 5 most frequent words in a table:",
]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to("npu:0")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Output: {result}\n")
