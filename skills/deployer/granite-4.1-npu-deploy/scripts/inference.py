#!/usr/bin/env python3
"""Granite 4.1 NPU Inference Script"""

import torch_npu
from torch_npu.contrib import transfer_to_npu

import os
import sys
import time
import argparse
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def check_npu():
    if not torch.npu.is_available():
        logger.error("NPU is not available. Please check CANN environment.")
        sys.exit(1)
    npu_count = torch.npu.device_count()
    logger.info(f"NPU available: True, device count: {npu_count}")
    for i in range(npu_count):
        props = torch.npu.get_device_properties(i)
        logger.info(f"  NPU:{i} name={props.name}, mem={props.total_memory / 1024**3:.1f}GB")


def load_model(model_path: str, device: str = "npu"):
    logger.info(f"Loading model from: {model_path}")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    dtype = torch.bfloat16 if torch.npu.is_available() else torch.float32
    logger.info(f"Using dtype: {dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    elapsed = time.time() - start
    logger.info(f"Model loaded in {elapsed:.1f}s")
    logger.info(f"Model device: {next(model.parameters()).device}")
    return model, tokenizer


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 128):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
    logger.info(f"Prompt: {prompt!r}")
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    elapsed = time.time() - start
    generated_tokens = outputs[0].shape[0] - inputs["input_ids"].shape[1]
    logger.info(f"Generated {generated_tokens} tokens in {elapsed:.2f}s "
                f"({generated_tokens / elapsed:.1f} tok/s)")
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Granite 4.1 NPU Inference")
    parser.add_argument("--model", type=str, required=True, help="Model ID or local path")
    parser.add_argument("--prompt", type=str, default="def hello_world():")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="npu")
    parser.add_argument("--cache-dir", type=str, default=None)
    args = parser.parse_args()

    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    check_npu()
    model, tokenizer = load_model(args.model, device=args.device)
    result = generate_text(model, tokenizer, prompt=args.prompt, max_new_tokens=args.max_new_tokens)
    print("\n" + "=" * 60)
    print(result)
    print("=" * 60)
    logger.info("NPU inference completed successfully.")


if __name__ == "__main__":
    main()
