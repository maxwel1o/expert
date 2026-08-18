#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ascend NPU Inference Script for facebook/mms-tts-*
Model: VitsModel (Facebook MMS Multilingual TTS)
Hardware: Huawei Ascend NPU
"""

import os
import sys
import time
import argparse

import torch
import numpy as np

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    torch_npu = None

from transformers import AutoTokenizer, VitsModel


def get_device():
    if torch_npu is not None and torch.npu.is_available():
        return torch.device("npu")
    return torch.device("cpu")


def load_model(model_path, device):
    print(f"Loading model from: {model_path}")
    print(f"Target device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = VitsModel.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded. Parameters: {param_count:,}")
    return tokenizer, model


def synthesize(text, tokenizer, model, device, output_path=None, speaking_rate=1.0):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        start = time.perf_counter()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        waveform = outputs.waveform[0]
        if speaking_rate != 1.0:
            waveform = torch.nn.functional.interpolate(
                waveform.unsqueeze(0).unsqueeze(0),
                scale_factor=1.0 / speaking_rate,
                mode="linear",
                align_corners=False,
            ).squeeze()
        else:
            waveform = waveform.cpu()
        latency = time.perf_counter() - start

    sampling_rate = model.config.sampling_rate
    print(f"Latency: {latency:.3f}s | Samples: {waveform.shape[0]} | SR: {sampling_rate}")

    if output_path:
        from scipy.io import wavfile
        wav_data = (waveform.numpy() * 32767).astype(np.int16)
        wavfile.write(output_path, sampling_rate, wav_data)
        print(f"Audio saved to: {output_path}")

    return waveform, sampling_rate


def main():
    parser = argparse.ArgumentParser(description="Ascend NPU TTS Inference")
    parser.add_argument("--model_path", type=str, default="./model")
    parser.add_argument("--text", type=str, default="Hello, welcome to the world of text to speech.")
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--speaking_rate", type=float, default=1.0)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    device = get_device()
    tokenizer, model = load_model(args.model_path, device)

    if args.benchmark:
        test_texts = [args.text] if args.text else [
            "Hello, welcome to the world of text to speech.",
        ]
        for i, t in enumerate(test_texts[:args.warmup]):
            synthesize(t, tokenizer, model, device)
        latencies = []
        for i, t in enumerate(test_texts[:args.iterations]):
            inputs = tokenizer(t, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            if device.type == "npu":
                torch.npu.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            if device.type == "npu":
                torch.npu.synchronize()
            latencies.append(time.perf_counter() - start)
        avg = sum(latencies) / len(latencies)
        print(f"\nBenchmark: {len(latencies)} iters, avg {avg:.3f}s")
    else:
        synthesize(args.text, tokenizer, model, device, output_path=args.output, speaking_rate=args.speaking_rate)


if __name__ == "__main__":
    main()
