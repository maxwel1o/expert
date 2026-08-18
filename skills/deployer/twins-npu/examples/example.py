#!/usr/bin/env python3
'''Twins NPU deployment example'''
import subprocess, sys, os

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'scripts')

def run_inference(model_name, device='npu'):
    subprocess.run([
        sys.executable, os.path.join(SCRIPTS_DIR, 'inference.py'),
        '--model-name', model_name,
        '--device', device,
        '--num-runs', '3'
    ])

def run_compare(model_name):
    subprocess.run([
        sys.executable, os.path.join(SCRIPTS_DIR, 'compare_cpu_npu.py'),
        '--model-name', model_name
    ])

if __name__ == '__main__':
    run_inference('twins_svt_small.in1k', 'npu')
