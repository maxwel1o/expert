"""Master script to process all PP-OCRv3 models serially."""

import sys
import os
import time
import json
import gc
import shutil
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'common'))
from model_configs import MODELS

BASE_DIR = '/opt/atomgit/ocr_npu_adapt'
MODELS_CACHE = '/opt/atomgit/ocr_models'
TEST_IMAGE = '/opt/atomgit/test_image.png'
TEST_REC_IMAGE = '/opt/atomgit/test_rec_image.png'

RESULTS = {}


def create_det_inference_script(model_dir, model_name, model_cache_path):
    """Create inference.py for detection model."""
    script = f'''"""PP-OCRv3 Text Detection inference - {model_name}."""

import sys, os, time, argparse, cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
from ocr_utils import preprocess_detection, postprocess_detection, create_session, run_inference

MODEL_PATH = "{model_cache_path}/model.onnx"

def detect_text(img_path, provider='CPUExecutionProvider', threshold=0.3):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {{img_path}}")
    input_tensor, scale = preprocess_detection(img)
    session = create_session(MODEL_PATH, provider)
    start = time.perf_counter()
    output = run_inference(session, input_tensor)
    elapsed = (time.perf_counter() - start) * 1000
    boxes = postprocess_detection(output, scale, img.shape, threshold)
    return boxes, elapsed

def main():
    parser = argparse.ArgumentParser(description='{model_name} Inference')
    parser.add_argument('--image', default='{TEST_IMAGE}')
    parser.add_argument('--provider', default='CPUExecutionProvider', choices=['CPUExecutionProvider','CANNExecutionProvider'])
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    print(f"[{{args.provider}}] Running {{args.image}}")
    boxes, elapsed = detect_text(args.image, args.provider, args.threshold)
    print(f"Inference: {{elapsed:.2f}} ms, boxes: {{len(boxes)}}")
    for i, box in enumerate(boxes):
        print(f"  Box {{i+1}}: [{{box[0]}}, {{box[1]}}, {{box[2]}}, {{box[3]}}]")
    if args.output and boxes:
        img = cv2.imread(args.image)
        for box in boxes:
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.imwrite(args.output, img)
        print(f"Saved to {{args.output}}")
    return boxes, elapsed

if __name__ == '__main__':
    main()
'''
    with open(os.path.join(model_dir, 'inference.py'), 'w') as f:
        f.write(script)


def create_rec_inference_script(model_dir, model_name, model_cache_path, dict_file):
    """Create inference.py for recognition model."""
    script = f'''"""PP-OCRv3 Text Recognition inference - {model_name}."""

import sys, os, time, argparse, cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
from ocr_utils import preprocess_recognition, postprocess_recognition, create_session, run_inference, load_char_dict

MODEL_PATH = "{model_cache_path}/model.onnx"
DICT_PATH = "{dict_file}"

_char_dict = None

def get_char_dict():
    global _char_dict
    if _char_dict is None:
        _char_dict = load_char_dict(DICT_PATH)
    return _char_dict

def recognize_text(img_path, provider='CPUExecutionProvider'):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {{img_path}}")
    input_tensor, orig_width = preprocess_recognition(img)
    session = create_session(MODEL_PATH, provider)
    start = time.perf_counter()
    output = run_inference(session, input_tensor)
    elapsed = (time.perf_counter() - start) * 1000
    char_dict = get_char_dict()
    text, confidence = postprocess_recognition(output, char_dict, orig_width)
    return text, confidence, elapsed

def main():
    parser = argparse.ArgumentParser(description='{model_name} Inference')
    parser.add_argument('--image', default='{TEST_REC_IMAGE}')
    parser.add_argument('--provider', default='CPUExecutionProvider', choices=['CPUExecutionProvider','CANNExecutionProvider'])
    args = parser.parse_args()

    print(f"[{{args.provider}}] Running {{args.image}}")
    text, conf, elapsed = recognize_text(args.image, args.provider)
    print(f"Inference: {{elapsed:.2f}} ms")
    print(f"Text: {{text}}")
    print(f"Confidence: {{conf:.4f}}")
    return text, conf, elapsed

if __name__ == '__main__':
    main()
'''
    with open(os.path.join(model_dir, 'inference.py'), 'w') as f:
        f.write(script)


def create_det_compare_script(model_dir, model_name, model_cache_path):
    """Create compare_cpu_npu.py for detection model."""
    script = f'''"""CPU vs NPU comparison for {model_name}."""

import sys, os, time, json, cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
from ocr_utils import preprocess_detection, postprocess_detection, create_session, run_inference

MODEL_PATH = "{model_cache_path}/model.onnx"
IMAGE_PATH = "{TEST_IMAGE}"

def compute_iou(box_a, box_b):
    x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-8)

def match_boxes(cpu_boxes, npu_boxes, iou_threshold=0.5):
    matches = []
    used_npu = set()
    for i, cb in enumerate(cpu_boxes):
        best_iou, best_j = 0.0, -1
        for j, nb in enumerate(npu_boxes):
            if j in used_npu:
                continue
            iou = compute_iou(cb, nb)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_threshold:
            matches.append((i, best_j, best_iou))
            used_npu.add(best_j)
    return matches

def main():
    print("=" * 60)
    print("{model_name}: CPU vs NPU Comparison")
    print("=" * 60)

    img = cv2.imread(IMAGE_PATH)
    input_tensor, scale = preprocess_detection(img)

    # CPU
    print("\\n[CPU]")
    cpu_session = create_session(MODEL_PATH, 'CPUExecutionProvider')
    cpu_times = []
    for _ in range(5):
        start = time.perf_counter()
        cpu_output = run_inference(cpu_session, input_tensor)
        cpu_times.append((time.perf_counter() - start) * 1000)
    cpu_time = np.mean(cpu_times[1:])
    cpu_boxes = postprocess_detection(cpu_output, scale, img.shape, threshold=0.3)
    print(f"  Time: {{cpu_time:.2f}} ms, Boxes: {{len(cpu_boxes)}}")
    print(f"  Output: min={{cpu_output.min():.6f}}, max={{cpu_output.max():.6f}}, mean={{cpu_output.mean():.6f}}")

    # NPU
    print("\\n[NPU]")
    npu_session = create_session(MODEL_PATH, 'CANNExecutionProvider')
    npu_times = []
    for _ in range(5):
        start = time.perf_counter()
        npu_output = run_inference(npu_session, input_tensor)
        npu_times.append((time.perf_counter() - start) * 1000)
    npu_time = np.mean(npu_times[1:])
    npu_boxes = postprocess_detection(npu_output, scale, img.shape, threshold=0.3)
    print(f"  Time: {{npu_time:.2f}} ms, Boxes: {{len(npu_boxes)}}")
    print(f"  Output: min={{npu_output.min():.6f}}, max={{npu_output.max():.6f}}, mean={{npu_output.mean():.6f}}")

    # Compare
    print("\\n[Precision]")
    raw_diff = np.abs(cpu_output - npu_output)
    max_diff = float(raw_diff.max())
    mean_diff = float(raw_diff.mean())
    rel_diff = mean_diff / (np.abs(cpu_output).mean() + 1e-8)

    print(f"  Max abs diff: {{max_diff:.8f}}")
    print(f"  Mean abs diff: {{mean_diff:.8f}}")
    print(f"  Relative diff: {{rel_diff*100:.4f}}%")

    matches = match_boxes(cpu_boxes, npu_boxes)
    match_rate = len(matches) / max(len(cpu_boxes), 1) * 100
    print(f"  CPU boxes: {{len(cpu_boxes)}}, NPU boxes: {{len(npu_boxes)}}")
    print(f"  Matched: {{len(matches)}}, Rate: {{match_rate:.1f}}%")

    for i, j, iou in matches:
        print(f"    Box {{i+1}}: CPU={{cpu_boxes[i]}}, NPU={{npu_boxes[j]}}, IoU={{iou:.4f}}")

    precision_ok = (mean_diff < 0.01) and (rel_diff * 100 < 1.0)
    boxes_ok = match_rate >= 90.0

    print(f"\\n{{'='*60}}")
    print(f"Precision:")
    print(f"  Mean abs diff < 0.01: {{'PASS' if mean_diff < 0.01 else 'FAIL'}} ({{mean_diff:.8f}})")
    print(f"  Relative diff < 1%: {{'PASS' if rel_diff*100 < 1.0 else 'FAIL'}} ({{rel_diff*100:.4f}}%)")
    print(f"  Box match >= 90%: {{'PASS' if boxes_ok else 'FAIL'}} ({{match_rate:.1f}}%)")
    overall = precision_ok and boxes_ok
    print(f"  Overall: {{'PASS - Error < 1%' if overall else 'FAIL'}}")
    print(f"{{'='*60}}")

    results = {{
        'model': '{model_name}',
        'task': 'ocr-detection',
        'cpu_time_ms': round(float(cpu_time), 2),
        'npu_time_ms': round(float(npu_time), 2),
        'cpu_boxes_count': len(cpu_boxes),
        'npu_boxes_count': len(npu_boxes),
        'matched_boxes': len(matches),
        'box_match_rate': round(match_rate, 1),
        'raw_max_diff': max_diff,
        'raw_mean_diff': mean_diff,
        'raw_rel_diff_pct': round(rel_diff * 100, 4),
        'precision_pass': bool(overall),
    }}
    with open(os.path.join(os.path.dirname(__file__), 'compare_result.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\\nSaved to compare_result.json")

if __name__ == '__main__':
    main()
'''
    with open(os.path.join(model_dir, 'compare_cpu_npu.py'), 'w') as f:
        f.write(script)


def create_rec_compare_script(model_dir, model_name, model_cache_path, dict_file):
    """Create compare_cpu_npu.py for recognition model."""
    script = f'''"""CPU vs NPU comparison for {model_name}."""

import sys, os, time, json, cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
from ocr_utils import preprocess_recognition, postprocess_recognition, create_session, run_inference, load_char_dict

MODEL_PATH = "{model_cache_path}/model.onnx"
DICT_PATH = "{dict_file}"
IMAGE_PATH = "{TEST_REC_IMAGE}"

def main():
    print("=" * 60)
    print("{model_name}: CPU vs NPU Comparison")
    print("=" * 60)

    img = cv2.imread(IMAGE_PATH)
    char_dict = load_char_dict(DICT_PATH)
    input_tensor, orig_width = preprocess_recognition(img)

    # CPU
    print("\\n[CPU]")
    cpu_session = create_session(MODEL_PATH, 'CPUExecutionProvider')
    cpu_times = []
    for _ in range(5):
        start = time.perf_counter()
        cpu_output = run_inference(cpu_session, input_tensor)
        cpu_times.append((time.perf_counter() - start) * 1000)
    cpu_time = np.mean(cpu_times[1:])
    cpu_text, cpu_conf = postprocess_recognition(cpu_output, char_dict, orig_width)
    print(f"  Time: {{cpu_time:.2f}} ms")
    print(f"  Text: '{{cpu_text}}', Conf: {{cpu_conf:.4f}}")
    print(f"  Output shape: {{cpu_output.shape}}, mean={{cpu_output.mean():.6f}}, max={{cpu_output.max():.6f}}")

    # NPU
    print("\\n[NPU]")
    npu_session = create_session(MODEL_PATH, 'CANNExecutionProvider')
    npu_times = []
    for _ in range(5):
        start = time.perf_counter()
        npu_output = run_inference(npu_session, input_tensor)
        npu_times.append((time.perf_counter() - start) * 1000)
    npu_time = np.mean(npu_times[1:])
    npu_text, npu_conf = postprocess_recognition(npu_output, char_dict, orig_width)
    print(f"  Time: {{npu_time:.2f}} ms")
    print(f"  Text: '{{npu_text}}', Conf: {{npu_conf:.4f}}")
    print(f"  Output shape: {{npu_output.shape}}, mean={{npu_output.mean():.6f}}, max={{npu_output.max():.6f}}")

    # Compare
    print("\\n[Precision]")
    raw_diff = np.abs(cpu_output - npu_output)
    max_diff = float(raw_diff.max())
    mean_diff = float(raw_diff.mean())
    rel_diff = mean_diff / (np.abs(cpu_output).mean() + 1e-8)

    text_match = cpu_text == npu_text
    conf_diff = abs(cpu_conf - npu_conf)

    print(f"  Max abs diff: {{max_diff:.8f}}")
    print(f"  Mean abs diff: {{mean_diff:.8f}}")
    print(f"  Relative diff: {{rel_diff*100:.4f}}%")
    print(f"  CPU text: '{{cpu_text}}'")
    print(f"  NPU text: '{{npu_text}}'")
    print(f"  Text match: {{text_match}}")
    print(f"  Confidence diff: {{conf_diff:.6f}}")

    precision_ok = (mean_diff < 0.01) and (rel_diff * 100 < 1.0)
    text_ok = text_match

    print(f"\\n{{'='*60}}")
    print(f"Precision:")
    print(f"  Mean abs diff < 0.01: {{'PASS' if mean_diff < 0.01 else 'FAIL'}} ({{mean_diff:.8f}})")
    print(f"  Relative diff < 1%: {{'PASS' if rel_diff*100 < 1.0 else 'FAIL'}} ({{rel_diff*100:.4f}}%)")
    print(f"  Text match: {{'PASS' if text_match else 'FAIL'}}")
    overall = precision_ok and text_ok
    print(f"  Overall: {{'PASS - Error < 1%' if overall else 'FAIL'}}")
    print(f"{{'='*60}}")

    results = {{
        'model': '{model_name}',
        'task': 'ocr-recognition',
        'cpu_time_ms': round(float(cpu_time), 2),
        'npu_time_ms': round(float(npu_time), 2),
        'cpu_text': cpu_text,
        'npu_text': npu_text,
        'cpu_confidence': round(float(cpu_conf), 4),
        'npu_confidence': round(float(npu_conf), 4),
        'text_match': text_match,
        'confidence_diff': round(float(conf_diff), 6),
        'raw_max_diff': max_diff,
        'raw_mean_diff': mean_diff,
        'raw_rel_diff_pct': round(rel_diff * 100, 4),
        'precision_pass': bool(overall),
    }}
    with open(os.path.join(os.path.dirname(__file__), 'compare_result.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\\nSaved to compare_result.json")

if __name__ == '__main__':
    main()
'''
    with open(os.path.join(model_dir, 'compare_cpu_npu.py'), 'w') as f:
        f.write(script)


def create_requirements(model_dir, model_name):
    """Create requirements.txt."""
    req = f"""# {model_name} NPU Inference Requirements
onnxruntime-cann>=1.24.0
numpy>=1.21.0
opencv-python-headless>=4.5.0
onnx>=1.12.0
"""
    with open(os.path.join(model_dir, 'requirements.txt'), 'w') as f:
        f.write(req)


def free_memory():
    """Free NPU and system memory."""
    gc.collect()
    try:
        import torch
        if hasattr(torch, "npu"):
            torch.npu.empty_cache()
    except Exception:
        pass


def process_model(model_name, config):
    """Process a single model: create scripts, run inference, compare."""
    print(f"\n{'#'*70}")
    print(f"# Processing: {model_name}")
    print(f"{'#'*70}")

    model_id = config['model_id']
    task = config['task']
    dir_name = config.get('dir_name', model_name)

    model_cache_path = os.path.join(MODELS_CACHE, model_id.replace('/', '/'))
    if not os.path.exists(model_cache_path):
        model_cache_path = os.path.join(MODELS_CACHE, *model_id.split('/'))

    model_dir = os.path.join(BASE_DIR, dir_name)
    os.makedirs(model_dir, exist_ok=True)

    # Find dict file for rec models
    dict_file = None
    if config.get('has_dict'):
        txt_files = [f for f in os.listdir(model_cache_path) if f.endswith('_dict.txt')]
        if txt_files:
            dict_file = os.path.join(model_cache_path, txt_files[0])
        else:
            print(f"  WARNING: No dict file found in {model_cache_path}")
            return {'model': model_name, 'status': 'FAIL - no dict file'}

    # Create scripts
    if task == 'ocr-detection':
        create_det_inference_script(model_dir, model_name, model_cache_path)
        create_det_compare_script(model_dir, model_name, model_cache_path)
    else:
        create_rec_inference_script(model_dir, model_name, model_cache_path, dict_file)
        create_rec_compare_script(model_dir, model_name, model_cache_path, dict_file)

    create_requirements(model_dir, model_name)

    # Run comparison
    compare_script = os.path.join(model_dir, 'compare_cpu_npu.py')
    print(f"  Running comparison...")
    start_time = time.time()
    ret = os.system(f"cd {model_dir} && python3 compare_cpu_npu.py 2>&1")
    elapsed = time.time() - start_time

    if ret != 0:
        print(f"  ERROR: Comparison failed with code {ret}")
        free_memory()
        return {'model': model_name, 'status': f'FAIL - comparison error (code {ret})'}

    # Read results
    result_file = os.path.join(model_dir, 'compare_result.json')
    if os.path.exists(result_file):
        with open(result_file) as f:
            result = json.load(f)
        result['elapsed_sec'] = round(elapsed, 1)
        result['model_dir'] = model_dir
        print(f"  Result: {'PASS' if result.get('precision_pass') else 'FAIL'}")
        print(f"  CPU: {result.get('cpu_time_ms', 'N/A')} ms, NPU: {result.get('npu_time_ms', 'N/A')} ms")
        free_memory()
        return result
    else:
        print(f"  ERROR: No result file generated")
        free_memory()
        return {'model': model_name, 'status': 'FAIL - no result file'}


def main():
    print("=" * 70)
    print("PP-OCRv3 Model Batch Processing for Ascend NPU")
    print("=" * 70)
    print(f"Total models: {len(MODELS)}")
    print(f"Base directory: {BASE_DIR}")
    print()

    all_results = {}
    failed = []
    passed = []

    for i, (model_name, config) in enumerate(MODELS.items()):
        print(f"\n[{i+1}/{len(MODELS)}] Processing {model_name}...")
        try:
            result = process_model(model_name, config)
            all_results[model_name] = result
            if result.get('precision_pass'):
                passed.append(model_name)
            else:
                failed.append((model_name, result.get('status', 'Unknown')))
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            all_results[model_name] = {'model': model_name, 'status': f'FAIL - {str(e)}'}
            failed.append((model_name, str(e)))
            free_memory()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total: {len(MODELS)}, Passed: {len(passed)}, Failed: {len(failed)}")
    print()
    print("| 模型名称 | NPU 推理 | CPU/NPU 误差 | 状态 |")
    print("|---|---|---|---|")
    for model_name in MODELS:
        r = all_results.get(model_name, {})
        status = 'PASS' if r.get('precision_pass') else r.get('status', 'FAIL')
        cpu_time = r.get('cpu_time_ms', 'N/A')
        npu_time = r.get('npu_time_ms', 'N/A')
        rel_diff = r.get('raw_rel_diff_pct', 'N/A')
        print(f"| {model_name} | {cpu_time}ms / {npu_time}ms | {rel_diff}% | {status} |")

    if failed:
        print(f"\nFailed models:")
        for name, reason in failed:
            print(f"  - {name}: {reason}")

    # Save summary
    summary_file = os.path.join(BASE_DIR, 'all_results.json')
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to: {summary_file}")


if __name__ == '__main__':
    main()
