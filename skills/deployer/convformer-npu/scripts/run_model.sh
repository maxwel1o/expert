#!/bin/bash
# Usage: bash run_model.sh <model_name>
# Example: bash run_model.sh convformer_b36.sail_in1k_384

set -e

MODEL_NAME="$1"
if [ -z "$MODEL_NAME" ]; then
    echo "Usage: $0 <model_name>"
    exit 1
fi

WORK_BASE="/opt/atomgit/convformer_workspace"
WORK_DIR="${WORK_BASE}/models/${MODEL_NAME}"
MODELSCOPE_CACHE="${WORK_BASE}/modelscope_cache"
mkdir -p "$WORK_DIR"

echo ""
echo "========================================="
echo "  Model: $MODEL_NAME"
echo "  Started at: $(date)"
echo "========================================="

# ==========================================
# Step 1: Download model from ModelScope
# ==========================================
echo ""
echo "[1/4] Downloading model from ModelScope..."

python3 << PYEOF
from modelscope.hub.snapshot_download import snapshot_download
import os, sys

model_name = "${MODEL_NAME}"
model_id = f"timm/{model_name}"
cache_dir = "${MODELSCOPE_CACHE}"

try:
    model_dir = snapshot_download(model_id, cache_dir=cache_dir)
    # Write model path to a file so the next steps can read it
    with open("${WORK_DIR}/model_path.txt", "w") as f:
        f.write(model_dir)
    print(f"Model downloaded to: {model_dir}")
except Exception as e:
    print(f"ModelScope download failed: {e}")
    print("Falling back to HuggingFace via hf-mirror...")
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    from huggingface_hub import snapshot_download as hf_download
    model_dir = hf_download(f"timm/{model_name}", cache_dir=cache_dir + "_hf")
    with open("${WORK_DIR}/model_path.txt", "w") as f:
        f.write(model_dir)
    print(f"Model downloaded via hf-mirror to: {model_dir}")
PYEOF

# ==========================================
# Step 2: CPU Inference
# ==========================================
echo ""
echo "[2/4] Running CPU inference..."

python3 << PYEOF
import os, sys, json, time, torch
import timm
from safetensors.torch import load_file
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

model_name = "${MODEL_NAME}"
work_dir = "${WORK_DIR}"

with open(os.path.join(work_dir, "model_path.txt")) as f:
    model_dir = f.read().strip()

# Find model weights
weights_path = os.path.join(model_dir, "model.safetensors")
if os.path.exists(weights_path):
    state_dict = load_file(weights_path)
    print("Loaded from safetensors")
else:
    weights_path = os.path.join(model_dir, "pytorch_model.bin")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    print("Loaded from pytorch_model.bin")

# Create model and load weights
model = timm.create_model(model_name, pretrained=False)
model.load_state_dict(state_dict, strict=True)
model.eval()
print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Prepare image
img_path = "${WORK_BASE}/test_image/cat.jpg"
img = Image.open(img_path).convert("RGB")
print(f"Image size: {img.size}")

# Get data config and transforms
config = resolve_data_config({}, model=model)
print(f"Data config: {config}")
transform = create_transform(**config)
input_tensor = transform(img).unsqueeze(0)
print(f"Input tensor shape: {input_tensor.shape}")

# CPU inference
print("Running CPU inference...")
with torch.no_grad():
    start = time.time()
    cpu_output = model(input_tensor)
    cpu_time = time.time() - start

cpu_probs = torch.nn.functional.softmax(cpu_output[0], dim=0)
cpu_top5 = cpu_probs.topk(5)

print(f"CPU inference time: {cpu_time:.4f}s")
print(f"CPU Top-5 indices: {cpu_top5.indices.tolist()}")
print(f"CPU Top-5 probs: {[round(float(v), 6) for v in cpu_top5.values.tolist()]}")

# Save CPU results
results = {
    "model": model_name,
    "cpu_output": [round(float(v), 8) for v in cpu_output[0].tolist()],
    "cpu_top5_indices": [int(v) for v in cpu_top5.indices.tolist()],
    "cpu_top5_probs": [round(float(v), 6) for v in cpu_top5.values.tolist()],
    "cpu_time_s": round(cpu_time, 4),
    "input_shape": list(input_tensor.shape)
}
with open(os.path.join(work_dir, "cpu_results.json"), "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("CPU results saved.")
PYEOF

echo "[2/4] CPU inference completed."

# ==========================================
# Step 3: NPU Inference
# ==========================================
echo ""
echo "[3/4] Running NPU inference..."

python3 << PYEOF
import os, sys, json, time, torch
import timm
from safetensors.torch import load_file
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

model_name = "${MODEL_NAME}"
work_dir = "${WORK_DIR}"

with open(os.path.join(work_dir, "model_path.txt")) as f:
    model_dir = f.read().strip()

# Find model weights
weights_path = os.path.join(model_dir, "model.safetensors")
if os.path.exists(weights_path):
    state_dict = load_file(weights_path)
else:
    weights_path = os.path.join(model_dir, "pytorch_model.bin")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

# Create model and load weights on CPU first
model = timm.create_model(model_name, pretrained=False)
model.load_state_dict(state_dict, strict=True)
model.eval()

# Move to NPU
model = model.npu()
print("Model moved to NPU.")

# Prepare image
img_path = "${WORK_BASE}/test_image/cat.jpg"
img = Image.open(img_path).convert("RGB")
config = resolve_data_config({}, model=model)
transform = create_transform(**config)
input_tensor = transform(img).unsqueeze(0).npu()
print(f"Input tensor shape: {input_tensor.shape}")

# Warmup NPU
print("Warming up NPU...")
with torch.no_grad():
    for _ in range(3):
        _ = model(input_tensor)
torch.npu.synchronize()

# NPU inference
print("Running NPU inference...")
with torch.no_grad():
    start = time.time()
    npu_output = model(input_tensor)
    torch.npu.synchronize()
    npu_time = time.time() - start

npu_probs = torch.nn.functional.softmax(npu_output[0], dim=0)
npu_top5 = npu_probs.topk(5)

print(f"NPU inference time: {npu_time:.4f}s")
print(f"NPU Top-5 indices: {npu_top5.indices.tolist()}")
print(f"NPU Top-5 probs: {[round(float(v), 6) for v in npu_top5.values.tolist()]}")

# Save NPU results
results = {
    "model": model_name,
    "npu_output": [round(float(v), 8) for v in npu_output[0].tolist()],
    "npu_top5_indices": [int(v) for v in npu_top5.indices.tolist()],
    "npu_top5_probs": [round(float(v), 6) for v in npu_top5.values.tolist()],
    "npu_time_s": round(npu_time, 4),
    "input_shape": list(input_tensor.shape)
}
with open(os.path.join(work_dir, "npu_results.json"), "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("NPU results saved.")
PYEOF

echo "[3/4] NPU inference completed."

# ==========================================
# Step 4: CPU/NPU Comparison
# ==========================================
echo ""
echo "[4/4] Comparing CPU vs NPU results..."

python3 << PYEOF
import os, json, torch, math

work_dir = "${WORK_DIR}"

with open(os.path.join(work_dir, "cpu_results.json")) as f:
    cpu_data = json.load(f)
with open(os.path.join(work_dir, "npu_results.json")) as f:
    npu_data = json.load(f)

cpu_out = torch.tensor(cpu_data["cpu_output"])
npu_out = torch.tensor(npu_data["npu_output"])

# Calculate metrics
mae = torch.abs(cpu_out - npu_out).mean().item()
mse = torch.pow(cpu_out - npu_out, 2).mean().item()
max_abs_err = torch.abs(cpu_out - npu_out).max().item()
cosine_sim = torch.nn.functional.cosine_similarity(cpu_out.unsqueeze(0), npu_out.unsqueeze(0)).item()
relative_err = (torch.abs(cpu_out - npu_out) / (torch.abs(cpu_out) + 1e-8)).mean().item() * 100

print(f"")
print(f"=== Precision Comparison: {cpu_data['model']} ===")
print(f"")
print(f"MAE (Mean Absolute Error):     {mae:.10f}")
print(f"MSE (Mean Squared Error):      {mse:.10f}")
print(f"Max Absolute Error:           {max_abs_err:.10f}")
print(f"Cosine Similarity:            {cosine_sim:.10f}")
print(f"Mean Relative Error:          {relative_err:.6f}%")
print(f"")

# Top-5 agreement
cpu_top5 = set(cpu_data["cpu_top5_indices"])
npu_top5 = set(npu_data["npu_top5_indices"])
top5_overlap = cpu_top5 & npu_top5
print(f"Top-5 overlap: {len(top5_overlap)}/5")
print(f"CPU Top-1: class={cpu_data['cpu_top5_indices'][0]}, prob={cpu_data['cpu_top5_probs'][0]}")
print(f"NPU Top-1: class={npu_data['npu_top5_indices'][0]}, prob={npu_data['npu_top5_probs'][0]}")
print(f"Top-1 match: {cpu_data['cpu_top5_indices'][0] == npu_data['npu_top5_indices'][0]}")
print(f"")

# Precision pass/fail (relative error < 1%)
passed = relative_err < 1.0
print(f"Requirement: NPU vs CPU error < 1%")
print(f"Result: {relative_err:.6f}% - {'PASSED' if passed else 'FAILED'}")
print(f"")

# Save comparison
comparison = {
    "model": cpu_data["model"],
    "mae": round(mae, 10),
    "mse": round(mse, 10),
    "max_abs_error": round(max_abs_err, 10),
    "cosine_similarity": round(cosine_sim, 10),
    "mean_relative_error_pct": round(relative_err, 6),
    "top5_overlap": len(top5_overlap),
    "top1_match": cpu_data["cpu_top5_indices"][0] == npu_data["npu_top5_indices"][0],
    "cpu_top1": {"class": cpu_data["cpu_top5_indices"][0], "prob": cpu_data["cpu_top5_probs"][0]},
    "npu_top1": {"class": npu_data["npu_top5_indices"][0], "prob": npu_data["npu_top5_probs"][0]},
    "cpu_time_s": cpu_data["cpu_time_s"],
    "npu_time_s": npu_data["npu_time_s"],
    "speedup": round(cpu_data["cpu_time_s"] / npu_data["npu_time_s"], 2) if npu_data["npu_time_s"] > 0 else 0,
    "passed": passed
}
with open(os.path.join(work_dir, "comparison.json"), "w") as f:
    json.dump(comparison, f, ensure_ascii=False, indent=2)
print("Comparison results saved.")
PYEOF

echo "[4/4] Comparison completed."

# ==========================================
# Cleanup
# ==========================================
echo ""
echo "Cleaning up..."
python3 -c "
import gc, torch
gc.collect()
if hasattr(torch, 'npu'):
    torch.npu.empty_cache()
print('Memory and NPU cache released.')
"

echo ""
echo "========================================="
echo "  Model $MODEL_NAME completed!"
echo "  Finished at: $(date)"
echo "========================================="
echo ""
