# Quick Start

## 1. Environment Setup

```bash
# Install dependencies
pip install torch torch_npu transformers safetensors Pillow numpy modelscope

# Check NPU
python3 -c "import torch; import torch_npu; print('NPU available:', torch.npu.is_available())"
```

## 2. Download Models

```python
from modelscope import snapshot_download
models = [
    "cubeai/cv_level1_protected_animals_classification",
    "cubeai/67_cat_breeds_image_detection",
    "cubeai/brain_model",
    "cubeai/133_dog_breeds_image_detection",
    "cubeai/bird_species_image_detection",
    "cubeai/bug_classifier",
    "cubeai/cv_edible_wild_plants_classification",
    "cubeai/cv_forest_pest_detection",
    "cubeai/100_butterfly_types_image_detection",
    "cubeai/215_mushroom_types_image_detection",
    "cubeai/birds_transform_full",
]
for m in models:
    snapshot_download(m)
```

## 3. Run Inference

```bash
# CPU
python scripts/run_inference.py \
  --model-path /path/to/model_dir \
  --image /path/to/image.jpg \
  --device cpu

# NPU
python scripts/run_inference.py \
  --model-path /path/to/model_dir \
  --image /path/to/image.jpg \
  --device npu
```

## 4. Compare Precision

```bash
python scripts/run_compare.py \
  --model-path /path/to/model_dir \
  --image /path/to/image.jpg
```

## 5. Batch Run All Models

```bash
bash scripts/run_all.sh
```
