#!/bin/bash
# PP-OCRv3 NPU Skill Usage Example

# 1. 环境准备
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  onnxruntime-cann numpy opencv-python-headless onnx modelscope

# 2. 下载所有模型
python scripts/download_models.py

# 3. 批量处理所有模型 (下载 + 推理 + 对比 + README)
python scripts/process_all_models.py

# 4. 查看结果
cat all_results.json

# 5. 单个模型推理示例
python scripts/inference_det.py \
  --model /path/to/model.onnx \
  --provider CANNExecutionProvider \
  --image test.png \
  --output result.png

# 6. 单个模型精度对比
cd <model_dir> && python compare_cpu_npu.py
