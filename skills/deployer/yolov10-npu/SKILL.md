---
name: yolov10-npu
description: >
  YOLOv10n 目标检测模型在华为昇腾 Ascend NPU 上的完整适配、推理与多阶段性能优化 Skill。
  涵盖：模型加载修复（weights_only=False）、NPU 推理适配（model.to('npu')）、FP16 半精度、
  torch.compile NPU 后端加速、运行时环境优化（TASK_QUEUE_ENABLE/CPU_AFFINITY）、
  NPU 后处理（避免 CPU 传输）、NPU 图像缩放（F.interpolate）、流式流水线（双缓冲线程预取）、
  精度验证（CPU vs NPU 输出一致性）及综合性能基准测试。从 3.2 FPS（CPU）优化至 69.5 FPS（NPU+stream），
  加速 21.5 倍。触发场景：YOLOv10 + Ascend/NPU 适配、YOLO + torch_npu 推理、
  目标检测 + 昇腾性能优化、视觉模型 + torch.compile + NPU。
metadata:
  short-description: YOLOv10n 昇腾 NPU 推理适配与优化
  category: NPU-Model-Optimization
  tags: [ascend, npu, yolov10, object-detection, pytorch, inference, optimization, torch-compile, fp16, stream-pipeline]
  hardware: [Ascend910]
  benchmark_single_fps: 60.2
  benchmark_stream_fps: 69.5
  speedup_vs_cpu: 21.5
compatibility: >
  Python 3.10+, PyTorch 2.3.1, torch_npu 2.3.1.post1,
  CANN 8.0.RC3, Ascend 910_9362 NPU.
license: AGPL-3.0
---

# YOLOv10n Ascend NPU 推理适配与性能优化

## TL;DR

将 YOLOv10n（260 万参数，COCO 80 类）从 CPU 迁移到华为昇腾 910 NPU。通过 6 阶段优化：

| 阶段 | 操作 | E2E FPS |
|:----|:----|:-------:|
| 0 | CPU FP32 基线 | 3.2 |
| 1 | NPU FP16 推理 | 49.0 |
| 2 | + torch.compile | 57.0 |
| 3 | + 运行时环境优化 | 60.2 |
| 4 | + NPU 后处理 | 微小提升 |
| 5 | + NPU 图像缩放 | 预处理降 2ms |
| **6** | **+ 流式流水线** | **69.5** |

**最终**: 端到端 14.38ms/图, 69.5 FPS, 相比 CPU 基线加速 **21.5×** 🏆

---

## 前置条件

- **硬件**: Ascend 910/910B NPU（至少 1 卡空闲）
- **软件**: 
  - CANN 8.0.RC3+（`source /usr/local/Ascend/ascend-toolkit/set_env.sh`）
  - Python 3.10+
  - PyTorch 2.3.1 + torch_npu 2.3.1.post1
  - `pip install pillow numpy torchvision`
- **模型权重**: [yolov10n.pt](https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt)
- **测试图像**: 自行准备或使用 COCO 验证集

---

## 流程总览

```
Step 0: 环境初始化 → 检查 NPU, 加载 CANN
Step 1: 安装依赖与权重 → pip install, 下载 yolov10n.pt
Step 2: 基础 NPU 推理 → model.to('npu'), 单图推理验证
Step 3: FP16 精度 → model.half(), 精度对比
Step 4: torch.compile → model = torch.compile(model, backend='npu')
Step 5: 运行时优化 → TASK_QUEUE_ENABLE, CPU_AFFINITY 等
Step 6: NPU 后处理 → 所有后处理在 NPU 上完成
Step 7: NPU 图像缩放 → F.interpolate 替代 cv2.resize
Step 8: 流式流水线 → 线程预取+双缓冲
Step 9: 精度验证 → CPU vs NPU 检测框对比
Step 10: 基准测试 → 完整 9 组配置性能测试
```

---

## Step 0: 环境初始化

```bash
# 0a. 加载 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 0b. 检查 NPU 可用性
npu-smi info

# 0c. 选择空闲卡（假设 npu:0 空闲）
export ASCEND_RT_VISIBLE_DEVICES=0

# 0d. 验证 torch_npu
python -c "
import torch
import torch_npu
print(f'torch={torch.__version__}, torch_npu={torch_npu.__version__}')
print(f'NPU available: {torch.npu.is_available()}')
print(f'NPU count: {torch.npu.device_count()}')
print(f'NPU name: {torch.npu.get_device_name(0)}')
"
```

**通过标准**: `NPU available: True`, 卡数 >= 1。

---

## Step 1: 安装依赖与权重

```bash
# 1a. 安装依赖
pip install pillow numpy torchvision

# 1b. 下载 YOLOv10n 权重
wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt -O yolov10n.pt
```

---

## Step 2: 基础 NPU 推理

创建 `infer_npu.py`:

```python
import torch
import torch_npu
from ultralytics import YOLO

model = YOLO("yolov10n.pt")
model.to("npu:0")
results = model("test.jpg")
results[0].show()
```

**通过标准**: 正常输出检测框（类别、置信度、坐标），无 NPU 相关错误。

---

## Step 3: FP16 半精度优化

```python
model = YOLO("yolov10n.pt").to("npu:0")
model.model.half()
```

验证精度:

```bash
python -c "
import torch, torch_npu
from PIL import Image
from ultralytics import YOLO

model_cpu = YOLO('yolov10n.pt')
model_npu = YOLO('yolov10n.pt').to('npu:0')
model_npu.model.half()

img = Image.open('test.jpg')
r_cpu = model_cpu(img)
r_npu = model_npu(img)

diff = (r_cpu[0].boxes.data - r_npu[0].boxes.data.cpu()).abs().max().item()
print(f'Detection box max diff: {diff:.6f}')
"
```

---

## Step 4: torch.compile NPU 后端加速

```python
model = YOLO("yolov10n.pt").to("npu:0")
model.model.half()
model.model = torch.compile(model.model, backend="npu")

# warmup (first compile ~20s)
for _ in range(3):
    _ = model("test.jpg")
```

**性能预期**: 推理延迟 2.78-2.89ms (346-360 FPS), E2E 60+ FPS。

---

## Step 5: 运行时环境变量优化

```bash
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export COMBINED_ENABLE=1
export ACL_OP_COMPILER_CACHE_MODE=1
export ACL_OP_COMPILER_CACHE_DIR=/tmp/npu_op_cache
mkdir -p /tmp/npu_op_cache
```

也可使用 `scripts/optimize_npu.sh`。

---

## Step 6: NPU 后处理

将 NMS/Box decode 放在 NPU 上执行，避免大张量 CPU 传输。

```python
# NPU 上执行后处理
preds = preds.transpose(-1, -2)
bboxes, scores, labels = ops.v10postprocess(preds, max_det, nc)
bboxes = ops.xywh2xyxy(bboxes)
filtered_cpu = filtered.float().cpu()  # 只有最终小结果
```

---

## Step 7: NPU 图像缩放

```python
img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to("npu:0")
img_tensor = F.interpolate(img_tensor, size=(640, 640))
```

---

## Step 8: 流式流水线

```python
from queue import Queue
import threading

def predict_stream(self, images):
    q = Queue(maxsize=2)
    n = len(images)

    def _prefetch():
        for i in range(n):
            q.put(self.preprocess(images[i]))
        q.put(None)

    threading.Thread(target=_prefetch, daemon=True).start()

    for i in range(n):
        item = q.get()
        if item is None:
            break
        tensor, shape, pad = item
        with torch.no_grad():
            preds = self.model(tensor)
        yield self.postprocess(preds, shape, pad)
```

---

## Step 9: 精度验证

```bash
python -c "
import torch, torch_npu
from PIL import Image
from ultralytics import YOLO

for img_path in ['test1.jpg', 'test2.jpg']:
    img = Image.open(img_path)
    m_cpu = YOLO('yolov10n.pt')
    r_cpu = m_cpu(img)

    m_npu = YOLO('yolov10n.pt').to('npu:0')
    m_npu.model.half()
    m_npu.model = torch.compile(m_npu.model, backend='npu')
    r_npu = m_npu(img)

    diff = (r_cpu[0].boxes.data - r_npu[0].boxes.data.cpu()).abs().max().item()
    print(f'{img_path}: max_diff={diff:.6f}')
    assert diff < 0.01
print('All precision checks passed!')
"
```

---

## Step 10: 基准测试

```bash
python scripts/yolov10_npu_benchmark.py --model yolov10n.pt --runs 50
```

**预期结果 (Ascend 910_9362)**:

| 配置 | E2E(ms) | FPS |
|:----|:-------:|:---:|
| CPU FP32 | 309.73 | 3.2 |
| NPU FP32 | 20.67 | 48.4 |
| NPU FP16+compile | 17.53 | 57.0 |
| + 运行时优化 | 16.60 | 60.2 |
| **+ 流式流水线 🏆** | **14.38** | **69.5** |

---

## 常见问题

### Q1: torch.load weights_only 错误

**修复**: `torch.load(weights, weights_only=False)`, 已在脚本中内置。

### Q2: 首次 torch.compile 慢

首次约 20s。预热 3 次后消除。设置 `ACL_OP_COMPILER_CACHE_MODE=1` 可缓存。

### Q3: NPU 内存不足

```python
torch.npu.empty_cache()  # 清理缓存
# 或使用 FP16 减少显存
```
