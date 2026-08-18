---
name: paddleocr-table-npu
description: >
  PaddleOCR 表格识别全链路 4 模型在昇腾 NPU 上的端到端部署与验证 Skill。
  覆盖版面检测(PP-DocLayout_plus-L)、印章文字检测(PP-OCRv4_server_seal_det)、
  表格分类(PP-LCNet_x1_0_table_cls)、表格单元格检测(RT-DETR-L_wired_table_cell_det)。
  支持自动下载、Paddle→ONNX→ATC→OM→ais_bench 推理、性能测试。
  当用户提到 PP-DocLayout、PP-OCRv4、PP-LCNet、RT-DETR、文档版面检测、
  印章检测、表格分类、表格单元格检测、昇腾 NPU、PaddleOCR 表格、OM 推理时触发。
metadata:
  short-description: PaddleOCR 表格全链路 4 模型昇腾 NPU 部署与验证
  category: NPU-Model-Deploy
  tags: [ascend, npu, ocr, table, pp-doclayout, seal-det, table-cls, table-cell, onnx, paddle]
---

# PaddleOCR 表格全链路 NPU 部署 Skill

## Skill Metadata

- **Skill Name**: PaddleOCR Table Ascend NPU Deployment
- **Version**: 1.0.0
- **Category**: Deployment / OCR / Ascend NPU
- **Target Platform**: Huawei Ascend NPU (310P3 / 910B3 系列)
- **Framework**: ATC + ais_bench 0.0.2 + PaddleX 3.3.13

## Skill Goal

自动完成 PaddleOCR 表格识别全链路 4 个模型在华为昇腾 NPU 上的部署、推理和验证。

4 个模型串行执行,每个模型跑完释放 NPU 显存,防溢出。

## Supported Models

| 模型名称 | 任务类型 | 输入尺寸 | 输入节点 | ModelScope ID |
| --- | --- | --- | --- | --- |
| **PP-DocLayout_plus-L** | 文档版面区域检测(RT-DETR-L) | 800x800 | 3 (im_shape/image/scale_factor) | `PaddlePaddle/PP-DocLayout_plus-L` |
| **PP-OCRv4_server_seal_det** | 印章文本检测(DBNet) | 640x640 | 1 (x) | `PaddlePaddle/PP-OCRv4_server_seal_det` |
| **PP-LCNet_x1_0_table_cls** | 表格图像分类(LCNet, 有线/无线) | 224x224 | 1 (x) | `PaddlePaddle/PP-LCNet_x1_0_table_cls` |
| **RT-DETR-L_wired_table_cell_det** | 表格单元格检测(RT-DETR-L) | 640x640 | 3 (im_shape/image/scale_factor) | `PaddlePaddle/RT-DETR-L_wired_table_cell_det` |

## Environment Requirements

- **Hardware**: Huawei Ascend NPU 310P3 / 910B3 系列
- **CANN**: 8.3.RC2 / 8.5.x
- **Python**: 3.11
- **PaddleX**: **强制 3.3.13**(仓里所有 patch / predictor.py 均按 3.3.x 写)
- **Key Dependencies**:
  - `paddlex[ocr]==3.3.13`, `paddlepaddle==3.2.2`, `paddleocr==3.3.2`
  - `paddle2onnx==2.1.0`, `onnx==1.16.1`
  - `ais_bench==0.0.2`, `aclruntime==0.0.2`, `msit`
  - `opencv-contrib-python==4.10.0.84`, `libgl1`
  - `modelscope`

## Skill Architecture

```
paddleocr-table-npu/
├── SKILL.md                     # 本文档
├── test-prompts.json            # 4 个测试 prompt
├── examples/
│   └── run_all_models.sh        # 串行跑 4 个模型的 shell 脚本
└── scripts/
    ├── model_configs.py         # 4 个模型配置表
    └── models/
        ├── doclayout/
        │   └── run_inference.py  # 跟仓里 infer.py 一样,调 paddleocr.LayoutDetection
        ├── seal_det/
        │   └── run_inference.py  # 跟仓里 infer.py 一样,调 paddlex create_predictor
        ├── table_cls/
        │   └── run_inference.py  # 跟仓里 infer.py 一样,调 ClasPredictor
        └── table_cell/
            └── run_inference.py  # 跟仓里 infer.py 一样,调 DetPredictor
```

**4 个 run_inference.py 严格按仓里 infer.py 风格**——只调 PaddleX 框架(LayoutDetection / create_predictor / ClasPredictor / DetPredictor),pre/post 由 PaddleX 从 `inference.yml` 读,不在 skill 里写自定义处理。

## Quick Start

### 1. 环境准备(一次性,4 个模型共用)

```bash
# 降级 PaddleX(关键!)
pip install --no-deps "paddlepaddle==3.2.2" "paddlex[ocr]==3.3.13" "paddleocr==3.3.2" "paddle2onnx==2.1.0"
pip install "onnx==1.16.1" modelscope "opencv-contrib-python==4.10.0.84"

# 系统级
apt-get update && apt-get install -y libgl1

# NPU 工具链
pip install msit && msit install surgeon
wget https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/ais_bench-0.0.2-py3-none-any.whl
wget https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/aclruntime-0.0.2-cp311-cp311-linux_aarch64.whl
pip install ais_bench-0.0.2-py3-none-any.whl aclruntime-0.0.2-cp311-cp311-linux_aarch64.whl

# 激活 CANN
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 2. 单个模型转换(以 doclayout 为例)

```bash
# 拉仓 + 拉权重
git clone https://atomgit.com/Ascend-SACT/PP-DocLayout_plus-L.git
modelscope download --model PaddlePaddle/PP-DocLayout_plus-L --local_dir /workspace/PP-DocLayout_plus-L

# Paddle -> ONNX
paddlex --paddle2onnx --paddle_model_dir /workspace/PP-DocLayout_plus-L --onnx_model_dir /workspace/PP-DocLayout_plus-L

# 改图优化
python3 -m auto_optimizer optimize \
    /workspace/PP-DocLayout_plus-L/inference.onnx \
    /workspace/PP-DocLayout_plus-L/inference_opt.onnx

# 静态化 batch -> 1
python3 -c "from shared.npu_om_utils import onnx_staticize_batch; \
    onnx_staticize_batch('/workspace/PP-DocLayout_plus-L/inference_opt.onnx', \
    '/workspace/PP-DocLayout_plus-L/inference_static.onnx')"

# ATC 转 OM
soc_version=$(npu-smi info | awk '/Name/{print $4; exit}')
atc --model=/workspace/PP-DocLayout_plus-L/inference_static.onnx \
    --framework=5 \
    --output=/workspace/PP-DocLayout_plus-L/inference_linux_aarch64 \
    --soc_version=${soc_version} \
    --input_shape="im_shape:1,2;image:1,3,800,800;scale_factor:1,2" \
    --log=info
```

### 3. 推理

```bash
cd /workspace/.claude/skills/paddleocr-table-npu
python3 scripts/models/doclayout/run_inference.py \
    --om /workspace/PP-DocLayout_plus-L/inference_linux_aarch64.om \
    --image /workspace/PP-DocLayout_plus-L-sact/test_Doclayout_plus.png \
    --score_thresh 0.3
```

### 4. 批量跑 4 个模型

```bash
chmod +x examples/run_all_models.sh
bash examples/run_all_models.sh
```

## PaddleX 版本兼容矩阵

| PaddleX 版本 | doclayout patch | seal_det cp | table_cls cp | table_cell patch | 建议 |
| --- | --- | --- | --- | --- | --- |
| **3.3.13**(推荐) | ✅ 4 hunk 全 apply | ✅ | ✅ | ✅ | 全程少改 site-packages |
| 3.7.x | ⚠️ 1 hunk,需手补 3 处 | ❌ port error | ❌ port error | ⚠️ 同左 | **不推荐** |

## 故障排查 / 速查

| 现象 | 解决 |
| --- | --- |
| `paddlex` 用 3.7.x 报 `ModuleNotFoundError: No module named 'paddlex.inference.models.base'` | **降级到 3.3.13**(`pip install paddlex[ocr]==3.3.13 paddlepaddle==3.2.2`) |
| paddle2onnx 装 onnxoptimizer 编译失败 | `pip install --no-deps paddle2onnx==2.1.0` |
| ATC 报 `opset::7::Conv` 没有 plugin | table_cls 默认 opset 7,用 `onnx_upgrade_opset` 升到 11+ |
## 故障排查 / 速查

| 现象 | 解决 |
| --- | --- |
| `paddlex` 报 `ModuleNotFoundError: No module named 'paddlex.inference.models.base'` | **降级到 3.3.13**(`pip install paddlex[ocr]==3.3.13 paddlepaddle==3.2.2`) |
| paddle2onnx 装 onnxoptimizer 编译失败 | `pip install --no-deps paddle2onnx==2.1.0` |
| ATC 报 `opset::7::Conv` 没有 plugin | table_cls 默认 opset 7,**转 OM 前升级到 11+**(用 `onnx.version_converter`) |
| OM 推理 `neesize:0 not match` | ONNX batch 是 -1 导致,**把 batch 维改 1,h/w 重命名**(ONNX 静态化) |
| `table_cls` 跑时报 `axis 1 is out of bounds for array of dimension 1` | 仓里 predictor.py 第 128 行的 `self.om_sess.infer(x, mode='dymshape', custom_sizes=100000000)` 在 ais_bench 0.0.2 + NPU 上输出 0 维假数据。**手动把 site-packages 里的 `image_classification/predictor.py` 第 128 行改为 `self.om_sess.infer(x)`**(去掉 `mode='dymshape', custom_sizes=100000000`) |
| libGL.so.1 报错 | `apt-get install -y libgl1` + `opencv-contrib-python==4.10.0.84` |
| cv2 IMREAD_COLOR 缺失 | 装 4.10.0.84(4.9 缺这个属性) |
| NPU 0 OOM | 改仓里 `predictor.py` 里的 `InferSession(0, ...)` 为空闲 device id,然后重新打 patch / cp |

## 注意事项

- **PaddleX 3.3.13 是硬要求**,3.7.x 目录结构重构了,仓里所有 patch / predictor.py 均不兼容
- **OM 文件名必须是 `inference_linux_aarch64.om`**(不是 `arch64`),跟仓里 predictor.py 写死的路径对齐
- **严格按仓里 README 路径**——4 个 `run_inference.py` 都直接调 PaddleX 框架(LayoutDetection / create_predictor / ClasPredictor / DetPredictor),pre/post 由 PaddleX 从 `inference.yml` 读,**不在 skill 里写自定义处理**
- **CANN/驱动/Py 强绑定**:不自行升级版本
- **整合 skill 的 4 个 run_inference.py 跟单 skill 完全等价**——单 skill 4 份 + 整合 1 份,内容一致,选哪个用都一样
