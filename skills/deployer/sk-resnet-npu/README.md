# SKNet NPU Deployment Skill

## 简介

本 Skill 用于在昇腾 Ascend910 NPU 上自动完成 SKNet（Selective Kernel Networks）系列图像分类模型的适配部署、推理验证、CPU/NPU 精度对比、终端截图生成和模型仓库发布。

## 支持的模型

| 模型名称 | 架构 | 参数量 | GitCode 仓库 |
| --- | --- | --- | --- |
| skresnext50_32x4d.ra_in1k | SKResNeXt50 | ~25M | [仓库链接](https://gitcode.com/m0_74196153/skresnext50_32x4d.ra_in1k-npu) |
| skresnet34.ra_in1k | SKResNet34 | ~22M | [仓库链接](https://gitcode.com/m0_74196153/skresnet34.ra_in1k-npu) |
| skresnet18.ra_in1k | SKResNet18 | ~12M | [仓库链接](https://gitcode.com/m0_74196153/skresnet18.ra_in1k-npu) |

## Skill 输入参数

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| model_name | string | 是 | timm 模型名称 |
| action | string | 否 | 执行操作（默认: all） |

支持的 action 值：
- `inference` — 运行 CPU 和 NPU 推理
- `compare` — 运行 CPU/NPU 精度对比
- `readme` — 生成 README 文档
- `screenshot` — 生成终端截图
- `publish` — 推送到 GitCode 模型仓库
- `all` — 执行全部操作

## Skill 输出结果

| 输出 | 格式 | 说明 |
| --- | --- | --- |
| inference_result | JSON | CPU/NPU 推理结果，含 Top-5 预测 |
| compare_result | JSON | CPU/NPU 精度对比结果 |
| screenshot | PNG | 模拟终端输出截图 |
| repo_url | URL | 模型仓库地址 |

## 环境要求

- **NPU**: Ascend910 (Atlas 800 A2)
- **CANN**: 8.5.1
- **Python**: 3.11+
- **PyTorch**: 2.9.0
- **torch_npu**: 2.9.0.post1
- **timm**: ≥0.9.0

## 执行 NPU 推理

### 方式一：Python 脚本

```bash
# 单个模型推理
python3 scripts/deploy.py --model skresnext50_32x4d.ra_in1k --action inference

# CPU 推理
python3 inference.py --model skresnext50_32x4d.ra_in1k --device cpu

# NPU 推理
python3 inference.py --model skresnext50_32x4d.ra_in1k --device npu
```

### 方式二：Shell 脚本

```bash
chmod +x scripts/run.sh
./scripts/run.sh skresnext50_32x4d.ra_in1k inference
```

## 执行 CPU/NPU 精度对比

```bash
python3 scripts/deploy.py --model skresnext50_32x4d.ra_in1k --action compare

# 或直接运行对比脚本
python3 compare_cpu_npu.py --model skresnext50_32x4d.ra_in1k
```

## 串行执行多个模型

为避免 NPU 显存爆炸，必须串行执行：

```bash
# 方式一：使用 deploy.py 串行模式
python3 scripts/deploy.py --serial --action all

# 方式二：使用串行脚本
bash examples/run_all_serial.sh

# 方式三：手动串行
python3 inference.py --model skresnext50_32x4d.ra_in1k --device npu
# 等待完成并释放资源
python3 -c "import gc; gc.collect(); import torch; torch.npu.empty_cache()"
python3 inference.py --model skresnet34.ra_in1k --device npu
# 等待完成并释放资源
```

## 生成 README

```bash
python3 scripts/deploy.py --model skresnext50_32x4d.ra_in1k --action readme
```

## 生成终端截图

```bash
python3 scripts/deploy.py --model skresnext50_32x4d.ra_in1k --action screenshot

# 或直接使用截图工具
python3 /opt/atomgit/terminal_screenshot.py \
    --input skresnext50_32x4d.ra_in1k_screenshot.txt \
    --output skresnext50_32x4d.ra_in1k-npu/terminal_screenshot.png
```

## 提交模型仓库

使用 GitCode API 创建模型仓库并推送代码：

```bash
# 创建仓库
curl -X POST "https://api.gitcode.com/api/v5/user/repos" \
  -H "private-token: ${ATOMGIT_USER_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"name": "{model_name}-npu", "description": "...", "visibility": "public"}'

# 推送代码
cd {model_name}-npu
git init
git checkout -b main
git add -A
git commit -m "Add {model_name} NPU adapted model"
git remote add origin https://auth:${ATOMGIT_USER_TOKEN}@gitcode.com/m0_74196153/{model_name}-npu.git
git push -u origin main
```

## 精度验证结果

所有三个模型在昇腾 NPU 上的推理结果与 CPU 一致，误差均 < 1%：

| 模型 | Max Prob Diff | Cosine Similarity | Top-1 一致 |
| --- | ---: | ---: | ---: |
| skresnext50_32x4d.ra_in1k | 0.039% | 0.999999 | ✓ |
| skresnet34.ra_in1k | 0.041% | 0.999999 | ✓ |
| skresnet18.ra_in1k | 0.112% | 0.999999 | ✓ |

## 资源释放

每个模型测试完成后必须释放资源：

```python
import gc
gc.collect()
if hasattr(torch, "npu"):
    torch.npu.empty_cache()
```

## 文件结构

```
skills/deployment/sk-resnet-npu/
├── skill.json           # Skill 元数据
├── README.md            # 本文件
├── scripts/
│   ├── deploy.py        # Python 部署入口
│   └── run.sh           # Shell 部署脚本
└── examples/
    ├── example_usage.md # 使用示例
    └── run_all_serial.sh# 串行运行脚本
```
