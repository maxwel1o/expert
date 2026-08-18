# Xception NPU Deployment Skill

## 概述

本 Skill 用于自动完成 **Xception 系列模型** 在华为昇腾 NPU（Ascend910B）上的部署、推理验证、精度测试、README 生成、终端截图生成以及 GitCode 模型仓库发布。

## 支持的模型列表

| # | 模型名称 | 参数量 | 输入尺寸 | GitCode 仓库 |
|---|---------|--------|---------|-------------|
| 1 | xception71.tf_in1k | 42.3M | 299×299 | [仓库](https://gitcode.com/m0_74196153/xception71.tf_in1k-npu) |
| 2 | xception65p.ra3_in1k | 39.8M | 299×299 | [仓库](https://gitcode.com/m0_74196153/xception65p.ra3_in1k-npu) |
| 3 | xception65.tf_in1k | 39.8M | 299×299 | [仓库](https://gitcode.com/m0_74196153/xception65.tf_in1k-npu) |
| 4 | xception65.ra3_in1k | 39.8M | 299×299 | [仓库](https://gitcode.com/m0_74196153/xception65.ra3_in1k-npu) |
| 5 | xception41p.ra3_in1k | 25.6M | 299×299 | [仓库](https://gitcode.com/m0_74196153/xception41p.ra3_in1k-npu) |
| 6 | xception41.tf_in1k | 25.6M | 299×299 | [仓库](https://gitcode.com/m0_74196153/xception41.tf_in1k-npu) |

## Skill 输入参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model_name | string | - | 模型名称（如 xception71.tf_in1k） |
| device | string | npu:0 | 推理设备（cpu / npu:0） |
| batch_size | int | 1 | 批处理大小 |
| skip_accuracy | bool | false | 是否跳过精度测试 |
| skip_push | bool | false | 是否跳过 GitCode 推送 |

## Skill 输出结果

- `inference.py`：模型推理脚本
- `compare_cpu_npu.py`：CPU/NPU 精度对比脚本
- `requirements.txt`：依赖清单
- `readme.md`：详细中文 README（含真实测试数据）
- `screenshot.html`：模拟终端输出截图
- `inference.log`：完整推理日志

## 环境要求

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision timm Pillow safetensors modelscope
```

昇腾 NPU 环境要求：
- torch_npu >= 2.0.0
- CANN 工具包
- Ascend910B 或兼容 NPU

## 执行 NPU 推理

每个模型目录包含独立的推理脚本：

```bash
cd /path/to/model_dir
python3 inference.py
```

脚本会自动检测 NPU 设备。通过传参指定设备：

```python
# 在 inference.py 中调用
result = run_inference(device="npu:0")  # NPU 推理
result = run_inference(device="cpu")    # CPU 推理
```

## 执行 CPU/NPU 精度对比

```bash
python3 compare_cpu_npu.py
```

该脚本会依次执行：
1. CPU 推理（记录输出和耗时）
2. NPU 推理（记录输出和耗时）
3. 对比 logits 和概率分布
4. 计算 cosine similarity、MAE、MaxAE
5. 检查 Top-1/Top-5 类别一致性
6. 验证概率误差 < 1%

## 串行执行多个模型（避免显存爆炸）

```bash
python3 scripts/run_all_serial.py
```

该脚本会串行遍历所有模型，每个模型执行完成后释放 NPU 显存和 CPU 内存，再处理下一个。

## 生成 README

```bash
python3 scripts/generate_readme.py --model_name xception71.tf_in1k --output readme.md
```

## 生成终端截图

```bash
python3 scripts/generate_screenshot.py --log inference.log --output screenshot.html
```

## GitCode 仓库发布

```bash
python3 scripts/publish_to_gitcode.py --model_name xception71.tf_in1k
```

## 模型仓库地址

所有模型仓库位于：https://gitcode.com/m0_74196153/{model_name}-npu

使用以下方式推送：

```bash
git remote add origin https://auth:${ATOMGIT_USER_TOKEN}@gitcode.com/m0_74196153/{model_name}-npu.git
git push -u origin main
```

## 验证结果摘要

| 模型 | CPU 延迟 | NPU 延迟 | 加速比 | 概率误差 |
|------|---------|---------|--------|---------|
| xception71.tf_in1k | 883.73ms | 9.56ms | 92.46× | 0.004% |
| xception65p.ra3_in1k | 730.63ms | 6.83ms | 106.99× | 0.00008% |
| xception65.tf_in1k | 690.07ms | 8.68ms | 79.50× | 0.015% |
| xception65.ra3_in1k | 750.09ms | 8.89ms | 84.37× | 0.0002% |
| xception41p.ra3_in1k | 476.78ms | 4.07ms | 117.10× | 0.005% |
| xception41.tf_in1k | 461.68ms | 5.73ms | 80.52× | 0.034% |

所有模型 NPU 与 CPU 推理结果误差 **< 1%**，满足精度要求。
