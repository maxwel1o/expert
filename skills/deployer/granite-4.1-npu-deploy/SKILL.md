---
name: granite-4.1-npu-deploy
description: >
  IBM Granite 4.1 系列大语言模型（3B / 8B / 30B）在华为昇腾 NPU 上的
  一键式部署与推理 Skill。基于 transformers + torch_npu + transfer_to_npu
  自动迁移，无需修改模型源码。涵盖环境准备、单卡/多卡推理、验证步骤
  和常见问题排查。当用户提到 Granite 4.1 昇腾部署、Granite NPU 推理、
  ibm-granite 模型适配时触发。
metadata:
  short-description: Granite 4.1 昇腾 NPU 部署与推理
  category: NPU-Model-Deploy
  tags: [ascend, npu, granite, ibm, llm, transformers, inference]
---

# IBM Granite 4.1 昇腾 NPU 部署与推理 Skill

本 Skill 提供 IBM Granite 4.1 系列（3B / 8B / 30B）稠密 Transformer 模型
在华为昇腾 NPU 上的标准化可复现部署流程。模型架构为 `GraniteForCausalLM`
（transformers 原生支持），无自定义 CUDA 算子，采用轻量级脚本化迁移。

## 前置条件

### 硬件与软件需求

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend910 系列（3B/8B 单卡即可，30B 需 2–4 卡） |
| OS | openEuler / Ubuntu / KylinOS（aarch64 或 x86_64） |
| CANN | >= 8.0（推荐 8.5.1） |
| Python | 3.9 – 3.11 |
| 依赖 | torch, torch_npu, transformers, accelerate |

### 磁盘空间

1. 确认磁盘空间 >= 120GB（30B 权重约 60GB + 8B 约 16GB + 3B 约 6GB + 临时空间）
2. 推理日志与输出约需 5GB

### 前置条件确认

- [确认] 使用 `npu-smi info` 确认 NPU 驱动已加载，至少一张卡状态为 Normal
- [确认] 使用 `python3 --version` 确认 Python 版本在 3.9–3.11 之间
- [确认] 使用 `cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg` 确认 CANN 版本 >= 8.0
- [确认] 磁盘剩余空间 >= 120GB（若下载全部三个模型）
- [确认] 如需运行 30B 模型：确认 `npu-smi info` 显示至少 2 张卡

若任一前置检查失败，需先解决环境问题再继续后续步骤。

## 模型规格速查

| 规格 | 3B | 8B | 30B |
|------|-----|-----|------|
| 参数量 | 3B | 8B | 30B |
| 隐藏维度 | 2560 | 4096 | 4096 |
| 层数 | 40 | 40 | 64 |
| Attention Heads | 20 | 32 | 32 |
| KV Heads (GQA) | 8 | 8 | 8 |
| Intermediate Size | 10240 | 12800 | 32768 |
| 上下文长度 | 131072 | 131072 | 131072 |
| 原生精度 | bfloat16 | bfloat16 | bfloat16 |
| 权重显存 | ~6 GB | ~16 GB | ~60 GB |
| 最低卡数 | 1 | 1 | 2（推荐 4） |

## 工作流程

### 流程总览

```
0. 环境初始化
→ 用户确认：NPU 状态正常
→ 1. 安装依赖
→ 用户确认：依赖安装成功
→ 2. NPU 基础验证
→ 用户确认：NPU 可用
→ 3. 下载模型权重
→ 4. 推理运行（单卡 / 多卡）
→ 用户确认：推理输出正常
→ 5. 验证输出
→ 6. 验收确认
```

按以下步骤顺序执行。**每步完成后检查对应通过标准**，不通过则排查修复或回退至上一步。

---

## 0. 环境初始化

```bash
# 加载 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 查看 NPU 状态
npu-smi info

# 设置设备可见性（单卡示例）
export ASCEND_RT_VISIBLE_DEVICES=0

# 多卡示例（30B 必需）
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

# HuggingFace 镜像（国内加速）
export HF_ENDPOINT=https://hf-mirror.com
```

### 错误处理

- 若 `source set_env.sh` 失败：确认 Ascend Toolkit 已安装，路径为 `/usr/local/Ascend/ascend-toolkit/set_env.sh`
- 若 `npu-smi info` 报错：确认驱动已安装，执行 `npu-smi -v` 查看驱动版本
- 若多卡环境下某些卡状态为 Abnormal：重启 NPU 驱动 `npu-smi reset -t <device_id>`（需 root 权限）
- 若 `HF_ENDPOINT` 设置后下载仍慢：检查网络连通性 `curl -I https://hf-mirror.com`

### Checkpoint：环境初始化通过标准

- [确认] `npu-smi info` 显示至少一张卡状态为 Normal
- [确认] `echo $ASCEND_RT_VISIBLE_DEVICES` 输出正确卡号
- [确认] 网络可访问 hf-mirror.com 或 huggingface.co

---

## 1. 安装依赖

```bash
pip install torch torch_npu transformers accelerate -i https://repo.huaweicloud.com/repository/pypi/simple/
```

版本要求：
- `torch` == `torch_npu`（如 2.9.0）
- `transformers` >= 4.50
- `accelerate` >= 0.20.0

### 错误处理

- 若 `pip install torch_npu` 失败：确认 CANN 版本与 torch_npu 版本匹配表
- 若 `accelerate` 未安装导致 `device_map` 报错：`pip install accelerate`
- 若版本冲突：创建独立虚拟环境 `python3 -m venv granite-env && source granite-env/bin/activate` 后重试
- 若 `ValueError: Using a device_map requires accelerate`：确认 accelerate 已安装且版本 >= 0.20.0

### Checkpoint：安装验证

- [确认] `python3 -c "import torch; import torch_npu; print(torch.__version__, torch_npu.__version__)"` 无报错
- [确认] `python3 -c "import transformers; import accelerate"` 无报错
- [确认] torch 和 torch_npu 版本一致

---

## 2. NPU 基础验证

```bash
python3 -c "
import torch
import torch_npu
print('torch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
print('NPU available:', torch.npu.is_available())
print('Device:', torch.npu.get_device_name(0))
"
```

**通过标准**：`torch.npu.is_available()` 返回 `True`，且无报错。

### 错误处理

- 若 `torch.npu.is_available()` 返回 False：
  1. 检查 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 是否已执行
  2. 检查 `ASCEND_RT_VISIBLE_DEVICES` 是否设置且包含有效卡号
  3. 检查 `pip list | grep torch_npu` 确认 torch_npu 已安装
- 若 `Segmentation fault`：尝试升级 CANN 至 8.5.1 以上

### Checkpoint：NPU 验证确认

- [确认] `torch.npu.is_available()` 返回 True
- [确认] 设备名称为 Ascend910 系列
- [确认] `torch.npu.device_count()` 返回 >= 所需卡数

---

## 3. 下载模型权重

首次运行会自动从 HuggingFace 下载，建议预先配置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

权重大小：
- 3B：~6 GB
- 8B：~16 GB
- 30B：~60 GB

### 错误处理

- 若下载超时：检查 `HF_ENDPOINT` 设置，或使用 `export HF_HUB_ENABLE_HF_TRANSFER=1` 启用加速
- 若磁盘空间不足：清理 `~/.cache/huggingface/hub/` 中的缓存
- 若 `OOM` during download：下载过程不占用 NPU，检查 CPU 内存是否充足

### Checkpoint：权重下载确认

- [确认] 模型文件已成功下载到缓存目录
- [确认] 磁盘空间充足，可继续后续推理

---

## 4. 推理运行

### 4.1 模型加载

使用 `transfer_to_npu` 自动将模型迁移到 NPU：

1. 导入 `from torch_npu.contrib import transfer_to_npu`
2. 使用 `AutoTokenizer` 加载分词器
3. 使用 `AutoModelForCausalLM` 加载模型
4. 设置 `device_map="npu"` 自动完成设备映射
5. 使用 `torch.bfloat16` 精度加载

### 4.2 推理脚本

脚本文件位于 `scripts/inference.py`，核心逻辑：

```python
#!/usr/bin/env python3
"""Granite 4.1 NPU Inference Script"""

import torch_npu
from torch_npu.contrib import transfer_to_npu

import os
import sys
import time
import argparse
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def check_npu():
    if not torch.npu.is_available():
        logger.error("NPU is not available. Please check CANN environment.")
        sys.exit(1)
    npu_count = torch.npu.device_count()
    logger.info(f"NPU available: True, device count: {npu_count}")
    for i in range(npu_count):
        props = torch.npu.get_device_properties(i)
        logger.info(f"  NPU:{i} name={props.name}, mem={props.total_memory / 1024**3:.1f}GB")


def load_model(model_path: str, device: str = "npu"):
    logger.info(f"Loading model from: {model_path}")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    dtype = torch.bfloat16 if torch.npu.is_available() else torch.float32
    logger.info(f"Using dtype: {dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=device,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    elapsed = time.time() - start
    logger.info(f"Model loaded in {elapsed:.1f}s")
    logger.info(f"Model device: {next(model.parameters()).device}")
    return model, tokenizer


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 128):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
    logger.info(f"Prompt: {prompt!r}")
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    elapsed = time.time() - start
    generated_tokens = outputs[0].shape[0] - inputs["input_ids"].shape[1]
    logger.info(f"Generated {generated_tokens} tokens in {elapsed:.2f}s "
                f"({generated_tokens / elapsed:.1f} tok/s)")
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Granite 4.1 NPU Inference")
    parser.add_argument("--model", type=str, required=True, help="Model ID or local path")
    parser.add_argument("--prompt", type=str, default="def hello_world():")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="npu")
    parser.add_argument("--cache-dir", type=str, default=None)
    args = parser.parse_args()

    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    check_npu()
    model, tokenizer = load_model(args.model, device=args.device)
    result = generate_text(model, tokenizer, prompt=args.prompt, max_new_tokens=args.max_new_tokens)
    print("\n" + "=" * 60)
    print(result)
    print("=" * 60)
    logger.info("NPU inference completed successfully.")


if __name__ == "__main__":
    main()
```

### 4.3 运行命令

**Granite 4.1-3B（单卡）**：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
python3 scripts/inference.py \
  --model ibm-granite/granite-4.1-3b \
  --prompt "def fibonacci(n):" \
  --max-new-tokens 64 \
  --device npu
```

**Granite 4.1-8B（单卡）**：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
python3 scripts/inference.py \
  --model ibm-granite/granite-4.1-8b \
  --prompt "def fibonacci(n):" \
  --max-new-tokens 64 \
  --device npu
```

**Granite 4.1-30B（多卡）**：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
python3 scripts/inference.py \
  --model ibm-granite/granite-4.1-30b \
  --prompt "def fibonacci(n):" \
  --max-new-tokens 64 \
  --device npu
```

> 30B 模型约需 60GB 显存，单卡 32GB 必然 OOM，`device_map="npu"` 会自动将权重分布到多张卡。

### 错误处理

- 若 30B 模型 OOM：确认 `ASCEND_RT_VISIBLE_DEVICES` 至少包含 2 张卡（推荐 4 张），或使用 `--max-memory` 限制每卡显存
- 若 `device_map` 报错：确认 accelerate 已安装，版本 >= 0.20.0
- 若模型加载卡死：30B 模型首次加载约需 600s，耐心等待；若超过 30 分钟无响应，检查 `npu-smi info` 确认无死锁
- 若多卡加载时某卡异常：检查卡间互联 `npu-smi info -t` 确认拓扑连通
- 若生成结果为空白或乱码：检查 tokenizer 配置，确认 `pad_token_id` 设置正确
- 若 `trust_remote_code` 报错：升级 transformers 至 >= 4.50（Granite 4.1 已原生支持）

### Checkpoint：推理确认

- [确认] 模型加载成功，日志显示 `Model loaded`
- [确认] `model.generate()` 正常输出文本
- [确认] 生成内容语义通顺
- [确认] 吞吐量在预期范围内（3B ~11 tok/s, 8B ~8 tok/s, 30B ~5 tok/s）
- [确认] 无 `CUDA error` / OOM 等 NPU 相关报错

---

## 5. 验证输出

**通过标准**：
- 模型加载成功，无 `CUDA` 相关报错
- `model.generate()` 正常输出文本
- 生成内容语义通顺（代码续写 / 文本生成）

**预期日志示例（3B）**：

```text
NPU available: True, device count: 1
  NPU:0 name=Ascend910B4, mem=32.0GB
Loading model from: ibm-granite/granite-4.1-3b
Using dtype: torch.bfloat16
Model loaded in 271.0s
Model device: npu:0
Generated 64 tokens in 5.76s (11.1 tok/s)
```

### 错误处理

- 若生成结果重复/短路：降低 `temperature` 或调整 `top_p` 参数
- 若生成为空：检查 `max_new_tokens` 是否设置正确，确认输入 prompt 正常
- 若日志显示 `pad_token_id` 警告：设置为 `tokenizer.eos_token_id` 即可

### Checkpoint：验证确认

- [确认] 生成文本不包含乱码或异常重复
- [确认] 日志中的生成速度符合预期

---

## 6. 性能参考

测试条件：Ascend910B4，bfloat16，batch_size=1，greedy / sample 生成。

| 指标 | 3B | 8B | 30B (4卡) |
|------|-----|-----|-----------|
| 加载时间 | ~271 s | ~350 s | ~600 s |
| 生成吞吐 | ~11.1 tok/s | ~8.0 tok/s | ~5.3 tok/s |
| 权重显存 | ~6 GB | ~16 GB | ~60 GB |

---

## 7. 验收确认

### 检查清单

- [ ] `npu-smi info` 显示设备正常
- [ ] `import torch_npu` 无报错
- [ ] 3B/8B 单卡推理正常输出
- [ ] 30B 多卡推理正常输出（需 >=2 卡）
- [ ] 生成结果语义通顺

### 回退策略

若任一检查项未通过：
1. 记录失败项和错误信息到日志文件
2. 根据失败项定位到对应的步骤
3. 执行该步骤的故障排除流程
4. 修复后重新运行验收确认

### 最终输出

部署成功后的交付物：
1. 推理日志（`inference_logs/` 目录，含 `eval.json` 格式的基准数据）
2. 生成结果样本（`generated_samples/` 目录）
3. 性能基准数据（可汇总到 `results.tsv` 进行趋势分析）

### 参考资源

- `test-prompts.json`：预定义的测试 prompt 与验证用例
- `scripts/inference.py`：标准化的推理脚本
- 精度验证可参考 `references/` 目录下的配置存档

---

## 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `NPU is not available` | CANN 环境未加载 | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| `ValueError: Using a device_map requires accelerate` | 缺少 accelerate | `pip install accelerate` |
| 30B OOM | 单卡显存不足 | 设置 `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3` 多卡运行 |
| 模型下载慢/失败 | 网络问题 | `export HF_ENDPOINT=https://hf-mirror.com` |
| `trust_remote_code` 报错 | 旧版 transformers | 升级 `transformers >= 4.50` |
| 多卡加载卡死 | device_map 分配不均 | 确保卡间互联正常，或限制 `max_memory` |
| 生成结果重复 | temperature 过高 | 降低 temperature 至 0.3–0.7 |
| 模型加载超时 | 权重较大 | 30B 首次加载约 600s，耐心等待 |

---

## 附录：资源文件

| 文件 | 用途 | 路径 |
|------|------|------|
| 推理脚本 | 模型加载与生成 | `scripts/inference.py` |
| 测试用例 | 验证 prompt 与预期结果 | `test-prompts.json` |

---

## 附录：Granite 4.1 NPU 适配要点速查

| 特征 | Granite 4.1 值 | 对 NPU 的影响 |
|------|---------------|--------------|
| 架构 | `GraniteForCausalLM` | transformers 原生支持，无需自定义代码 |
| 注意力 | GQA (MQA-like) | 标准算子，无需特殊处理 |
| 精度 | bfloat16 | Ascend910B 原生支持 bf16，无需转换 |
| 自定义 CUDA | 无 | `transfer_to_npu` 即可自动迁移 |
| RoPE | 标准 | 无需替换 |
| 激活函数 | SiLU | 原生支持 |
| 设备映射 | `device_map="npu"` | 自动完成多卡权重分布 |
