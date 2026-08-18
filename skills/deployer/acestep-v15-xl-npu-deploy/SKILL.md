---
name: acestep-v15-xl-npu-deploy
description: >
  ACE-Step v15 XL 系列音乐生成模型（4B DiT）在华为昇腾 NPU 上的完整部署与推理验证 Skill。
  涵盖环境准备、依赖安装、模型加载、NPU 推理验证、精度对比验证的全流程。
  支持 acestep-v15-xl-base、sft、turbo 三个模型变体。
  当用户提到 ACE-Step 部署昇腾、ACE-Step NPU 推理、音乐生成模型 NPU 适配时触发。
metadata:
  short-description: ACE-Step v15 XL 昇腾 NPU 部署与 bf16 推理验证
  category: NPU-Model-Deploy
  tags: [ascend, npu, ace-step, music-generation, dit, pytorch, inference, transformers, audio]
---

# ACE-Step v15 XL 昇腾 NPU 部署与 bf16 推理验证 Skill

本 Skill 提供 ACE-Step v15 XL 系列音乐生成模型在华为昇腾 NPU 上的
完整部署、推理验证和精度对比的标准化可复现流程。

支持的模型：

| 模型 | 参数量 | 推理步数 | CFG | 特点 |
|------|--------|----------|-----|------|
| `ACE-Step/acestep-v15-xl-base` | 4B | 50 | 是 | 支持全部任务，diversity 最高 |
| `ACE-Step/acestep-v15-xl-sft` | 4B | 50 | 是 | 音质更高，diversity 中等 |
| `ACE-Step/acestep-v15-xl-turbo` | 4B | 8 | 否 | 速度最快，无 CFG |

## 前置条件

### 硬件与软件需求

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend910 系列（建议 >= 24GB HBM） |
| OS | openEuler / Ubuntu / KylinOS（aarch64 或 x86_64） |
| CANN | >= 8.0（推荐 8.3.RC1 及以上） |
| Python | 3.9 – 3.13 |
| 网络 | 首次运行需联网下载模型权重 |

### 磁盘空间

1. 确认磁盘空间 >= 80GB
2. 模型缓存目录约需 18.8GB * 3 = 56.4GB
3. 评估日志与输出约需 5GB

### 前置条件确认

- [确认] 使用 `npu-smi info` 确认 NPU 驱动已加载，至少一张卡状态为 Normal
- [确认] 使用 `python3 --version` 确认 Python 版本 >= 3.9
- [确认] 使用 `cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg` 确认 CANN 版本 >= 8.0
- [确认] 磁盘剩余空间 >= 80GB（模型缓存约 18.8GB * 3 = 56.4GB + 临时空间）

若任一前置检查失败，需先解决环境问题再继续后续步骤。

## 工作流程

### 流程总览

```
0. 环境初始化
→ 用户确认：NPU 状态正常
→ 1. 安装依赖（torch_npu + transformers + vector_quantize_pytorch + einops）
→ 用户确认：依赖安装成功
→ 2. NPU 验证
→ 用户确认：NPU 可用
→ 3. 基础推理验证（training_loss + generate_audio）
→ 用户确认：推理结果合理
→ 4. 精度对比验证（NPU vs CPU，支持 --skip-cpu 跳过）
→ 5. 串行评测（多模型顺序执行，防止 OOM）
→ 6. 验收确认
```

按以下各节顺序执行，每步完成后再进入下一步。**每步执行后检查对应通过标准**，不通过则回退至上一步排查。

---

## 0. 环境初始化

```bash
# 加载 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 选择空闲 NPU（先用 npu-smi info 查看各卡占用，选空闲卡）
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=0   # 替换为实际空闲卡号

# 设置 ModelScope 镜像（国内加速）
export VLLM_USE_MODELSCOPE=true

# 华为 pip 镜像（国内加速）
export PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
```

### 错误处理

- 若 `source set_env.sh` 失败：确认 Ascend Toolkit 已安装，路径为 `/usr/local/Ascend/ascend-toolkit/set_env.sh`
- 若 `npu-smi info` 报错：确认驱动已安装，执行 `npu-smi -v` 查看驱动版本，检查 `npu-firmware` 状态
- 若所有卡均被占用：等待其他任务释放，或使用 `ps aux | grep python` 确认占用进程后协商释放
- 若 `npu-smi` 回显无 Normal 状态卡：重启 npu-smi 驱动 `npu-smi reset -t 0`（需 root 权限）

### Checkpoint：环境初始化通过标准

- [确认] `npu-smi info` 显示至少一张卡状态为 Normal
- [确认] `echo $ASCEND_RT_VISIBLE_DEVICES` 输出正确卡号
- [确认] 网络可访问 huggingface.co 或 modelscope.cn

---

## 1. 安装依赖

```bash
pip install torch torch_npu transformers vector-quantize-pytorch einops
```

安装完成后 `torch` 与 `torch_npu` 版本应一致（如均为 2.9.0）。

### 错误处理

- 若 `pip install torch_npu` 失败：确认 CANN 版本与 torch_npu 版本匹配表，参考 [Ascend PyTorch 版本对应关系](https://gitee.com/ascend/pytorch)
- 若 `vector-quantize-pytorch` 安装失败：尝试 `pip install git+https://github.com/lucidrains/vector-quantize-pytorch.git`
- 若 pip 超时：使用华为镜像 `pip install -i https://repo.huaweicloud.com/repository/pypi/simple/ torch torch_npu ...`
- 若版本冲突：创建独立虚拟环境 `python3 -m venv ace-step-env && source ace-step-env/bin/activate` 后重试

### Checkpoint：安装验证

- [确认] `python3 -c "import torch; import torch_npu; print(torch.__version__, torch_npu.__version__)"` 无报错
- [确认] torch 和 torch_npu 版本号一致

---

## 2. NPU 基础验证

运行以下 Python 代码确认 NPU 环境可用：

```bash
python3 -c "
import torch
import torch_npu
print('torch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
a = torch.randn(3, 4).npu()
print(a + a)
print('NPU device:', torch.npu.get_device_name(0))
"
```

**通过标准**：输出包含 `device='npu:0'` 的 Tensor 且无报错。

### 错误处理

- 若报错 `No module named 'torch_npu'`：
  1. 检查 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 是否已执行
  2. 检查 `pip list | grep torch_npu` 确认已安装
  3. 检查 CANN 版本与 torch_npu 是否匹配
  4. 尝试 `python3 -c "import sys; print(sys.path)"` 确认路径正确
- 若报错 `RuntimeError: NPU not available`：确认 `npu-smi info` 可看到设备且 `ASCEND_RT_VISIBLE_DEVICES` 设置正确
- 若报错 `Segmentation fault`：可能为 CANN/PyTorch 版本不匹配，尝试升级 CANN 至 8.3.RC1 以上

### Checkpoint：NPU 验证确认

- [确认] 无 ImportError/AttributeError
- [确认] 张量运算成功，输出 `device='npu:0'`
- [确认] `torch.npu.device_count()` 返回值 > 0

---

## 3. 基础推理验证

### 3.1 模型加载

使用 Hugging Face `transformers` 库加载模型权重和配置：

1. 调用 `AutoConfig.from_pretrained()` 加载模型配置
2. 调用 `AutoModel.from_pretrained()` 加载模型权重，使用 `trust_remote_code=True` 启用自定义代码
3. 设置 `torch_dtype=torch.bfloat16` 使用混合精度
4. 调用 `model.npu()` 将模型迁移到 NPU 设备

### 3.2 推理脚本

脚本文件位于 `scripts/inference.py`，核心逻辑：

```python
import torch
import torch_npu
from transformers import AutoConfig, AutoModel

model_path = "/path/to/acestep-v15-xl-base"
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_path,
    config=config,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
model = model.npu()
model.eval()

# 构造 dummy 输入（text2music 模式）
inputs = {
    "text_hidden_states": torch.randn(1, 77, 1024, dtype=torch.bfloat16, device="npu"),
    "text_attention_mask": torch.ones(1, 77, dtype=torch.bfloat16, device="npu"),
    "lyric_hidden_states": torch.randn(1, 123, 1024, dtype=torch.bfloat16, device="npu"),
    "lyric_attention_mask": torch.ones(1, 123, dtype=torch.bfloat16, device="npu"),
    "refer_audio_acoustic_hidden_states_packed": torch.randn(2, 750, 64, dtype=torch.bfloat16, device="npu"),
    "refer_audio_order_mask": torch.LongTensor([0, 0]).to("npu"),
    "src_latents": torch.randn(1, 250, 64, dtype=torch.bfloat16, device="npu"),
    "chunk_masks": torch.ones(1, 250, 64, dtype=torch.bfloat16, device="npu"),
    "is_covers": torch.tensor([0], dtype=torch.long, device="npu"),
}

# training_loss 前向验证
with torch.no_grad():
    outputs = model.training_losses(
        hidden_states=inputs["src_latents"],
        attention_mask=torch.ones(1, 250, dtype=torch.bfloat16, device="npu"),
        chunk_masks=inputs["chunk_masks"],
        silence_latent=torch.randn(1, 250, 64, dtype=torch.bfloat16, device="npu"),
        cfg_ratio=0.15,
        **inputs,
    )
    loss = outputs["diffusion_loss"]
    print(f"Loss: {loss.item():.6f}")

# generate_audio 生成验证
with torch.no_grad():
    gen_outputs = model.generate_audio(
        silence_latent=torch.randn(1, 250, 64, dtype=torch.bfloat16, device="npu"),
        seed=42,
        infer_steps=50,
        diffusion_guidance_scale=7.0,
        **inputs,
    )
    latents = gen_outputs["target_latents"]
    print(f"Generated latents shape: {latents.shape}")
```

### 3.3 运行推理

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1

# Base 模型（50步 + CFG）
python3 scripts/inference.py \
  --model-path ./model_cache/ACE-Step/acestep-v15-xl-base \
  --device npu --dtype bfloat16 --mode both \
  --infer-steps 50 --guidance-scale 7.0 --seed 42

# SFT 模型（50步 + CFG）
python3 scripts/inference.py \
  --model-path ./model_cache/ACE-Step/acestep-v15-xl-sft \
  --device npu --dtype bfloat16 --mode both \
  --infer-steps 50 --guidance-scale 7.0 --seed 42

# Turbo 模型（8步，无 CFG）
python3 scripts/inference.py \
  --model-path ./model_cache/ACE-Step/acestep-v15-xl-turbo \
  --device npu --dtype bfloat16 --mode both \
  --infer-steps 8 --guidance-scale 1.0 --seed 42
```

**通过标准**：
- `training_loss` 输出合理的 loss 值（Base ~0.4，SFT ~0.8，Turbo ~0.06）
- `generate_audio` 输出 shape 为 `(batch, seq_len, 64)` 的 latents
- 无 NPU 相关报错

### 错误处理

- 若 `training_loss` 返回 NaN：降低 learning rate 相关参数，或确认 bf16 精度模式下输入数据和模型均在 NPU 上
- 若 `generate_audio` 显存溢出（OOM）：减少 batch size、改用更少推理步数、或使用 `torch.npu.empty_cache()` 清理缓存
- 若 `trust_remote_code=True` 报安全警告：已通过 `AutoConfig.from_pretrained` 的白名单机制放行，可忽略
- 若模型下载失败：使用 `modelscope` 代替 huggingface，设置 `export VLLM_USE_MODELSCOPE=true` 或手动下载到 model_cache 目录
- 若 `vector_quantize_pytorch` 依赖缺失：`pip install vector-quantize-pytorch`，若仍缺失尝试重新安装 transformers
- 若 `SetPrecisionMode error 500001`：未加载 CANN 环境，重新执行 `source set_env.sh`

### Checkpoint：推理确认

- [确认] 三个模型的 loss 值均在预期范围内
- [确认] generate_audio 输出 shape 正确
- [确认] 无 RuntimeError / OOM / CUDA error（注意 NPU 错误类似 CUDA error 格式）

### 3.4 各模型预期性能

| 模型 | 推理步数 | 单次 loss 前向 | 显存占用 | 输出维度 |
|------|----------|---------------|----------|----------|
| acestep-v15-xl-base | 50 | ~2s | ~20GB | (B, T, 64) |
| acestep-v15-xl-sft | 50 | ~2s | ~20GB | (B, T, 64) |
| acestep-v15-xl-turbo | 8 | ~2s | ~20GB | (B, T, 64) |

---

## 4. 精度对比验证

### 4.1 脚本说明

精度验证通过对比 NPU 与 CPU 推理结果的 loss 值，计算相对误差百分比。

### 4.2 验证脚本

脚本文件位于 `scripts/verify_accuracy.py`，核心逻辑：

```python
# 1. 在 NPU 上推理
model_npu = load_model(model_path, "npu", torch.bfloat16)
loss_npu = run_forward(model_npu, inputs, silence_latent)

# 2. 清理 NPU 内存
del model_npu
torch.npu.empty_cache()

# 3. 在 CPU 上推理（基线，大模型可能 OOM）
model_cpu = load_model(model_path, "cpu", torch.bfloat16)
loss_cpu = run_forward(model_cpu, inputs_cpu, silence_cpu)

# 4. 对比
rel_err = abs(loss_npu - loss_cpu) / (abs(loss_cpu) + 1e-8) * 100
```

### 4.2 运行验证

```bash
# Base
python3 scripts/verify_accuracy.py \
  --model-path ./model_cache/ACE-Step/acestep-v15-xl-base \
  --seed 42 --threshold 1.0

# SFT
python3 scripts/verify_accuracy.py \
  --model-path ./model_cache/ACE-Step/acestep-v15-xl-sft \
  --seed 42 --threshold 1.0

# Turbo（注意实际路径名是 acestep-v15-xl-turbo）
python3 scripts/verify_accuracy.py \
  --model-path ./model_cache/ACE-Step/acestep-v15-xl-turbo \
  --seed 42 --threshold 1.0
```

### 4.3 精度结果参考

| 模型 | NPU Loss | 相对误差阈值 | 结果 |
|------|----------|-------------|------|
| acestep-v15-xl-base | 0.425781 | < 1% | PASS |
| acestep-v15-xl-sft | 0.777344 | < 1% | PASS |
| acestep-v15-xl-turbo | 0.059814 | < 1% | PASS |

**通过标准**：NPU 推理功能正常，bf16 精度相对误差 < 1%。

说明：4B 模型在 CPU 上推理可能因内存不足（OOM）而无法完成对比，验证脚本支持 `--skip-cpu` 选项跳过 CPU 对比，仅验证 NPU 推理功能正确性。

### 错误处理

- 若 CPU 对比 OOM：添加 `--skip-cpu` 跳过 CPU 对比，仅验证 NPU 功能
- 若 rel_err > 1%：检查 bf16 精度是否对齐，确认 NPU 与 CPU 的计算图一致，检查 dropout/training 模式
- 若 NPU 内存清理不彻底导致后续模型 OOM：在模型切换间执行 `torch.npu.empty_cache()` 并等待 5 秒

### Checkpoint：精度验证确认

- [确认] 每个模型的精度验证均报告 PASSED
- [确认] 相对误差 <= 1%（或使用了 --skip-cpu 确认 NPU 推理功能正常）
- [确认] 无 NPU OOM 错误

---

## 5. 串行评测

多模型评测时，为避免 NPU 显存溢出，必须串行执行：

```bash
for model in acestep-v15-xl-base acestep-v15-xl-sft acestep-v15-xl-turbo; do
  echo "=== Running $model ==="
  python3 scripts/verify_accuracy.py \
    --model-path ./model_cache/ACE-Step/$model \
    --seed 42 --threshold 1.0 --skip-cpu
  python3 -c "import torch; import torch_npu; torch.npu.empty_cache(); print('Cache cleared')"
done
```

### 错误处理

- 若串行执行中途失败：检查失败模型的具体错误，修复后从失败模型继续（而非从头开始），使用 `export ASCEND_RT_VISIBLE_DEVICES=<新卡>` 切换到空闲卡
- 若执行过程中显存泄漏：每轮循环后执行 `torch.npu.empty_cache()` 和 `torch.npu.synchronize()`
- 若 python 进程退出后显存未释放：使用 `npu-smi info` 确认，若仍有残留进程用 `kill -9 <pid>` 清理

### Checkpoint：串行评测确认

- [确认] 全部三个模型均 PASSED
- [确认] 无 OOM/显存泄漏
- [确认] 清理缓存后显存占用恢复至基线水平

---

## 6. 验收确认

### 检查清单

完成以下检查清单即为部署成功：

1. [ ] `npu-smi info` 显示设备正常
2. [ ] `import torch_npu` 无报错
3. [ ] `inference.py` 输出正常 loss 和 latents shape
4. [ ] `verify_accuracy.py` 报告 `PASSED`
5. [ ] 相对误差 < 1%（或功能验证通过）

### 回退策略

若任一检查项未通过：
1. 记录失败项和错误信息到日志文件
2. 根据失败项定位到对应的步骤
3. 执行该步骤的故障排除流程
4. 修复后重新运行验收确认

### 最终输出

部署成功后的交付物：
1. 精度验证报告（`accuracy_report.json`）
2. 推理日志（`inference_logs/` 目录）
3. 模型配置快照（`model_config.yaml`）

---

## 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `No module named 'torch_npu'` | 未安装或 CANN 环境未加载 | `source set_env.sh` 后重装 torch_npu |
| `No module named 'vector_quantize_pytorch'` | 缺少依赖 | `pip install vector-quantize-pytorch` |
| `SetPrecisionMode error 500001` | CANN 环境未加载 | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| 模型下载失败 | 网络问题 | 使用 ModelScope：`snapshot_download('ACE-Step/acestep-v15-xl-base')` |
| OOM | 模型过大（18.8GB bf16） | 串行执行，每次推理后 `torch.npu.empty_cache()` |
| CPU 对比 OOM | 4B 模型超出 CPU 内存 | 使用 `--skip-cpu` 仅验证 NPU 功能 |
| 多卡抢占冲突 | 默认都用 0 号卡 | `npu-smi info` 选空闲卡 |
| loss 为 NaN | bf16 溢出或输入异常 | 检查输入 dtype 和 device 是否一致 |
| 精度误差 > 1% | 计算图差异 | 确认 dropout 模式关闭，seed 一致 |

---

## 附录：资源文件

| 文件 | 用途 | 路径 |
|------|------|------|
| 推理脚本 | 模型加载与推理 | `scripts/inference.py` |
| 精度验证脚本 | NPU vs CPU 精度对比 | `scripts/verify_accuracy.py` |
| 测试用例 | 验证 prompt 与预期结果 | `test-prompts.json` |

---

## 附录：ACE-Step NPU 适配要点速查

| 特征 | ACE-Step 值 | 对 NPU 适配的影响 |
|------|-------------|------------------|
| 架构 | DiT (Diffusion Transformer) | 使用 transformers AutoModel 加载 |
| 自定义代码 | `trust_remote_code=True` | 需要 `vector_quantize_pytorch` 依赖 |
| 精度 | bf16 | NPU 原生支持 bf16 |
| 输入格式 | 多模态（text + lyric + audio） | 需构造 dummy 输入进行验证 |
| 推理方式 | Flow Matching (ODE/SDE) | `generate_audio` 方法 |
| 注意力类型 | Causal + Sliding Window | 模型内部已处理 |
| 任务支持 | Base 全任务 / SFT&Turbo 标准任务 | 根据需求选择模型变体 |
