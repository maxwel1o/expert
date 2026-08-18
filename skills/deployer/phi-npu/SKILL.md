---
name: phi-npu
description: Microsoft Phi 系列模型（phi-1、phi-1.5、phi-2）在昇腾 NPU 上的部署与验证。涵盖 vLLM-Ascend 服务启动、精度评测、性能基准测试及最佳实践。
keywords:
  - phi
  - phi-1
  - phi-1.5
  - phi-2
  - microsoft
  - text-generation
  - code-generation
  - ascend
  - npu
  - vllm-ascend
  - torch-npu
  - cann
metadata:
  author: model-agent
  version: "1.0.0"
  supported-chip: Ascend910B4
  core-scripts: scripts/inference.py,scripts/check_accuracy.py,scripts/perf_test.py
  skill-type: model-deployment
allowed-tools: Bash(*) Python3(*)
---

# Microsoft Phi 系列 昇腾 NPU 部署 Skill

在昇腾 NPU（Ascend910B4）上部署和验证 Microsoft Phi 系列模型（phi-1、phi-1.5、phi-2），
涵盖 vLLM-Ascend 服务化部署、transformers 直接推理、精度验证和性能基准测试。

## 输入参数

| 参数名 | 必填 | 类型 | 说明 | 示例值 |
|--------|------|------|------|--------|
| model_name | 是 | string | 模型名称 | phi-2 |
| dtype | 否 | string | 推理精度 | float16 |
| port | 否 | int | vLLM 服务端口 | 8000 |
| max_new_tokens | 否 | int | 最大生成长度 | 64 |
| prompt | 否 | string | 输入提示词 | "def fibonacci(n):" |

## 使用约束

1. 仅支持 Ascend910B4 芯片，需安装 torch_npu + CANN 工具链。
2. phi-1 原始权重为 fp32，vLLM 部署需 `--dtype float16`。
3. 首次 vLLM 启动需 PIECEWISE 编译（约 1-2 分钟）。
4. 模型权重推荐从 ModelScope 下载（国内网络更稳定）。
5. 单卡部署，不支持多卡张量并行。

## 执行工作流

### Step 1: 环境检查与 NPU 状态确认

在执行部署之前，先确认昇腾 NPU 环境是否就绪：

```bash
# 检查 NPU 设备数量和型号
npu-smi info

# 验证 torch_npu 是否可用
python3 -c "import torch; import torch_npu; print('NPU可用:', torch.npu.is_available(), '设备:', torch.npu.get_device_name(0))"

# 检查 CANN 版本
cat /usr/local/Ascend/version.cfg | grep -i version

# 设置 NPU 环境变量
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
```

1. 执行 `npu-smi info`，确认 NPU 状态为正常。
2. 检查 torch_npu 版本与 CANN 版本匹配。
3. 确认显存充足（phi-2 约需 5.3 GiB，phi-1/1.5 约需 2.7 GiB）。
4. 设置推理所需环境变量。

### Step 2: 模型权重准备

从 ModelScope 下载 Phi 系列模型权重：

```bash
# 安装 model scope
pip install modelscope

# 下载 phi-2 权重（示例）
python3 -c "from modelscope import snapshot_download; snapshot_download('microsoft/phi-2', local_dir='/opt/models/phi-2')"

# 下载 phi-1.5 权重
python3 -c "from modelscope import snapshot_download; snapshot_download('microsoft/phi-1_5', local_dir='/opt/models/phi-1_5')"

# 下载 phi-1 权重
python3 -c "from modelscope import snapshot_download; snapshot_download('microsoft/phi-1', local_dir='/opt/models/phi-1')"
```

1. 确认下载路径与脚本配置路径一致。
2. 检查权重文件完整性（检查目录结构和文件列表）。
3. phi-1 权重为 fp32，约需 5 GB 存储空间。

### Step 3: vLLM-Ascend 服务化部署

启动 vLLM 推理服务（根据模型选择对应端口）：

```bash
# phi-1 启动
vllm serve /opt/models/phi-1 \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 --seed 42 \
  --served-model-name phi-1 \
  --max-num-seqs 64 --max-model-len 2048 \
  --trust-remote-code --dtype float16 \
  --gpu-memory-utilization 0.90 --no-enable-prefix-caching

# phi-1.5 启动
vllm serve /opt/models/phi-1_5 \
  --host 0.0.0.0 --port 8001 \
  --tensor-parallel-size 1 --seed 42 \
  --served-model-name phi-1_5 \
  --max-num-seqs 64 --max-model-len 2048 \
  --trust-remote-code --dtype float16 \
  --gpu-memory-utilization 0.90 --no-enable-prefix-caching

# phi-2 启动
vllm serve /opt/models/phi-2 \
  --host 0.0.0.0 --port 8002 \
  --tensor-parallel-size 1 --seed 42 \
  --served-model-name phi-2 \
  --max-num-seqs 64 --max-model-len 2048 \
  --trust-remote-code --dtype float16 \
  --gpu-memory-utilization 0.90 --no-enable-prefix-caching
```

1. 等待服务就绪后，使用 curl 验证服务可用性。
2. 检查 NPU 显存占用是否合理。

### Step 4: 服务验证与 API 测试

验证 vLLM 服务是否正常运行：

```bash
# 检查可用模型列表
curl http://127.0.0.1:8000/v1/models

# Smoke 测试 - 代码生成
curl http://127.0.0.1:8002/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "phi-2", "prompt": "def fibonacci(n):", "max_tokens": 64, "temperature": 0.0}'

# 文本生成测试
curl http://127.0.0.1:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "phi-1_5", "prompt": "Machine learning is", "max_tokens": 64, "temperature": 0.0}'
```

1. 确认 API 返回结果格式正确。
2. 检查生成文本质量和输出长度是否符合预期。
3. 如果服务启动失败，检查日志中的 CANN 错误信息。

### Step 5: 直接推理与精度基线（transformers）

使用 transformers 进行 NPU 直接推理，作为精度基线：

```bash
# 使用统一推理脚本
python3 scripts/inference.py phi-2 --prompt "def fibonacci(n):" --max-new-tokens 100

# 使用各模型独立脚本
python3 model_configs/phi-2/inference.py

# CPU 基线推理（需要下载权重到 CPU）
python3 scripts/inference.py phi-2 --prompt "def fibonacci(n):" --max-new-tokens 100 --device cpu
```

1. 分别记录 NPU 和 CPU 的生成结果。
2. 确认生成结果语义一致性。
3. 如果显存不足（OOM），减小 max-new-tokens 或使用批处理策略。

### Step 6: 精度对比验证

运行 NPU 与 CPU 精度对比测试：

```bash
# 执行精度检查
python3 scripts/check_accuracy.py phi-2 --output results/phi-2_accuracy.json

# phi-1 精度检查
python3 scripts/check_accuracy.py phi-1 --output results/phi-1_accuracy.json

# phi-1.5 精度检查
python3 scripts/check_accuracy.py phi-1_5 --output results/phi-1_5_accuracy.json
```

验证指标阈值（所有指标需同时满足）：

1. Top-1 准确率 >= 99%，确认 NPU 预测结果与 CPU 一致。
2. 余弦相似度 >= 0.99，确保 logit 分布贴近。
3. KL 散度 < 0.01，验证概率分布差异在可接受范围内。
4. 如果精度不达标，检查 dtype 设置和算子兼容性。

### Step 7: 性能基准测试与结果记录

执行性能基准测试，记录关键指标：

```bash
# phi-2 性能测试
python3 scripts/perf_test.py phi-2

# phi-1 性能测试
python3 scripts/perf_test.py phi-1

# phi-1.5 性能测试
python3 scripts/perf_test.py phi-1_5
```

1. 记录吞吐量（tok/s）、延迟（s）和峰值显存。
2. 检查 NPU 显存峰值是否在合理范围。
3. 对于不同输入长度（32/64/128 tokens），分别记录性能数据。
4. 保存 benchmark 结果到 `results/` 目录。

### Step 8: 生成 README 与发布到仓库

将验证结果整理并生成 README：

```bash
# 创建结果输出目录
mkdir -p results/ benchmarks/

# 复制精度和性能数据到发布目录
cp scripts/check_accuracy.py results/phi-2_accuracy.json 2>/dev/null || true

# 生成汇总 README（如使用模板）
python3 -c "
import json
with open('results/phi-2_accuracy.json') as f:
    data = json.load(f)
print(json.dumps(data, indent=2))
" > benchmarks/summary.json

# 提交结果到仓库
git add results/ benchmarks/ references/
git commit -m 'chore: add phi-npu benchmark results and accuracy data'
git push
```

1. 确认所有测试结果已归档到 `results/` 和 `benchmarks/`。
2. 验证 `evals.json` 或 `references/benchmark_results.md` 中的数据完整性。
3. 如果推送失败，检查仓库权限和远程配置。

## 执行检查点与用户确认

| 序号 | 检查点 | 确认操作 | 通过标准 | 失败处理 |
|------|--------|----------|----------|----------|
| 1 | NPU 环境检查 | 执行 npu-smi info 确认设备状态 | 设备状态为正常，驱动版本匹配 | 暂停部署，检查驱动安装 |
| 2 | torch_npu 可用性 | 运行 Python 检查 torch.npu.is_available() | 返回 True | 检查 torch_npu 安装 |
| 3 | 权重下载完整性 | 检查权重目录文件列表 | 文件完整，无损坏 | 重新下载 |
| 4 | vLLM 服务启动 | 检查服务进程和 API 响应 | curl /v1/models 返回 200 | 检查日志，确认端口未被占用 |
| 5 | API 推理验证 | 执行 smoke test | 返回正确格式的生成结果 | 检查 model 参数和 dtype 设置 |
| 6 | 精度对比测试 | check_accuracy.py 执行完毕 | top-1 >= 99%, KL < 0.01 | 检查 fp16 精度设置 |
| 7 | 性能基准测试 | perf_test.py 执行完毕 | 数值合理，无异常 | 检查显存占用和 batch 设置 |
| 8 | 结果归档确认 | 检查 results/ 和 references/ | 所有文件已提交 | 重新运行测试生成数据 |

## 异常处理与回滚策略

| 异常场景 | 可能原因 | 检测方法 | 处理策略 | 回滚/恢复操作 |
|----------|----------|----------|----------|---------------|
| NPU 设备未找到 | 驱动未加载或昇腾设备故障 | npu-smi info 返回空 | 尝试重试加载驱动 | 检查 dmesg 确认硬件状态 |
| torch_npu 导入失败 | torch_npu 未安装或版本不匹配 | Python import 报错 | fallback 到 CPU 模式 | 使用 pip 重新安装 torch_npu |
| 显存不足 (OOM) | 模型过大或并发过多 | 报 CUDA Out of Memory 错误 | 降低 max-model-len 或 gpu-memory-utilization | 使用较小 batch 或切换到 CPU |
| vLLM 服务启动失败 | CANN 版本不兼容 | 服务日志报错 | 检查 CANN 版本，降级 vllm-ascend | 回滚到上一版本 vllm-ascend |
| 精度验证失败 | dtype 设置错误 | top-1 准确率 < 99% | 检查 dtype 参数（phi-1 需 float16） | 重新运行带正确 dtype 的测试 |
| 推理结果异常 | 模型权重损坏或路径错误 | 生成乱码/空结果 | 重新下载权重 | 清除缓存后重试下载 |
| 服务端口冲突 | 端口被其他进程占用 | curl 连接失败 | 更换端口号 | kill 占用进程或使用不同端口 |
| 权重下载超时 | 网络不稳定 | snapshot_download 失败 | 设置代理或使用国内镜像 | retry 下载，切换 modelscope 源 |
| Python 脚本报错 | 参数传递错误或 API 变更 | 脚本运行抛出异常 | 检查参数格式和脚本兼容性 | 回退到调试模式逐个排查 |
| 性能指标异常 | NPU 频率未达到峰值 | 吞吐量显著低于预期 | 检查 NPU 温度和功耗 | 降频保护后重试，等待恢复 |
| Git 提交失败 | 权限不足或远程不可用 | git push 返回错误 | 检查仓库权限 | 使用 SSH key 重新认证 |
| CANN 算子编译失败 | ascend 算子不兼容 | PIECEWISE 编译阶段报错 | 降低 vllm-ascend 版本 | 使用 transformers 直接推理作为回滚 |

## 资源与评测产物

| 资源路径 | 类型 | 说明 |
|----------|------|------|
| `scripts/inference.py` | 推理脚本 | 统一的 NPU 推理测试脚本（支持 phi-1/phi-1.5/phi-2） |
| `scripts/check_accuracy.py` | 精度检查脚本 | NPU vs CPU 精度对比工具 |
| `scripts/perf_test.py` | 性能测试脚本 | 吞吐量、延迟与显存基准测试 |
| `model_configs/phi-1/run_vllm.sh` | 启动脚本 | phi-1 vLLM 服务化部署启动脚本 |
| `model_configs/phi-1.5/run_vllm.sh` | 启动脚本 | phi-1.5 vLLM 服务化部署启动脚本 |
| `model_configs/phi-2/run_vllm.sh` | 启动脚本 | phi-2 vLLM 服务化部署启动脚本 |
| `references/benchmark_results.md` | 参考文档 | 精度与性能基准数据报告 |
| `results/*.json` | 评测产物 | 精度对比 JSON 结果文件 |
| `benchmarks/summary.json` | 评测产物 | 性能基准测试汇总数据 |
| `evals.json` | 评测产物 | 综合评测结果（运行时生成） |
| `SKILL.md` | 技能文件 | 当前部署技能文档 |

## 验证结果摘要

### 精度（NPU vs CPU）

| 模型 | Top-1 准确率 | 余弦相似度 | KL 散度 | 状态 |
|------|-------------|-----------|---------|------|
| phi-1 | 100% | 1.000000 | ~3e-8 | PASS |
| phi-1.5 | 100% | 0.999996 | ~8e-6 | PASS |
| phi-2 | 100% | 0.99999 | ~9e-6 | PASS |

### 性能（NPU + transformers）

| 模型 | 吞吐量 | 延迟(64 tok) | NPU 峰值内存 |
|------|--------|------------|-------------|
| phi-1 | ~26 tok/s | ~2.5s | 2.7 GiB |
| phi-1.5 | ~27 tok/s | ~2.4s | 2.7 GiB |
| phi-2 | ~20 tok/s | ~3.1s | 5.3 GiB |

### 测试 Prompt

以下是精度与性能测试中使用的 benchmark 测试 prompt：

1. "def fibonacci(n):"
2. "Machine learning is a field of"
3. "Write a Python function to sort a list:"
4. "The capital of France is Paris."
5. "Quantum computing is"

## 注意事项

1. **phi-1 fp32 转 fp16**：phi-1 原始权重为 float32，Ascend 的 `aclnnFusedInferAttentionScoreV3` 算子不支持 float32 输入，vLLM 部署需 `--dtype float16`。
2. **首次编译**：vLLM 首次启动需 PIECEWISE 编译（约 1-2 分钟），后续可复用缓存。
3. **权重来源**：推荐从 ModelScope 下载（国内网络稳定）。
4. **精度说明**：float16 模型的微小 logit 差异属于混合精度推理的固有数值误差，Top-1 预测完全一致。
5. **实测建议**：在昇腾 910B4 上实测验证，确保 eval 结果符合 benchmark 阈值。
