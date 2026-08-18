# vLLM-Ascend 环境验证清单

部署前和排障时使用此清单逐项检查环境。每项检查包含：Purpose（目的）、Recommended command（推荐命令）、Pass criteria（通过标准）、Failure next step（失败处理）。

---

## 一、环境验证

**Purpose**: 确认操作系统、Python 版本满足 vLLM-Ascend 运行要求

**Recommended command**:
```bash
python3 -c "import platform; print(f'OS: {platform.system()} {platform.release()} ({platform.machine()})'); print(f'Python: {platform.python_version()}')"
```

**Pass criteria**:
- OS 为 Linux
- Python >= 3.9
- 架构为 x86_64 或 aarch64

**Failure next step**: 升级 Python 到 3.9+ 或使用正确的操作系统环境

---

## 二、NPU 可见性验证

**Purpose**: 确认 NPU 设备物理可用且软件可访问

**Recommended command**:
```bash
npu-smi info
python3 -c "import torch, torch_npu; print(f'NPU available: {torch.npu.is_available()}'); print(f'NPU count: {torch.npu.device_count()}')"
```

**Pass criteria**:
- `npu-smi info` 输出所有 NPU 状态为 OK
- `torch.npu.is_available()` 返回 True
- `torch.npu.device_count()` 与 npu-smi 显示数量一致

**Failure next step**:
- npu-smi 失败 → 检查 NPU 驱动安装和加载 (`lsmod | grep npu`)
- torch.npu 不可用 → 检查 torch_npu 安装和 CANN 环境变量
- 设备数量不一致 → 检查 ASCEND_VISIBLE_DEVICES 是否限制了可见设备

---

## 三、Python 包版本验证

**Purpose**: 确认 PyTorch、torch_npu、CANN 版本兼容

**Recommended command**:
```bash
python3 -c "
import torch; print(f'PyTorch: {torch.__version__}')
import torch_npu; print(f'torch_npu: {torch_npu.__version__}')
import os; print(f'ASCEND_HOME_PATH: {\"SET\" if os.environ.get(\"ASCEND_HOME_PATH\") else \"NOT SET\"}')
"
cat $ASCEND_HOME_PATH/version.cfg 2>/dev/null || echo 'CANN version file not found'
```

**Pass criteria**:
- PyTorch 和 torch_npu 版本匹配（见兼容性矩阵）
- ASCEND_HOME_PATH 已设置
- CANN 版本与 torch_npu 版本兼容

**Failure next step**: 参考兼容性矩阵重新安装正确版本组合

**兼容性矩阵**:

| CANN 版本 | torch_npu 版本 | PyTorch 版本 |
|-----------|---------------|-------------|
| 8.1.RC1 | 2.5.1 | 2.5.1 |
| 8.0.T6 | 2.4.0 | 2.4.0 |
| 8.0.T5 | 2.3.1 | 2.3.1 |

---

## 四、vLLM-Ascend import 验证

**Purpose**: 确认 vLLM-Ascend 包可正确导入且版本匹配

**Recommended command**:
```bash
python3 -c "
import vllm; print(f'vLLM: {vllm.__version__}')
import importlib.metadata
try:
    ver = importlib.metadata.version('vllm-ascend')
    print(f'vLLM-Ascend: {ver}')
except:
    print('vLLM-Ascend: not found (may be merged into vllm)')
"
```

**Pass criteria**:
- vLLM 成功导入
- vLLM-Ascend 成功导入（或已合并到 vllm 主包）
- vLLM 和 vLLM-Ascend 版本匹配

**Failure next step**: `pip install vllm==<VER> vllm-ascend==<VER>` 安装匹配版本

---

## 五、dummy load 验证

**Purpose**: 验证模型架构可被 vLLM-Ascend 加载（不加载真实权重）

**Recommended command**:
```bash
vllm serve <MODEL_PATH> --load-format dummy --max-model-len 64 --device npu --port <PORT>
```

**Pass criteria**:
- 服务启动无报错
- 日志中显示模型加载完成

**Failure next step**:
- 模型架构不支持 → 检查 vLLM-Ascend 支持的模型列表
- dtype 不支持 → 添加 `--dtype float16`
- OOM → 减小 `--max-model-len` 或使用 `--tensor-parallel-size`

---

## 六、real weight load 验证

**Purpose**: 验证真实模型权重可正确加载

**Recommended command**:
```bash
vllm serve <MODEL_PATH> --max-model-len 64 --enforce-eager --device npu --port <PORT>
```

**Pass criteria**:
- 服务启动无报错
- 日志中显示权重加载完成
- 无 shape mismatch 或 safetensors 错误

**Failure next step**:
- OOM → 使用 tensor parallel 或减小 max-model-len
- shape 不匹配 → 检查 config.json 与权重文件是否对应
- safetensors 错误 → 验证文件完整性或重新下载

---

## 七、OpenAI-compatible API 验证

**Purpose**: 验证 vLLM 服务的 OpenAI 兼容 API 可用

**Recommended command**:
```bash
# 检查 models 端点
curl http://localhost:<PORT>/v1/models

# 检查 chat completions 端点
curl http://localhost:<PORT>/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"<MODEL_PATH>","messages":[{"role":"user","content":"Hello"}],"max_tokens":16,"temperature":0}'
```

**Pass criteria**:
- /v1/models 返回模型列表 JSON
- /v1/chat/completions 返回正常生成结果（choices 非空）

**Failure next step**:
- 500 错误 → 检查服务端日志
- 超时 → 检查 NPU 利用率和请求排队
- 空输出 → 检查 tokenizer 和 sampler 配置

---

## 八、eager / graph 对比验证

**Purpose**: 确认 ACLGraph 模式和 eager 模式均可用且输出一致

**Recommended command**:
```bash
# eager mode
vllm serve <MODEL_PATH> --enforce-eager --device npu --port <PORT_EAGER>
curl http://localhost:<PORT_EAGER>/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"<MODEL_PATH>","messages":[{"role":"user","content":"What is 1+1?"}],"max_tokens":8,"temperature":0}'
# 记录输出 A

# graph mode (default)
vllm serve <MODEL_PATH> --device npu --port <PORT_GRAPH>
curl http://localhost:<PORT_GRAPH>/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"<MODEL_PATH>","messages":[{"role":"user","content":"What is 1+1?"}],"max_tokens":8,"temperature":0}'
# 记录输出 B
```

**Pass criteria**:
- 两种模式均成功启动
- 输出 A 和 B 在 temperature=0 时一致

**Failure next step**:
- graph mode 失败但 eager 成功 → 按 E-OP-002 处理，使用 `--enforce-eager` 作为 workaround
- 两种模式输出不一致 → 可能是算子精度差异，对比 logits

---

## 九、性能 sanity check

**Purpose**: 验证基本推理性能在合理范围

**Recommended command**:
```bash
# 开启性能优化环境变量
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=1
export COMBINED_ENABLE=1

# 简单延迟测试
time curl -s http://localhost:<PORT>/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"<MODEL_PATH>","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"temperature":0}' > /dev/null
```

**Pass criteria**:
- 单次请求延迟在合理范围（根据模型大小判断）
- NPU 利用率 > 0%（通过 `watch -n 1 npu-smi info` 观察）

**Failure next step**:
- NPU 利用率为 0% → 检查是否在 CPU 上运行
- 延迟异常高 → 检查 host-bound、开启 TASK_QUEUE_ENABLE
- 吞吐低 → 检查 batch size、开启 CPU_AFFINITY_CONF

---

## 十、多卡 tensor parallel sanity check

**Purpose**: 验证多卡 tensor parallel 通信正常

**Recommended command**:
```bash
vllm serve <MODEL_PATH> --tensor-parallel-size <TP_SIZE> --device npu --port <PORT>
```

**Pass criteria**:
- 服务启动无 HCCL 错误
- 所有 rank 正常初始化
- API 请求正常返回

**Failure next step**:
- HCCL timeout → 检查 NPU 拓扑 (`npu-smi info -t topo`)、设置 HCCL_CONNECT_TIMEOUT
- rank OOM → 减小 max-model-len 或使用更大 TP size
- 通信错误 → 检查 HCCL_IF_IP、MASTER_ADDR、MASTER_PORT 配置

---

## 十一、多模态模型额外验证

**Purpose**: 验证多模态模型的 image/video 输入处理正常

**Recommended command**:
```bash
curl http://localhost:<PORT>/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model":"<MODEL_PATH>",
        "messages":[{"role":"user","content":[
            {"type":"text","text":"Describe this image"},
            {"type":"image_url","image_url":{"url":"https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"}}
        ]}],
        "max_tokens":64
    }'
```

**Pass criteria**:
- 请求成功返回
- 返回内容包含对图片的描述

**Failure next step**:
- PIL 导入失败 → `pip install Pillow`
- processor 加载失败 → 检查 transformers 和 timm 版本
- image 处理报错 → 检查 processor 配置和图片格式支持

---

## 检查结果模板

```
=== vLLM-Ascend 环境检查报告 ===
日期: YYYY-MM-DD HH:MM:SS

[硬件]
NPU 型号:       Ascend910B
NPU 数量:       X
NPU 状态:       OK / FAULT
驱动版本:       XX.X.X

[软件]
OS:             Linux x.x.x (aarch64/x86_64)
Python:         3.11.x
PyTorch:        2.5.1
torch_npu:      2.5.1
CANN:           8.1.RC1
vLLM:           0.9.0
vLLM-Ascend:    0.9.0
transformers:   4.45.x

[环境变量]
ASCEND_HOME_PATH: /usr/local/Ascend/ascend-toolkit/latest  [OK]
LD_LIBRARY_PATH:  ...  [OK]
PYTORCH_NPU_ALLOC_CONF: max_split_size_mb:256  [OK]

[验证]
G0 环境信息完整性:     PASS / FAIL
G1 NPU 可见性:         PASS / FAIL
G2 Python 包 import:   PASS / FAIL
G3 vLLM-Ascend import: PASS / FAIL
G4 dummy load:         PASS / FAIL
G5 real weight load:   PASS / FAIL
G6 /v1/models:         PASS / FAIL
G7 /v1/chat/completions: PASS / FAIL
G8 eager vs graph:     PASS / FAIL
G9 tensor parallel:    PASS / FAIL
G10 performance:       PASS / FAIL
G11 multimodal:        PASS / FAIL

=== 检查结论: PASS / FAIL (详见上述标记) ===
```
