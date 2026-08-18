# vLLM-Ascend 常见错误模式库

本文档收录 vLLM-Ascend 部署和推理中的常见错误模式、根因和解决方案。排障时优先在此匹配。

---

## 总览异常表格

| Failure Class | Symptoms | Common Signals | First Checks | Fallback | Escalation |
|---|---|---|---|---|---|
| 服务启动失败 | 进程崩溃、端口占用、启动超时 | OSError: Address already in use; 进程退出码非 0 | lsof -i :PORT; 检查日志尾部 | 更换端口; 减小 max-model-len | 检查系统资源限制 ulimit |
| 模型加载失败 | OOM、权重 shape 不匹配、safetensors 解析失败 | RuntimeError: size mismatch; SafetensorError | 检查 config.json; 验证权重文件完整性 | --load-format dummy; 单卡加载 | 模型文件损坏需重新下载 |
| tokenizer / processor 加载失败 | tokenizer.json 缺失、processor 初始化报错 | FileNotFoundError; AutoTokenizer.from_pretrained 失败 | 检查 tokenizer 文件是否存在 | 使用默认 tokenizer 路径 | 模型包不完整 |
| CANN 环境变量缺失 | ASCEND_HOME_PATH not found; libascend_hal.so 缺失 | ModuleNotFoundError; ImportError | echo $ASCEND_HOME_PATH; ls $ASCEND_HOME_PATH/lib64 | 手动 export 环境变量 | CANN 未安装或损坏 |
| 版本不匹配 | AttributeError; 不支持的参数; ABI 错误 | torch vs torch_npu 版本不一致; vLLM vs vLLM-Ascend 不匹配 | pip show torch torch_npu vllm vllm-ascend | 重新安装兼容版本组合 | 参考兼容性矩阵 |
| NPU 不可见 | No available npu device; npu-smi 报错 | torch.npu.is_available() 返回 False | npu-smi info; ls /dev/davinci*; groups | 检查驱动加载; 检查用户权限 | 驱动未安装或设备故障 |
| ACLGraph 捕获失败 | ACLGraph capture failed; 推理随机崩溃 | 动态 shape 操作; CANN 版本不兼容 | --enforce-eager 对比测试 | 保持 eager mode | 等待 CANN / vLLM-Ascend 更新 |
| eager 可运行但 graph 失败 | eager mode 正常，默认模式报错 | ACLGraph 相关 RuntimeError | 对比两种模式输出 | --enforce-eager 作为 workaround | 已知限制 |
| CUDA-only 算子不兼容 | Triton 报错; CUDA 算子未注册 | RuntimeError: not supported on NPU; Triton 导入 | 检查算子是否有 torch_npu 替代 | 使用 eager mode; 等待社区适配 | 模型架构不支持 |
| 量化模型加载失败 | Quantization method not supported; 量化权重 shape 错误 | ValueError: quantization; ascend/awq/gptq/fp8 | 检查 quantization_config; 确认量化方式受支持 | 使用非量化版本 | 量化方式未适配 |
| 多模态输入失败 | image/video 处理报错; processor 异常 | PIL 导入失败; image processor 配置错误 | 检查 Pillow/timm 版本; 检查 processor 配置 | 使用纯文本模式 | 多模态模型未完全适配 |
| /v1/chat/completions 异常 | API 返回 500/503; 超时; 空输出 | detail: Internal Server Error; TimeoutError | 检查服务端日志; 简化请求 payload | 减小 max_tokens; 使用 /v1/completions | 服务端 bug |
| tensor parallel 多卡异常 | HCCL 超时; rank 不一致; 负载不均 | RuntimeError: HCCL timeout; rank OOM | npu-smi info -t topo; HCCL_DEBUG=1 | 单卡复现; 降低 TP size | 通信拓扑或网络问题 |
| 性能异常 | 吞吐低、延迟高、NPU 利用率低 | TTFT 过高; TPOT 过高; host-bound | npu-smi info; top; TASK_QUEUE_ENABLE=1 | --enforce-eager; 减小 batch | 需要 profiler 深度分析 |

---

## 一、环境与安装类

### E-ENV-001: torch_npu 导入失败

**错误特征**:
```
ModuleNotFoundError: No module named 'torch_npu'
```
或
```
ImportError: libascend_hal.so: cannot open shared object file
```

**根因**: torch_npu 未安装，或安装的 torch_npu 与 PyTorch / CANN 版本不匹配。

**解决方案**:
1. 确认版本兼容性：torch_npu 版本必须与 PyTorch 版本严格对应
2. 重新安装：
   ```bash
   pip install torch==2.5.1 torch_npu==2.5.1
   ```
3. 如果 libascend_hal.so 缺失，检查 CANN 安装完整性：
   ```bash
   ls $ASCEND_HOME_PATH/lib64/libascend_hal.so
   ```

---

### E-ENV-002: NPU 设备不可见

**错误特征**:
```
RuntimeError: No available npu device found
```
或 `npu-smi info` 报错：
```
Failed to initialize npu device
```

**根因**: NPU 驱动未安装/未加载、用户无权限、设备被其他进程占用。

**解决方案**:
1. 检查驱动状态：
   ```bash
   npu-smi info
   lsmod | grep npu
   ```
2. 检查用户组权限：
   ```bash
   groups  # 应包含 HwHiAiUser 或 ascend
   ```
3. 检查设备文件：
   ```bash
   ls -la /dev/davinci*
   ```
4. 如果设备被占用，kill 相关进程后重试

---

### E-ENV-003: CANN 版本不兼容

**错误特征**:
```
ASCEND_HOME_PATH not found
```
或推理时报各种 ACL 错误码（如 507001、507002）。

**根因**: CANN 未正确安装、ASCEND_HOME_PATH 未设置、版本与 torch_npu 不匹配。

**解决方案**:
1. 设置环境变量：
   ```bash
   export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
   export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
   export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$LD_LIBRARY_PATH
   ```
2. 验证 CANN 版本：
   ```bash
   cat $ASCEND_HOME_PATH/version.cfg
   ```
3. 确认版本兼容性矩阵

---

### E-ENV-004: vLLM-Ascend 版本不匹配

**错误特征**:
```
AttributeError: module 'vllm' has no attribute 'xxx'
```
或启动时提示不支持的参数。

**根因**: vLLM-Ascend 版本与 vLLM 主版本不匹配，或安装了原生 vLLM 而非 vLLM-Ascend。

**解决方案**:
1. 确认安装的是 vLLM-Ascend：
   ```bash
   pip show vllm | grep -i ascend
   ```
2. 按官方文档安装正确版本组合：
   ```bash
   pip install vllm==0.9.0 vllm-ascend==0.9.0
   ```

---

## 二、服务启动类

### E-START-001: 端口占用

**错误特征**:
```
OSError: [Errno 98] Address already in use
```

**根因**: 默认端口 8000 被占用。

**解决方案**:
```bash
# 查找占用进程
lsof -i :8000
# 或更换端口
vllm serve /path/to/model --port 8001 --device npu
```

---

### E-START-002: 模型路径不存在

**错误特征**:
```
FileNotFoundError: [Errno 2] No such file or directory: '/path/to/model/config.json'
```
或
```
OSError: /path/to/model does not appear to have a file named config.json
```

**根因**: 模型路径错误、路径权限不足、模型文件不完整。

**解决方案**:
1. 检查路径和文件：
   ```bash
   ls -la /path/to/model/
   ls -la /path/to/model/config.json
   ```
2. 确认包含必需文件：config.json, tokenizer.json/tokenizer_config.json, 模型权重文件
3. 检查路径中是否有软链接断裂

---

### E-START-003: OOM 启动失败

**错误特征**:
```
RuntimeError: NPU out of memory. Tried to allocate xxx MiB
```
或启动过程中进程被 OOM killer 终止。

**根因**: 模型太大超出单卡 HBM 容量、KV cache 预分配过大。

**解决方案**:
1. 使用 tensor parallel 分卡：
   ```bash
   vllm serve /path/to/model --tensor-parallel-size 2 --device npu
   ```
2. 减小 max-model-len：
   ```bash
   vllm serve /path/to/model --max-model-len 2048 --device npu
   ```
3. 设置内存分配策略：
   ```bash
   export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
   ```
4. 使用量化模型减少权重占用

---

### E-START-004: dtype 不支持

**错误特征**:
```
ValueError: dtype xxx is not supported on NPU
```
或
```
RuntimeError: xxx: unsupported dtype
```

**根因**: 某些 dtype（如 bfloat16 在旧版 CANN）在 NPU 上不支持。

**解决方案**:
1. 使用 float16 替代：
   ```bash
   vllm serve /path/to/model --dtype float16 --device npu
   ```
2. 检查 CANN 版本是否支持目标 dtype
3. bfloat16 需要 CANN 8.1+ 和对应 torch_npu 版本

---

### E-START-005: 分布式启动失败

**错误特征**:
```
RuntimeError: Address already in use (hccl)
```
或
```
RuntimeError: NCCL/HCCL error: unhandled system error
```

**根因**: HCCL 通信端口冲突、多卡拓扑配置错误、网络不通。

**解决方案**:
1. 设置独立的 HCCL 端口范围：
   ```bash
   export HCCL_IF_IP=xxx.xxx.xxx.xxx
   export MASTER_ADDR=xxx.xxx.xxx.xxx
   export MASTER_PORT=29500
   ```
2. 检查 NPU 间通信：
   ```bash
   npu-smi info -t topo
   ```
3. 对于多机场景，确保网络互通和 RDMA 配置正确

---

## 三、模型加载类

### E-LOAD-001: safetensors 加载失败

**错误特征**:
```
safetensors_rust.SafetensorError: Error while deserializing header
```
或
```
RuntimeError: Error(s) in loading state_dict
```

**根因**: 模型文件损坏、下载不完整、safetensors 版本不兼容。

**解决方案**:
1. 验证文件完整性：
   ```bash
   sha256sum /path/to/model/*.safetensors
   ```
2. 重新下载模型
3. 升级 safetensors：
   ```bash
   pip install --upgrade safetensors
   ```

---

### E-LOAD-002: 权重 shape 不匹配

**错误特征**:
```
RuntimeError: Error(s) in loading state_dict for XxxForCausalLM:
    size mismatch for xxx.weight: copying a param with shape xxx
```

**根因**: 模型权重与 config.json 中定义的架构不匹配，常见于分片权重未完整下载。

**解决方案**:
1. 检查 config.json 中的模型参数是否与权重一致
2. 确认所有分片文件都已下载：
   ```bash
   ls -la /path/to/model/model*.safetensors
   ```
3. 尝试使用 transformers 加载验证：
   ```python
   from transformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained("/path/to/model")
   ```

---

### E-LOAD-003: 模型架构不支持

**错误特征**:
```
ValueError: Model architectures ['XxxForCausalLM'] are not supported for now.
```

**根因**: 当前 vLLM-Ascend 版本不支持该模型架构。

**解决方案**:
1. 查看 vLLM-Ascend 支持的模型列表：
   ```bash
   python -c "from vllm.model_executor.models import _MODELS; print(list(_MODELS.keys()))"
   ```
2. 升级到最新版 vLLM-Ascend
3. 如果是新架构，可能需要等待社区适配或自行注册模型

---

### E-LOAD-004: 量化模型加载失败

**错误特征**:
```
ValueError: Quantization method 'xxx' is not supported
```
或加载量化权重时 shape 不匹配。

**根因**: vLLM-Ascend 不支持该量化方式、量化配置缺失、权重格式不兼容。

**解决方案**:
1. 确认量化方式受支持：
   - vLLM-Ascend 支持: `awq`, `gptq`, `fp8`, `ascend` (W8A8)
2. 检查是否有 quantize_config.json 或 quantization_config
3. 对于 Ascend W8A8 量化，使用 `--quantization ascend`
4. 对于 AWQ/GPTQ，确保安装了对应的量化库

---

## 四、推理请求类

### E-INFER-001: 推理 OOM

**错误特征**:
```
RuntimeError: NPU out of memory during inference
```
或请求过程中进程崩溃。

**根因**: 输入序列过长、batch 过大、KV cache 空间不足。

**解决方案**:
1. 减小 max_model_len 或请求中的 max_tokens
2. 启用 chunked prefill（如果版本支持）：
   ```bash
   vllm serve /path/to/model --enable-chunked-prefill --device npu
   ```
3. 设置 VLLM_WORKER_MULTIPROC_METHOD=spawn 避免 fork 导致内存复制
4. 降低 gpu_memory_utilization（默认 0.9）：
   ```bash
   vllm serve /path/to/model --gpu-memory-utilization 0.85 --device npu
   ```

---

### E-INFER-002: 推理超时

**错误特征**:
```
TimeoutError: Request timed out
```
或客户端收到 504 超时。

**根因**: 模型推理太慢、请求排队过长、首 token 延迟过高。

**解决方案**:
1. 检查 NPU 利用率：
   ```bash
   watch -n 1 npu-smi info
   ```
2. 如果 NPU 利用率低但延迟高，可能是 host-bound：
   ```bash
   export TASK_QUEUE_ENABLE=1
   export CPU_AFFINITY_CONF=1
   export COMBINED_ENABLE=1
   ```
3. 检查是否在 ACLGraph 模式下运行（默认开启）
4. 尝试 eager mode 排查：
   ```bash
   vllm serve /path/to/model --enforce-eager --device npu
   ```

---

### E-INFER-003: 输出乱码/空输出

**错误特征**: 推理返回结果为乱码、重复 token、或空字符串。

**根因**: tokenizer 配置错误、模型精度问题、sampler 配置异常。

**解决方案**:
1. 验证 tokenizer 加载：
   ```python
   from transformers import AutoTokenizer
   tok = AutoTokenizer.from_pretrained("/path/to/model")
   print(tok.decode(tok.encode("Hello")))
   ```
2. 检查 dtype 是否正确（有些模型必须用 float32 或 bfloat16）
3. 尝试固定采样参数排除 sampler 问题：
   ```json
   {"temperature": 0, "top_p": 1.0, "max_tokens": 64}
   ```

---

### E-INFER-004: API 返回 500

**错误特征**:
```json
{"detail": "Internal Server Error"}
```

**根因**: 服务端未捕获的异常，需查看 vLLM 服务端日志。

**解决方案**:
1. 查看 vLLM 服务端日志（通常在启动终端或日志文件中）
2. 检查请求 payload 格式是否正确
3. 如果是模型特有问题，尝试用更简单的请求复现

---

## 五、算子与兼容性类

### E-OP-001: 算子不支持

**错误特征**:
```
RuntimeError: xxx: not supported on NPU
```
或
```
ACL error: 507001
```

**根因**: 模型使用的算子在当前 CANN / torch_npu 版本中未实现或不支持。

**解决方案**:
1. 检查算子是否有 torch_npu 替代实现
2. 升级 CANN 和 torch_npu 到最新版本
3. 如果是自定义算子，需要注册到 ACL
4. 尝试 eager mode 排除编译优化问题

---

### E-OP-002: ACLGraph 模式异常

**错误特征**:
```
RuntimeError: ACLGraph capture failed
```
或推理时随机崩溃。

**根因**: ACLGraph 不支持某些动态形状操作、CANN 版本不兼容。

**解决方案**:
1. 强制 eager mode：
   ```bash
   vllm serve /path/to/model --enforce-eager --device npu
   ```
2. 升级 CANN 到 8.1.RC1+ 版本
3. 如果 eager mode 正常但 ACLGraph 失败，这是已知限制，保持 eager mode

---

### E-OP-003: 自定义算子注册失败

**错误特征**:
```
RuntimeError: Could not load custom op library
```
或
```
ImportError: cannot import name 'xxx' from 'vllm._C'
```

**根因**: vLLM-Ascend 的自定义算子编译失败或 ABI 不匹配。

**解决方案**:
1. 重新安装 vLLM-Ascend：
   ```bash
   pip install --force-reinstall vllm-ascend
   ```
2. 检查 GCC 版本和 ABI 兼容性
3. 从源码编译：
   ```bash
   VLLM_TARGET_DEVICE=ascend pip install -e .
   ```

---

## 六、精度与性能类

### E-PREC-001: 输出与 GPU 差异过大

**错误特征**: 同一输入在 NPU 和 GPU 上的输出差异 > 1%。

**根因**: dtype 不一致、算子实现差异、随机种子未固定。

**解决方案**:
1. 确保两端使用相同 dtype（float16 vs float16）
2. 固定随机种子：
   ```python
   torch.manual_seed(42)
   torch.npu.manual_seed(42)
   ```
3. 检查是否有 NPU 特有的精度损失（某些融合算子精度略低）
4. 对比 logits 而非最终文本输出

---

### E-PREC-002: NaN/Inf 输出

**错误特征**: 推理输出包含 NaN 或 Inf 值。

**根因**: 数值溢出、模型权重异常、输入数据异常。

**解决方案**:
1. 检查输入数据是否包含异常值
2. 使用 float32 排查精度问题：
   ```bash
   vllm serve /path/to/model --dtype float32 --device npu
   ```
3. 检查模型权重：
   ```python
   import torch
   state_dict = torch.load("/path/to/model/model.safetensors")
   for k, v in state_dict.items():
       if torch.isnan(v).any() or torch.isinf(v).any():
           print(f"异常权重: {k}")
   ```

---

### E-PERF-001: 吞吐低于预期

**错误特征**: NPU 利用率低、吞吐量远低于同等 GPU。

**根因**: host-bound、未开启性能优化参数、数据预处理瓶颈。

**解决方案**:
1. 开启性能优化：
   ```bash
   export TASK_QUEUE_ENABLE=1
   export CPU_AFFINITY_CONF=1
   export COMBINED_ENABLE=1
   ```
2. 使用 profiler 定位瓶颈：
   ```bash
   vllm serve /path/to/model --device npu --enforce-eager  # 先排除 ACLGraph
   ```
3. 检查 CPU 是否成为瓶颈（top/htop 查看 CPU 利用率）
4. 尝试增大 batch size（调整 max-num-seqs）

---

### E-PERF-002: 首 token 延迟 (TTFT) 过高

**错误特征**: 首个 token 生成时间远超预期。

**根因**: prefill 阶段计算量大、未启用 chunked prefill、prompt 过长。

**解决方案**:
1. 启用 chunked prefill：
   ```bash
   vllm serve /path/to/model --enable-chunked-prefill --device npu
   ```
2. 减小 max_model_len 减少 prefill 计算量
3. 检查是否在 ACLGraph 模式下（首次有编译开销）

---

### E-PERF-003: KV cache 分配失败

**错误特征**:
```
RuntimeError: Failed to allocate KV cache
```
或启动时 OOM。

**根因**: HBM 空间不足以容纳模型权重 + KV cache。

**解决方案**:
1. 减小 max_model_len：
   ```bash
   vllm serve /path/to/model --max-model-len 2048 --device npu
   ```
2. 减小 gpu_memory_utilization 留出余量
3. 使用 tensor parallel 分散 KV cache 到多卡
4. 使用量化减少模型权重占用

---

## 七、分布式与通信类

### E-DIST-001: HCCL 通信超时

**错误特征**:
```
RuntimeError: HCCL timeout: rank xxx
```
或进程 hang 住不动。

**根因**: NPU 间通信故障、网络不通、死锁。

**解决方案**:
1. 检查 NPU 拓扑：
   ```bash
   npu-smi info -t topo
   ```
2. 设置 HCCL 超时时间：
   ```bash
   export HCCL_CONNECT_TIMEOUT=1200
   ```
3. 开启 HCCL 调试：
   ```bash
   export HCCL_DEBUG=1
   ```
4. 检查防火墙和网络配置（多机场景）

---

### E-DIST-002: Tensor Parallel 不均匀

**错误特征**: 某些 NPU 利用率远高于其他，或个别 rank OOM。

**根因**: 模型参数无法被 TP size 整除、某些层在特定 rank 上计算量更大。

**解决方案**:
1. 确认模型 hidden_size 能被 TP size 整除
2. 使用 2 的幂次作为 TP size（1, 2, 4, 8）
3. 检查是否是已知的 load balancing 问题
