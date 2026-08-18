---
name: vllm-ascend-acl-graph-warmup
description: >
  vLLM-Ascend ACL Graph 编译预热与首跑优化 Skill。
  解决首次启动/请求时 30-60s ACL Graph 编译延迟问题，
  涵盖 compilation-config 模式选择、预热策略、编译缓存复用、
  日志解读。当用户提到 vLLM 首请求慢、ACL Graph、图编译、
  首次启动延迟、warmup 时触发。
metadata:
  short-description: vLLM-Ascend ACL Graph 编译预热与首跑优化
  category: NPU-Inference-Optimization
  tags: [ascend, npu, vllm, acl-graph, compilation, warmup, first-run, optimization]
---

# vLLM-Ascend ACL Graph 编译预热与首跑优化 Skill

本 Skill 解决 vLLM-Ascend 部署中最常见的高频问题：
**首次请求延迟极高（30-60s），用户误以为服务卡死或部署失败**。

以 `google/gemma-3-270m-it` 和 `SpatialLM-Llama-1B-LLM`
在 Atlas 800 A2 (NPU 910B4) 上的验证为参考案例。

## 问题现象

| 现象 | 说明 |
|------|------|
| 服务启动后第一个请求耗时 30-60s | ACL Graph 正在编译算子融合图 |
| 后续请求速度稳定 | 编译产物已缓存，复用即可 |
| 日志中出现 `Graph compile` 或 `ACL` 相关耗时 | 确认是图编译阶段 |
| 客户端/测试脚本超时 | 默认 timeout 不足以覆盖编译时间 |

## 根本原因

vLLM-Ascend 使用华为 ACL (Ascend Computing Language) Graph 模式
执行推理。在首次遇到新的输入 shape / 图结构时，
CANN 运行时需将计算图编译为 NPU 可执行的二进制，该过程：

- **耗时**：30-60s（取决于模型大小和 CANN 版本）
- **CPU 占用高**：编译过程主要在 CPU 侧执行
- **一次性**：同一 shape 的图编译完成后会缓存，后续复用

## 流程总览

```
0. 确认是否为 ACL Graph 编译问题
→ 1. 选择 compilation-config 模式
→ 2. 执行预热（warmup）
→ 3. 调整客户端超时
→ 4. 验证编译缓存复用
```

---

## 0. 确认是否为 ACL Graph 编译问题

查看 vLLM serve 日志，若包含以下关键字，则确认为 ACL Graph 编译：

```
[INFO] Graph compile start
[INFO] Graph compile success, cost time: 32456 ms
[INFO] Acl compile op kernel
```

**非编译问题的延迟**：
- 模型权重加载到 HBM（通常 5-10s，一次性）
- tokenizer 下载/加载（首次启动，网络相关）
- 客户端网络延迟

---

## 1. 选择 compilation-config 模式

vLLM-Ascend 通过 `--compilation-config` 控制 ACL Graph 编译行为。

### 1.1 模式对比

| 模式 | 值 | 说明 | 首跑延迟 | 适用场景 |
|------|-----|------|---------|---------|
| **FULL_DECODE_ONLY** | `{"cudagraph_mode":"FULL_DECODE_ONLY"}` | 仅对 decode 阶段做全图编译 | ~30s | 🌟 **推荐**，大多数生成模型 |
| **NO_CUDAGRAPH** | `{"cudagraph_mode":"NO_CUDAGRAPH"}` | 禁用图编译，逐算子执行 | 无 | 调试、shape 极度变化的场景 |
| **PIECEWISE** | 自动启用 | 分片编译，NPU 默认行为 | ~30-40s | 无法关闭，自动叠加 |

### 1.2 推荐配置

```bash
vllm serve /path/to/model \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  ...
```

### 1.3 其他相关参数

| 参数 | 作用 |
|------|------|
| `--no-enable-prefix-caching` | 禁用 prefix caching，避免额外的图变体编译 |
| `--additional-config '{"enable_cpu_binding":true}'` | CPU 绑核，加速编译过程 |

---

## 2. 执行预热（Warmup）

### 2.1 为什么需要预热

生产环境中，若服务刚启动就接收真实流量，第一个用户会遇到
30-60s 的超时。预热策略是在服务启动后主动触发一次推理，
让 ACL Graph 在真实用户请求前完成编译。

### 2.2 手动预热

服务启动后，在服务端本地执行：

```bash
# 等待服务端口就绪
sleep 5

# 发送一个 warm-up 请求
curl -sf http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"model-name","messages":[{"role":"user","content":"Hi"}],"max_tokens":16}' \
  --max-time 120

echo "Warmup complete"
```

### 2.3 自动化预热脚本

使用本 Skill 提供的脚本：

```bash
python scripts/warmup.py \
  --api-url http://127.0.0.1:8000/v1/chat/completions \
  --model-name gemma-3-270m-it \
  --max-tokens 16 \
  --timeout 120
```

脚本特性：
- 自动轮询等待服务就绪
- 支持设置超长超时（默认 120s）
- 打印编译/响应时间，便于判断是否为首次编译

### 2.4 预热请求设计

| 参数 | 推荐值 | 原因 |
|------|--------|------|
| max_tokens | 16-32 | 足够触发 decode 图编译，又不耗时过长 |
| temperature | 0 | 确定性输出，便于验证 |
| prompt | 任意短文本 | `"Hi"` 即可 |

---

## 3. 调整客户端超时

### 3.1 测试脚本超时

性能测试脚本应设置充足的首次请求超时：

```python
import requests

# 首次请求给 120s，后续请求可缩短到 30s
resp = requests.post(url, json=payload, timeout=120)
```

### 3.2 生产网关超时

若前端有 Nginx / API Gateway，确保其超时 > 60s：

```nginx
# nginx.conf
proxy_connect_timeout 120s;
proxy_send_timeout 120s;
proxy_read_timeout 120s;
```

---

## 4. 验证编译缓存复用

### 4.1 检查方法

重启服务后再次发送相同 shape 的请求：

1. **首次请求**（服务刚启动）：
   - 日志中出现 `Graph compile start`
   - 耗时 30-60s

2. **第二次请求**（相同 shape）：
   - 日志中**无** `Graph compile` 字样
   - 耗时稳定（如 gemma-270m 约 22s）

3. **重启服务后**（同一台机器）：
   - 若 CANN 缓存有效，首次请求可能缩短到 5-10s
   - 若缓存被清理，仍需完整编译

### 4.2 缓存位置

ACL Graph 编译缓存由 CANN 运行时自动管理，通常位于：

```
~/.cache/ascend/
/usr/local/Ascend/driver/cache/
```

> **注意**：跨 CANN 版本升级时缓存可能失效，需重新编译。

---

## 5. 验收确认

- [ ] 服务启动后，warmup 脚本在 120s 内成功完成
- [ ] warmup 完成后，后续请求延迟稳定（无 30s+ 跳变）
- [ ] 客户端/网关超时已调整为 > 60s
- [ ] 日志中确认 `Graph compile success` 出现一次后不再出现

---

## 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 每次请求都慢，不止第一次 | 输入 shape 每次变化，触发重新编译 | 固定 `max_model_len` 和 batch size，使用 `FULL_DECODE_ONLY` |
| warmup 请求也超时 | 模型过大或 CANN 版本过旧 | 增加 timeout 到 180s，或升级 CANN |
| 重启后又要编译 30s | 编译缓存被清理或跨版本失效 | 正常现象，可在启动脚本中内置 warmup |
| `compilation-config` 报错 | JSON 格式错误 | 确保外层单引号 + 内层双引号：`'{"cudagraph_mode":"FULL_DECODE_ONLY"}'` |

---

## 参考

- ACL Graph 设计文档：<https://docs.vllm.ai/projects/ascend/zh-cn/v0.18.0/developer_guide/Design_Documents/ACL_Graph.html>
- 本 Skill 脚本：`scripts/warmup.py`
