# EngineCore NPU 初始化失败排查参考

## 常见错误码

| 错误码 | 含义 | 常见根因 |
|--------|------|----------|
| 107001 | aclInit 失败 | CANN 环境未正确初始化、驱动不匹配 |
| 507008 | 无效设备 ID | ASCEND_RT_VISIBLE_DEVICES 与容器映射冲突 |

## 根因速查表

### 1. fork 多进程导致子进程无法继承 NPU 上下文
- **现象**：主进程 `torch_npu.npu.is_available()` 为 True，子进程 init 失败
- **解决**：`export VLLM_WORKER_MULTIPROC_METHOD=spawn`

### 2. 单卡容器设置了 ASCEND_RT_VISIBLE_DEVICES
- **现象**：设备文件存在但子进程报 Invalid device ID
- **解决**：`unset ASCEND_RT_VISIBLE_DEVICES`

### 3. CANN 日志目录不可写
- **现象**：aclInit 返回权限相关错误
- **解决**：`mkdir -p ~/ascend/log && chmod 755 ~/ascend/log`

### 4. vllm bench 子进程不受 spawn 控制
- **现象**：`vllm bench latency` 仍然失败
- **解决**：改用 `vllm serve` + `vllm bench serve` 模式

## 参考文档

- [vLLM-Ascend 官方文档](https://docs.vllm.ai/projects/ascend/)
- [CANN 错误码手册](https://www.hiascend.com/document/)
