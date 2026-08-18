---
name: ascend-modelscope-mirror
description: >
  昇腾 NPU 模型权重下载镜像与加速配置 Skill。
  解决国内环境访问 HuggingFace 超时/失败问题，提供 ModelScope、HF-Mirror 等替代方案的配置方法、
  离线缓存管理策略及权重完整性校验手段。
  当用户提到模型下载慢、HuggingFace 连接超时、ModelScope、权重缓存、
  snapshot_download 等问题时触发。
metadata:
  short-description: NPU 模型权重下载镜像与加速配置
  category: NPU-Model-Deploy
  tags: [ascend, npu, modelscope, huggingface, download, mirror, cache, deployment]
---

# 昇腾 NPU 模型权重下载镜像与加速配置 Skill

本 Skill 解决国内环境部署 vLLM-Ascend 时，从 HuggingFace 下载模型权重遇到的网络问题。
以 `google/gemma-3-270m-it` 和 `manycore-research/SpatialLM-Llama-1B`
在 Atlas 800 A2 (NPU 910B4) 上的部署经验为参考。

## 问题背景

国内网络环境访问 HuggingFace (huggingface.co) 经常出现：
- 连接超时 (`Connection timed out`)
- 下载中断 (`Read timed out`)
- SSL 证书错误
- 速度极慢 (< 10 KB/s)

本 Skill 提供 3 种解决方案，按推荐程度排序。

---

## 方案一：ModelScope（推荐）

ModelScope 是阿里开源的模型平台，在国内访问稳定，且与 HuggingFace 生态兼容。

### vLLM 集成配置

```bash
export VLLM_USE_MODELSCOPE=true
```

设置后，vLLM 的 `AutoModel` 加载流程会自动将 HuggingFace 模型 ID 映射到 ModelScope 对应模型，无需修改代码。

### 手动下载

```python
from modelscope import snapshot_download

# 下载完整模型到本地目录
model_dir = snapshot_download(
    "google/gemma-3-270m-it",      # ModelScope 模型 ID
    cache_dir="/opt/atomgit/weights"  # 自定义缓存目录
)
print(f"模型已下载到: {model_dir}")
```

### 指定版本/分支

```python
model_dir = snapshot_download(
    "google/gemma-3-270m-it",
    revision="master",    # 或具体 commit hash / tag
    cache_dir="/opt/atomgit/weights"
)
```

### 只下载特定文件

```python
from modelscope import snapshot_download

model_dir = snapshot_download(
    "google/gemma-3-270m-it",
    cache_dir="/opt/atomgit/weights",
    allow_patterns=["*.safetensors", "config.json", "tokenizer.json"]
)
```

---

## 方案二：HF-Mirror 镜像

如果不使用 ModelScope，可通过镜像站加速 HuggingFace 访问。

### 设置镜像 Endpoint

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 配合 huggingface-cli 使用

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download google/gemma-3-270m-it \
  --local-dir /opt/atomgit/weights/gemma-3-270m-it \
  --local-dir-use-symlinks False
```

### 环境变量组合

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1   # 启用高速传输（需安装 hf-transfer）
```

---

## 方案三：离线缓存管理

当网络完全不可用时，可预先在有网络的环境下载模型，再通过磁盘/U 盘迁移到目标 NPU 服务器。

### 默认缓存路径

| 工具 | 默认缓存路径 |
|------|-------------|
| HuggingFace Hub | `~/.cache/huggingface/hub` |
| ModelScope | `~/.cache/modelscope/hub` |

### 自定义缓存目录

```bash
# HuggingFace
export HF_HOME=/opt/atomgit/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/opt/atomgit/cache/huggingface/hub

# ModelScope
export MODELSCOPE_CACHE=/opt/atomgit/cache/modelscope
```

### 缓存迁移步骤

1. 在有网络的服务器下载模型
2. 打包缓存目录或模型文件夹
3. 传输到 NPU 服务器
4. 在 NPU 服务器上指定本地路径加载

```bash
# vLLM 直接加载本地路径
vllm serve /opt/atomgit/weights/gemma-3-270m-it \
  --tensor-parallel-size 1 \
  --dtype bfloat16
```

---

## 权重完整性校验

下载完成后应验证权重文件完整性，避免损坏导致运行时错误。

### 校验文件列表

标准 HuggingFace 模型至少应包含：

```
config.json
tokenizer.json  (或 tokenizer_config.json + vocab 文件)
*.safetensors    (或 pytorch_model.bin)
generation_config.json  (可选)
```

### 快速校验脚本

```python
import os
import json

def check_model_weights(model_path: str) -> bool:
    """检查模型权重目录是否完整。"""
    required = ["config.json"]
    optional = ["tokenizer.json", "tokenizer_config.json"]

    has_weights = any(
        f.endswith(".safetensors") or f.startswith("pytorch_model")
        for f in os.listdir(model_path)
    )

    missing = [f for f in required if not os.path.exists(os.path.join(model_path, f))]

    if missing:
        print(f"[FAIL] 缺失必需文件: {missing}")
        return False
    if not has_weights:
        print("[FAIL] 未找到权重文件 (.safetensors 或 pytorch_model.bin)")
        return False

    # 检查 config.json 可解析
    try:
        with open(os.path.join(model_path, "config.json")) as f:
            config = json.load(f)
        print(f"[OK] 架构: {config.get('architectures', ['unknown'])[0]}")
    except Exception as e:
        print(f"[WARN] config.json 解析失败: {e}")

    print("[OK] 权重目录检查通过")
    return True

if __name__ == "__main__":
    import sys
    check_model_weights(sys.argv[1])
```

### safetensors 校验

```python
from safetensors import safe_open

# 验证每个 safetensors 文件可读
for f in os.listdir(model_path):
    if f.endswith(".safetensors"):
        with safe_open(os.path.join(model_path, f), framework="pt") as fd:
            print(f"[OK] {f}: keys={list(fd.keys())}")
```

---

## 常见下载报错与修复

| 报错信息 | 原因 | 修复方案 |
|---------|------|---------|
| `Connection timed out` / `Read timed out` | HuggingFace 被墙或网络不稳定 | `export VLLM_USE_MODELSCOPE=true` 或 `export HF_ENDPOINT=https://hf-mirror.com` |
| `SSLCertVerificationError` | SSL 证书验证失败 | `export CURL_CA_BUNDLE=` 或 `export REQUESTS_CA_BUNDLE=` 临时禁用；或更新系统 CA 证书 |
| `RepoNotFoundError` | 模型 ID 错误或平台未同步 | 确认 ModelScope 上存在该模型；部分 HF 新模型可能未同步到 MS |
| `OSError: no space left on device` | 缓存盘空间不足 | `export HF_HOME=/data/xxx` 或 `export MODELSCOPE_CACHE=/data/xxx` 指向大容量分区 |
| `File exists` 或权限错误 | 缓存目录权限问题 | 确保当前用户对缓存目录有读写权限；`chmod -R 755 <cache_dir>` |
| 下载进度卡在 0% | DNS 解析失败 | 更换 DNS 为 `223.5.5.5` / `114.114.114.114`；或使用 ModelScope |

---

## 一键检查脚本

使用本 Skill 提供的脚本检查下载环境：

```bash
python scripts/download_env_check.py
```

脚本功能：
- 检测网络连通性（HuggingFace / ModelScope / HF-Mirror）
- 检查环境变量配置
- 验证缓存目录权限与空间

---

## 完整配置模板

### 启动脚本模板（国内环境）

```bash
#!/bin/bash
# 国内环境模型下载加速配置

# 推荐：使用 ModelScope
export VLLM_USE_MODELSCOPE=true

# 可选：自定义缓存目录（大容量分区）
export MODELSCOPE_CACHE=/opt/atomgit/cache/modelscope

# 备选：使用 HF-Mirror
# export HF_ENDPOINT=https://hf-mirror.com
# export HF_HOME=/opt/atomgit/cache/huggingface

# 下载并启动
vllm serve google/gemma-3-270m-it \
  --tensor-parallel-size 1 \
  --dtype bfloat16
```

---

## 参考

- ModelScope 文档：<https://www.modelscope.cn/docs/intro/model-download>
- HuggingFace Hub 文档：<https://huggingface.co/docs/huggingface_hub/guides/download>
- HF-Mirror：<https://hf-mirror.com/>
