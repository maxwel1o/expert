# 部署指南

## 快速启动

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
python3 scripts/inference.py --device npu --image input.jpg
```

## 模型权重

从 ModelScope 下载：
```python
from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download('timm/convnext_base.clip_laion2b')
```

## 精度验证

运行 `python3 scripts/eval_benchmark.py` 对比 CPU/NPU 输出差异。
