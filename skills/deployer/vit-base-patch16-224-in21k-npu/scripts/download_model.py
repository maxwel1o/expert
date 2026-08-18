#!/usr/bin/env python3
"""从 ModelScope 下载 vit-base-patch16-224-in21k 模型权重"""
import os
import sys

# 设置 ModelScope 缓存目录
os.environ['MODELSCOPE_CACHE'] = '/opt/atomgit/vit-base-patch16-224-npu/model_weights'

from modelscope import snapshot_download

model_id = 'google/vit-base-patch16-224-in21k'
local_dir = snapshot_download(model_id, cache_dir=os.environ['MODELSCOPE_CACHE'])
print(f"Model downloaded to: {local_dir}")

# 写入路径记录
with open('/opt/atomgit/vit-base-patch16-224-npu/model_weights/model_path.txt', 'w') as f:
    f.write(local_dir)
