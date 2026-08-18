# SKNet NPU 部署示例

## 环境要求

- Ascend910 NPU
- Python 3.11+
- PyTorch 2.x + torch_npu
- timm ≥ 0.9.0

## 安装依赖

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple timm torchvision Pillow numpy safetensors
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple modelscope
```

## 下载模型权重

```python
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('timm/skresnext50_32x4d.ra_in1k', cache_dir='./cache')
snapshot_download('timm/skresnet34.ra_in1k', cache_dir='./cache')
snapshot_download('timm/skresnet18.ra_in1k', cache_dir='./cache')
```

## 运行单个模型

```bash
# 单模型全部流程（推理+对比+截图）
python3 scripts/deploy.py --model skresnext50_32x4d.ra_in1k --action all

# 仅精度对比
python3 scripts/deploy.py --model skresnet34.ra_in1k --action compare

# 仅推理
python3 scripts/deploy.py --model skresnet18.ra_in1k --action inference
```

## 串行运行所有模型

```bash
# 串行运行全部3个模型（避免显存爆炸）
python3 scripts/deploy.py --model skresnext50_32x4d.ra_in1k --serial --action all
```

## 使用 Shell 脚本

```bash
chmod +x scripts/run.sh
./scripts/run.sh skresnext50_32x4d.ra_in1k all
./scripts/run.sh skresnet34.ra_in1k compare
```

## 推理代码示例

```python
import torch
import timm

# NPU 推理
model = timm.create_model("skresnext50_32x4d.ra_in1k", pretrained=True)
model = model.to("npu:0")
model.eval()

from PIL import Image
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img = Image.open("test.jpg").convert("RGB")
input_tensor = transform(img).unsqueeze(0).to("npu:0")

with torch.no_grad():
    output = model(input_tensor)
print(output.argmax(dim=1))
```

## 已发布模型仓库

| 模型 | GitCode 仓库 |
| --- | --- |
| skresnext50_32x4d.ra_in1k | https://gitcode.com/m0_74196153/skresnext50_32x4d.ra_in1k-npu |
| skresnet34.ra_in1k | https://gitcode.com/m0_74196153/skresnet34.ra_in1k-npu |
| skresnet18.ra_in1k | https://gitcode.com/m0_74196153/skresnet18.ra_in1k-npu |

## 注意事项

1. 多个模型必须串行执行，防止 NPU 显存或内存爆炸
2. 每个模型测试完成后释放资源：`torch.npu.empty_cache()` + `gc.collect()`
3. 模型权重优先从 ModelScope 下载，必要时使用 hf-mirror
4. README 中的精度数据必须是真实测试结果
