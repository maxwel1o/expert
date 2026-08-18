# Xception NPU Deployment Usage Examples

## 1. 运行单个模型的 CPU 推理

```python
from modelscope.hub.snapshot_download import snapshot_download
import torch
import timm
from PIL import Image
from safetensors.torch import load_file

model_name = "xception71.tf_in1k"
model_path = snapshot_download(f"timm/{model_name}")

state_dict = load_file(f"{model_path}/model.safetensors")
model = timm.create_model(model_name, pretrained=False)
model.load_state_dict(state_dict, strict=True)
model = model.eval()

img = Image.new("RGB", (299, 299), color=(128, 128, 128))
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

input_tensor = transforms(img).unsqueeze(0)
with torch.no_grad():
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)
    top5_probs, top5_idxs = torch.topk(probs, k=5)

print("Top-5 predictions:")
for i in range(5):
    print(f"  {i+1}: class {top5_idxs[0][i].item()} ({top5_probs[0][i].item()*100:.2f}%)")
```

## 2. 运行单个模型的 NPU 推理

```python
import torch
import timm
from PIL import Image
from safetensors.torch import load_file
from modelscope.hub.snapshot_download import snapshot_download

model_name = "xception71.tf_in1k"
model_path = snapshot_download(f"timm/{model_name}")

state_dict = load_file(f"{model_path}/model.safetensors")
model = timm.create_model(model_name, pretrained=False)
model.load_state_dict(state_dict, strict=True)
model = model.eval().to("npu:0")

img = Image.new("RGB", (299, 299), color=(128, 128, 128))
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

input_tensor = transforms(img).unsqueeze(0).to("npu:0")
with torch.no_grad():
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)
    top5_probs, top5_idxs = torch.topk(probs, k=5)

print("Top-5 predictions:")
for i in range(5):
    print(f"  {i+1}: class {top5_idxs[0][i].item()} ({top5_probs[0][i].item()*100:.2f}%)")
```

## 3. 串行运行全部 6 个模型

```bash
python3 scripts/run_all_serial.py
```

## 4. CPU/NPU 精度对比

```bash
cd /path/to/model_directory
python3 compare_cpu_npu.py
```

## 5. 生成终端截图

```bash
python3 scripts/generate_screenshot.py --log inference.log --output screenshot.html
```

## 6. 发布到 GitCode

```bash
export ATOMGIT_USER_TOKEN="your_token"
cd /path/to/model_directory
git init
git remote add origin https://auth:${ATOMGIT_USER_TOKEN}@gitcode.com/m0_74196153/{model_name}-npu.git
git add -A
git commit -m "Add {model_name} NPU adaptation"
git branch -M main
git push -u origin main
```
