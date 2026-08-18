# SE-Net NPU 部署 Skill

## 简介

本 Skill 提供 SE-Net 系列模型在昇腾 NPU 上的自动化部署、推理测试、CPU/NPU 精度对比、README 生成和模型仓库发布能力。

支持以下 9 个 SE-Net 变体模型：

| 模型名称 | 架构 | 参数量 | GitCode 仓库 |
|----------|------|--------|-------------|
| seresnet50.a1_in1k | SE-ResNet50 | 28M | [seresnet50.a1_in1k-npu](https://gitcode.com/m0_74196153/seresnet50.a1_in1k-npu) |
| seresnet50.ra2_in1k | SE-ResNet50 | 28M | [seresnet50.ra2_in1k-npu](https://gitcode.com/m0_74196153/seresnet50.ra2_in1k-npu) |
| seresnet50.a3_in1k | SE-ResNet50 | 28M | [seresnet50.a3_in1k-npu](https://gitcode.com/m0_74196153/seresnet50.a3_in1k-npu) |
| seresnet50.a2_in1k | SE-ResNet50 | 28M | [seresnet50.a2_in1k-npu](https://gitcode.com/m0_74196153/seresnet50.a2_in1k-npu) |
| seresnet33ts.ra2_in1k | SE-ResNet33ts | 16M | [seresnet33ts.ra2_in1k-npu](https://gitcode.com/m0_74196153/seresnet33ts.ra2_in1k-npu) |
| seresnet152d.ra2_in1k | SE-ResNet152d | 67M | [seresnet152d.ra2_in1k-npu](https://gitcode.com/m0_74196153/seresnet152d.ra2_in1k-npu) |
| senet154.gluon_in1k | SENet154 | 115M | [senet154.gluon_in1k-npu](https://gitcode.com/m0_74196153/senet154.gluon_in1k-npu) |
| sehalonet33ts.ra2_in1k | SE-HaLO-Net33ts | 22M | [sehalonet33ts.ra2_in1k-npu](https://gitcode.com/m0_74196153/sehalonet33ts.ra2_in1k-npu) |
| sebotnet33ts_256.a1h_in1k | SE-BotNet33ts | 21M | [sebotnet33ts_256.a1h_in1k-npu](https://gitcode.com/m0_74196153/sebotnet33ts_256.a1h_in1k-npu) |

## 输入参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model_name | str | seresnet50.a1_in1k | 模型名称（必须为以上列表中的模型） |
| device | str | npu | 推理设备：cpu 或 npu |
| image | str | test_image.jpg | 输入图像路径 |

## 输出结果

- 推理日志：模型加载信息、推理延迟、Top-5 预测结果
- 精度对比报告：CPU vs NPU 的误差指标表格
- 终端截图：模拟终端输出的推理过程截图
- 模型仓库：每个模型推送到独立的 GitCode 模型仓库

## 环境要求

- 昇腾 NPU 设备（Ascend 910B/910）
- CANN 8.5.1+
- PyTorch 2.0+ / torch_npu
- timm 0.9+

## 使用方法

### 1. 单个模型推理

```bash
export HF_ENDPOINT=https://hf-mirror.com
python3 examples/example.py --model-name seresnet50.a1_in1k --device npu
```

### 2. 执行 CPU/NPU 精度对比

```bash
# 先 CPU 推理
python3 examples/example.py --model-name seresnet50.a1_in1k --device cpu --output output_cpu.pt
# 再 NPU 推理
python3 examples/example.py --model-name seresnet50.a1_in1k --device npu --output output_npu.pt
# 对比
python3 -c "
import torch
cpu = torch.load('output_cpu.pt')
npu = torch.load('output_npu.pt')
diff = torch.abs(cpu - npu)
print(f'Max error: {diff.max().item():.8f}')
print(f'Relative error: {diff.max().item()/torch.abs(cpu).max().item()*100:.6f}%')
"
```

### 3. 串行执行多个模型

为避免 NPU 显存爆炸，模型必须串行执行。批量处理脚本：

```bash
for model in seresnet50.a1_in1k seresnet50.ra2_in1k seresnet50.a3_in1k; do
    echo "Processing $model..."
    python3 examples/example.py --model-name "$model" --device npu
    python3 -c "import gc, torch; gc.collect(); torch.npu.empty_cache()"
    sleep 3
done
```

### 4. 生成终端截图

```bash
python3 /opt/atomgit/terminal_screenshot.py \
    --input inference_output.txt \
    --output terminal_screenshot.png
```

### 5. 发布模型仓库到 GitCode

```bash
# 创建模型仓库（每个模型独立）
curl -X POST "https://api.gitcode.com/api/v5/user/repos" \
  -H "Authorization: Bearer ${ATOMGIT_USER_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "seresnet50.a1_in1k-npu",
    "repository_type": "model",
    "visibility": "public"
  }'

# 推送代码
git remote add origin https://auth:${ATOMGIT_USER_TOKEN}@gitcode.com/m0_74196153/seresnet50.a1_in1k-npu.git
git branch -M main
git push -u origin main
```

## Skill 文件结构

```
skills/deployment/senet-npu/
├── skill.json          # Skill 元数据
├── README.md           # 本文档
├── scripts/
│   └── inference.sh    # 推理部署脚本
└── examples/
    └── example.py      # 使用示例
```

## 精度验证结论

所有 9 个 SE-Net 模型在昇腾 NPU 上的推理精度均通过验证：
- **NPU 与 CPU 推理结果误差 < 1%**
- 所有模型相对误差均 < 0.2%
- Top-5 类别完全一致（5/5）
- 余弦相似度 > 0.99999

## 性能数据

| 模型 | CPU 延迟 (ms) | NPU 延迟 (ms) | 加速比 |
|------|:----------:|:----------:|:----:|
| seresnet50.a1_in1k | 198.98 | 7.33 | 27.1x |
| seresnet50.ra2_in1k | 199.76 | 7.02 | 28.5x |
| seresnet50.a3_in1k | 121.23 | 6.76 | 17.9x |
| seresnet50.a2_in1k | 203.58 | 7.04 | 28.9x |
| seresnet33ts.ra2_in1k | 228.91 | 5.41 | 42.3x |
| seresnet152d.ra2_in1k | 789.71 | 68.36 | 11.6x |
| senet154.gluon_in1k | 1069.19 | 21.95 | 48.7x |
| sehalonet33ts.ra2_in1k | 205.91 | 6.56 | 31.4x |
| sebotnet33ts_256.a1h_in1k | 230.21 | 6.78 | 34.0x |

## 注意事项

1. 串行执行：多个模型必须串行推理，防止 NPU 显存溢出
2. 释放资源：每个模型推理后执行 `gc.collect()` 和 `torch.npu.empty_cache()`
3. 模型缓存：预训练权重缓存在 `~/.cache/huggingface/hub/` 目录，首次下载较慢
4. 镜像加速：设置 `HF_ENDPOINT=https://hf-mirror.com` 加速模型下载
