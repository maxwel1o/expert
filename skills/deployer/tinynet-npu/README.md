# TinyNet NPU 部署 Skill

## 概述

本 Skill 用于在昇腾 NPU 上自动完成 TinyNet 系列模型（tinynet_e / tinynet_d / tinynet_c / tinynet_b / tinynet_a）的：
- 模型权重下载与推理（CPU/NPU）
- CPU vs NPU 精度对比验证
- 推理结果与精度报告生成
- 终端截图生成

## 支持的模型列表

| 模型名称 | 模型仓库地址 |
|----------|-------------|
| tinynet_e.in1k | [tinynet_e-npu](https://gitcode.com/m0_74196153/tinynet_e-npu) |
| tinynet_d.in1k | [tinynet_d-npu](https://gitcode.com/m0_74196153/tinynet_d-npu) |
| tinynet_c.in1k | [tinynet_c-npu](https://gitcode.com/m0_74196153/tinynet_c-npu) |
| tinynet_b.in1k | [tinynet_b-npu](https://gitcode.com/m0_74196153/tinynet_b-npu) |
| tinynet_a.in1k | [tinynet_a-npu](https://gitcode.com/m0_74196153/tinynet_a-npu) |

## Skill 输入参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| model | string | 是 | 模型名称（如 `tinynet_e`、`tinynet_d` 等） |
| device | string | 否 | 推理设备 `cpu` 或 `npu`，默认为 `npu` |

## 依赖环境

- Python 3.10+
- PyTorch 2.0+
- torch_npu（昇腾 NPU 插件）
- timm 0.9.0+
- torchvision
- modelscope（用于下载权重）

## 如何使用

### 1. 安装依赖

```bash
pip install torch torch_npu timm torchvision Pillow numpy modelscope safetensors
```

### 2. 执行 NPU 推理

```bash
cd scripts/
python inference.py --model tinynet_e --device npu
```

### 3. 执行 CPU/NPU 精度对比

```bash
cd scripts/
python compare_cpu_npu.py --model tinynet_e
```

### 4. 串行执行多个模型（避免显存爆炸）

```bash
# 按模型从小到大的顺序执行
for model in tinynet_e tinynet_d tinynet_c tinynet_b tinynet_a; do
    echo "=== Processing $model ==="
    python compare_cpu_npu.py --model $model
    # 自动释放 NPU 显存
    python -c "import torch; torch.npu.empty_cache()"
done
```

## 输出结果

- `{model}_cpu_result.json` - CPU 推理结果
- `{model}_npu_0_result.json` - NPU 推理结果
- `{model}_compare_result.json` - CPU/NPU 精度对比结果
- `{model}_terminal_output.txt` - 模拟终端输出文本
- `{model}_terminal_output.html` - 终端输出 HTML 截图

## 精度测试指标

每次精度对比自动计算以下指标：

1. **余弦相似度 (Cosine Similarity)** - 衡量输出向量的方向一致性
2. **最大绝对误差 (Max Absolute Error)** - 所有元素中的最大差异
3. **平均绝对误差 (Mean Absolute Error)** - 所有元素的平均差异
4. **均方根误差 (RMSE)** - 输出差异的均方根
5. **平均相对误差 (Relative Error)** - 相对差异百分比
6. **Top-1 一致性** - 最高概率类别是否一致
7. **Top-5 重叠数** - 前 5 预测类别的重叠数

## 完整测试结果

| 模型 | CPU 耗时 | NPU 耗时 | 加速比 | 余弦相似度 | Top-1 一致 |
|------|---------|---------|--------|-----------|-----------|
| tinynet_e | 9.20 ms | 4.63 ms | 1.99x | 0.99999863 | ✓ |
| tinynet_d | 14.11 ms | 4.36 ms | 3.24x | 0.99999845 | ✓ |
| tinynet_c | 22.53 ms | 10.80 ms | 2.09x | 0.99999851 | ✓ |
| tinynet_b | 34.78 ms | 7.67 ms | 4.53x | 0.99999976 | ✓ |
| tinynet_a | 46.69 ms | 7.78 ms | 6.00x | 0.99999964 | ✓ |

所有模型 NPU 与 CPU 推理误差 < 1%，精度验证通过。

## 模型仓库发布

本 Skill 自动使用 GitCode API 发布模型仓库：

```python
# 创建模型仓库示例
POST https://api.gitcode.com/api/v5/user/repos
{
    "name": "tinynet_e-npu",
    "repository_type": "model",
    "visibility": "public"
}

# 推送代码
git remote add origin https://auth:${ATOMGIT_USER_TOKEN}@gitcode.com/<username>/<repo>.git
git push -u origin main
```

已发布仓库：
- https://gitcode.com/m0_74196153/tinynet_e-npu
- https://gitcode.com/m0_74196153/tinynet_d-npu
- https://gitcode.com/m0_74196153/tinynet_c-npu
- https://gitcode.com/m0_74196153/tinynet_b-npu
- https://gitcode.com/m0_74196153/tinynet_a-npu

## 注意事项

1. 模型按 tinynet_e → tinynet_d → tinynet_c → tinynet_b → tinynet_a 的顺序从轻到重执行
2. 每个模型测试完成后自动释放 NPU 显存，避免显存爆炸
3. 模型权重通过 ModelScope 下载，无需访问 HuggingFace
4. README 遵循 Apache-2.0 许可，包含 YAML frontmatter 标签
