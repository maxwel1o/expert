---
description: ConvFormer NPU 部署 Skill - 用于 ConvFormer 系列模型在昇腾 NPU 上的统一部署与验证
---

# ConvFormer NPU 部署 Skill

本 Skill 提供 ConvFormer 系列图像分类模型在昇腾 Ascend NPU 上的完整部署流程。

## 快速开始

```bash
# 安装依赖
pip install timm torchvision Pillow safetensors modelscope

# 运行 NPU 推理
python scripts/inference.py --model convformer_b36.sail_in1k_384 --device npu --image test.jpg

# CPU/NPU 精度对比
python scripts/compare_cpu_npu.py --model convformer_b36.sail_in1k_384 --image test.jpg
```

## 目录结构

```
convformer-npu/
├── SKILL.md                 # Skill 定义文件
├── README.md                # 本文档
├── scripts/
│   ├── run_model.sh          # 单模型全流程脚本
│   ├── inference.py          # 推理脚本模板
│   ├── compare_cpu_npu.py    # 精度对比脚本模板
│   ├── generate_readme.py    # README 生成器
│   └── model_list.txt        # 支持的模型列表
└── examples/
    ├── example_inference.py  # 推理示例
    └── example_batch.sh      # 批量串行示例
```

详情请见 `SKILL.md`。
