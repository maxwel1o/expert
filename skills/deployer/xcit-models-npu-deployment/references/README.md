# XCiT 系列模型 GitCode 仓库参考

本目录包含 XCiT 系列 5 个模型的 GitCode 模型仓库地址及关键信息。

## 模型仓库列表

| # | 模型名称 | timm 标识 | 参数量 | GitCode 仓库 |
|---|---------|-----------|:------:|-------------|
| 1 | XCiT-Tiny-12/16 | `xcit_tiny_12_p16_384.fb_dist_in1k` | 6.7M | https://gitcode.com/gcw_C8PI9e90/xcit_tiny_12_p16_384-npu |
| 2 | XCiT-Tiny-12/8 | `xcit_tiny_12_p8_384.fb_dist_in1k` | 6.7M | https://gitcode.com/gcw_C8PI9e90/xcit_tiny_12_p8_384-npu |
| 3 | XCiT-Small-12/8 | `xcit_small_12_p8_224.fb_in1k` | 26.2M | https://gitcode.com/gcw_C8PI9e90/xcit_small_12_p8_224-npu |
| 4 | XCiT-Medium-24/16 | `xcit_medium_24_p16_384.fb_dist_in1k` | 84.4M | https://gitcode.com/gcw_C8PI9e90/xcit_medium_24_p16_384-npu |
| 5 | XCiT-Large-24/8 | `xcit_large_24_p8_224.fb_in1k` | 188.9M | https://gitcode.com/gcw_C8PI9e90/xcit_large_24_p8_224-npu |

## 仓库结构

每个模型仓库包含以下文件：

```
{model_name}-npu/
├── readme.md                 # 中文适配验证报告（含精度数据和性能数据）
├── inference.py              # NPU/CPU 推理脚本
├── accuracy_eval.py          # 50 样本精度验证脚本
├── compare_cpu_npu.py        # CPU/NPU 精度对比入口
├── accuracy_result.json      # 精度验证结果（JSON）
├── requirements.txt          # Python 依赖
└── terminal_screenshot.png   # 终端输出截图
```

## 精度验证结果汇总

| 模型 | 最大误差 | 余弦相似度 | Top-1一致率 |
|------|:-------:|:---------:|:----------:|
| xcit_tiny_12_p16_384 | 0.042 | 0.99998730 | 74.0% |
| xcit_tiny_12_p8_384 | 0.074 | 0.99994910 | 58.0% |
| xcit_small_12_p8_224 | 0.063 | 0.99998361 | 76.0% |
| xcit_medium_24_p16_384 | 0.069 | 0.99995112 | 70.0% |
| xcit_large_24_p8_224 | 0.086 | 0.99992698 | 80.0% |

所有模型余弦相似度 > 0.9999，误差远小于 1%。

## 模型来源

- ModelScope: https://www.modelscope.cn/models/timm
- HuggingFace: https://huggingface.co/timm
