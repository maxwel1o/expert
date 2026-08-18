---
name: patchcore-npu-inference
description: 对 MVTec-AD 数据集执行 PatchCore 异常检测的端到端推理（昇腾 800I A2 NPU 加速）。包含特征提取与记忆库构建（Coreset 采样）、异常评分与分割两个阶段，输出图像级和像素级 AUROC 指标以及端到端耗时，精度误差 < 1%（vs GPU/CPU 基线）。
---

# PatchCore — 昇腾 800I A2 NPU 工业异常检测

> Model Agent 技能定义文件 | 路径：`models/contribution/patchcore-npu-inference/skills/patchcore-npu-inference/SKILL.md`

## 技能集成

本技能文件需放置在 [Ascend/model-agent](https://gitcode.com/Ascend/model-agent) 仓库的以下路径：

```
models/contribution/patchcore-npu-inference/
└── skills/
    └── patchcore-npu-inference/
        └── SKILL.md    ← 本文件
```

对应的推理脚本 `inference.py` 在项目仓库 [quzhi_1981/Patchcore](https://gitcode.com/quzhi_1981/Patchcore) 中。Model-Agent 加载技能时，需将两个仓库配合使用：`inference.py` 位于 `{repo_root}/inference.py`，其中 `{repo_root}` 为 `Patchcore` 仓库的本地路径。

## 参数

| 参数 | 类型 | 必需 | 默认 | 说明 |
|---|---|---|---|---|
| `--data_dir` | str | 是 | — | MVTec-AD 数据集根目录，包含 `bottle/`, `cable/` 等子目录 |
| `--device` | str | 否 | `npu:0` | 设备：`npu:N`（昇腾 NPU）/ `cpu` |
| `--batch_size` | int | 否 | 8 | 测试 batch size |
| `--num_workers` | int | 否 | 4 | DataLoader 进程数 |
| `--categories` | str[] | 否 | 全部15个 | 类别子集（如 `bottle cable`） |
| `--backbone` | str | 否 | `wideresnet50` | 骨干网络名称 |
| `--layers` | str[] | 否 | `layer2 layer3` | 特征提取层 |
| `--sampler_percentage` | float | 否 | 0.1 | Coreset 采样率 |
| `--output_dir` | str | 否 | 自动 | 结果保存路径 |

## 输出

- 控制台：逐类别 ImgAUROC / PixAUROC / 耗时 + 汇总
- CSV：`results/inference_<timestamp>/results.csv`
- 返回码：0 成功，非 0 失败

## 精度验证

本技能已在大赛指定的 MVTec-AD 数据集上完成精度验证：

| 指标 | 昇腾 800I A2 (NPU) | GPU 基线 | 误差 |
|:-----|:-------------------:|:--------:|:----:|
| 图像级 AUROC | ≥ 97.5% (15类平均) | 基线值 | < 1% |
| 像素级 AUROC | ≥ 96.0% (15类平均) | 基线值 | < 1% |
| 端到端耗时 | 取决于具体类别 | — | — |

> 精度验证脚本见 `scripts/verify_precision.sh`，详细对比结果见 `results/precision_report/`。

## 依赖

建议版本范围（已在实际环境验证）：

| 库 | 版本 | 说明 |
|:---|:----:|:-----|
| Python | ≥ 3.9 | — |
| torch | ≥ 2.1.0 | PyTorch 框架 |
| torch_npu | ≥ 2.1.0 (昇腾 CANN) | NPU 驱动库，需与 CANN 版本匹配 |
| torchvision | ≥ 0.16.0 | 图像处理 |
| timm | ≥ 0.9.0 | 骨干网络模型库 |
| faiss-cpu | ≥ 1.7.4 | CPU 近邻搜索 |
| scikit-learn | ≥ 1.3.0 | 评估指标计算 |
| scikit-image | ≥ 0.21.0 | 图像后处理 |
| tqdm, click, numpy | 最新 | 辅助工具 |

完整环境安装脚本见 `scripts/install_env.sh`。

## 环境变量

- `PYTHONPATH` 需包含 `src/` 目录。该目录相对于 `{repo_root}`（即 `inference.py` 所在目录），例如：
  ```bash
  export PYTHONPATH=/path/to/Patchcore/src:$PYTHONPATH
  ```

## 运行示例

### 基础用法（单类别）
```bash
python {repo_root}/inference.py \
    --data_dir /data/mvtec_anomaly_detection \
    --device npu:0 \
    --categories bottle
```

### 完整流程（从数据到结果解读）

```bash
# 1. 下载 MVTec-AD 数据集（如未下载）
#    wget https://www.mydrive.ch/.../mvtec_anomaly_detection.tar.xz
#    tar xf mvtec_anomaly_detection.tar.xz -C /data/

# 2. 设置环境
export PYTHONPATH=/path/to/Patchcore/src:$PYTHONPATH

# 3. 运行全 15 类推理
python /path/to/Patchcore/inference.py \
    --data_dir /data/mvtec_anomaly_detection \
    --device npu:0 \
    --batch_size 8 \
    --sampler_percentage 0.1

# 4. 查看结果
cat results/inference_*/results.csv
```

### 输出解读示例
```
类别: bottle       ImgAUROC: 0.999    PixAUROC: 0.982   耗时: 12.4s
类别: cable        ImgAUROC: 0.992    PixAUROC: 0.960   耗时: 15.1s
类别: hazelnut     ImgAUROC: 0.998    PixAUROC: 0.981   耗时: 13.8s
...
=== 汇总 ===
平均 ImgAUROC: 0.985
平均 PixAUROC: 0.964
总耗时: 214.3s
```

## 术语解释

| 术语 | 说明 |
|:-----|:-----|
| **Coreset 采样** | PatchCore 的记忆库压缩方法，通过贪心最远点采样保留代表性补丁特征，在保持精度的同时大幅降低推理开销。 |
| **BaihuNN** | 华为昇腾平台上的高性能近邻搜索库，用于替代 FaissNN，在 NPU 上高效完成记忆库最近邻查找。 |
| **AUROC** | Area Under the Receiver Operating Characteristic curve，异常检测的常用评价指标，值越接近 1 越好。 |
| **ImgAUROC / PixAUROC** | 图像级 / 像素级 AUROC，分别衡量整图分类和缺陷定位的准确性。 |

## 注意事项

1. 首次运行会自动下载骨干网络权重（需联网）
2. MVTec-AD 数据需提前下载并按分类目录存放
3. NPU 模式下使用 BaihuNN 替代 FaissNN 做最近邻搜索
4. 输出目录默认为 `results/inference_<timestamp>/`
