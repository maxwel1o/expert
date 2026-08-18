# SelecSLS 系列模型昇腾 NPU 部署 Skill

## 简介

本 Skill 用于自动完成 **SelecSLS 系列图像分类模型** 在昇腾 NPU 上的部署、推理、CPU/NPU 精度验证、README 文档生成、终端截图生成和模型仓库发布。

### 支持的模型

| 模型名称 | 参数量 | 模型仓库 |
|:---|:---:|:---|
| `selecsls60b.in1k` | 32.8M | [selecsls60b-npu](https://gitcode.com/m0_74196153/selecsls60b-npu) |
| `selecsls60.in1k` | 30.7M | [selecsls60-npu](https://gitcode.com/m0_74196153/selecsls60-npu) |
| `selecsls42b.in1k` | 32.5M | [selecsls42b-npu](https://gitcode.com/m0_74196153/selecsls42b-npu) |

## 环境要求

- Python 3.11+
- PyTorch 2.0+
- torch_npu (用于 NPU 推理)
- timm 1.0+
- Pillow, numpy

## 使用方法

### 1. 安装依赖

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple timm torch_npu pillow numpy
```

### 2. 单个模型推理

```bash
# CPU 推理
python3 scripts/inference.py --model selecsls60b.in1k --image test_image.jpg --device cpu

# NPU 推理
python3 scripts/inference.py --model selecsls60b.in1k --image test_image.jpg --device npu
```

### 3. CPU/NPU 精度对比

```bash
python3 scripts/compare_cpu_npu.py --model selecsls60b.in1k --image test_image.jpg
```

### 4. 串行执行多个模型

使用 `scripts/run_all.sh` 可串行执行全部 3 个模型，避免显存爆炸：

```bash
bash scripts/run_all.sh
```

该脚本会依次：
1. 对每个模型执行 CPU 推理
2. 对每个模型执行 NPU 推理
3. 对每个模型执行 CPU/NPU 精度对比
4. 生成终端截图
5. 释放内存和 NPU 显存后处理下一个模型

### 5. 生成 README 文档

精度验证通过后，可根据测试结果编写模型 README（参考各模型仓库中的 README.md）。

### 6. 生成终端截图

```bash
python3 /path/to/terminal_screenshot.py --input terminal_output.txt --output terminal_screenshot.png
```

### 7. 发布模型仓库到 GitCode

创建和推送模型仓库：

```bash
# 使用 GitCode API 创建仓库
curl -X POST "https://api.gitcode.com/api/v5/user/repos" \
  --header "PRIVATE-TOKEN: ${ATOMGIT_USER_TOKEN}" \
  --header "Content-Type: application/json" \
  --data '{"name": "{model_name}-npu", "license": "cc-by-4.0"}'

# 推送代码
git init
git checkout -b main
git remote add origin "https://auth:${ATOMGIT_USER_TOKEN}@gitcode.com/{username}/{repo}.git"
git push -u origin main
```

## 输入参数

| 参数 | 类型 | 必填 | 说明 |
|:---|:---|:---:|:---|
| `--model` | string | 是 | 模型名称 |
| `--device` | string | 否 | cpu 或 npu（默认 npu） |
| `--image` | string | 否 | 测试图像路径 |

## 输出结果

- `cpu_results.json` — CPU 推理结果（top-5 类别、置信度、推理耗时）
- `npu_results.json` — NPU 推理结果
- `compare_results.json` — CPU/NPU 精度对比结果
- `terminal_screenshot.png` — 模拟终端输出截图
- `README.md` — 模型中文文档

## 精度要求

NPU 与 CPU 推理结果误差必须 < 1%。

## 串行执行说明

为防止 NPU 显存或内存爆炸，多个模型必须串行执行：

1. 完成一个模型的全部推理和精度测试
2. 释放资源（`gc.collect()`, `torch.npu.empty_cache()`）
3. 再处理下一个模型

## 注意事项

1. 使用 ModelScope 或 hf-mirror.com 下载模型权重（HF 官方可能无法访问）
2. 不要在精度测试完成前提交模型仓库
3. 每个模型独立提交仓库，不混合多个模型
4. 模型仓库使用 `main` 分支
5. README 需包含真实测试数据和截图
