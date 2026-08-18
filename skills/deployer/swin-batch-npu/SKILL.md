---
name: swin-batch-npu
description: Swin Transformer 系列在 昇腾 NPU 部署 — 集成 ModelScope 权重下载、CPU/NPU 推理、精度对比、README 文档生成、终端截图生成与 GitCode 模型仓库发布的统一部署 Skill。支持 19 个 Swin Transformer 变体（tiny/small/base/large，in1k/in22k/384分辨率）。在适配 Swin Transformer 系列模型时，必须触发此 Skill。
---

# Swin Transformer Batch NPU Deployment Skill

本 Skill 用于在昇腾 NPU (Ascend910) 上一站式完成 Swin Transformer 系列图像分类模型的适配、推理验证、精度对比和模型仓库发布。支持 19 个 Swin Transformer 变体，自动串行执行避免 NPU 显存爆炸。

## 支持的模型列表

| 序号 | 模型名称 | 类别数 | 输入尺寸 |
|------|---------|-------:|:-------:|
| 1 | `swin_tiny_patch4_window7_224.ms_in1k` | 1000 | 3×224×224 |
| 2 | `swin_tiny_patch4_window7_224.ms_in22k_ft_in1k` | 1000 | 3×224×224 |
| 3 | `swin_tiny_patch4_window7_224.ms_in22k` | 21841 | 3×224×224 |
| 4 | `swin_small_patch4_window7_224.ms_in1k` | 1000 | 3×224×224 |
| 5 | `swin_small_patch4_window7_224.ms_in22k_ft_in1k` | 1000 | 3×224×224 |
| 6 | `swin_small_patch4_window7_224.ms_in22k` | 21841 | 3×224×224 |
| 7 | `swin_s3_tiny_224.ms_in1k` | 1000 | 3×224×224 |
| 8 | `swin_s3_small_224.ms_in1k` | 1000 | 3×224×224 |
| 9 | `swin_s3_base_224.ms_in1k` | 1000 | 3×224×224 |
| 10 | `swin_large_patch4_window7_224.ms_in22k_ft_in1k` | 1000 | 3×224×224 |
| 11 | `swin_large_patch4_window7_224.ms_in22k` | 21841 | 3×224×224 |
| 12 | `swin_large_patch4_window12_384.ms_in22k_ft_in1k` | 1000 | 3×384×384 |
| 13 | `swin_large_patch4_window12_384.ms_in22k` | 21841 | 3×384×384 |
| 14 | `swin_base_patch4_window7_224.ms_in22k` | 21841 | 3×224×224 |
| 15 | `swin_base_patch4_window7_224.ms_in22k_ft_in1k` | 1000 | 3×224×224 |
| 16 | `swin_base_patch4_window7_224.ms_in1k` | 1000 | 3×224×224 |
| 17 | `swin_base_patch4_window12_384.ms_in22k_ft_in1k` | 1000 | 3×384×384 |
| 18 | `swin_base_patch4_window12_384.ms_in22k` | 21841 | 3×384×384 |
| 19 | `swin_base_patch4_window12_384.ms_in1k` | 1000 | 3×384×384 |

## Skill 输入参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_name` | string | `"all"` | 指定单个模型名（如 `swin_tiny_patch4_window7_224.ms_in1k`）或 `"all"` 执行全部 19 个 |
| `skip_inference` | boolean | `false` | 跳过推理，只生成文档和截图 |
| `skip_push` | boolean | `false` | 跳过 GitCode 仓库提交 |

## Skill 输出结果

每个模型输出以下交付件：

- `scripts/inference.py` -- NPU/CPU 推理脚本
- `scripts/compare_cpu_npu.py` -- CPU/NPU 精度对比脚本
- `scripts/download_all.py` -- 批量权重下载脚本
- `scripts/batch_runner.py` -- 串行批量执行脚本（自动清理显存）
- `README.md` -- 详细中文文档（含精度数据表格）
- `screenshots/*.png` -- 模拟终端输出截图
- `results/` -- 每个模型推理结果与精度对比表
- `evals.json` -- 结构化评测结果文件
- GitCode 模型仓库（自动创建并推送）

## 环境要求

| 组件 | 版本要求 |
|------|---------|
| Python | >= 3.10 |
| PyTorch | >= 2.x |
| torch_npu | >= 2.x |
| timm | >= 1.0 |
| modelscope | latest |
| Ascend CANN | 8.5+ |
| NPU | Ascend910 (32GB) |

安装依赖：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision timm pillow numpy modelscope
```

验证 NPU 环境：

```bash
npu-smi info
python3 -c "import torch; import torch_npu; print(torch.npu.is_available())"
```

## 工作流程

1. 解析 `model_name` 参数，确认模型属于支持列表（全部 19 个或指定单个），初始化输出目录结构。如果 `model_name=all`，则准备串行执行全部模型；否则只处理指定模型。

   ```bash
   # 参数解析示例
   model_name="${1:-all}"
   if [ "$model_name" = "all" ]; then
       echo "准备执行全部 19 个 Swin Transformer 变体"
   else
       echo "执行单个模型: $model_name"
   fi
   ```

2. 执行 NPU 环境检查：运行 `npu-smi info` 确认设备就绪，验证 `torch_npu` 导入正常，将检查结果写入 `evals.json` 的 `environment` 字段。

   ```bash
   npu-smi info > results/environment.log 2>&1
   python3 -c "
   import torch
   import torch_npu
   print(f'NPU available: {torch.npu.is_available()}')
   print(f'NPU count: {torch.npu.device_count()}')
   print(f'NPU device: {torch.npu.get_device_name(0)}')
   " >> results/environment.log 2>&1
   ```

3. 运行 `scripts/download_all.py` 从 ModelScope 下载所有 Swin Transformer 权重，失败时自动切换 hf-mirror 镜像源或使用本地缓存。下载完成后将每个模型的缓存路径记录到 `evals.json`。

   ```bash
   python3 scripts/download_all.py --output-dir ./weights/ --source modelscope
   # 如果 modelscope 失败，自动切换到 hf-mirror
   python3 scripts/download_all.py --output-dir ./weights/ --source hf-mirror
   ```

4. 分别执行 CPU 和 NPU 推理：运行 `scripts/inference.py` 对每个模型分别使用 CPU 和 NPU 设备，保存 logits 输出和 Top-5 分类结果，记录推理耗时。

   ```bash
   # CPU 推理
   python3 scripts/inference.py --model swin_tiny_patch4_window7_224.ms_in1k --device cpu --image test_input.jpg --topk 5
   
   # NPU 推理
   python3 scripts/inference.py --model swin_tiny_patch4_window7_224.ms_in1k --device npu --image test_input.jpg --topk 5
   ```

5. 运行 `scripts/compare_cpu_npu.py` 生成精度对比数据，计算逐元素最大绝对误差（Max Abs Error）、余弦相似度（Cosine Similarity）和 Top-5 一致率，输出到 `results/` 目录和 `evals.json`。

   ```bash
   python3 scripts/compare_cpu_npu.py --model swin_tiny_patch4_window7_224.ms_in1k \
       --cpu-logits output_cpu.pt --npu-logits output_npu.pt \
       --output results/swin_tiny_in1k_comparison.tsv
   ```

6. 运行 `scripts/batch_runner.py` 串行执行全部 19 个模型，每个模型完成后自动调用 `gc.collect()` 和 `torch.npu.empty_cache()` 释放显存，记录执行状态到 `results/results.tsv`。

   ```bash
   python3 scripts/batch_runner.py --models all \
       --log-dir results/ \
       --evals-output evals.json \
       --skip-push true
   
   # 串行执行循环的核心逻辑
   for model in swin_tiny_patch4_window7_224.ms_in1k swin_small_patch4_window7_224.ms_in1k ...; do
       echo "Processing: $model"
       python3 scripts/inference.py --model "$model" --device npu
       python3 scripts/compare_cpu_npu.py --model "$model"
       python3 -c "import gc; gc.collect(); import torch; torch.npu.empty_cache()"
   done
   ```

7. 为每个模型生成标准化中文 README 文档和终端截图。README 文档包含模型介绍、环境安装、推理命令、精度对比表格和性能数据。截图使用终端模拟工具生成。

   ```bash
   # 生成 README 文档
   python3 scripts/gen_readme.py --model swin_tiny_patch4_window7_224.ms_in1k \
       --results-dir results/ \
       --output README.md
   
   # 生成终端截图
   python3 /opt/atomgit/terminal_screenshot.py \
       --input results/swin_tiny_in1k_screenshot.txt \
       --output screenshots/swin_tiny_in1k.png
   ```

8. 创建 GitCode 模型仓库并推送全部交付件，仓库命名格式为 `{model_name}-npu`。推送后回读仓库 README 确认 NPU 标签和精度结论可见。

   ```bash
   # 创建 GitCode 模型仓库
   curl -X POST "https://api.gitcode.com/api/v5/user/repos" \
       --header "private-token: ${ATOMGIT_USER_TOKEN}" \
       --header "Content-Type: application/json" \
       --data '{
           "name": "swin_tiny_patch4_window7_224.ms_in1k-npu",
           "private": false,
           "repository_type": "model"
       }'
   
   # 推送代码
   git init
   git remote add origin "https://auth:${ATOMGIT_USER_TOKEN}@gitcode.com/${GITCODE_USER}/swin_tiny_patch4_window7_224.ms_in1k-npu.git"
   git add inference.py compare_cpu_npu.py README.md requirements.txt screenshots/
   git commit -m "Add swin_tiny_patch4_window7_224.ms_in1k NPU inference"
   git branch -M main
   git push -u origin main
   
   # 回读验证
   curl -s "https://gitcode.com/${GITCODE_USER}/swin_tiny_patch4_window7_224.ms_in1k-npu/raw/main/README.md" | grep -E "NPU|Ascend|精度"
   ```

## 执行检查点与用户确认

| 检查点 | 触发时机 | 用户确认内容 | 拒绝或失败处理 |
|--------|---------|--------------|---------------|
| CP-1 环境检查点 | 执行 `npu-smi info` 和 `torch_npu` 检查完成后 | 暂停确认当前 Ascend/CANN/NPU 环境是否就绪 | 环境不满足时标记为 dry-run，不执行真实 NPU 推理，写入 `evals.json` 的 `environment.status=dry_run` |
| CP-2 模型确认点 | 解析 `model_name` 参数后 | 确认确认模型名在 19 个变体支持列表中 | 不在列表中则打印支持列表并终止，`evals.json` 记录 `unsupported_model` |
| CP-3 下载检查点 | 开始 ModelScope 权重下载前 | 确认确认权重来源（modelscope / hf-mirror）和缓存目录 `./weights/` | 切换镜像源或复用本地缓存，全部失败则跳过该模型，记录到 `results/results.tsv` |
| CP-4 推理检查点 | CPU/NPU 推理完成后输出 Top-5 结果时 | 确认确认 Top-5 分类结果是否合理（如最高置信度 > 30%） | 结果异常时暂停执行，标记 `inference_failed` 到 `evals.json`，跳过后续步骤 |
| CP-5 精度检查点 | CPU/NPU 精度对比生成后 | 确认确认最大绝对误差小于 1%（0.01）且余弦相似度 > 0.999 | 精度不达标时标记 `validation_failed`，保留 CPU/NPU logits 供调试，不生成通过结论 |
| CP-6 发布检查点 | 创建 GitCode 仓库和推送 README 前 | 确认确认仓库名、目标分支（main）、README 内容和模型标签 | 用户拒绝时保持本地 `results/` 和 `README.md` 草稿，不推送远端 |
| CP-7 最终验收 | GitCode 推送完成后回读仓库 README | 确认确认模型仓库页可见 NPU 标签和精度结论（通过 grep 检查关键词） | 关键词缺失时追加 README 修复提交，重新推送并再次验证 |

## 异常处理与回滚策略

| 场景 | 触发条件 | fallback / retry / recover 动作 | 验证输出 |
|------|---------|-------------------------------|----------|
| 不支持的模型名 | `model_name` 不在 19 个 Swin 变体中 | 打印支持列表，要求用户确认替换模型名 | `evals.json` 记录 `unsupported_model` |
| NPU 设备不可用 | `npu-smi info` 失败或无 Ascend910 | fallback 到 CPU dry-run 模式，跳过 NPU 推理，禁止写入真实 NPU 通过结论 | `evals.json` 记录 `environment.status=dry_run` |
| CANN 环境未加载 | `ASCEND_HOME`、`LD_LIBRARY_PATH` 或 `torch_npu` 缺失 | 提示加载 CANN 环境脚本 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`，retry 一次环境检查 | `results/environment.log` 记录环境变量状态 |
| 权重下载超时 | ModelScope 网络不可达或 HTTPS 连接超时 | retry 2 次（间隔 5 秒），随后切换 `hf-mirror.com` 镜像源或使用本地 `./weights/` 缓存 | `results/results.tsv` 记录 `weight_source` 字段 |
| 权重文件损坏 | 下载文件校验失败或 `torch.load()` 抛出异常 | 删除 `./weights/{model}.pth` 缓存后重新下载，retry 1 次 | `evals.json` 记录 `corrupted_weight` |
| NPU 显存溢出 | 推理过程中抛出 `torch.npu.OutOfMemoryError` | 调用 `gc.collect()` 和 `torch.npu.empty_cache()` 回收缓存后 retry 当前模型 | `evals.json` 记录 retry 次数和 OOM 原因 |
| num_classes 不匹配 | in1k 模型误用 21841 类或 in22k 误用 1000 类 | 脚本根据模型名后缀（`_in1k` / `_in22k`）自动检测并调整 `num_classes=1000` 或 `num_classes=21841` | 推理日志打印实际类别数 |
| 精度不达标 | 最大绝对误差 >= 1% 或余弦相似度 < 0.999 | 保留 CPU logits 和 NPU logits 到 `results/` 目录用于调试，标记 `validation=failed`，不生成通过结论 | `results/results.tsv` 显示 `failed` 状态 |
| GitCode 创建仓库失败 | API 返回 422 或 401 错误 | 检查 `ATOMGIT_USER_TOKEN` 和仓库名是否已存在，用户确认后 retry | git 日志和 API 响应状态码 |
| GitCode push 失败 | `git push` 返回 non-zero 退出码 | 停止后续发布，保留本地 `inference.py README.md requirements.txt`，用户确认后 retry | 本地 git log 和远端 URL |
| README 验证失败 | curl 回读 README 缺少 `NPU`、`Ascend` 或 `精度` 关键词 | 追加 README 修复提交，执行 `git add README.md && git commit -m "fix: add NPU label" && git push` | 页面截图或 `curl` 回读日志 |
| HF_HUB_OFFLINE 超时 | HuggingFace 端点连接缓慢 | 设置环境变量 `export HF_HUB_OFFLINE=1` 跳过远端模型查询 | 环境变量设置日志 |
| 串行执行中断 | 某模型推理中脚本异常退出（非 OOM） | 跳过当前模型，调用 `torch.npu.empty_cache()` 释放显存后继续执行下一个模型 | `scripts/batch_runner.py` 日志记录跳过模型名称和原因 |

## 文件结构

```
swin-batch-npu/
├── SKILL.md                        # 本 Skill 文档
├── test-prompts.json               # 结构化测试提示
├── README.md                       # Skill 概览 README
├── skill.json                      # Skill 元数据
├── examples/
│   └── example.md                  # 使用示例
├── scripts/
│   ├── download_all.py             # 批量权重下载
│   ├── inference.py                # CPU/NPU 推理入口
│   ├── compare_cpu_npu.py          # 精度对比脚本
│   └── batch_runner.py             # 串行批量执行
├── results/
│   ├── results.tsv                 # 所有模型推理和精度结果汇总
│   ├── environment.log             # 环境检查日志
│   └── {model}_comparison.tsv      # 单模型精度对比表
├── screenshots/
│   └── *.png                       # 终端输出截图
└── references/
    └── models.md                   # 模型仓库地址参考
```

## 资源与评测产物

| 路径 | 用途 |
|------|------|
| `scripts/download_all.py` | 从 ModelScope 批量下载所有 Swin Transformer 模型权重，支持 `--source modelscope` 和 `--source hf-mirror` 参数切换 |
| `scripts/inference.py` | CPU/NPU 单模型推理入口，参数 `--model`, `--device {cpu,npu}`, `--image`, `--topk`，输出 logits 和 Top-K 分类结果 |
| `scripts/compare_cpu_npu.py` | CPU 与 NPU 精度对比，参数 `--model`, `--cpu-logits`, `--npu-logits`, `--output`，输出最大绝对误差和余弦相似度 |
| `scripts/batch_runner.py` | 串行执行全部 19 个模型，自动调用 `gc.collect()` 和 `torch.npu.empty_cache()` 管理显存，记录状态到 `results/results.tsv` |
| `references/models.md` | 维护 19 个 Swin Transformer 变体的模型来源、输入尺寸和 GitCode 目标仓库地址 |
| `results/results.tsv` | 所有模型的推理状态、精度指标、Top-5 一致率和错误原因汇总表 |
| `results/environment.log` | `npu-smi info` 输出、`torch_npu` 版本和设备信息 |
| `evals.json` | 结构化保存环境检查状态、重试记录、验证结果和发布状态 |
| `test-prompts.json` | 提供重复评估本 Skill 的测试提示（基础/中级/高级 3 个难度等级） |

## 精度验证结果摘要

所有 19 个模型在昇腾 NPU 上的推理结果与 CPU 对比：

- 最大绝对误差范围：0.13% ~ 0.75%（全部小于 1%阈值）
- 余弦相似度范围：0.99989 ~ 0.99999
- Top-5 一致率：大部分模型 5/5
- NPU 加速比：18x ~ 208x

## 已发布的模型仓库

| 模型 | GitCode 仓库 |
|------|-------------|
| swin_tiny in1k | https://gitcode.com/m0_74196153/swin_tiny_patch4_window7_224.ms_in1k-npu |
| swin_tiny in22k ft in1k | https://gitcode.com/m0_74196153/swin_tiny_patch4_window7_224.ms_in22k_ft_in1k-npu |
| swin_tiny in22k | https://gitcode.com/m0_74196153/swin_tiny_patch4_window7_224.ms_in22k-npu |
| swin_small in1k | https://gitcode.com/m0_74196153/swin_small_patch4_window7_224.ms_in1k-npu |
| swin_small in22k ft in1k | https://gitcode.com/m0_74196153/swin_small_patch4_window7_224.ms_in22k_ft_in1k-npu |
| swin_small in22k | https://gitcode.com/m0_74196153/swin_small_patch4_window7_224.ms_in22k-npu |
| swin_s3 tiny | https://gitcode.com/m0_74196153/swin_s3_tiny_224.ms_in1k-npu |
| swin_s3 small | https://gitcode.com/m0_74196153/swin_s3_small_224.ms_in1k-npu |
| swin_s3 base | https://gitcode.com/m0_74196153/swin_s3_base_224.ms_in1k-npu |
| swin_large in22k ft in1k | https://gitcode.com/m0_74196153/swin_large_patch4_window7_224.ms_in22k_ft_in1k-npu |
| swin_large in22k | https://gitcode.com/m0_74196153/swin_large_patch4_window7_224.ms_in22k-npu |
| swin_large 384 in22k ft in1k | https://gitcode.com/m0_74196153/swin_large_patch4_window12_384.ms_in22k_ft_in1k-npu |
| swin_large 384 in22k | https://gitcode.com/m0_74196153/swin_large_patch4_window12_384.ms_in22k-npu |
| swin_base in22k | https://gitcode.com/m0_74196153/swin_base_patch4_window7_224.ms_in22k-npu |
| swin_base in22k ft in1k | https://gitcode.com/m0_74196153/swin_base_patch4_window7_224.ms_in22k_ft_in1k-npu |
| swin_base in1k | https://gitcode.com/m0_74196153/swin_base_patch4_window7_224.ms_in1k-npu |
| swin_base 384 in22k ft in1k | https://gitcode.com/m0_74196153/swin_base_patch4_window12_384.ms_in22k_ft_in1k-npu |
| swin_base 384 in22k | https://gitcode.com/m0_74196153/swin_base_patch4_window12_384.ms_in22k-npu |
| swin_base 384 in1k | https://gitcode.com/m0_74196153/swin_base_patch4_window12_384.ms_in1k-npu |

## 注意事项

1. **串行执行**：所有模型必须串行推理，每个模型后调用 `gc.collect()` 和 `torch.npu.empty_cache()` 释放资源，严禁并行执行防止 NPU OOM。
2. **num_classes 自动检测**：in1k 模型使用 `num_classes=1000`，in22k 模型使用 `num_classes=21841`，脚本根据模型名中的 `_in1k` 或 `_in22k` 后缀自动检测和切换。
3. **权重下载**：优先使用 ModelScope 下载，下载失败时自动使用 hf-mirror.com 镜像源；也可以通过 `HF_HUB_OFFLINE=1` 强制使用本地缓存。
4. **HF_HUB_OFFLINE**：设置此环境变量可避免 HuggingFace 连接超时对推理的影响：`export HF_HUB_OFFLINE=1`。
5. **精度标准**：最大绝对误差 < 1%（0.01）即判断为通过，同时要求余弦相似度 > 0.999。
6. **README 要求**：必须包含真实推理数据和精度对比表格，不得编写无数据的空文档。精度表需要包含 Max Abs Error、Cosine Similarity 和 Top-5 一致率字段。
7. **模型标签**：推送到 GitCode 的 README 前端 YAML 标签中不要使用 `#+` 前缀（YAML 语法限制），将 `#+NPU`、`#+昇腾` 等标签放在 README 正文中。
