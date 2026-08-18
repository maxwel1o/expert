---
name: neutts-nano-npu-deploy
description: >
  NeuTTS-Nano 多语言 TTS 模型系列（英/德/法/西）在昇腾 NPU 上的部署 Skill。
  涵盖环境准备、模型下载、neutts + torch_npu 推理、精度验证与性能基准测试。
  当用户提到 NeuTTS、NeuTTS-Nano、TTS NPU 部署、语音合成 NPU 推理、neutts-nano 时触发。
metadata:
  short-description: NeuTTS-Nano 多语言 TTS 昇腾 NPU 部署
  category: NPU-Model-Deploy
  tags: [ascend, npu, neutts, tts, text-to-speech, llama, pytorch, multilingual]
---

# NeuTTS-Nano 多语言 TTS 昇腾 NPU 部署 Skill

本 Skill 提供 **NeuTTS-Nano** 多语言 TTS 模型系列在华为昇腾 NPU 上的完整部署、
推理验证和精度验证流程。支持 4 种语言：

| 语言 | 模型 | HuggingFace |
|------|------|-------------|
| 英语 (en-us) | neutts-nano | [neuphonic/neutts-nano](https://huggingface.co/neuphonic/neutts-nano) |
| 德语 (de) | neutts-nano-german | [neuphonic/neutts-nano-german](https://huggingface.co/neuphonic/neutts-nano-german) |
| 法语 (fr-fr) | neutts-nano-french | [neuphonic/neutts-nano-french](https://huggingface.co/neuphonic/neutts-nano-french) |
| 西班牙语 (es) | neutts-nano-spanish | [neuphonic/neutts-nano-spanish](https://huggingface.co/neuphonic/neutts-nano-spanish) |

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend910 系列（至少 1 卡，32GB HBM） |
| OS | openEuler / Ubuntu（aarch64 或 x86_64） |
| CANN | >= 8.0（推荐 25.5.1） |
| Python | 3.9 – 3.13 |
| 网络 | 首次运行需联网下载模型权重（~913MB/个）及 NeuCodec |

## 详细执行步骤

0. 环境初始化 — 加载 CANN 环境，设置 `ASCEND_RT_VISIBLE_DEVICES`，配置镜像加速
1. 安装依赖 — pip 安装 `neutts`, `soundfile`, `torch_npu`
2. NPU 验证 — 通过 `torch_npu` 确认设备就绪
3. 模型下载 — 从 HuggingFace 下载目标语言模型权重
4. 基础推理验证 — 运行 `inference.py` 生成对应语言 WAV 文件
5. 精度验证 — 执行 `verify_deterministic.py` 对比 CPU 与 NPU logit
6. 验收确认 — 检查所有检查点清单
7. 异常处理 — 根据错误场景执行对应的回滚策略
8. 资源归档 — 将评测产物整理至 `reports/` 和 `results/` 目录
9. 环境清理 — 释放 NPU 显存，关闭无关进程
10. 多语言扩展 — 切换模型目录，对其他语言重复步骤 3-5
11. 性能基准测试 — 记录各语言推理耗时与加速比
12. 结果汇总 — 生成全语言精度对比表
13. 故障排查 — 参考常见问题表，定位并修复异常
14. 复查确认 — 重新运行所有检查点，确保无遗漏
15. 文档整理 — 更新 README 记录实测数据

按以下各节顺序执行，每步完成后再进入下一步。

---

## 流程总览

```bash
# 一键串行执行（需根据实际路径调整）
export SKILL_DIR=<skill_dir>
cd /tmp/neutts-workdir && \
  source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
  export ASCEND_RT_VISIBLE_DEVICES=0 && \
  export HF_ENDPOINT=https://hf-mirror.com && \
  pip install neutts soundfile && \
  python3 -c "import torch; import torch_npu; print('NPU ready:', torch_npu.__version__)" && \
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download('neuphonic/neutts-nano', local_dir='./neutts-nano')" && \
  cp ${SKILL_DIR}/scripts/inference.py . && \
  python3 inference.py --model ./neutts-nano --text "Hello world." --output output_en.wav && \
  python3 ${SKILL_DIR}/scripts/verify_deterministic.py
```

---

## 0. 环境初始化

```bash
# 加载 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 选择空闲 NPU
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=0

# HuggingFace 镜像（国内加速）
export HF_ENDPOINT=https://hf-mirror.com

# 华为 pip 镜像（国内加速）
export PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
```

**验证环境**:
```bash
python3 -c "import torch; print('PyTorch OK:', torch.__version__)"
```

---

## 1. 安装依赖

```bash
pip install neutts soundfile
```

安装完成后验证版本：

```bash
python3 -c "import torch; import torch_npu; print('torch:', torch.__version__); print('torch_npu:', torch_npu.__version__)"
```

**预期输出**：`torch` 与 `torch_npu` 版本一致（如 2.9.0）。

---

## 2. NPU 基础验证

```bash
python3 -c "
import torch
import torch_npu
print('torch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
print('NPU count:', torch.npu.device_count())
print('Device:', torch.npu.get_device_name(0))
a = torch.randn(3, 4).npu()
print(a + a)
"
```

**通过标准**：输出包含 `device='npu:0'` 的 Tensor 且无报错。

---

## 3. 模型下载

选择目标语言，使用 HF 国内镜像下载权重（~913MB/个）：

```bash
export HF_ENDPOINT=https://hf-mirror.com

# 英语
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('neuphonic/neutts-nano', local_dir='./neutts-nano')"

# 德语
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('neuphonic/neutts-nano-german', local_dir='./neutts-nano-german')"

# 法语
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('neuphonic/neutts-nano-french', local_dir='./neutts-nano-french')"

# 西班牙语
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('neuphonic/neutts-nano-spanish', local_dir='./neutts-nano-spanish')"
```

---

## 4. 基础推理验证

复制推理脚本到工作目录：

```bash
cp <skill_dir>/scripts/inference.py .
```

运行推理（`--model` 指向目标语言目录，语言参数自动适配）：

**英语**:
```bash
python3 inference.py --model ./neutts-nano --text "Hello world, this is a test." --output output_en.wav
```

**德语**:
```bash
python3 inference.py --model ./neutts-nano-german --text "Hallo Welt, dies ist ein Test." --output output_de.wav
```

**法语**:
```bash
python3 inference.py --model ./neutts-nano-french --text "Bonjour le monde, ceci est un test." --output output_fr.wav
```

**西班牙语**:
```bash
python3 inference.py --model ./neutts-nano-spanish --text "Hola mundo, esta es una prueba." --output output_es.wav
```

> **说明**：脚本内置 espeak-ng 回退补丁（文本直通模式），无需安装 espeak-ng 即可运行。

**通过标准**：
- 成功生成 output_XX.wav（24kHz 单声道）
- 运行过程无 NPU 报错

---

## 5. 精度验证

NPU 与 CPU 在同权重、同输入下的 backbone 前向 logits 对比。

```bash
python3 scripts/verify_deterministic.py
```

**通过标准**：

| 指标 | 阈值 | 实测值 |
|------|------|--------|
| 相对误差 | < 1% | **< 0.0001%** |
| Top-1 token 一致率 | 100% | **100%** |

各语言实测精度：

| 模型 | 相对误差 | Top-1 | 加速比 |
|------|---------|-------|--------|
| neutts-nano | 0.00008% | 100% | 4.03x |
| neutts-nano-german | 0.00005% | 100% | 42.86x |
| neutts-nano-french | 0.00007% | 100% | 40.67x |
| neutts-nano-spanish | 0.00007% | 100% | 21.45x |

> **说明**：NeuTTS 使用 `do_sample=True, temperature=1.0` 随机采样，波形对比无意义。
> 正确方法是比较相同输入下 backbone 的 logit 输出。详情见 `reports/verification_report_deterministic.json`。

---

## 6. 验收确认

- [ ] `npu-smi info` 显示设备正常
- [ ] `import torch_npu` 无报错
- [ ] `inference.py` 成功生成 wav 文件
- [ ] 精度验证相对误差 < 1%
- [ ] 生成的语音可听懂且与输入文本一致
- [ ] 所有4种语言均验证通过
- [ ] 实测结果记录在 `reports/` 目录下

---

## 7. 异常处理与回滚策略

| 异常场景 | 可能原因 | 处理方案 | 回滚操作 |
|----------|----------|----------|----------|
| `torch_npu` 导入失败 | CANN 环境未加载或 torch_npu 未安装 | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 后 `pip install torch_npu` | 回退到步骤 0，重新配置环境 |
| 模型下载失败 | 网络不可达或 HF 镜像不可用 | 切换 `HF_ENDPOINT=https://hf-mirror.com`；若仍失败改用 `git lfs clone` | 检查网络连通性，重试步骤 3 |
| NPU 显存不足 (OOM) | 文本过长超过 2048 tokens | 缩短输入文本至 200 tokens 以内；检查其他进程是否占用 NPU | `npu-smi info` 查看占用，释放后重试步骤 4 |
| `neutts` pip 安装失败 | Python 版本不兼容或依赖冲突 | 确认 Python 3.9-3.13；使用 `pip install --upgrade neutts` | 回退到步骤 1，检查 Python 版本 |
| 推理输出无声或异常 | espeak-ng 缺失导致音素化异常 | 脚本内置回退补丁，无需 espeak-ng；如仍需可 `apt-get install espeak-ng` | 重试推理，添加 `--debug` 参数 |
| NPU 推理报错 `device-side assert` | 模型与 NPU 算子不兼容 | 确认 CANN 版本 >= 8.0；尝试 `torch.npu.set_device(0)` | `npu-smi info` 检查设备状态 |
| 精度验证失败 (相对误差 > 1%) | 权重加载错误或 dtype 不匹配 | 确认模型完整下载；检查 `torch.set_default_dtype(torch.float16)` | 重新下载模型权重，重试步骤 5 |
| 多语言模型切换错误 | 模型路径指向错误的语言目录 | 确认 `--model` 参数指向正确的子目录 | 检查目录结构，重新执行步骤 4 对应语言 |
| `soundfile` 写入失败 | 磁盘空间不足或权限问题 | `df -h` 检查磁盘；`chmod +w .` 确保可写 | 清理磁盘后重试步骤 4 |
| Git LFS 下载中断 | 网络不稳定或 LFS 配额超限 | `GIT_LFS_SKIP_SMUDGE=1` 跳过 LFS，用 `huggingface_hub` 替代 | 切换到 HF API 下载，重试步骤 3 |
| CANN 版本不兼容 | `npu-smi info` 报驱动错误 | 升级 CANN 至 >= 8.0；确认驱动与固件匹配 | `npu-smi info -t health` 诊断 |
| `torch_npu` 版本与 torch 不匹配 | pip 安装了错误的版本组合 | `pip install torch_npu==2.9.0` 与 torch 版本对齐 | 卸载后固定版本重新安装 |

---

## 8. 执行检查点与用户确认

| 检查点 | 确认项 | 验证命令 | 通过条件 |
|--------|--------|----------|----------|
| 环境初始化完成 | NPU 设备就绪 | `npu-smi info` | 显示 NPU 设备且状态正常 |
| 依赖安装完成 | torch_npu 可导入 | `python3 -c "import torch_npu; print(torch_npu.__version__)"` | 无 ImportError |
| NPU 基础验证通过 | NPU 计算正常 | `python3 -c "import torch; import torch_npu; print(torch.randn(3,4).npu() + torch.randn(3,4).npu())"` | 输出 tensor 在 npu:0 上 |
| 模型下载完成 | 权重文件完整 | `ls ./neutts-nano/*.safetensors 2>/dev/null || ls ./neutts-nano/*.bin` | 至少一个模型文件存在 |
| 基础推理通过 | WAV 文件生成 | `file output_en.wav` | 文件类型为 "RIFF (little-endian) data, WAVE" |
| 精度验证通过 | 相对误差 < 1% | `python3 scripts/verify_deterministic.py` | 脚本输出 `all_pass: true` |
| 多语言测试完成 | 全部4种语言验证 | 依次运行 en/de/fr/es 推理命令 | 各自生成对应 WAV 文件且无报错 |
| 验收确认完成 | 所有检查点通过 | 确认清单全部勾选 | 无未通过的检查项 |
| 资源归档确认 | 评测产物已保存 | `ls reports/` | 包含 verification_report_deterministic.json |

---

## 9. 资源与评测产物

| 资源 | 路径 | 说明 |
|------|------|------|
| 推理脚本 | `scripts/inference.py` | NeuTTS-Nano NPU 推理（espeak-ng 回退补丁） |
| 精度验证脚本 | `scripts/verify_deterministic.py` | CPU vs NPU logit 对比 |
| 精度评测报告 | `reports/verification_report_deterministic.json` | 各语言相对误差、Top-1 一致率、加速比 |
| 测试提示词 | `test-prompts.json` | 多语言测试 prompt 模板 |
| 推理产出示例 | `results/` | 生成的 WAV 文件及推理日志 |
| 模型权重缓存 | `references/` | HuggingFace 模型引用与版本记录 |
| 评测结果汇总 | `evals.json` | 全语言精度对比汇总表 |
| 环境配置快照 | `scripts/setup.sh` | 一键环境初始化脚本 |

---

## 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `No module named 'torch_npu'` | 未安装或 CANN 环境未加载 | `source set_env.sh` 后重装 torch_npu |
| 模型下载失败 | 网络问题 | 使用 `HF_ENDPOINT=https://hf-mirror.com` |
| OOM | 文本过长 | 缩短输入文本（< 200 tokens） |
| 推理结果异常 | 音素化质量 | 安装 `apt-get install espeak-ng` |
| 多语言切换失败 | 模型目录错误 | 确认 `--model` 指向正确语言子目录 |

---

## 附录：模型适配要点速查

| 特征 | NeuTTS-Nano 值 | 说明 |
|------|----------------|------|
| 基座模型 | LLaMA backbone (~229M / ~117M active) | Causal LM 自回归生成 |
| 音频编解码器 | NeuCodec (50Hz, 单码本) | 24kHz 单声道输出 |
| 精度 | float16 (NPU) | 推理精度无损 |
| 推理框架 | neutts 1.1.0 + torch_npu | 零代码修改 |
| 上下文窗口 | 2048 tokens | ~30s 音频 |
| 适配策略 | espeak-ng monkey-patch + torch_npu 设备注册 | 无需修改模型代码 |
