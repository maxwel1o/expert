---
name: sambert-hifigan-tts-npu
description: >
  Sambert-HifiGAN 语音合成模型在昇腾 NPU 上的完整部署、推理与精度测试 Skill。
  支持 7 个模型（中文 4 模型 7 声音、英文美式 3 模型 4 声音）的自动下载、NPU 推理、
  CPU/NPU 精度对比、测试报告生成、终端截图生成和 GitCode 模型仓库发布。
  可在任意 Ascend910 系列服务器上一键复现。
  当用户提到 Sambert、HifiGAN、TTS、语音合成、昇腾、NPU 时触发。
metadata:
  short-description: Sambert-HifiGAN TTS 昇腾 NPU 部署与精度测试
  category: NPU-Model-Deploy
  tags: [ascend, npu, sambert, hifigan, tts, speech, pytorch, inference]
---

# Sambert-HifiGAN TTS 昇腾 NPU 部署与精度测试 Skill

本 Skill 提供阿里通义语音实验室 Sambert-HifiGAN 系列语音合成模型在华为昇腾 Ascend910 NPU 上的完整部署、推理验证和 CPU/NPU 精度对比的标准化可复现流程。

## 支持的模型列表

| # | 模型名称 | 声音 | 语言 | 采样率 |
|---|----------|------|:----:|:------:|
| 1 | `speech_sambert-hifigan_tts_zh-cn_16k` | zhibei_emo, zhitian_emo, zhiyan_emo, zhizhe_emo | zh-cn | 16kHz |
| 2 | `speech_sambert-hifigan_tts_zhitian_emo_zh-cn_16k` | zhitian_emo | zh-cn | 16kHz |
| 3 | `speech_sambert-hifigan_tts_zhiyan_emo_zh-cn_16k` | zhiyan_emo | zh-cn | 16kHz |
| 4 | `speech_sambert-hifigan_tts_zhizhe_emo_zh-cn_16k` | zhizhe_emo | zh-cn | 16kHz |
| 5 | `speech_sambert-hifigan_tts_en-us_16k` | andy, annie | en-us | 16kHz |
| 6 | `speech_sambert-hifigan_tts_andy_en-us_16k` | andy | en-us | 16kHz |
| 7 | `speech_sambert-hifigan_tts_annie_en-us_16k` | annie | en-us | 16kHz |

### GitCode 模型仓库地址

| 模型 | 仓库地址 |
|------|---------|
| zh-cn_16k | [speech_sambert-hifigan_tts_zh-cn_16k-npu](https://gitcode.com/gcw_C8PI9e90/speech_sambert-hifigan_tts_zh-cn_16k-npu) |
| zhitian_emo_zh-cn_16k | [speech_sambert-hifigan_tts_zhitian_emo_zh-cn_16k-npu](https://gitcode.com/gcw_C8PI9e90/speech_sambert-hifigan_tts_zhitian_emo_zh-cn_16k-npu) |
| zhiyan_emo_zh-cn_16k | [speech_sambert-hifigan_tts_zhiyan_emo_zh-cn_16k-npu](https://gitcode.com/gcw_C8PI9e90/speech_sambert-hifigan_tts_zhiyan_emo_zh-cn_16k-npu) |
| zhizhe_emo_zh-cn_16k | [speech_sambert-hifigan_tts_zhizhe_emo_zh-cn_16k-npu](https://gitcode.com/gcw_C8PI9e90/speech_sambert-hifigan_tts_zhizhe_emo_zh-cn_16k-npu) |
| en-us_16k | [speech_sambert-hifigan_tts_en-us_16k-npu](https://gitcode.com/gcw_C8PI9e90/speech_sambert-hifigan_tts_en-us_16k-npu) |
| andy_en-us_16k | [speech_sambert-hifigan_tts_andy_en-us_16k-npu](https://gitcode.com/gcw_C8PI9e90/speech_sambert-hifigan_tts_andy_en-us_16k-npu) |
| annie_en-us_16k | [speech_sambert-hifigan_tts_annie_en-us_16k-npu](https://gitcode.com/gcw_C8PI9e90/speech_sambert-hifigan_tts_annie_en-us_16k-npu) |

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend910 系列（至少 1 卡） |
| OS | openEuler / Ubuntu / KylinOS（aarch64 或 x86_64） |
| CANN | >= 8.0 |
| Python | 3.11+ |
| 网络 | 首次运行需联网下载模型权重（~500MB/模型） |

## 流程总览

```
0. 环境初始化
→ 1. 安装依赖（PyTorch + torch_npu + KAN-TTS）
→ 2. 配置 ttsfrd stub 和语言资源
→ 3. 下载模型权重
→ 4. CPU 推理验证
→ 5. NPU 推理验证
→ 6. CPU/NPU 精度对比
→ 7. 串行执行多个模型
→ 8. 生成终端截图
→ 9. 生成 README 并发布模型仓库
```

按以下各节顺序执行，每步完成后再进入下一步。

> **重要**：多个模型必须串行执行，每个模型测试完成后释放 NPU 显存和 CPU 内存，再处理下一个模型，避免显存爆炸。

## 执行检查点与用户确认

在执行会修改环境、下载大文件或发布模型仓库的步骤前，必须先暂停并向用户确认。每个 checkpoint 都要给出当前输入、预期输出和失败 fallback。

1. **环境 checkpoint**：运行 `npu-smi info`、`python3 --version` 和 `python3 -c "import torch_npu"`；如果 NPU 不可用，则暂停并提示用户切换到 CPU dry-run 验证，不继续执行 NPU 推理。
2. **模型下载 checkpoint**：下载前确认目标模型列表、磁盘空间和缓存目录；如果下载失败，则 retry 1 次并记录失败模型，不要删除已完成模型。
3. **推理 checkpoint**：每个模型先跑 CPU smoke test，再跑 NPU smoke test；如果 CPU 失败，则不要进入 NPU；如果 NPU 失败，则保留 CPU 结果并进入 fallback 排查。
4. **发布 checkpoint**：只有当 `publish=true` 且用户明确确认仓库名、可见性和 token 权限后，才允许执行 GitCode 创建仓库与 `git push`。

## 异常处理与回滚策略

| 场景 | 判断方式 | fallback / recover | 回滚要求 |
|------|----------|--------------------|----------|
| `npu-smi info` 失败 | 命令非 0 或无 Ascend 设备 | 切换为 CPU dry-run，输出缺失 NPU 的原因 | 不写入 NPU benchmark 结果 |
| `torch_npu` 导入失败 | `ModuleNotFoundError` 或版本不匹配 | 重新安装匹配 CANN 的 `torch_npu`，retry 后仍失败则停止 | 保留安装日志到 `results/logs/` |
| 模型下载失败 | ModelScope/GitCode 返回网络错误 | retry 1 次，仍失败则跳过该模型并标记 `download_failed` | 不删除已有缓存 |
| CPU 推理失败 | 生成音频为空或脚本报错 | 停止该模型 NPU 流程，先修复 CPU 输入/依赖 | 回滚临时脚本改动 |
| NPU 推理失败 | NPU OOM、算子不支持或精度异常 | 释放显存、缩短文本、单模型 retry；必要时回退 CPU 算子 | 记录 `npu_failed`，不覆盖 CPU 结果 |
| 发布失败 | GitCode API 或 `git push` 失败 | 检查 token/remote/仓库名，retry 前必须用户确认 | 删除本地临时 remote，避免误推 |

## 资源与评测产物

本 skill 自带 `scripts/`、`examples/` 和 `requirements.txt` 资源。执行时优先复用这些资源，不要临时拼接不可追踪命令。

1. `scripts/download_models.sh`：下载模型权重，失败日志写入 `results/logs/download.log`。
2. `scripts/inference.py`：单模型 CPU/NPU 推理入口，输出音频到 `results/audio/`。
3. `scripts/compare_cpu_npu.py`：生成 `results/results.json` 和 `compare_report.txt`。
4. `scripts/run_all.sh`：串行运行多模型 eval / benchmark，禁止并行。
5. `examples/quickstart.sh`：最小 smoke test，可用于 CI 或人工复现。
6. `test-prompts.json`：用于 ascend-skills-eval 对该 skill 做结构评估和 dry_run 实测表现验证。

每次执行结束必须汇总：成功模型数、失败模型数、失败原因、是否发生 fallback、是否需要用户再次确认后 retry。

---

## 0. 环境初始化

```bash
# 加载 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 选择空闲 NPU
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=0   # 替换为实际空闲卡号

# 创建工作目录
mkdir -p ~/sambert-tts-workdir && cd ~/sambert-tts-workdir
```

---

## 1. 安装依赖

### 1.1 安装 PyTorch 和 torch_npu

```bash
pip install torch soundfile pyyaml numpy scipy
pip install torch_npu
```

### 1.2 安装 KAN-TTS

```bash
git clone https://github.com/alibaba-damo-academy/KAN-TTS.git /tmp/KAN-TTS
cd /tmp/KAN-TTS
# 绕过 python_requires < 3.9 检查
sed -i "s/python_requires.*//" setup.py
pip install -e .
```

### 1.3 安装其他依赖

```bash
pip install pypinyin g2p pyyaml
```

### 1.4 安装 ttsfrd stub

```bash
mkdir -p /tmp/ttsfrd_stub/ttsfrd
```

将本项目提供的 `ttsfrd_stub.py` 复制到：

```bash
cp ttsfrd_stub.py /tmp/ttsfrd_stub/ttsfrd/__init__.py
```

### 1.5 配置 EnUS 语言资源（英文模型需要）

```bash
# 如果从模型仓库中已包含，直接复制
cp -r languages/EnUS /opt/atomgit/.local/lib/python3.11/site-packages/kantts/preprocess/languages/EnUS
```

然后修改 KAN-TTS 的 `languages/__init__.py`，添加 EnUS 条目（如果不存在）。

---

## 2. 模型下载

```bash
# 下载单个模型（以 zh-cn_16k 为例）
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('iic/speech_sambert-hifigan_tts_zh-cn_16k',
                  cache_dir='/path/to/modelscope/hub')

# 或使用 download_models.sh 脚本批量下载
bash download_models.sh
```

---

## 3. 单模型推理

### 3.1 CPU 推理

```bash
python3 inference.py \
  --model speech_sambert-hifigan_tts_zh-cn_16k \
  --voice zhitian_emo \
  --text "北京今天天气怎么样" \
  --device cpu \
  --output output_cpu.wav
```

### 3.2 NPU 推理

```bash
python3 inference.py \
  --model speech_sambert-hifigan_tts_zh-cn_16k \
  --voice zhitian_emo \
  --text "北京今天天气怎么样" \
  --device npu \
  --output output_npu.wav
```

### 3.3 英文模型推理

```bash
# 英文模型使用 g2p 库做 ARPAbet 音素转换
python3 inference.py \
  --model speech_sambert-hifigan_tts_en-us_16k \
  --voice andy \
  --text "Hello world" \
  --device cpu \
  --output output_cpu.wav
```

---

## 4. CPU/NPU 精度对比

```bash
python3 compare_cpu_npu.py \
  --model-dir /path/to/modelscope/hub/models/iic/speech_sambert-hifigan_tts_zh-cn_16k \
  --voice zhitian_emo \
  --text "北京今天天气怎么样"
```

### 评估指标

| 指标 | 说明 | 判定标准 |
|------|------|---------|
| SNR（信噪比） | 信号与噪声功率之比 | > 20 dB 为通过 |
| 相关系数 | 波形线性相关性 | 越接近 1.0 越好 |
| RMSE（均方根误差） | 逐样本误差的均方根 | 越小越好 |

---

## 5. 串行执行多个模型

使用 `batch_inference.py` 自动串行处理所有模型：

```bash
python3 batch_inference.py --text "北京今天天气怎么样"
```

该脚本会自动：
1. 发现已下载的模型列表
2. 对每个模型的每个声音串行执行 CPU 和 NPU 推理
3. 进行精度对比
4. 释放内存（`gc.collect()` + `torch.npu.empty_cache()`）
5. 生成汇总结果 JSON

**串行流程伪代码：**

```python
for model_name, voices in models:
    for voice in voices:
        # 加载模型
        sambert, hifigan, ling_unit = load_model(...)
        # CPU 推理
        audio_cpu = synthesize(device='cpu')
        # NPU 推理
        audio_npu = synthesize(device='npu')
        # 精度对比
        snr, corr = compare(audio_cpu, audio_npu)
        # 释放资源
        del sambert, hifigan, ling_unit
        gc.collect()
        torch.npu.empty_cache()
```

---

## 6. 生成终端截图

```bash
python3 -c "
from terminal_screenshot import take_screenshot
# 截取精度对比结果
take_screenshot('comparison_result.png', 'python3 compare_cpu_npu.py ...')
"
```

---

## 7. 发布模型仓库到 GitCode

### 7.1 使用 GitCode API 创建仓库

```bash
curl -X POST "https://api.gitcode.com/api/v5/user/repos?access_token=${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "speech_sambert-hifigan_tts_zh-cn_16k-npu",
    "repository_type": "model",
    "visibility_level": 0
  }'
```

### 7.2 推送代码

```bash
git init
git branch -M main
git remote add origin https://auth:${TOKEN}@gitcode.com/${USER}/${REPO_NAME}.git
git add -A
git commit -m "Add ${MODEL_NAME} NPU adaptation"
git push -u origin main
```

### 7.3 使用 gitcode-publish Skill

或者通过调用内置的 `gitcode-publish` Skill 来发布：

```bash
# 参见 gitcode-publish Skill 文档
```

---

## Skill 输入参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | string | 全部 | 指定单个模型，不指定则处理全部 |
| `text` | string | "北京今天天气怎么样" | 推理文本 |
| `device` | string | "npu" | 推理设备 |
| `publish` | boolean | false | 是否发布到 GitCode |

## Skill 输出结果

- `results/results.json` — 所有模型的精度对比结果汇总
- `results/audio/` — CPU 和 NPU 输出音频文件
- `compare_report.txt` — 单模型精度报告
- 模型仓库已推送到 GitCode

## 已知问题

1. **zhida / zhisha / zhiyue 模型**：使用不同结构（PhoneSet.xml 直接配置，无 am/voc 独立 checkpoint），需额外适配工作
2. **英式英语模型（luca / luna）**：使用英式英语音素集，需额外 British English G2P 支持
3. **NSF 模型（cally_en-us_24k）**：24kHz 采样率，含有 Neural Source Filter 组件，结构不同
4. **SamBERT 自回归解码器在 NPU 上性能较差**：由于 autoregressive 解码的顺序依赖，CPU 上更快；HiFiGAN 的 CNN 架构在 NPU 上有显著加速
