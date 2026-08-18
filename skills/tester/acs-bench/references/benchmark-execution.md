---
name: acs-bench-benchmark
description: "LLM推理服务压测：使用 acs-bench 对 ModelArts MaaS 推理服务进行性能基准测试，支持固定并发、爬坡模式和速率跑坡模式"
trigger: "压测, benchmark, acs-bench, 压力测试, 性能测试, prof, 推理压测"
related_skills:
  - acs-bench-workflow
  - acs-bench-peak-finding
---

# acs-bench LLM 推理压测 Skill

## 概述

使用 `acs-bench prof` 对 LLM 推理服务进行性能压测，支持两种模式：
- **固定并发模式**：指定 concurrency 和 request-rate，跑一轮
- **爬坡模式**：使用 `--use-climb` 逐步增加并发，寻找最优性能点

## 环境信息

- **节点**: maas-poc
- **工作目录**: `$PROF_ROOT`
- **Conda环境**: `$CONDA_ENV`
- **环境激活**: `source $WORK_ROOT/$CONDA_ENV/bin/activate`
- **acs-bench路径**: `$WORK_ROOT/$CONDA_ENV/bin/acs-bench`
- **Tokenizer (Qwen/LongCat)**: `$PROF_ROOT/models/LongCat-Flash-Chat`
- **Tokenizer (LongCat-Flash-Chat)**: `$PROF_ROOT/models/LongCat-Flash-Chat` (vocab=128K, max_len=131072)
- **Tokenizer (DeepSeek V3-0324)**: `$PROF_ROOT/models/DeepSeek-V3-0324` (vocab=128000, max_len=131072)
- **Tokenizer (DeepSeek V3)**: `$PROF_ROOT/models/DeepSeek-V3` (旧版，非0324)
- **Tokenizer (DeepSeek V4 Flash)**: `$PROF_ROOT/models/DeepSeek-V4-Flash` (从hf-mirror下载)
- **Provider配置**: `$PROF_ROOT/conf/provider.yaml`（含 api_key + base_url + model_name）
- **Provider (DeepSeek V3)**: `$PROF_ROOT/conf/provider_deepseek_v3.yaml`
- **Provider (DeepSeek V4 Flash 官方API)**: `$PROF_ROOT/conf/provider_deepseek_v4_flash.yaml` (base_url=https://api.deepseek.com/v1)
- **Provider (LongCat-Flash-Chat)**: `$PROF_ROOT/conf/provider_longcat.yaml` (ModelArts MaaS, 同api_key/model_name)
- **Provider (LongCat-Flash-Chat)**: `$PROF_ROOT/conf/provider_longcat.yaml`（ModelArts MaaS，同api_key/model_name）
- **数据集目录**: `$PROF_ROOT/dataset/mt_dataset/`（变长）、`$PROF_ROOT/dataset/fixed_length/`（定长）
- **结果目录**: `$PROF_ROOT/result/csv/`
- **报告目录**: `$PROF_ROOT/result/report/`
- **日志目录**: `$PROF_ROOT/log/`
- **场景配置**: `$PROF_ROOT/conf/scenarios/`
- **网络环境**: ⚠️ huggingface.co不可达，需用 `HF_ENDPOINT=https://hf-mirror.com` 下载模型/tokenizer
- **HuggingFace CLI**: ⚠️ `huggingface-cli` 已废弃，用 `hf download` 代替；`--include` 参数在新CLI中无效，需逐个传文件名作位置参数（如 `hf download repo tokenizer.json --local-dir dir`）
- **Tokenizer下载**: `HF_ENDPOINT=https://hf-mirror.com huggingface-cli download <repo> --local-dir <path> --include "tokenizer.json" "tokenizer_config.json" "special_tokens_map.json"`

### Tokenizer下载指南

不同DeepSeek模型版本需要对应的tokenizer，不可混用（tokenization结果不同）。

```bash
# 从HuggingFace镜像下载tokenizer（仅tokenizer文件，不下载模型权重）
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
  deepseek-ai/<REPO_NAME> \
  tokenizer.json tokenizer_config.json special_tokens_map.json \
  --local-dir $PROF_ROOT/models/<MODEL_NAME> \
  --local-dir-use-symlinks False
```

**⚠️ Tokenizer下载陷阱**：
- 必须用 `HF_ENDPOINT=https://hf-mirror.com`（huggingface.co不可达）
- 不同模型版本的tokenizer不可混用：V3-0324的tokenizer不能用于V4-Flash压测（token计数不准）
- 下载前需确认HuggingFace repo名（如`deepseek-ai/DeepSeek-V4-Flash`），不确定时问用户
- 只需下载tokenizer相关文件（tokenizer.json等），无需下载模型权重

> ⚠️ 执行 acs-bench 前必须先激活环境：`source $WORK_ROOT/$CONDA_ENV/bin/activate`

## Provider 配置 (provider.yaml)

### ModelArts MaaS 推理服务（默认）

```yaml
providers:
  - id: 'mt_test'
    name: 'xds'
    api_key: '<API_KEY>'
    base_url: 'https://api.modelarts-maas.com/v1'
    model_name: '<MODEL_ID>'
```

### DeepSeek 官方 API

```yaml
providers:
  - id: 'mt_test'
    name: 'deepseek-<model>'
    api_key: '<API_KEY>'        # sk-... 格式
    base_url: 'https://api.deepseek.com/v1'
    model_name: '<MODEL_NAME>'  # 如 deepseek-v4-flash, deepseek-chat 等
```

**DeepSeek官方API vs ModelArts MaaS差异**：
- base_url不同：`https://api.deepseek.com/v1` vs `https://api.modelarts-maas.com/v1`
- api_key格式不同：`sk-...` vs MaaS平台key
- model_name不同：直接用模型名（如`deepseek-v4-flash`）vs MaaS资源ID（如`f17067bd-...`）
- 无需model_name的UUID格式，直接使用模型公开名称

> ⚠️ 修改 api_key 和 model_name 后再执行压测
> 📁 实际配置文件：`$PROF_ROOT/conf/provider.yaml`，模板：`$PROF_ROOT/conf/provider.yaml.template`

**DeepSeek官方API Provider配置**（非ModelArts MaaS）：

```yaml
providers:
  - id: 'mt_test'
    name: 'deepseek-v4-flash'
    api_key: '<DEEPSEEK_API_KEY>'
    base_url: 'https://api.deepseek.com/v1'
    model_name: 'deepseek-v4-flash'   # 直接使用模型名，非ModelArts的UUID
```

## 数据集类型

### 定长数据集（默认）

当用户未指定数据集时，**默认采用定长场景**。定长场景指输入和输出均为固定token数（如"输入10k-输出400"）。

定长数据需要生成，两种方式：
1. **使用已有定长数据**：`./prof-test/dataset/` 下有按输入长度组织的目录（如 `in10240_n10000/`、`in8k_n10000/` 等）
2. **用 acs-bench generate 重新生成**：
```bash
source $WORK_ROOT/$CONDA_ENV/bin/activate
acs-bench generate dataset \
  --dataset-type random \
  --input-length 10240 \
  --num-requests 10000 \
  --tokenizer $PROF_ROOT/models/LongCat-Flash-Chat \
  --trust-remote-code \
  --output-path ./prof-test/dataset/in10240_n10000/10240.json
```

3. **用 DS-V3-0324 tokenizer 生成**（不同tokenizer的数据集不可混用，目录加 `_dsv30324` 后缀区分）：
```bash
source $WORK_ROOT/$CONDA_ENV/bin/activate
acs-bench generate dataset \
  -il 10240 \
  -n 10000 \
  -t $PROF_ROOT/models/DeepSeek-V3-0324 \
  -o ./dataset/fixed_length/in10240_n10000_dsv30324/10240.json
```
> ⚠️ 生成速度约14 it/s，10000条约12分钟，建议后台执行

**⚠️ 数据集生成 output-path 陷阱**：acs-bench `--output-path` 参数需到文件级别（如 `10240.json`），但实际行为是**创建以文件名为名的子目录**，最终文件路径 = `output-path/filename`。例如 `-o ./dataset/fixed_length/in10240_n10000_dsv30324/10240.json` → 实际文件在 `./dataset/fixed_length/in10240_n10000_dsv30324/10240.json/10240.json`。设计 `-o` 路径时需考虑此双重嵌套行为。

**已有定长数据目录：**

- **`in2560_n10000/`**: 2560 — 10000
- **`in4k_n10000/`**: 4096 — 10000
- **`in8k_n10000/`**: 8192 — 10000
- **`in9k_n10000/`**: 9216 — 10000
- **`in10240_n10000/`**: 10240 — 10000
- **`in11k_n10000/`**: 11264 — 10000
- **`in12k_n10000/`**: 12288 — 10000
- **`in13k_n10000/`**: 13312 — 10000
- **`in14k_n10000/`**: 14336 — 10000
- **`in15k_n10000/`**: 15360 — 10000
- **`in10240_n10000_dsv30324/`**: 10240 — 10000

> 定长场景默认 num-requests=10000，output-length 由用户指定，ignore-eos=True（强制输出定长）

### 变长数据集

位于 `./prof-test/dataset/mt_dataset/`：

- **`data_n3838_avg11944.json`**: 3838条，avg输入11944 tokens，**唯一保持CSV原始顺序** — 基准对照
- **`data_n3838_avg11944_cut_n2820_avg10443.json`**: keep_error裁剪后2820条，avg 10443（已shuffle） — -
- **`data_n3838_avg11944_cut_n2699_avg10240.json`**: keep_max_n裁剪后2699条，avg 10240（已shuffle） — -
- **`data_n3838_avg11944_cut_keep_max_n_n2699_avg10240.json`**: keep_max_n裁剪后2699条，avg 10240（**保序，去重后与基准顺序一致**） — 保序压测
- **`data_n3838_avg11944_cut_keep_error_e2.0_shuffled_n2820_avg10443.json`**: keep_error裁剪+shuffle，2820条 — -
- **`data_n3838_avg11944_cut_keep_max_n_shuffled_n2699_avg10240.json`**: keep_max_n裁剪+shuffle，2699条 — -
- **`data_shuffled_n3838_avg11944.json`**: shuffle版3838条 — -
- **`data_shuffled_cut_keep_error_e2.0_n2820_avg10443.json`**: shuffle+keep_error裁剪，2820条 — -
- **`data_shuffled_cut_keep_max_n_n2699_avg10240.json`**: shuffle+keep_max_n裁剪，2699条（**压测实际使用**） — 
- **`data_shuffled_cut_keep_error_e2.0_n2820_avg10443.json`**: shuffle+keep_error裁剪，2820条，avg 10443 — 
- **`data_n3838_avg11944_cut_keep_error_e2.0_shuffled_n2820_avg10443.json`**: keep_error裁剪+shuffle，2820条，avg 10443 — 
- **`data_n3838_avg11944_cut_keep_error_e2.0_n2820_avg10443.json`**: keep_error裁剪保序，2820条，avg 10443（**保序**） — 保序压测

> ⚠️ 保序数据集：`data_n3838_avg11944.json`、`data_n3838_avg11944_cut_keep_max_n_n2699_avg10240.json`、`data_n3838_avg11944_cut_keep_error_e2.0_n2820_avg10443.json`；其余均已被 shuffle

## 标准化脚本（~/prof/scripts/）

- **`run_benchmark.sh`**: 单次压测执行 — `bash scripts/run_benchmark.sh -c 400 -r 26`
- **`run_scenario.sh`**: 场景化压测（读YAML自动配参） — `bash scripts/run_scenario.sh -s fixedlen-10k-400-deepseek-v3 -c 400 -r 26` ⚠️ `-r`可选，省略=不限速
- **`run_multi_scene.sh`**: 多场景批量执行 — `bash scripts/run_multi_scene.sh -c "350,400" -r "20,22"`
- **`gen_fixed_dataset.sh`**: 定长数据集生成 — `bash scripts/gen_fixed_dataset.sh -l 10240`
- **`parse_result.py`**: CSV结果解析与汇总 — `python3 scripts/parse_result.py --dir result/csv/`
- **`summary_csv.py`**: 汇总CSV生成（用户指定列） — `python3 scripts/summary_csv.py --dir result/csv/ -o result/report/summary.csv`
- **`parse_benchmark_results.py`**: 阶段映射+异常过滤+前缀模式+A vs B对比 — `python3 scripts/parse_benchmark_results.py --today --max-fail 0.05`
- **`validate_csv_report.py`**: CSV报告校验（列完整性+关键列非空+数值合理性） — `python3 scripts/validate_csv_report.py --dir result/csv/ --today`
- **`verify_dataset_order.py`**: 数据集顺序验证 — `python3 scripts/verify_dataset_order.py src.json sub.json`

> **脚本层级**：`parse_result.py`(底层) → `summary_csv.py`(汇总) → `parse_benchmark_results.py`(业务)，详见 `acs-bench-workflow` skill

### run_benchmark.sh 参数

```
必填: -c concurrency  -r request-rate
可选: -d dataset  -t varlen|fixedlen  -o output-length  -n num-requests
      -e epochs  -w warmup  -i ignore-eos  -l log-prefix  -p provider  -k tokenizer
```

### run_scenario.sh 参数（场景化执行，推荐DSV3使用）

```
必填: -s scenario  -c concurrency
可选: -r request-rate  省略则不限速（命令中不传--request-rate）
      -d dataset   -e epochs  -w warmup  -n num-requests  -l log-prefix  -p (dry-run)

跑坡(climb)参数:
      --climb           启用climb跑坡模式（--use-climb）
      --climb-mode MODE 跑坡模式: linear(线性)/static, 默认linear
      --growth-rate N   每步并发增长量, 默认0
      --growth-interval MS  增长间隔(毫秒), 默认1000
      --init-concurrency N  初始并发数, 默认等于-c
```

> ⚠️ `-r request-rate` 为可选参数（2026-05-11起）。省略时命令中不传 `--request-rate`，即不限速尽快发送。日志中 rate 部分显示为 `rnolimit`

**跑坡(climb)模式说明**：acs-bench原生支持并发爬坡，无需修改源码。参数直接透传给`acs-bench prof`的`--use-climb`系列参数。跑坡从`--init-concurrency`起步，每`--growth-interval`毫秒增加`--growth-rate`个并发，直到达到`-c`指定的最大并发。一次跑坡可替代多轮固定并发测试，高效获取QPS/时延随并发变化的完整曲线。

**⚠️ growth-rate 与 request-rate 互斥规则**：
- `--growth-rate`通过逐步增加并发来推高负载
- `--request-rate`硬限请求速率，并发再高也被限速
- **同时使用时，并发增长无法实际推高QPS（被request-rate卡死），跑坡曲线失真**
- **正确做法：二选一**
  - **跑坡找上限** → 用climb增长并发，**不限request_rate**（省略`-r`参数）
  - **验证目标RPM** → 用固定并发 + request_rate，**不用climb**

**⚠️ run_scenario.sh CMD构建**：2026-05-13修复：原`$(if...)`模式在双引号CMD字符串中引号转义错误，导致空`REQUEST_RATE`时仍输出`"--request-rate "`使acs-bench报错。已改用bash数组`CMD_ARGS=()`构建命令，彻底避免引号转义问题。**禁止回退到字符串拼接+eval模式**。

**场景名对应 conf/scenarios/<name>.yaml**，脚本自动读取数据集/tokenizer/provider/ignore-eos等参数。
DSV3场景自动使用_dsv30324专用定长数据集。

**-d 参数说明**：覆盖YAML中的数据集名，用于摸高轮次的注入数据集。例如：
- `-d data_n3838_avg11944_r04_uid` → 使用unique模式注入数据
- `-d data_n3838_avg11944_r01` → 使用shared模式注入数据
- 不指定-d → 使用YAML中定义的源数据集

**LongCat场景列表：**
- **`varlen-10k-600-longcat`** | 类型=混长(保序) | 数据集=data_n3838_avg11944_cut_keep_max_n_n2699_avg10240 | n=2699 | output=600
- **`varlen-3838-600-longcat`** | 类型=混长(保序) | 数据集=data_n3838_avg11944 | n=3838 | output=600 | 缓存命中场景 | QPS天花板≈31.5 | e2e≤4s不可达

**LongCat varlen-3838-600 基准数据（缓存命中, 2026-05-13）：**
- **c=160**: QPS=25.42 | AVG_E2E=5.889s | TP99_E2E=12.06s
- **c=165**: QPS=25.97 | AVG_E2E=5.964s | TP99_E2E=12.11s ← **e2e≤6s甜点**
- **c=168**: QPS=25.40 | AVG_E2E=6.119s | TP99_E2E=12.43s（超6s）
- **结论**: e2e≤6s约束下最大QPS≈25.97（≈RPM 1558），甜点c=165 | 爬坡基线: c=168 QPS=25.4 AVG_E2E=6.1s

**DSV3场景列表：**
- **`fixedlen-10k-400-deepseek-v3`** | 类型=定长 | 数据集=in10240_n10000_dsv30324 | n=10000 | output=400
- **`varlen-10k-600-deepseek-v3`** | 类型=混长 | 数据集=data_shuffled_cut_keep_max_n_n2699_avg10240 | n=2699 | output=600
- **`varlen-3838-600-deepseek-v3`** | 类型=混长(保序) | 数据集=data_n3838_avg11944 | n=3838 | output=600

> ⚠️ **run_benchmark.sh 硬编码陷阱**：脚本默认 `TOKENIZER=LongCat-Flash-Chat` 和 `PROVIDER=provider.yaml`（Qwen3-32b）。对 DSV3-0324 场景，必须通过 `-p $PROF_ROOT/conf/provider_deepseek_v3.yaml` 覆盖 provider，且需修改脚本内 TOKENIZER 变量或增加 `-k` 参数支持。否则 token 计数和 API 调用都会指向错误模型。

**⚠️ 新模型压测Provider复用模式**：当新模型与已有模型共用同一ModelArts MaaS服务（api_key/model_name/base_url相同）时，只需创建新provider YAML（如`provider_longcat.yaml`），将`name`字段改为模型名，其余字段沿用。tokenizer需切换到对应模型路径。

### run_multi_scene.sh 参数

```
模式1: -c "C1,C2" -r "R1,R2"  (矩阵组合)
模式2: -f scenes.txt           (场景文件，每行: concurrency rate [purpose])
通用: -d dataset -t type -o output-length -n num-requests -e epochs -w warmup -i ignore-eos
```

### parse_result.py 参数

```
单文件: python3 parse_result.py <csv_file>
目录:   python3 parse_result.py --dir <dir> [--sort time|qps] [--json]
```

### summary_csv.py 参数（汇总CSV生成）

```
目录:   python3 summary_csv.py --dir <dir> [-o output.csv] [--sort time|qps|rpm] [--log-dir <log_dir>]
```

**输出列**: 执行时间, 场景(混长/定长), 请求数, 输入长度, 输出长度, 最大并发, 压测QPS, AVG_TTFT(s), TP90_TTFT(s), AVG_TPOT(s), TP90_TPOT(s), AVG_E2E(s), TP90_E2E(s), 输入TPS, 输出TPS, 总TPS, total_time, 实际QPS, RPM

**字段说明**（详见 `references/summary-csv-field-mapping.md`）:
- 场景：数据集输入长度一致→定长，不一致→混长（日志文件名含`in{n}_n{m}`→定长）
- **压测QPS**：压测命令中的request-rate（提取优先级：①日志文件名→②bash history→③空）
- **实际QPS**：结果CSV中的QPS字段
- 输入TPS = 总TPS - 输出TPS
- 输出TPS = Output_Token_Throughput(tokens/s)
- 总TPS = Total_Token_Throughput(tokens/s)
- RPM = 实际QPS × 60

> ⚠️ 压测QPS ≠ 实际QPS：前者是命令参数（发送速率），后者是服务端实际处理速率

## 命令模板

### 1. 固定并发压测（推荐日常使用，使用标准化脚本）

```bash
cd $PROF_ROOT
source $WORK_ROOT/$CONDA_ENV/bin/activate

# 变长场景
bash scripts/run_benchmark.sh -c 350 -r 20 \
  -d data_shuffled_cut_keep_max_n_n2699_avg10240 -o 600

# 定长场景
bash scripts/run_benchmark.sh -c 400 -r 26 \
  -t fixedlen -d in10240_n10000 -o 400 -n 10000 -i True
```

### 1a. 固定并发压测（原始命令，用于调试/自定义）

```bash
cd $PROF_ROOT
source $WORK_ROOT/$CONDA_ENV/bin/activate

# 设置参数
dataset="data_shuffled_cut_keep_max_n_n2699_avg10240"
date=$(date +%Y%m%d_%H%M)
log_file="log/run_${dataset}_c350_r20_${date}.log"

# 执行压测
nohup acs-bench prof \
  --tokenizer $PROF_ROOT/models/LongCat-Flash-Chat \
  --trust-remote-code \
  --benchmark-save-path "./result/csv/" \
  --epochs 1 \
  --warmup 0 \
  --num-requests 2699 \
  --concurrency-backend threading-pool \
  --backend openai-chat \
  --input-path "./dataset/mt_dataset/${dataset}.json" \
  --output-length 600 \
  --ignore-eos False \
  --concurrency 350 \
  --request-rate 20 \
  --provider ./conf/provider.yaml \
  -D > ${log_file} 2>&1 &
```

### 2. 爬坡模式压测（寻找最优并发）

```bash
cd $PROF_ROOT
source $WORK_ROOT/$CONDA_ENV/bin/activate

acs-bench prof \
  --dataset-type custom \
  --backend openai-chat \
  --use-climb \
  --climb-mode linear \
  --concurrency-backend threading-pool \
  --provider ./conf/provider.yaml \
  --input-path "./dataset/mt_dataset/data_n3838_avg11944.json" \
  --warmup 2 \
  --epochs 1 \
  --growth-rate 6 \
  --init-concurrency 6 \
  --concurrency 1024,2048,3072,4096 \
  --num-requests 24576,24576,24576,24576 \
  --input-length 50 \
  --output-length 120960 \
  --use-spec-decode \
  --num-spec-tokens 1 \
  --num-scheduler-steps 1 \
  --benchmark-save-path "./result/csv/"
```

### 2a. 跑坡模式压测（并发爬坡，探测QPS天花板）— 原生climb模式

acs-bench原生支持并发爬坡，**无需修改源码**，直接使用`--use-climb`系列参数：

```bash
cd $PROF_ROOT
source $WORK_ROOT/$CONDA_ENV/bin/activate

# 方式1：直接用acs-bench命令
# 从c=10起步，每30s增加20并发，直到c=400，不限request-rate（让并发自然驱动QPS）
acs-bench prof \
  --tokenizer $PROF_ROOT/models/DeepSeek-V3-0324 \
  --trust-remote-code \
  --benchmark-save-path "./result/csv/" \
  --epochs 1 --warmup 0 --num-requests 3838 \
  --concurrency-backend threading-pool --backend openai-chat \
  --input-path "./dataset/mt_dataset/data_n3838_avg11944.json" \
  --output-length 600 --ignore-eos False \
  --concurrency 400 \
  --use-climb --climb-mode linear \
  --growth-rate 20 --growth-interval 30000 \
  --init-concurrency 10 \
  --provider ./conf/provider_deepseek_v3.yaml -D

# 方式2：用run_scenario.sh（climb参数透传）
bash scripts/run_scenario.sh \
  -s varlen-3838-600-deepseek-v3 \
  -c 400 \
  --climb --climb-mode linear \
  --growth-rate 20 --growth-interval 30000 \
  --init-concurrency 10 \
  -l ramp
```

**climb参数说明**：
- `--use-climb` / `--climb`: 启用爬坡模式
- `--climb-mode linear`: 线性增长（推荐），另有`static`模式
- `--growth-rate N`: 每步增加N个并发连接
- `--growth-interval MS`: 每步持续毫秒数（建议30000=30s，足够稳态采样）
- `--init-concurrency N`: 初始并发数（起步点，建议10~20）

**跑坡设计要点**：
- `-c`设为预期最大并发（跑坡终点），`--init-concurrency`设为起点
- **禁止同时指定`-r`（request_rate）**：climb模式通过并发增长推高负载，限速会卡死QPS增长，跑坡曲线失真。跑坡时省略`-r`参数，让并发自然驱动QPS
- **growth-rate安全上限**：`growth-rate ≤ ceil(c × 0.4)`，超过后瞬态冲击明显（e2e微超目标）。实测：同c=165下g=60达标，g=70微超0.006s
- `--growth-interval`建议1000ms（快速爬坡，大部分请求在稳态执行）
- 一次跑坡约 `(max_c - init_c) / growth_rate × interval_ms / 1000` 秒
- 结果按时间窗口（每interval一个窗口）提取各并发下的QPS/e2e
- **c与e2e非线性**：c接近系统容量时每+1并发e2e增量加速（边际递增），甜点附近需小步验证
- 结果按时间窗口（每interval一个窗口）提取各并发下的QPS/e2e

**⚠️ run_scenario.sh CMD构建陷阱（已修复）**：旧版用双引号字符串+`$(if...)`模式构建CMD，空变量时`--request-rate`仍会泄漏到命令中（引号转义导致`\"--request-rate \"`被eval为单个参数）。已改用bash数组（`CMD_ARGS=()` + `CMD_ARGS+=()`）重构，彻底避免引号问题。禁止回退到字符串拼接方式

> ⚠️ **禁止为跑坡修改acs-bench源码**：原生`--use-climb`已完整支持并发爬坡。如需速率爬坡(r递增)，那是不同需求，需确认acs-bench是否已支持`--rate-growth`参数

### 3. 多 request-rate 循环压测（使用标准化脚本）

```bash
cd $PROF_ROOT

# 矩阵模式：自动组合 concurrency × rate
bash scripts/run_multi_scene.sh -c "350,400" -r "20,22"

# 场景文件模式
bash scripts/run_multi_scene.sh -f conf/scenes.txt
```

## 关键参数说明

- **`--tokenizer`**: 分词器路径，用于精确计算token — `$PROF_ROOT/models/LongCat-Flash-Chat`
- **`--trust-remote-code`**: 信任远程代码（tokenizer需要） — -
- **`--backend`**: 后端类型 — `openai-chat`
- **`--concurrency-backend`**: 并发实现方式 — `threading-pool`
- **`--concurrency`**: 并发数 — `300`
- **`--request-rate`**: 请求速率（req/s），**可省略** — `23`
- **`--num-requests`**: 总请求数 — `2699`
- **`--output-length`**: 期望输出长度 — `600`
- **`--ignore-eos`**: 是否忽略EOS持续生成 — `False`
- **`--epochs`**: 测试轮数 — `1`
- **`--warmup`**: 预热轮数 — `0`
- **`--provider`**: API配置文件 — `./conf/provider.yaml`
- **`--benchmark-save-path`**: 结果保存路径 — `./result/csv/`
- **`-D`**: 调试模式，输出详细日志 — -
- **`--use-climb`**: 启用爬坡模式 — -
- **`--climb-mode`**: 爬坡策略 — `linear`
- **`--growth-rate`**: 爬坡增长率（并发爬坡） — `6`
- **`--rate-growth`**: 请求速率每步增长量（速率爬坡，待实现） — `5`
- **`--rate-growth-step`**: 速率增长步长秒数（待实现） — `30`

## 结果解读

压测完成后，结果保存为 CSV 文件，关键指标：

- **`QPS`**: 每秒查询数 — 越高越好
- **`Output_Token_Throughput`**: 输出token吞吐量 (tokens/s) — 核心性能指标
- **`Total_Token_Throughput`**: 总token吞吐量 (tokens/s) — 包含输入+输出
- **`AVG_TTFT`**: 平均首token延迟 (s) — 用户体感
- **`TP99_TTFT`**: P99首token延迟 (s) — 长尾延迟
- **`AVG_TPOT`**: 平均token间延迟 (s) — 生成速度
- **`AVG_E2E`**: 平均端到端延迟 (s) — 整体耗时
- **`Fail_Rate`**: 失败率 — 应为0

### 查看最新结果

```bash
# 查看最新CSV
ls -lt $PROF_ROOT/result/csv/*.csv | head -1

# 查看最新日志
ls -lt $PROF_ROOT/log/*.log | head -1

# 实时跟踪运行中的日志
tail -f $PROF_ROOT/log/<log_file>

# 解析结果汇总
python3 $PROF_ROOT/scripts/parse_result.py --dir $PROF_ROOT/result/csv/ --sort qps
```

## 执行步骤

1. **激活环境**：`source $WORK_ROOT/$CONDA_ENV/bin/activate`
2. **确认环境**：确保 `acs-bench` 可用
2. **检查 provider.yaml**：确认 `$PROF_ROOT/conf/provider.yaml` 中 api_key 和 model_name 正确
3. **选择数据集**：根据测试目标选择合适的数据集文件
4. **设置参数**：调整 concurrency、request-rate、output-length 等
5. **执行压测**：使用 `run_benchmark.sh` 或 nohup 后台运行
6. **监控进度**：`tail -f` 跟踪日志
7. **分析结果**：使用 `parse_result.py` 解析 result/csv/ 下的 CSV 文件

## 常见场景

### 场景1：验证服务基本可用性
```bash
cd $PROF_ROOT
# 小规模快速验证：100请求，并发10
bash scripts/run_benchmark.sh -c 10 -r 5 -n 100 -o 100
```

### 场景2：长文本高并发压测
```bash
cd $PROF_ROOT
# 2699请求，并发300，输出600 token
bash scripts/run_benchmark.sh -c 300 -r 22 -o 600
```

### 场景3：不同并发梯度对比
```bash
cd $PROF_ROOT
# 使用多场景脚本
bash scripts/run_multi_scene.sh -c "100,200,300,500" -r "23"
```

## 多组对比测试模式

当需要跑多组配置对比时（如不同concurrency/request-rate组合）：

1. **设计用例**：列出所有配置，一次性与用户确认
2. **串行执行**：逐组后台运行，`notify_on_complete=true`，完成后自动启动下一组
3. **结果汇总**：所有组跑完后，输出对比表格，含QPS、AVG_E2E、TP90/95/99_E2E、AVG_TTFT、Output吞吐、Fail_Rate
4. **分析结论**：指出最优配置及各配置适用场景

### 日志命名规范

```
# 标准格式（2026-05-01起）
log/run_{dataset}_c{concurrency}_r{rate}_{YYYYMMDD_HHMM}.log
log/run_peak_{dataset}_c{concurrency}_r{rate}_{YYYYMMDD_HHMM}.log      # 摸高
log/run_stability_{dataset}_c{concurrency}_r{rate}_run{N}_{YYYYMMDD_HHMM}.log  # 稳定性验证
log/run_scene{N}_{dataset}_c{concurrency}_r{rate}_{YYYYMMDD_HHMM}.log      # 多场景

# 定长场景（dataset含in{n}_n{m}标识）
log/run_in10240_n10000_c400_r26_20260501_1617.log

# 变长场景（dataset含avg/shuffled等标识）
log/run_data_shuffled_cut_keep_max_n_n2699_avg10240_c350_r20_20260501_0858.log
```

**命名规范要点**：
- 日期时间：`YYYYMMDD_HHMM` 格式，表示压测**启动时间**（早于CSV中的Execution_Time完成时间）
- 并发：`c{concurrency}`，如 `c350`、`c400`
- 请求速率：`r{rate}`，如 `r20`、`r26`
- 定长标识：数据集名含 `in{n}_n{m}`（如 `in10240_n10000`），用于区分定长/混长场景
- 变长标识：数据集名含 `avg`/`shuffled`/`cut` 等关键词

### CSV结果文件命名规范

```
result/csv/summary_{provider}_{concurrency_backend}_{YYYY-MM-DD_HH_MM_SS}.csv
# 示例：summary_mt_test_parallel_threading-pool_2026-05-01_16_30_45.csv
```

**CSV与日志的关联**：
- CSV的Execution_Time（第1列）= 压测**完成时间**
- 日志文件名中的时间 = 压测**启动时间**
- 关联方式：通过时间窗口（0~120分钟）+ 并发数模糊匹配
- 压测QPS（request-rate）提取优先级：①日志文件名 → ②bash history命令 → ③标记为空

### 场景推断规则

- **日志文件名含 `in{n}_n{m}`**: 定长
- **CSV Input_Length列非空**: 定长
- **其他**: 混长

### 串行后台执行模式

逐组串行执行，每组用 `terminal(background=true, notify_on_complete=true)` 启动：

```
1. 启动第1组 → 等待完成通知
2. 读取最新CSV结果 → 提取关键指标 → 简报用户
3. 启动第2组 → 等待完成通知
4. ...重复直到所有组完成
5. 输出汇总对比表 + 分析结论
```

### CSV结果快速提取

结果CSV为单行数据（第2行），关键字段位置：

- **Concurrency(第6列)**: 并发数 — 确认配置
- **QPS(第57列)**: 每秒查询数 — 核心吞吐
- **AVG_E2E(第49列)**: 平均端到端延迟 — 核心时延
- **AVG_TTFT(第14列)**: 平均首token延迟 — 排队指标
- **TP99_E2E(第47列)**: P99端到端延迟 — 长尾
- **Fail_Rate(第58列)**: 失败率 — 必须为0

> ⚠️ 列号可能随acs-bench版本变化，建议用header名索引而非硬编码列号。关键字段名：`QPS`, `AVG_E2E(s)`, `AVG_TTFT(s)`, `TP99_E2E(s)`, `Fail_Rate`, `Concurrency`

> 实际使用时用 `read_file` 读取CSV，直接解析对应列值

## 数据集顺序问题

- ⚠️ **现有数据集全部被 shuffle，不保持 CSV 原始顺序**（即使文件名不含 "shuffled"）
- `data_n3838_avg11944.json` 是唯一与 CSV 原始顺序一致的数据集
- `trans_to_json.py` 默认 `shuffle=True`（L42），裁剪后执行 shuffle 打乱顺序
- `cut_keep_max_n` 函数本身保持原始顺序（按原始索引重建），但后续 shuffle 步骤破坏顺序
- 如需保序数据集，运行 `trans_to_json.py` 时加 `--no-shuffle`
- 详细分析见 `references/dataset-order-analysis.md`

## 执行规范

- ⚠️ 压测命令**必须与用户对齐确认后才可执行**，不可直接运行
- ⚠️ 确认交互用编号简选（如 `1确认 / 2取消`），用户回复编号后立即执行，不加多余文字
- ⚠️ **同一provider（后端）的压测任务禁止同时执行，必须逐轮串行**：共享同一推理服务时，并行压测互相干扰（争抢服务端资源），导致QPS/e2e数据不可信。每轮必须等上轮完成后，再启动下一轮
- ✅ 长时间压测使用后台执行（`background=true` + `notify_on_complete=true`），前台会被新消息中断
- ⚠️ `acs-bench` 不在默认 PATH，需用完整路径：`$WORK_ROOT/$CONDA_ENV/bin/acs-bench`
- ⚠️ `nohup ... &` 后台运行时，确保 log 文件路径正确，便于后续查看
- ⚠️ 高并发下可能触发服务端限流，关注 Fail_Rate
- ⚠️ `--ignore-eos False` 表示尊重EOS正常停止；设为 True 则强制生成到 output-length
- ⚠️ **不同tokenizer生成的定长数据集不可混用**（tokenization结果不同），DS-V3数据集目录加 `_dsv3` 后缀区分
- ⚠️ **DeepSeek V4 Flash reasoning tokens**：该模型含reasoning机制，AVG_Completion_Tokens中大部分是reasoning tokens（实测345/400），实际content tokens仅约55。解读QPS/吞吐时需注意：Output_Throughput含reasoning tokens，实际有效内容吞吐远低于此
- ⚠️ **extract_subset.py 参数**：实际参数为 `--src/--dst/--count`（非 `--source/--num/--output`）；对大文件(>100MB)可能只提取1条（流式解析bug），需用Python ijson或手动流式解析替代
- ⚠️ **不同模型版本需对应tokenizer**：DeepSeek V3-0324的tokenizer不能用于V4-Flash，需下载对应版本tokenizer（见Tokenizer下载指南）
- ⚠️ **缓存命中场景也必须注入数据**：用`--prefix-mode shared`注入（同前缀，KV Cache命中前缀部分），而非不注入。不注入导致跨轮KV Cache效应，指标虚高。缓存命中vs不命中的区别仅在`--prefix-mode shared/unique`，两种场景每轮都必须重新注入
- ✅ 爬坡模式用于寻优；日常验证使用固定并发模式
- ⚠️ 爬坡模式耗时较长
- ⚠️ 数据集文件名中的数字含义：`n2820` = 2820条数据，`avg10443` = 平均输入10443 tokens
- ⚠️ 基准数据有重复内容（24组），顺序对比需去重后验证
- 📖 官方文档：https://support.huaweicloud.com/bestpractice-modelarts/modelarts_llm_infer_5906032.html#section0

## 数据准备（trans_to_json.py）

数据集由 `prof-test/trans_to_json.py` 从 CSV 生成，详见 `references/data-pipeline.md`。

生成后数据集位于 `$PROF_ROOT/dataset/mt_dataset/`。

关键要点：
- 顺序问题详见数据集顺序问题章节
- 基准数据 `data_n3838_avg11944.json` 存在重复内容（24组重复，共139条），顺序对比时需去重后验证
- 生成 keep_max_n 保序数据集命令示例：
```bash
source $WORK_ROOT/$CONDA_ENV/bin/activate
cd $PROF_ROOT
python3 scripts/trans_to_json.py  # 脚本路径以实际部署位置为准 \
  -i $PROF_ROOT/dataset/mt_dataset/data_n3838_avg11944.json \
  -o $PROF_ROOT/dataset/mt_dataset/ \
  -n data_n3838_avg11944 \
  -f json --json_field input \
  -t $PROF_ROOT/models/LongCat-Flash-Chat \
  -cm keep_max_n -ta 10240 \
  --no_shuffle -s 1000
```
- 生成 keep_error 保序数据集命令示例：
```bash
source $WORK_ROOT/$CONDA_ENV/bin/activate
cd $PROF_ROOT
python3 scripts/trans_to_json.py  # 脚本路径以实际部署位置为准 \
  -i $PROF_ROOT/dataset/mt_dataset/data_n3838_avg11944.json \
  -o $PROF_ROOT/dataset/mt_dataset/ \
  -n data_n3838_avg11944 \
  -f json --json_field input \
  -t $PROF_ROOT/models/LongCat-Flash-Chat \
  -cm keep_error -ta 10240 -e 2.0 \
  --no_shuffle -s 1000
```

> ⚠️ **`-n` 命名陷阱**：trans_to_json.py 会自动在 `-n` 名称后追加裁剪模式信息（如 `_cut_keep_error_e2.0_n2820_avg10443`）。因此 `-n` 必须只写**基础名**（如 `data_n3838_avg11944`），不要包含裁剪信息，否则输出文件名会重复拼接。例：`-n data_n3838_avg11944_cut_keep_error_e2.0_n2820_avg10240` → 输出变成 `..._cut_keep_error_e2.0_n2820_avg10240_cut_keep_error_e2.0_n2820_avg10443.json`（错误）

> ⚠️ **`-ta` 与实际 avg 的差异**：keep_error 模式下，`-ta` 是目标下限，实际 avg 可能高于 -ta。如 `-ta 10240 -e 2.0` 目标范围 [10035, 10445]，实际 avg 可能是 10443。输出文件名中的 avg 数字反映实际值，非 -ta 值
- 📖 官方文档：https://support.huaweicloud.com/bestpractice-modelarts/modelarts_llm_infer_5906032.html#section0

## 合规性

> ⚠️ 2026-05-12审计发现21项不合规，详见 `acs-bench-workflow → references/skill-compliance-audit-20260512.md`
> 主要问题：~20处Markdown表格(飞书不可渲染)、沟通规范属memory、归档task.md引用过时、历史数据写入主体、auto-decision逻辑

## 关联 Skills

- `acs-bench-workflow` — 全流程编排：6步法、Check流程、数据注入、报告输出规范
- `acs-bench-peak-finding` — 摸高策略：7步法、动态调整、稳定性验证

## 支撑文件

- `references/data-pipeline.md` — trans_to_json.py 数据管道详解：CSV→JSON 流程、cut_keep_max_n 保序性分析、现有数据集顺序问题、常用命令
- `references/benchmark-results-20260501.md` — 2026-05-01 压测结果：8组测试数据、保序/shuffled对比、寻优结论
- `references/benchmark-results-fixedlen-10k-400-20260501.md` — 2026-05-01 定长10k-400压测：6组摸高数据、e2e≤10s最优配置
- `references/longcat-flash-chat-benchmark-20260512.md` — LongCat-Flash-Chat混长场景压测实录：冒烟基线、7组摸高数据、e2e卡点甜点、QPS天花板≈31.8
- `references/longcat-varlen-3838-600-benchmark-20260513.md` — LongCat varlen-3838-600缓存命中爬坡实录：c=160/165/168三组、e2e≤6s甜点c=165/QPS=25.97
- `references/longcat-varlen-3838-600-ramp-20260513.md` — LongCat varlen-3838-600爬坡压测：c=168 QPS=25.4 E2E=6.1s、climb参数(growth-rate=30/interval=1000)
- `scripts/verify_dataset_order.py` — 验证子集是否保持原始数据集相对顺序：`python verify_dataset_order.py <source.json> <subset.json>`

## 任务事实持久化

> ⚠️ 任务事实（环境路径、QPS天花板、数据集属性、甜点结果）**不存memory**，存文件体系：

- `~/.hermes/task.md` — ⚠️ 已归档清空（2026-05-12），归档位置：$PROF_ROOT/archive/TASK_archive_20260512.md
- `$PROF_ROOT/TASK.md` — ⚠️ 已归档清空
- `$PROF_ROOT/COMPLETE_TASK.md` — ⚠️ 已归档清空

详见 `acs-bench-workflow` skill → `references/task-fact-management.md`

## 标准化工作目录（~/prof/）

```
~/prof/
├── conf/                        # 配置文件
│   ├── provider.yaml            # API Provider 配置
│   ├── provider.yaml.template   # Provider 模板
│   └── scenarios/               # 场景配置
│       ├── varlen-10k-600.yaml
│       └── fixedlen-10k-400.yaml
├── dataset/                     # 数据集
│   ├── mt_dataset/              # 变长数据集
│   ├── built_in/                # 内置LongBench数据集
│   └── fixed_length/            # 定长数据集
├── result/                      # 压测结果
│   ├── csv/                     # CSV 结果文件
│   └── report/                  # 汇总报告
├── log/                         # 运行日志
├── scripts/                     # 标准化脚本
│   ├── run_benchmark.sh         # 单次压测执行
│   ├── run_multi_scene.sh       # 多场景批量执行
│   ├── gen_fixed_dataset.sh     # 定长数据集生成
│   ├── parse_result.py          # CSV结果解析与汇总
│   ├── summary_csv.py          # 汇总CSV生成（用户指定列：场景/TPS/RPM等）
│   ├── validate_csv_report.py  # CSV报告校验（列完整性+关键列非空+数值合理性）
│   └── verify_dataset_order.py  # 数据集顺序验证
└── README.md
```

## 标准化压测流程（SOP）

当用户提出压测需求时，严格按以下流程执行：

### 第1步：明确测试需求

向用户确认以下**必填项**（缺项必须追问）：

- ****数据集****: 定长(输入长度-输出长度) 或 变长(裁剪模式+保序)；**未指定时默认定长** — 10k-400
- ****目标指标****: 目标QPS/RPM 或 目标时延 — 1700RPM, E2E<10s
- ****并发范围****: 初始并发、最大并发 — 300~450
- ****请求速率范围****: request-rate 范围 — 20~22
- ****输出长度****: output-length — 600

**可选项**（有默认值）：

- **epochs**: 1
- **warmup**: 0
- **ignore-eos**: False
- **num-requests**: = 数据集条数

### 第2步：明确测试场景

根据需求向用户确认场景选择，提供以下场景供选择：

- ****基准验证****: 首次测试/服务变更后 — 1~2组固定配置，验证服务可用
- ****寻优测试****: 需找最优并发/速率 — 2×2交叉矩阵 + 并发上探
- ****对比测试****: 换数据集/换配置 — 复现基准最优 + 扩展对比
- ****极限测试****: 试探系统天花板 — 逐步加压直到QPS下降/失败率上升

### 第3步：设计测试用例

**寻优矩阵设计规则：**
1. 在预估最优点附近做 **并发 × 速率** 交叉（通常 2×2 或 2×3）
2. 加一组**并发上探**（c+50 或 c+100）验证收益
3. 换数据集时**必须复现基准最优配置**，确认趋势一致
4. num-requests 必须与数据集条数匹配

**用例输出格式：**

```
| 编号 | 并发 | request-rate | num-requests | 目的 |
|------|------|-------------|-------------|------|
| ① | 350 | 20 | 2699 | 基准最优 |
| ② | ... | ... | ... | ... |
```

### 第3.5步：数据注入（每轮压测前必须执行）

**→ 完整规范见 `acs-bench-workflow` 第2b步**

每轮压测前必须执行数据注入，生成新一轮数据集，避免KV Cache命中导致指标虚高：

1. **执行注入**：`python3 scripts/inject_round_identifier.py --source <源数据集> --model <模型> --scene <场景> --round <轮次> --prefix-mode <shared|unique> --output <输出路径>`
2. **压测命令指向注入后数据集**：`-d <注入数据集名>`
3. **压测完成后删除注入数据**：`rm <注入数据集文件>`

**完整工作流**：`数据注入 → 压测执行 → CSV校验 → 结果解析 → 删除注入数据`，禁止跳过任何环节

### 第4步：输出测试命令

生成完整可执行命令，包含：
- 环境激活
- 数据集变量
- 日志命名（含 dataset_c{并发}_r{速率}_日期）
- 完整 acs-bench prof 命令

**向用户确认**：1确认执行 / 2调整用例

### 第5步：执行测试

- 每组用 `terminal(background=true, notify_on_complete=true)` 后台执行
- 完成回调中：读CSV → 提取指标 → 简报 → 启动下一组
- 全部完成后输出汇总表

### 第6步：输出测试结果与报告

**单组结果表：**
```
| 指标 | 值 |
|------|-----|
| QPS | 18.22 |
| AVG_E2E | 5.72s |
| ... | ... |
```

**汇总对比表（所有组）：**
```
| 编号 | 配置 | QPS | AVG_E2E | TP90_E2E | TP99_E2E | AVG_TTFT | Fail |
```

**分析结论：**
- 最优配置及推荐理由
- 各配置适用场景
- 与历史数据对比（如有）
- 是否达到目标指标，未达到的原因和建议

---

## 寻优方法论

### 核心发现

1. **request-rate是时延的决定因素**：rate对e2e影响远大于concurrency
2. **并发超过最优值后收益微弱**：QPS不再随concurrency提升
3. **高并发+高rate已过载**：QPS下降，时延恶化
4. **Shuffled vs Ordered趋势一致**

### 推荐配置

- **keep_max_n (avg10240)** | 场景=时延优先 | 并发=350 | request-rate=20 | 预期QPS=~18.2 | 预期AVG_E2E=~5.7s
- **keep_max_n (avg10240)** | 场景=均衡 | 并发=400 | request-rate=21 | 预期QPS=~18.6 | 预期AVG_E2E=~6.6s
- **keep_max_n (avg10240)** | 场景=吞吐优先 | 并发=350 | request-rate=22 | 预期QPS=~18.7 | 预期AVG_E2E=~9.3s
- **keep_error (avg10443)** | 场景=时延优先 | 并发=400 | request-rate=20 | 预期QPS=~17.9 | 预期AVG_E2E=~6.0s
- **keep_error (avg10443)** | 场景=均衡 | 并发=400 | request-rate=21 | 预期QPS=~18.0 | 预期AVG_E2E=~6.5s

### 测试用例设计方法

设计规则同SOP第3步寻优矩阵设计规则

### 经验参数参考

基于历史实测（avg输入 10240~10443 tokens, output 600 tokens）：

- **request-rate**: 20 — 时延优先；21均衡；22+时延恶化
- **concurrency**: 350~400 — 超过400收益微弱
- **output-length**: 600 — 业务相关，需用户指定
- **num-requests**: =数据集条数 — 全量测试
- **epochs**: 1 — 单轮即可
- **warmup**: 0 — 生产环境无需预热

### QPS天花板参考

**变长场景（output=600）：**

- **10240 tokens** | QPS天花板=~18.5 | ≈RPM=~1110 | 最优配置=c=350/r=20, E2E=5.72s
- **10443 tokens** | QPS天花板=~18.0 | ≈RPM=~1080 | 最优配置=c=400/r=20, E2E=6.03s

> request-rate是时延决定因素（r=20远优于r=22）；1700RPM目标无法达到需服务端扩容

**定长场景（input=10240, output=400, ignore-eos=True）：**

- **c=400, r=26** | QPS=~25.4 | AVG_E2E=~8.9s | 场景=**e2e≤10s最优**
- **c=400, r=27** | QPS=~26.3 | AVG_E2E=~10.5s | 场景=QPS最高但e2e超标
- **c=350, r=25** | QPS=~24.5 | AVG_E2E=~8.2s | 场景=保守稳定

> 定长10k-400场景：e2e≤10s约束下最大QPS≈25.4（≈1521 RPM），每+1 request-rate约+1.2s e2e
> ⚠️ 以上为当前服务配置下的天花板，扩容后需重新摸高

### 定长场景寻优方法论（2026-05-01实测）

**场景：input=10240, output=400, e2e≤10s, 目标max QPS**

摸高路径（6组测试）：

- **①** | 并发=350 | rate=25 | QPS=24.51 | AVG_E2E=8.19s | 达标=✅ | 分析=基准起点
- **②** | 并发=400 | rate=30 | QPS=26.35 | AVG_E2E=12.79s | 达标=❌ | 分析=双高过载
- **③** | 并发=350 | rate=28 | QPS=21.58 | AVG_E2E=13.99s | 达标=❌ | 分析=并发不足致排队，QPS反降
- **④** | 并发=400 | rate=25 | QPS=24.38 | AVG_E2E=8.17s | 达标=✅ | 分析=提并发无收益（与①一致）
- **⑤** | 并发=400 | rate=27 | QPS=26.29 | AVG_E2E=10.54s | 达标=❌ | 分析=仅超0.54s，接近甜点
- **⑥** | 并发=400 | rate=26 | QPS=25.35 | AVG_E2E=8.93s | 达标=✅ | 分析=**甜点确认**

关键发现：
1. **request-rate是e2e线性决定因素**：r=25→26→27，e2e从8.17→8.93→10.54s，每+1 rate约+1.2s e2e
2. **并发在rate≤26时无收益**：c=350 vs c=400 在r=25下QPS几乎相同（24.51 vs 24.38）
3. **高rate下需足够并发**：r=28时c=350严重排队（QPS反降至21.58），c=400更稳
4. **r=28是过载拐点**：并发不足+速率过高→TTFT飙升→QPS下降
5. **寻优策略**：先固定并发找rate甜点（二分法），再验证并发上探是否有收益

### 多组顺序执行模式

执行方式同多组对比测试→串行后台执行模式

