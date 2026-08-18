---
name: acs-bench-workflow
description: "LLM推理压测全流程：从场景输入→数据准备→压测执行→结果解析→报告输出→回传的完整SOP"
trigger: "压测流程, 压测工作流, benchmark workflow, 压测SOP, prof workflow"
related_skills:
  - acs-bench-benchmark
  - acs-bench-peak-finding
---

# acs-bench 压测全流程 SOP

## 概述

统一管理 LLM 推理服务压测的完整生命周期，从场景定义到结果回传。本 skill 是 `acs-bench-benchmark` 和 `acs-bench-peak-finding` 的上层编排。

## 工作目录

```
~/prof/  (即 $PROF_ROOT/)
├── conf/                        # 配置
│   ├── provider.yaml            # API Provider（含敏感信息）
│   ├── provider.yaml.template   # 模板
│   └── scenarios/               # 场景定义
├── dataset/                     # 数据集
│   ├── mt_dataset/              # 变长
│   ├── built_in/                # 内置
│   └── fixed_length/            # 定长
├── result/
│   ├── csv/                     # 原始CSV结果
│   └── report/                  # 汇总报告
├── log/                         # 运行日志
├── scripts/                     # 标准化脚本
└── README.md
```

## 压测 Check 流程（任务启动前必走）

接受压测任务后，按以下5步依次执行，确保配置正确、方案清晰、命令可复现。**每步输出必须对齐用户确认后，才进入下一步。**

### Check 1：确认任务输入

向用户收集/确认以下信息（缺项必须追问）：

- **模型** | 必填=✅ | 说明=模型名及版本 | 示例=DeepSeek-V3-0324
- **model_name** | 必填=✅ | 说明=Provider中的模型ID | 示例=`f17067bd-...`
- **api_key** | 必填=✅ | 说明=API密钥（确认已填入provider YAML） | 示例=已填
- **场景** | 必填=✅ | 说明=定长/混长，输入输出长度 | 示例=定长10k-400 / 混长avg10240-600
- **目标指标** | 必填=❌ | 说明=e2e上限、QPS/RPM目标（可后续指定） | 示例=e2e≤10s, max QPS
- **摸高策略** | 必填=❌ | 说明=3档e2e sweep / 单目标 | 示例=默认3档（≤10s/≤10.5s/≤11s）

**⚠️ 快速模式**：当用户已明确给出模型、数据集、场景等关键输入时，当用户已明确给出关键输入时，可提议跳过逐项追问，但需用户确认。不要对已确认的信息重复确认。目标指标可标注"冒烟后确定"，不必阻塞流程。

**输出：** 任务输入汇总表。

### Check 2：确定压测场景与参数

根据任务输入，逐场景确定：

- **场景名**: 格式：`{type}-{input}-{output}-{model}`，如 `fixedlen-10k-400-deepseek-v3`
- **dataset_type**: 定长→`fixedlen`，混长→`varlen`
- **dataset**: 定长：`in{len}_n{num}[_dsv30324]`；混长：具体数据集文件名
- **num_requests**: = 数据集条数
- **output_length**: 用户指定
- **ignore_eos**: 定长→True，混长→False
- **tokenizer**: 按模型选择，DSV3→`$PROF_ROOT/models/DeepSeek-V3-0324`
- **provider**: 按模型选择，DSV3→`./conf/provider_deepseek_v3.yaml`
- **起始并发/速率**: 无历史数据时保守起步（Qwen3历史值×0.5），有历史时基于历史微调

**输出：** 场景参数表（每个场景一行）。

### Check 3：设计压测方案

根据场景和目标，设计完整执行方案：

1. **冒烟测试**（Phase 0）：验证服务连通性，获取单请求基线指标（TTFT/TPOT/e2e/completion_tokens）
   - **目的**：①确认推理服务可正常响应；②获取TTFT/TPOT基本水准，作为后续摸高的参考基线；③校准e2e目标可行性
   - **默认参数**：`c=1, n=10`（单并发+10条数据，足够取平均且无并发干扰）
   - ⚠️ 冒烟后校准e2e目标：若单请求e2e已超目标，必须先调整目标再摸高
   - ⚠️ ignore_eos=False时不可用output_length×0.9估算completion_tokens：模型自然命中EOS停止，实际tokens远小于output_length。必须跑冒烟获取实际AVG_COMPLETION_TOKENS后再估算e2e
2. **QPS天花板探测**（Phase 1 阶段1）：固定中等并发，单变量递增r，测出系统实际QPS上限
   - ⚠️ 禁止双变量(c,r)同时递增：过度并发致排队恶化使QPS下降，误判天花板偏低
   - 正确方法：固定c（如c=270~300），递增r（20→30→40→50→60），QPS不再随r增长即天花板
   - ⚠️ 注意服务端rate limit：若出现大量"Request timeout: Request is limited by rate limit"，需降低rate重测
3. **e2e卡点摸高**（Phase 1~N 阶段2）：基于QPS天花板，反向调参逼近各档e2e目标
   - 每场景：从天花板附近反向逼近，逐步降低并发/rate直到e2e达标
   - 多档e2e目标：一次sweep递增中依次捕获各档甜点
   - 摸高过程中可提议免逐轮确认，但首次需用户授权
   - ⚠️ **每轮摸高必须执行完整工作流**：数据注入 → 压测执行 → CSV校验 → 结果解析 → 删除注入数据（详见第2b步），禁止跳过任何环节
4. **摸高后校验**（两步，缺一不可）：
   - **甜点校验（并发瓶颈校验）**：找到最优QPS后，设置 `c_verify = ceil(target_e2e × (optimal_QPS + 1))`，保持request_rate不变再压一轮。若QPS增长→并发不足是瓶颈，需以c_verify为新并发重新寻优；若QPS不增长→甜点确认，并发非瓶颈（详见 acs-bench-peak-finding 并发设置策略→甜点校验）
   - **稳定性验证**：甜点确认后，同配置至少跑3次，QPS波动<±5%，e2e波动<±10%，校验不通过则该甜点不可信
5. **汇总报告**：所有场景甜点配置 + 推荐方案，输出为CSV文件（summary_csv.py生成，详见第5步）

**输出：** 执行流程图 + 起始点设计表。

### Check 4：打印 Scenarios 压测配置

逐场景输出YAML配置内容，格式：

```yaml
# 场景名: {name}
dataset_type: {type}
dataset: {dataset}
num_requests: {n}
output_length: {output}
ignore_eos: {eos}

tokenizer: {tokenizer_path}
trust_remote_code: True
provider: {provider_path}

recommended:
  latency_priority:
    concurrency: {c}
    request_rate: {r}
  balanced:
    concurrency: {c}
    request_rate: {r}
```

**同时输出汇总对比表：**

- **dataset_type** | S1=... | S2=... | S3=...
- **dataset** | S1=... | S2=... | S3=...
- **num_requests** | S1=... | S2=... | S3=...
- **output_length** | S1=... | S2=... | S3=...
- **ignore_eos** | S1=... | S2=... | S3=...

**验证项：**
- [ ] 每个场景YAML文件存在于 `conf/scenarios/`
- [ ] 数据集文件存在且条数匹配
- [ ] provider YAML中api_key和model_name已填
- [ ] tokenizer路径存在
- [ ] dry-run通过（`bash scripts/run_scenario.sh -s <scenario> -c 10 -r 5 -p`）

### Check 5：打印压测命令

逐场景输出完整可执行命令，两种格式均需输出：

**方式1：run_scenario.sh（必须采用）**
```bash
# 环境激活
cd $PROF_ROOT
source $WORK_ROOT/$CONDA_ENV/bin/activate

# 冒烟测试
bash scripts/run_scenario.sh -s {scenario} -c 10 -r 5 -n 50

# 摸高起始点
bash scripts/run_scenario.sh -s {scenario} -c {concurrency} -r {rate} -l peak
```

**方式2：acs-bench 原始命令（完整展开，供审计/复现）**
```bash
acs-bench prof \
  --tokenizer {tokenizer} \
  --trust-remote-code \
  --benchmark-save-path "./result/csv/" \
  --epochs 1 --warmup 0 \
  --num-requests {n} \
  --concurrency-backend threading-pool \
  --backend openai-chat \
  --input-path "{dataset_path}" \
  --output-length {output} \
  --ignore-eos {eos} \
  --concurrency {c} \
  --request-rate {r} \
  --provider {provider_path} \
  -D
```

**同时输出：**
- 基础配置表（model_name / api_key / base_url / tokenizer / 工作目录 / 结果目录）
- 摸高策略表（场景 / 起始并发 / 起始rate / e2e目标档位）

---

**Check 1~5 全部对齐确认后，方可启动 Phase 0 冒烟测试。**

## 全流程 6 步法

### 第1步：场景输入（Scene Definition）

用户需提供以下信息（缺项必须追问）：

- **场景名** | 必填=✅ | 说明=标识本次压测 | 示例=`varlen-10k-600` / `fixedlen-10k-400`
- **数据集** | 必填=✅ | 说明=定长(输入长度-输出长度) 或 变长(数据集名) | 示例=`10k-400` / `data_shuffled_...`
- **目标指标** | 必填=✅ | 说明=目标QPS/RPM 或 目标时延 | 示例=`1700RPM, E2E<10s`
- **并发范围** | 必填=✅ | 说明=初始~最大并发 | 示例=`300~450`
- **速率范围** | 必填=✅ | 说明=request-rate 范围 | 示例=`20~26`
- **输出长度** | 必填=✅ | 说明=output-length | 示例=`600` / `400`
- **测试模式** | 必填=❌ | 说明=基准验证/寻优/对比/极限 | 示例=由用户指定，未指定时需询问

**场景自动映射**：

- **`varlen-10k-600`** | 配置文件=`conf/scenarios/varlen-10k-600.yaml` | 数据集=`data_shuffled_cut_keep_max_n_n2699_avg10240` | 默认参数=n=2699, o=600, eos=False
- **`fixedlen-10k-400`** | 配置文件=`conf/scenarios/fixedlen-10k-400.yaml` | 数据集=`in10240_n10000` | 默认参数=n=10000, o=400, eos=True

### 第2步：数据准备（Data Preparation）

#### 2a. 基础数据集检查

根据场景检查数据集是否就绪：

```bash
cd $PROF_ROOT

# 检查变长数据集
ls dataset/mt_dataset/${DATASET}.json

# 检查定长数据集
ls dataset/fixed_length/in${INPUT_LEN}_n${NUM}/

# 如需生成定长数据集（DS-V3-0324 tokenizer）
# -o 指向目录，{INPUT_LEN}.json 自动生成在目录下
# 目录命名：in{INPUT_LEN}_n{NUM}_dsv30324[_cache{RATIO}]
acs-bench generate dataset -il ${INPUT_LEN} -n ${NUM} \
  -t $PROF_ROOT/models/DeepSeek-V3-0324 \
  -trc \
  -o ./dataset/fixed_length/in${INPUT_LEN}_n${NUM}_dsv30324/

# 如需生成带前缀缓存的定长数据集（60%缓存示例）
# prefix-length = input-length × 60%
acs-bench generate dataset -il ${INPUT_LEN} -n ${NUM} \
  -pl ${PREFIX_LEN} \
  -t $PROF_ROOT/models/DeepSeek-V3-0324 \
  -trc \
  -o ./dataset/fixed_length/in${INPUT_LEN}_n${NUM}_dsv30324_cache60/
```

#### 2b. 轮次数据注入（每轮压测前必须执行）

**目的**：每轮压测使用不同数据，避免服务端 KV Cache 命中导致指标虚高。

**方法**：在源数据集每条 input content 最前面注入唯一轮次标识 `[模型_场景_轮次_时间]`，生成新一轮数据集。

**两种前缀模式**：

- **`shared`（默认）** | 标识格式=`[模型_场景_轮次_时间]` | 说明=所有数据共用相同前缀 | 适用场景=轮次间区分，同轮次内数据相同前缀
- **`unique`** | 标识格式=`[模型_场景_轮次_时间_数据id]` | 说明=每条数据不同前缀（含数据id） | 适用场景=需要每条请求都不同，彻底避免KV Cache命中

**标识示例**：
- shared：`[DSV3_S3_R01_20260508_0930]`（3838条数据全部相同）
- unique：`[DSV3_S3_R01_20260508_0930_0000]` ~ `[DSV3_S3_R01_20260508_0930_3837]`（3838条各不相同）

**命令**：

```bash
cd $PROF_ROOT

# === shared模式（默认）===

# S3场景（3838保序）第1轮，所有数据相同前缀
python3 scripts/inject_round_identifier.py \
  --source dataset/mt_dataset/data_n3838_avg11944.json \
  --model DSV3 --scene S3 --round 1 --prefix-mode shared \
  --output dataset/mt_dataset/data_n3838_avg11944_r01.json

# S2场景（2699保序）第1轮
python3 scripts/inject_round_identifier.py \
  --source dataset/mt_dataset/data_n3838_avg11944_cut_keep_max_n_n2699_avg10240.json \
  --model DSV3 --scene S2 --round 1 --prefix-mode shared \
  --output dataset/mt_dataset/data_n3838_avg11944_cut_keep_max_n_n2699_avg10240_r01.json

# === unique模式 ===

# S3场景第1轮，每条数据不同前缀（含数据id）
python3 scripts/inject_round_identifier.py \
  --source dataset/mt_dataset/data_n3838_avg11944.json \
  --model DSV3 --scene S3 --round 1 --prefix-mode unique \
  --output dataset/mt_dataset/data_n3838_avg11944_r01_uid.json

# 第N轮（更换轮次号即可）
python3 scripts/inject_round_identifier.py \
  --source dataset/mt_dataset/data_n3838_avg11944.json \
  --model DSV3 --scene S3 --round N --prefix-mode unique
```

**输出文件命名规范**：
- shared模式：`{源数据集名}_r{轮次号:02d}.json`（如 `data_n3838_avg11944_r01.json`）
- unique模式：`{源数据集名}_r{轮次号:02d}_uid.json`（如 `data_n3838_avg11944_r01_uid.json`）

**⚠️ 关键规范**：
- 每轮压测前必须执行数据注入，生成新一轮数据集
- 压测命令中 `--input-path` 必须指向注入后的数据集，而非源数据集
- ⚠️ **每一次压测均需使用新注入数据集**：即使同一轮次内多次压测（如摸高调参），每次压测前都必须重新注入数据，禁止复用已压测过的注入数据集
- **缓存命中场景也必须注入数据**：用`shared`模式（同前缀，KV Cache可命中前缀部分），而非不注入。不注入会导致跨轮KV Cache效应（前轮数据缓存残留），指标虚高
- **缓存命中 vs 缓存不命中**的区别仅在注入模式：
  - `shared`（缓存命中）：同轮数据共用相同前缀，KV Cache命中前缀部分，TTFT更低
  - `unique`（缓存不命中）：每条数据不同前缀（含data_id），KV Cache无法命中，反映真实无缓存性能
- 两种场景每轮都必须重新注入，轮次号递增，禁止复用
- 跨轮次必须重新注入，确保标识不同
- `shared`模式：轮次间区分，所有数据共用相同前缀
- `unique`模式：每条请求都不同，彻底避免KV Cache命中
- 标识注入位置：每条数据第一条 user message 的 content 最前面
- 标识格式：`[模型_场景_轮次_时间]\n` 或 `[模型_场景_轮次_时间_数据id]\n` + 原始content
- 注入后数据集条数不变，仅 content 前缀不同
- ⚠️ **每次压测必须使用新注入数据集**：每次压测前重新注入数据，不可复用已压测过的注入数据集
- ⚠️ **缓存命中场景也必须注入数据**：用`shared`模式注入（同前缀，KV Cache命中前缀部分），而非不注入。不注入会导致跨轮KV Cache效应，指标虚高。缓存命中vs不命中的区别仅在`--prefix-mode shared/unique`
- **⚠️ 已用数据及时删除**：压测完成后删除该轮注入数据集，释放磁盘空间（源数据集保留）
- **⚠️ 必须执行的工作流**：每次压测 = `数据注入 → 压测执行 → CSV校验 → 结果解析 → 删除注入数据`，禁止跳过任何环节

**数据集状态检查清单**：
- [ ] 源数据集文件存在
- [ ] 条数与 num-requests 匹配
- [ ] 变长数据集：确认 shuffle/保序 属性
- [ ] 定长数据集：确认 input-length 正确
- [ ] 本轮注入数据集已生成（`_r{N}.json` 存在且条数匹配）
- [ ] 注入标识格式正确（抽查首条 content 开头）

### 第3步：压测执行（Benchmark Execution）

#### 3a. 单次压测

**必须采用方式：run_scenario.sh（场景化脚本，自动读取YAML配置，自动保存日志）**

> run_scenario.sh 自动处理日志命名、环境激活、残留进程检查，且支持场景YAML参数复用。禁止直接使用 acs-bench 原始命令，仅当run_scenario.sh无法满足需求时经用户确认后方可使用。

```bash
cd $PROF_ROOT
source $WORK_ROOT/$CONDA_ENV/bin/activate

# 基础压测（使用YAML中定义的数据集）
bash scripts/run_scenario.sh -s <scenario_name> -c ${CONCURRENCY} -r ${RATE}

# 不限速压测（省略 -r，不传 --request-rate）
bash scripts/run_scenario.sh -s <scenario_name> -c ${CONCURRENCY}

# 使用注入数据集（-d 覆盖数据集名，用于摸高轮次）
bash scripts/run_scenario.sh -s <scenario_name> -c ${CONCURRENCY} -r ${RATE} \
  -d <injected_dataset_name> -l peak

# dry-run（仅打印命令不执行）
bash scripts/run_scenario.sh -s <scenario_name> -c ${CONCURRENCY} -r ${RATE} -p
```

**run_scenario.sh 参数**：
- **-s**: ✅ — 场景名（对应 conf/scenarios/<name>.yaml）
- **-c**: ✅ — 并发数
- **-r**: ❌ — 请求速率 (req/s)，省略则不限速（不传--request-rate）
- **-d**: ❌ — 覆盖数据集名（支持注入数据集）
- **-l**: ❌ — 日志前缀（如 peak/stability）
- **-n**: ❌ — 覆盖总请求数
- **-e**: ❌ — 覆盖测试轮数
- **-w**: ❌ — 覆盖预热轮数
- **-p**: ❌ — dry-run模式

**备选方式：run_benchmark.sh（通用脚本，需手动指定所有参数）**

```bash
cd $PROF_ROOT
bash scripts/run_benchmark.sh -c ${CONCURRENCY} -r ${RATE} \
  [-t varlen|fixedlen] -d ${DATASET} -o ${OUTPUT_LEN} -n ${NUM}
```

**DSV3场景列表：**
→ 详见 acs-bench-benchmark DSV3场景列表

#### 3b. 摸高测试

按 `acs-bench-peak-finding` skill 的两阶段7步法执行：
1. 高压探测QPS天花板 → 2. 确认天花板 → 3. 设定e2e目标 → 4. 计算并发下限 → 5. 设置初始rate → 6. 单变量递进摸高 → 7. 甜点校验（并发瓶颈校验） → 8. 稳定性验证

**执行规范**：
- ⚠️ 压测命令执行前必须与用户对齐确认
- ⚠️ 长时间压测用 `background=true` + `notify_on_complete=true`
- ⚠️ 多组测试串行执行（共享服务端）
- ⚠️ 每组前检查残留进程：`ps aux | grep acs-bench | grep -v grep`

### 第4步：结果解析（Result Parsing）

#### 4a. 结构化结果解析（推荐）

使用 `parse_benchmark_results.py` 自动解析CSV、匹配阶段、过滤异常、双场景对比：

```bash
cd $PROF_ROOT

# 解析今天所有结果（自动映射阶段，过滤Fail>5%异常）
python3 scripts/parse_benchmark_results.py --today --max-fail 0.05

# 仅解析shared/unique模式
python3 scripts/parse_benchmark_results.py --today --mode shared
python3 scripts/parse_benchmark_results.py --today --mode unique

# A vs B对比表（前缀匹配 vs 前缀不匹配）
python3 scripts/parse_benchmark_results.py --today --compare --max-fail 0.05

# 解析指定日期
python3 scripts/parse_benchmark_results.py --date 2026-05-08 --max-fail 0.05
```

**参数说明**：
- **`--today`**: 解析今天的结果 — -
- **`--date`**: 指定日期 YYYY-MM-DD — -
- **`--csv-dir`**: CSV目录 — `./result/csv/`
- **`--mode`**: 过滤模式：shared/unique/all — `all`
- **`--compare`**: 输出A vs B对比表 — -
- **`--max-fail`**: 最大Fail_Rate阈值（过滤异常） — `1.0`（不过滤）
- **`--stage-map`**: 自定义阶段映射JSON文件 — 内置DEFAULT_STAGE_MAP

**阶段映射**：脚本内置 `DEFAULT_STAGE_MAP`，通过CSV文件名时间戳(`_YYYY-MM-DD_HH_MM_SS.csv`)自动匹配阶段名、前缀模式、并发、速率。新增阶段时需更新映射。

**前缀模式自动推断**（`summary_csv.py` 增强逻辑）：
- **优先级**：`DEFAULT_STAGE_MAP`显式映射 > 日志文件名推断 > 空
- **日志推断规则**：日志文件名含`_uid` → `unique(前缀不匹配)`，否则 → `shared(前缀匹配)`
- **适用场景**：新增压测结果不在stage_map中时，自动从日志文件名推断前缀模式，无需手动更新映射
- **`match_log_file`返回值**：`(request_rate, is_fixedlen, prefix_mode)` 三元组（原为二元组）

**输出示例**：
```
阶段           前缀          c   r     QPS      E2E     TP99     TTFT   OutTput   Fail
------------------------------------------------------------------------------------------
A-P0冒烟       shared     10   5    1.20    6.77s   14.89s    1.27s     264.8  0.0%
A-P1天花板      shared   1000  50   33.40   18.18s   36.36s    9.48s    8935.5  0.0%
B-P0冒烟       unique     10   5    1.18    6.83s   15.02s    1.20s     266.9  0.0%
```

#### 4b. 原始CSV解析（底层）

```bash
cd $PROF_ROOT

# 单文件解析
python3 scripts/parse_result.py result/csv/summary_xxx.csv

# 目录汇总（按QPS排序）
python3 scripts/parse_result.py --dir result/csv/ --sort qps

# JSON格式输出（供程序处理）
python3 scripts/parse_result.py --dir result/csv/ --json

# 汇总CSV生成（用户指定列：场景/TPS/RPM等）
python3 scripts/summary_csv.py --dir result/csv/ -o result/report/summary.csv --sort time
```

**关键指标提取**：

→ 详见 acs-bench-benchmark 结果解读

#### 4c. CSV报告校验（每轮压测后必须执行）

使用 `validate_csv_report.py` 校验压测输出CSV的完整性，确保关键指标列数据存在且合理。

```bash
cd $PROF_ROOT

# 报告模式（默认）：仅校验最终交付报告（result/report/*_benchmark_*.csv，19/20列）
python3 scripts/validate_csv_report.py --report
python3 scripts/validate_csv_report.py --report --today

# 原始模式：校验74列原始CSV（result/csv/summary_*.csv）
python3 scripts/validate_csv_report.py --raw
python3 scripts/validate_csv_report.py --raw --today

# 校验单个文件（自动识别类型）
python3 scripts/validate_csv_report.py result/report/S3_shared_benchmark_20260508.csv

# 严格模式：-1占位值也报错
python3 scripts/validate_csv_report.py --report --strict
```

> ⚠️ **报告模式仅校验 `*_benchmark_*.csv`**（最终交付报告），自动排除摸高简报（`*_summary_*.csv`）和历史文件

**支持的报告格式**：
- **19** | 格式名=报告19列 | 来源脚本=summary_csv.py | 说明=标准汇总报告
- **20** | 格式名=报告20列(含前缀模式) | 来源脚本=summary_csv.py | 说明=新版，增加前缀模式列
- **13** | 格式名=摸高简报13列 | 来源脚本=parse_benchmark_results.py | 说明=摸高过程简报，含占位行

**校验内容**：
- **文件存在且非空**: CSV文件完整写入 — 重新压测
- **列数匹配**: 19/20/13列（自动识别） — 检查生成脚本版本
- **数据行存在**: 有实际结果数据 — 重新压测
- **关键列非空**: 19/20列12列关键指标；13列仅校验表头 — 排查压测异常
- **数值合理性**: QPS∈[0,10000]、E2E∈[0,600]s等 — 排查数据异常

**19/20列关键列**：执行时间, 场景, 请求数, 输出长度, 最大并发, AVG_TTFT, AVG_TPOT, AVG_E2E, 输出TPS, 总TPS, 实际QPS, RPM

**⚠️ 校验铁律**：每轮压测完成后、结果解析前，必须先执行CSV校验。校验不通过则该轮结果不可信，需排查原因后重跑。

#### 4d. 结果沉淀规范

每轮压测完成后，必须：
1. **解析结果**：获取实际QPS、E2E、TTFT等关键指标
2. **更新阶段映射**：若新增阶段，在 `parse_benchmark_results.py` 的 `DEFAULT_STAGE_MAP` 中添加映射
3. **更新task.md**：将结果记录到压测任务进度文档
4. **判断是否达标**：对比e2e目标，决定下一步调参方向

### 第5步：报告输出（Report Generation）

**⚠️ 强制规范：压测结果必须按本步骤完整格式输出，不可只输出裸数据/解析表。用户会检查是否遵循流程规范。**

**⚠️ 报告生成必须使用skill脚本，禁止手动逐行解析CSV**：
1. `python3 scripts/validate_csv_report.py --raw --today` — 校验CSV完整性（每轮压测后必做）
2. `python3 scripts/parse_benchmark_results.py --today --stage-map <json>` — 结构化解析（需构建stage_map映射CSV时间戳→阶段信息）
3. `python3 scripts/summary_csv.py --dir <csv_dir> -o result/report/${SCENE}_${PREFIX}_benchmark_${DATE}.csv` — 生成正式CSV报告（19/20列结构化）
4. 报告文件写入 `result/report/` 目录，**格式必须为CSV**，禁止TXT/MD等其他格式

手动解析CSV的问题：遗漏字段、格式不一致、无法复用校验逻辑。脚本已处理74列原始CSV的所有边界情况。stage_map构建方法：从CSV文件名提取时间戳(YYYY-MM-DD_HH_MM)，映射到(阶段名, 前缀模式, c, r)元组，写入JSON文件传给--stage-map参数。

**⚠️ 报告格式铁律**：
- 报告必须是CSV格式（summary_csv.py生成20列），**绝对禁止TXT/MD格式**
- 必须用脚本生成CSV
- summary_csv.py生成的CSV如有BOM头，需去除后再校验（`python3 -c "..."` 去BOM）

#### 汇总对比表

```
| 编号 | 前缀模式 | 配置(c/r) | QPS | AVG_E2E | TP99_E2E | AVG_TTFT | Output_Throughput | Fail_Rate |
|------|---------|-----------|-----|---------|----------|----------|-------------------|-----------|
| ① | shared(前缀匹配) | 350/20 | 18.22 | 5.72s | 8.1s | 0.8s | 10932 | 0% |
| ② | unique(前缀不匹配) | 350/20 | 17.85 | 6.10s | 8.5s | 1.0s | 10780 | 0% |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |
```

**前缀模式列说明**：
- `shared(前缀匹配)`：所有数据共用相同前缀，KV Cache可命中前缀部分，TTFT可能更低
- `unique(前缀不匹配)`：每条数据不同前缀，KV Cache无法命中，反映真实无缓存性能

#### 分析结论

**必须包含以下5项，缺一不可：**

1. **最优配置**及推荐理由
2. **各配置适用场景**（时延优先/均衡/吞吐优先）
3. **与历史数据对比**（如有）
4. **是否达到目标指标**，未达到的原因和建议
5. **QPS天花板**估计

#### 报告保存

**⚠️ 报告必须落盘到 `result/report/` 目录，不可仅输出到对话。**

命名规范（**⚠️ 强制公式，禁止手写文件名**）：
```
# 变量定义
SCENE = S2/S3/AB_compare等场景标识
PREFIX = shared/unique等前缀模式标识
DATE = YYYYMMDD（当天日期）

# CSV（summary_csv.py生成，19/20列结构化数据）
result/report/${SCENE}_${PREFIX}_benchmark_${DATE}.csv

# 示例：S3_shared_benchmark_20260508.csv
# 示例：S3_unique_benchmark_20260508.csv
# 示例：S3_AB_compare_20260508.csv
```

**⚠️ 每次保存报告前必须按公式生成文件名，禁止手动拼文件名**。

报告内容必须包含workflow第5步规定的全部5项：汇总对比表、最优配置、适用场景、历史对比、达标情况+QPS天花板。

```bash
# 验证报告已保存
ls result/report/
```

### 第6步：结果回传（Result Delivery）

#### 6a. 飞书推送

压测完成后，自动推送结果到飞书：

```bash
# 通过 Hermes send_message 推送到飞书
# target: feishu:oc_f7dfc3dac03d33fa68f42fb906cc9be8
```

#### 6b. 定时简报（Cron Job）

如需定时简报，需新建cron job。

#### 6c. 结果归档

```bash
# 压测结果已自动保存在:
# - CSV: $PROF_ROOT/result/csv/
# - 报告: $PROF_ROOT/result/report/
# - 日志: $PROF_ROOT/log/
```

## 场景配置文件规范

场景配置文件位于 `conf/scenarios/`，YAML格式：

```yaml
# 场景名: varlen-10k-600
dataset_type: varlen              # varlen | fixedlen
dataset: data_shuffled_...        # 数据集名称
num_requests: 2699                # 请求数
output_length: 600                # 输出长度
ignore_eos: False                 # 忽略EOS

# 推荐配置（基于历史实测）
recommended:
  latency_priority:               # 时延优先
    concurrency: 350
    request_rate: 20
    expected_qps: 18.2
    expected_e2e: 5.7s
  balanced:                       # 均衡
    concurrency: 400
    request_rate: 21
    expected_qps: 18.6
    expected_e2e: 6.6s
```

## 基础规范（引用 acs-bench-benchmark）

以下规范以 `acs-bench-benchmark` skill 为准，本skill不重复定义：
- **环境信息**：节点/Conda/Tokenizer/Provider/目录布局 → 详见 benchmark 环境信息
- **日志命名规范**：4种格式 + 命名要点 → 详见 benchmark 日志命名规范
- **CSV结果文件命名规范**：格式 + CSV与日志关联 → 详见 benchmark CSV结果文件命名规范
- **场景推断规则**：定长/混长判定 → 详见 benchmark 场景推断规则
- **DSV3场景列表**：3场景定义 → 详见 benchmark DSV3场景列表
- **acs-bench原始命令模板**：完整参数展开 → 详见 benchmark 命令模板
- **脚本清单与参数**：8脚本 + 参数说明 → 详见 benchmark 标准化脚本
- **关键指标定义**：8指标完整定义 → 详见 benchmark 结果解读
- **数据集清单**：定长+变长完整清单 → 详见 benchmark 数据集类型

本skill仅补充以下 workflow 特有规范：
- 数据注入规范（第2b步）
- 结果沉淀规范（第4c步）
- 报告输出规范（第5步）
- 结果回传规范（第6步）
- 执行铁律（做事优先级/报告格式/改skill规范）

## 任务状态文件管理

任务事实（环境路径、基线数据、进度、甜点结果）**不存入memory**，按以下文件层级组织：

```
~/.hermes/task.md              ← ⚠️ 已归档清空（2026-05-12），归档位置：$PROF_ROOT/archive/TASK_archive_20260512.md
<project>/TASK.md              ← ⚠️ 已归档清空
<project>/COMPLETE_TASK.md     ← ⚠️ 已归档清空
```

**职责划分**：

- **`~/.hermes/task.md`**: 压测环境（路径/模型/RAM/fd限制）+ 任务挂钩表 — 跨任务持久
- **`<project>/TASK.md`**: 当前场景进度、已有数据、推理、下一步、命令模板 — 任务进行中，完成后归档到COMPLETE
- **`<project>/COMPLETE_TASK.md`**: 已完成场景的甜点汇总、冒烟基线、报告文件路径 — 任务完成后长期保留

**挂钩格式**（`~/.hermes/task.md`中）：
```markdown
| 任务 | 状态 | 任务文件 | 完成文件 |
|------|------|----------|----------|
| DSV3-0324 S3+S2摸高 | 进行中 | $PROF_ROOT/TASK.md | $PROF_ROOT/COMPLETE_TASK.md |
```

**memory只存工作习惯/偏好/沟通规范**，不存任务事实。任务事实的分类去向：
- 环境信息 → `~/.hermes/task.md`
- 进行中进度 → `<project>/TASK.md`
- 已完成结果 → `<project>/COMPLETE_TASK.md`
- 数据规范/流程 → skill
- 服务器网络等通用约束 → skill环境信息

### 任务归档流程

当压测任务全部完成或需清空当前任务状态时，执行归档：

1. **创建归档目录**：`mkdir -p $PROF_ROOT/archive/`
2. **合并归档**：将3个task文件内容合并写入 `$PROF_ROOT/archive/TASK_archive_{YYYYMMDD}.md`，按以下结构组织：
   - §1 全局任务索引（原 `~/.hermes/task.md`）
   - §2 进行中任务（原 `$PROF_ROOT/TASK.md`）
   - §3 已完成任务（原 `$PROF_ROOT/COMPLETE_TASK.md`）
3. **清空原文件**：
   - `~/.hermes/task.md` → 仅保留空壳+归档位置引用
   - `$PROF_ROOT/TASK.md` → 无活跃任务标记
   - `$PROF_ROOT/COMPLETE_TASK.md` → 已归档标记
4. **归档文件格式**：飞书友好，用列表/键值对格式而非Markdown表格

**归档触发条件**：用户明确要求清空/归档，或新压测任务与旧任务无关需全新开始

## 执行铁律

- ⚠️ **做事优先级：①memory → ②TASK.md → ③skill**，禁止自作主张；不确定必须问用户，不可自行决定格式/方案/规范
- ⚠️ **报告格式必须遵循已有规范**：CSV（summary_csv.py）+ TXT（parse_result.py），不可发明新格式（如md）
- ⚠️ **改skill时必须先读原有规范**，不可覆盖写入自己的偏好
- ✅ **大范围修改前先呈现问题清单供用户选择修复范围**，不可直接全量执行

## 结果解析脚本层级关系

三个脚本功能递进，不是重复：

- **`parse_result.py`** | 层级=底层 | 功能=原始CSV解析，16关键字段 | 输出=TXT/JSON | 使用场景=单文件查看、目录浏览
- **`summary_csv.py`** | 层级=汇总层 | 功能=关联日志推断场景/压测QPS，19列结构化 | 输出=CSV | 使用场景=报告落盘、跨批次对比
- **`parse_benchmark_results.py`** | 层级=业务层 | 功能=阶段映射+异常过滤+前缀模式+A vs B对比 | 输出=TXT | 使用场景=摸高过程快速看结果

关系链：`parse_result.py` → `summary_csv.py` → `parse_benchmark_results.py`

## 执行规范

- ⚠️ 压测命令执行前必须与用户对齐确认，采用编号简选（1确认/2调整）
- ⚠️ 确认后立即执行，不加多余文字
- 摸高过程中可提议免逐轮确认，但首次需用户授权
- ⚠️ **同一provider（后端）的压测任务禁止同时执行，必须逐轮串行**：共享同一推理服务时，并行压测互相干扰，导致数据不可信
- ⚠️ 长时间压测用 `background=true` + `notify_on_complete=true`
- ⚠️ 每组前检查残留进程：`ps aux | grep acs-bench | grep -v grep`
- ⚠️ `conf/provider.yaml` 含 API Key，不要提交到版本控制
- ✅ **每次压测使用新注入数据集**：每次压测前重新注入数据
- ✅ **压测完成后删除该轮注入数据集**，释放磁盘空间（源数据集保留）
- ⚠️ **必须执行的工作流**：每次压测 = `数据注入 → 压测执行 → CSV校验 → 结果解析 → 删除注入数据`，禁止跳过任何环节
- ⚠️ **CSV校验必须在结果解析前执行**：`python3 scripts/validate_csv_report.py --dir result/csv/ --today`，校验不通过则结果不可信
- ⚠️ **CSV校验必须每轮执行，不可批量补验**：摸高过程中每轮压测完成后立即校验该轮CSV，而非全部跑完后再批量校验
- ⚠️ **脚本优先沉淀到skill**：新增或修改脚本时，先保存到压测skill（`scripts/`目录），再部署到 `$PROF_ROOT/scripts/` 执行。确保skill始终是脚本的single source of truth
- 📖 官方文档：https://support.huaweicloud.com/bestpractice-modelarts/modelarts_llm_infer_5906032.html#section0

→ 沟通规范详见memory

## 数据集与汇总CSV

数据集清单、汇总CSV脚本（summary_csv.py 20列定义、场景推断、压测QPS提取）详见 `acs-bench-benchmark` skill。

- ⚠️ **保序数据集优先**：保序数据集为默认选择。S2场景YAML已改为保序数据集`data_n3838_avg11944_cut_keep_max_n_n2699_avg10240`（非shuffled），S3使用`data_n3838_avg11944`。设计压测方案时默认使用保序数据集
- ⚠️ **执行顺序S3→S2**：当多场景摸高时，执行顺序：S3(3838)→S2(2699)，而非按数据量从小到大

## 合规性

> ⚠️ 2026-05-12审计发现42项不合规，详见 `references/skill-compliance-audit-20260512.md`
> 主要问题已修复ill、5处与benchmark重复内容

## 关联 Skills

- `acs-bench-benchmark` — 压测执行细节、命令模板、参数说明、环境信息、日志/CSV规范（本skill的基础定义层）
- `acs-bench-peak-finding` — 摸高测试策略、动态调整、稳定性验证（本skill的策略层）

## Workflow特有脚本

- **`inject_round_identifier.py`**: 轮次数据注入（shared/unique） — `scripts/inject_round_identifier.py`
- **`cleanup_injected_data.py`**: 注入数据清理（--stats/--dry-run/--all） — `scripts/cleanup_injected_data.py`
- **`parse_benchmark_results.py`**: 阶段映射+异常过滤+双场景对比 — `scripts/parse_benchmark_results.py`
- **`validate_csv_report.py`**: CSV报告校验（列完整性+关键列非空+数值合理性） — `scripts/validate_csv_report.py`
- **`extract_subset.py`**: 数据集子集提取（⚠️ 参数为 `--src/--dst/--count`，大文件可能只提取1条，需手动Python替代） — `python3 scripts/extract_subset.py --src SRC --dst DST --count N`
- **`migrate_and_cleanup.sh`**: 磁盘迁移清理（历史） — `scripts/migrate_and_cleanup.sh`

**脚本层级关系**：`parse_result.py`(底层解析) → `summary_csv.py`(汇总+场景推断+前缀模式) → `parse_benchmark_results.py`(业务层:阶段映射+对比)；`validate_csv_report.py`(校验层:独立于解析链，压测后第一时间执行)
底层脚本详见 `acs-bench-benchmark` 标准化脚本章节。

> 本skill是上层编排，benchmark/peak-finding是下层能力。数据注入(2b)、结果解析(4)、报告输出(5)的规范以本skill为准

## 支撑文件

- `references/output-columns-spec.md` — 压测输出数据列规格：原始CSV 74列、parse_result.py 16字段、summary_csv.py 19列的完整定义与索引映射
- `references/dsv3-0324-three-scenario-peak-plan.md` — DSV3-0324 三场景摸高计划（定长10k-400、混长2699、混长3838）— 注意：优先保序数据集、执行顺序S3(3838)→S2(2699)
- `references/s3-round-injection-plan.md` — S3场景轮次数据注入压测方案（每轮重新构造数据集）
- `references/task-fact-management.md` — 任务事实持久化规范：环境→task.md，完成→COMPLETE_TASK.md，进行中→TASK.md
- `references/longcat-setup-session-20260511.md` — LongCat-Flash-Chat压测环境搭建实录：provider复用模式、场景配置、冒烟命令
- `references/longcat-peak-finding-20260512.md` — LongCat摸高结果：4档甜点配置、QPS瓶颈在c不在r、E2E对r敏感度极低、并发天花板c≈174
- `references/script-hierarchy.md` — 结果解析三层脚本关系（parse_result→summary_csv→parse_benchmark_results）、使用场景映射、挂钩情况
- `references/skill-compliance-audit-methodology.md` — Skill合规审计方法论：12项检查规则、执行流程、表格转换规则、历史审计记录
- `references/task-state-file-templates.md` — 任务状态文件层级模板（~/.hermes/task.md / TASK.md / COMPLETE_TASK.md）及memory→skill→task文件的事实分类去向
- `references/skill-compliance-audit-20260512.md` — 2026-05-12三skill合规性审计：80项不合规、12类规则、环境验证、修复优先级
- `references/dsv3-s3-shared-peak-session-20260512c.md` — DSV3-0324 S3 shared摸高修正版：天花板探测修正(27→31)、并发公式高估发现、6档甜点+1700RPM验证