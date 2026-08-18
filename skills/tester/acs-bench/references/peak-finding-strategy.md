---
name: acs-bench-peak-finding
description: "LLM推理服务摸高测试：系统性寻找最优并发和QPS配置，达到目标RPM/e2e指标"
trigger: "摸高, peak, 寻优, 最优并发, peak-finding, 摸高测试"
related_skills:
  - acs-bench-workflow
  - acs-bench-benchmark
---

# acs-bench 摸高测试 Skill

## 概述

系统性寻找 LLM 推理服务的最优并发和请求速率配置，使实际性能达到目标指标（RPM/QPS、e2e时延、TTFT等）。

## 基础规范（引用）

- **执行规范**（确认/后台/串行/残留检查）→ 以 `acs-bench-benchmark` 执行规范为准
- **数据注入/结果解析/报告输出** → 以 `acs-bench-workflow` 第2b/4/5步为准
- 本skill仅定义摸高特有规范：7步法(含报告输出与回传)/并发下限/单变量递进/动态调整/甜点校验(并发瓶颈校验)/稳定性验证

## 核心原理

### 三大约束

1. **并发设置策略**（甜点摸高阶段）：
   - **核心公式**：`concurrency = ceil(target_e2e × peak_QPS)`
     - `target_e2e`：当前测试档位的e2e目标（如测e2e≤4s档→target_e2e=4s）
     - `peak_QPS`：天花板探测阶段获得的最大QPS（系统固有吞吐上限）
   - **原理**：peak_QPS代表系统最大处理能力，concurrency=target_e2e×peak_QPS确保在目标e2e下并发足以支撑系统满载吞吐，既不不足（卡QPS）也不过剩（推高TTFT）
   - **调参方式**：并发按公式固定后，**仅调request_rate**寻找最优QPS甜点
   - **⚠️ 公式是起点非终点**：公式给出初始c值，但实际最优c可能更低（高估案例：e2e≤6s公式c=186实测最优c=162）。**推荐验证**：同时测c×0.8/c/c×1.2三点，取QPS最高且e2e达标的c作为最优并发
   - **⚠️ 低档位(e2e接近单请求延迟)公式严重高估**：当target_e2e接近单请求处理时间时，e2e主要由处理时间决定而非排队延迟，公式假设的线性关系不成立。实测：e2e≤4.5s公式c=140实测c=68(高估2倍)；e2e≤7s公式c=221实测c=208(偏高6%)；e2e≤8s公式c=252实测c=251(准确)；e2e≤10s公式c=310实测c=310(准确)。低档位应从低并发起步逐步摸高，不可盲信公式
   - **甜点校验（必须执行）**：找到最优QPS后，设置 `c_verify = ceil(target_e2e × (optimal_QPS + 1))`，保持request_rate不变再压一轮。若QPS增长→并发不足是瓶颈，需以c_verify为新并发重新寻优；若QPS不增长→甜点确认，并发非瓶颈
   - ⚠️ **target_e2e必须使用当前测试档位自身的e2e目标**，**禁止用全局最宽松档替代**
   - ⚠️ **过度并发危害**：concurrency远超公式值时，TTFT/e2e膨胀（更多请求排队），反而恶化时延
   - ⚠️ **peak_QPS必须来自天花板探测实测**，不可估算
   - 例：peak_QPS=31.8，e2e≤4s → c=ceil(4×31.8)=128；e2e≤10s → c=ceil(10×31.8)=318
   - **调参前必须先算concurrency**，不可凭感觉设并发

2. **QPS天花板**：系统有固有QPS上限（服务端处理能力）
   - request-rate 设置过高无意义，实际QPS受系统限制
   - request-rate 设置过低则人为限制了吞吐

3. **并发双刃剑**：
   - 并发太低 → 压力不够，无法达到系统极限
   - 并发太高 → TTFT膨胀，用户体验恶化

### 摸高策略（7步法 — 三阶段）

**阶段1：探测QPS天花板（两种方式可选）**
1. **方式A：固定并发单变量递增r**（传统方式）：选择中等并发（如c=270~300），固定c不变，单变量递增r（r=20→30→40→50→60），每轮记录QPS和e2e。确认QPS天花板：当QPS不再随r单调递增（r↑但QPS↓或持平），前一轮最高QPS即为peak_QPS
   - ⚠️ **禁止双变量(c,r)同时递增探测天花板**：过度并发致排队恶化使QPS下降，误判天花板偏低（实测偏差可达15%）。详见"天花板探测修正方法"章节
2. **方式B：跑坡测试**（推荐，更高效）：使用acs-bench原生`--use-climb`参数，一次跑坡得到完整QPS-并发曲线。**无需修改源码**，原生即支持。详见 `acs-bench-benchmark` §2a
   - **爬坡参数推荐**：`--init-concurrency 1 --growth-rate 30~60 --growth-interval 1000`（快速爬坡，1秒/步，步长30~60并发）。`-c`用公式计算：`ceil(target_e2e × target_QPS)`
   - **⚠️ 跑坡模式禁止同时指定`--request-rate`（`-r`）**：`--growth-rate`通过并发增长推高负载，`--request-rate`硬限请求速率，两者互斥。同时使用时并发增长无法推高QPS（被限速卡死），跑坡曲线失真。**正确做法：跑坡时省略`-r`，让并发自然驱动QPS；验证目标RPM时用固定并发+request_rate，不用climb**
   - **快速爬坡参数推荐**：`--init-concurrency 1 --growth-rate 30 --growth-interval 1000`（从c=1起步，每秒+30并发，快速到达目标并发）。适合快速探测场景，爬坡阶段仅数秒
   - **稳态爬坡参数推荐**：`--init-concurrency 10 --growth-rate 20 --growth-interval 30000`（从c=10起步，每30s+20并发，每步有足够稳态采样）。适合需要精细QPS-并发曲线的场景

**阶段2：e2e卡点摸高（反向调参）**
2. **设定e2e目标+计算并发**：
   - 多档e2e上限（如≤5s/≤6s/≤10s/≤11s）
   - **甜点摸高阶段并发**：`concurrency = ceil(target_e2e × peak_QPS)`（peak_QPS=步骤1确认的天花板QPS，target_e2e=当前档位e2e目标；并发固定后仅调request_rate）
   - **天花板探测阶段并发下限**：`concurrency_min = ceil(request_rate × target_e2e)`（request_rate=命令-r参数值，target_e2e=当前档位e2e目标）
3. **单变量递进摸高**：
   - 设置初始request-rate：略低于天花板（如 `ceil(peak_QPS × 0.9)`）
   - 固定并发，**仅调request_rate**寻找最优QPS甜点
   - **小步递进**（r步长1~5），每轮观察实际QPS、e2e、TTFT
   - 动态调整：QPS不足+e2e可接受→加r；e2e超标→降r；QPS达标+e2e达标→甜点
4. **甜点校验（并发瓶颈校验，必须执行）**：找到最优QPS后，设置 `c_verify = ceil(target_e2e × (optimal_QPS + 1))`，保持request_rate不变再压一轮。若QPS增长→并发不足是瓶颈，需以c_verify为新并发重新寻优；若QPS不增长→甜点确认，并发非瓶颈
5. **稳定性验证**：甜点确认后，每个配置至少跑3次，观察波动范围（QPS ±5%, e2e ±10%）

**阶段3：报告输出与回传（摸高完成后必须自动执行）**
6. **生成CSV报告**：
   - 执行 `python3 scripts/validate_csv_report.py --raw --today` 校验CSV完整性
   - 执行 `python3 scripts/summary_csv.py --dir result/csv/ -o result/report/${SCENE}_${PREFIX}_benchmark_${DATE}.csv` 生成正式CSV报告（20列）
   - 报告必须落盘到 `result/report/` 目录
   - ⚠️ **禁止手动逐行解析CSV生成报告**，必须用脚本
7. **输出分析结论+回传**：
   - 输出5项分析结论（详见下方"报告分析结论5项"）
   - 推送报告到飞书（`send_message` → feishu:oc_f7dfc3dac03d33fa68f42fb906cc9be8）
   - 摸高实录保存到skill references
   - ⚠️ **摸高全部完成后必须自动执行步骤6~7**，不可等用户要求。遗漏此步骤是常见流程错误

### 报告分析结论5项（步骤7必须包含）

1. **最优配置**及推荐理由
2. **各配置适用场景**（时延优先/均衡/吞吐优先）
3. **与历史数据对比**（如有）
4. **是否达到目标指标**，未达到的原因和建议
5. **QPS天花板**估计

### 测试用例设计模板

**输入参数：**
- `target_RPM`: 目标RPM
- `target_e2e`: e2e上限(秒)
- `target_TTFT`: TTFT上限(秒，可选)
- `system_QPS_baseline`: 已知系统QPS基线
- `dataset`: 数据集文件
- `num_requests`: 每轮请求数

**计算过程（天花板探测阶段）：**
```
target_QPS = target_RPM / 60
concurrency_min = ceil(request_rate × target_e2e)  # 天花板探测阶段专用；request_rate=命令-r参数值, target_e2e = 当前测试档位自身的e2e目标
initial_request_rate = ceil(target_QPS × 1.05)
```

**甜点摸高阶段并发设定：**
```
concurrency = ceil(target_e2e × peak_QPS)  # peak_QPS=天花板探测实测值；并发固定后仅调request_rate
```

**梯度轮次设计（至少3轮）：**
- R1: concurrency=concurrency_min, request-rate=initial_request_rate（理论下限）
- R2: concurrency=concurrency_min+50, request-rate=initial_request_rate+2（中等余量）
- R3: concurrency=concurrency_min+100, request-rate=initial_request_rate+4（激进推高）

根据R1-R3结果，可追加R4+微调。

## 结果评估矩阵

- **QPS≥target 且 e2e≤target** → ✅ 达标，进入甜点校验→稳定性验证
- **QPS<target 且 e2e≤target** → ⚠️ 压力不足，增加并发或request-rate
- **QPS≥target 且 e2e>target** → ⚠️ 过载，降低并发或rate
- **QPS<target 且 e2e>target** → ❌ 系统瓶颈，系统无法达到目标，需优化服务端

## 摸高实战案例

**案例1（Qwen3-32b on ModelArts MaaS）：**
- **结论：** 系统QPS天花板~19，无法达到1700 RPM
- **推荐配置：** concurrency=220, request-rate=25, QPS均值18.70 (~1122 RPM), e2e均值10.30s

**案例2（定长10k-400, e2e≤10s）：**
- **结论：** 最优c=400/r=26, QPS=25.35(≈1521 RPM), e2e=8.93s

**案例3（DSV3-0324 S3 shared, varlen-3838-600, 2026-05-12）：**
- **天花板探测修正**：双变量递增误判peak_QPS=27，修正为31（固定c=270单变量递增r）
- **并发公式高估**：e2e≤6s公式c=186实测最优c=162；e2e≤10s公式c=310实测最优c=270
- **甜点汇总**：e2e≤5s→c=155/r=21/QPS=19.60；e2e≤6s→c=162/r=27/QPS=24.34；e2e≤10s→c=270/r=45/QPS=31.04；e2e≤11s→c=341/r=50/QPS=32.12
- **1700RPM验证**：c=270/r=35→QPS=29.94(RPM=1796)✅

**摸高方法论：** 先高压探测QPS天花板→反向调参逼近e2e目标→单变量递进→稳定性验证。实战案例见 references/peak-finding-session-20260430.md 和 references/benchmark-results-20260501.md

## 甜点校验与稳定性验证

### 甜点校验（并发瓶颈校验，必须执行）

找到最优QPS后，必须执行甜点校验：
- 设置 `c_verify = ceil(target_e2e × (optimal_QPS + 1))`
- 保持request_rate不变，用c_verify再压一轮
- **若QPS增长** → 并发不足是瓶颈，需以c_verify为新并发重新寻优
- **若QPS不增长** → 甜点确认，并发非瓶颈，进入稳定性验证

### 稳定性验证

甜点确认后，在最优配置下连续跑3+次，观察：
- QPS波动范围（期望 < ±5%）
- e2e波动范围（期望 < ±10%）
- TTFT波动范围
- 失败率是否为0

## 执行命令模板

**推荐方式：run_scenario.sh（场景化脚本，自动读取YAML配置）**

```bash
cd $PROF_ROOT
source $WORK_ROOT/$CONDA_ENV/bin/activate

# 基础压测（使用YAML中定义的数据集）
bash scripts/run_scenario.sh -s varlen-3838-600-deepseek-v3 \
  -c ${CONCURRENCY} -r ${REQUESTRATE} -l peak

# 使用注入数据集（-d 覆盖数据集名）
bash scripts/run_scenario.sh -s varlen-3838-600-deepseek-v3 \
  -c ${CONCURRENCY} -r ${REQUESTRATE} \
  -d data_n3838_avg11944_r04_uid -l peak

# S2场景
bash scripts/run_scenario.sh -s varlen-10k-600-deepseek-v3 \
  -c ${CONCURRENCY} -r ${REQUESTRATE} \
  -d data_n3838_avg11944_cut_keep_max_n_n2699_avg10240_r01 -l peak

# dry-run（仅打印命令不执行）
bash scripts/run_scenario.sh -s varlen-3838-600-deepseek-v3 \
  -c ${CONCURRENCY} -r ${REQUESTRATE} -d ${INJECTED_DATASET} -l peak -p
```

**run_scenario.sh 参数说明**：
- **-s** (✅必填): 场景名（对应 conf/scenarios/<name>.yaml）
- **-c** (✅必填): 并发数
- **-r** (✅必填): 请求速率 (req/s)
- **-d** (❌选填): 覆盖数据集名（支持注入数据集，如 data_n3838_avg11944_r04_uid）
- **-l** (❌选填): 日志前缀（如 peak/stability）
- **-n** (❌选填): 覆盖总请求数
- **-p** (❌选填): dry-run模式

**备选方式：原始命令（调试/自定义场景）**

```bash
cd $PROF_ROOT
source $WORK_ROOT/$CONDA_ENV/bin/activate

dataset="data_n3838_avg11944_r04_uid"
date=$(date +%Y%m%d_%H%M)
log_file="log/run_peak_${dataset}_c${CONCURRENCY}_r${REQUESTRATE}_${date}.log"

nohup acs-bench prof \
  --tokenizer $PROF_ROOT/models/DeepSeek-V3-0324 \
  --trust-remote-code \
  --benchmark-save-path "./result/csv/" \
  --epochs 1 \
  --warmup 0 \
  --num-requests 3838 \
  --concurrency-backend threading-pool \
  --backend openai-chat \
  --input-path "./dataset/mt_dataset/${dataset}.json" \
  --output-length 600 \
  --ignore-eos False \
  --concurrency ${CONCURRENCY} \
  --request-rate ${REQUESTRATE} \
  --provider ./conf/provider_deepseek_v3.yaml \
  -D > ${log_file} 2>&1 &
```

## 每轮调参前必做检查（Pre-Round Checklist）

**⚠️ 这是最高优先级规范，违反即浪费轮次。每轮启动压测前必须逐项检查：**

1. **加载skill**：调用 `skill_view(name='acs-bench-peak-finding')` 确保遵循最新规范
2. **数据注入**：执行 `inject_round_identifier.py` 生成新一轮数据集（→ 完整规范见 `acs-bench-workflow` 第2b步）。禁止复用已压测过的注入数据集（KV Cache效应）
3. **CSV校验**：上轮压测结果必须先通过 `python3 scripts/validate_csv_report.py <csv_file>` 校验，不通过则结果不可信
3. **计算并发下限（按阶段选用公式）**：
   - **天花板探测阶段**：`concurrency_min = ceil(request_rate × target_e2e)`（**request_rate = 命令-r参数值**，直接取命令参数；**target_e2e = 当前测试档位自身的e2e目标**，不可用实测e2e或估算QPS）
   - **甜点摸高阶段**：`concurrency = ceil(target_e2e × peak_QPS)`（**peak_QPS = 天花板探测实测值**，**target_e2e = 当前档位自身目标**；并发按此公式固定后，仅调request_rate）
4. **验证并发充足**：`concurrency > concurrency_min`（天花板探测阶段），否则**先加并发再推rate**，不可用不足的并发跑
5. **单变量递进检查**：与上一轮对比，只改变c或r中的一个，不可同时调两个
6. **打印调参推理**：输出计算过程（如"天花板探测: r=50, target_e2e=19s → concurrency_min=ceil(50×19)=950, c=1000>950 ✅" 或 "甜点摸高: peak_QPS=31.8, target_e2e=10s → c=ceil(10×31.8)=318, 固定c调r"）

**典型错误（天花板探测阶段）：**
- R3结果QPS=28.2/e2e=8.82s，推r=50时错误地用expected_QPS≈35估算，算出concurrency_min=665
- **正确做法**：r=50是命令参数，直接用：concurrency_min=ceil(50×19)=950，c必须≥950（如c=1000）
- **根本错误**：用估算QPS代替request_rate，引入不确定性，多次导致并发不足

## 执行规范

- 执行规范（串行/后台/残留检查）→ 以 acs-bench-benchmark 执行规范为准
- ⚠️ **request_rate最低下限≥20**：低于20的rate无实际意义（QPS太低），且无法反映真实服务压力。设计e2e sweep时r起始值从20起
- ⚠️ **每轮压测后必须分析结果再设计下一轮**：禁止批量预设多轮参数一次性执行。正确流程：执行1轮→解析结果→分析e2e/QPS趋势→调整r或c→确认后执行下一轮
- ⚠️ **执行前必须与用户对齐确认命令**，不可直接执行；多组用例可一起确认
- ⚠️ **逐轮执行**：先跑1轮，分析结果，再与用户确认下一轮参数（动态调整）

## 动态调整策略（核心）

摸高不是固定梯度扫参，而是**根据每轮结果动态调整下一轮**：

1. 跑完1轮后，提取 actual_QPS、AVG_E2E、AVG_TTFT、Fail_Rate
2. 对照目标判定（见评估矩阵），确定调整方向：
   - QPS不足 + e2e可接受 → 增加并发或request-rate
   - QPS达标 + e2e超标 → 降低并发
   - QPS不足 + e2e超标 → 系统瓶颈，需降低request-rate减轻压力
   - QPS达标 + e2e可接受 → ✅ 甜点，进入甜点校验→稳定性验证
3. 与用户对齐下一轮参数后执行
4. 重复直到找到甜点

## 执行流程规范

### 1. 命令确认（必须）
- 压测命令**必须与用户对齐确认后才可执行**，不可直接运行
- 多组测试用例可一起确认

### 2. 后台执行
- 长时间压测命令必须用 `background=true` + `notify_on_complete=true` 执行
- 前台执行会被用户新消息中断（exit code 130），导致压测中断
- 每轮测试前确认无残留进程：`ps aux | grep acs-bench | grep -v grep`

### 3. 动态调整（核心策略）
- 摸高不是固定梯度，而是**根据每轮结果动态调整下一轮参数**
- 评估矩阵指导调整方向：
  - QPS不达标 + e2e不超标 → 增加并发或request-rate
  - QPS达标 + e2e超标 → 降低并发
  - QPS不达标 + e2e超标 → 系统瓶颈，需优化服务端
- 每轮出结果后，先分析再设计下一轮，与用户确认后执行

## 反思与经验

### 执行逻辑要点
- **单变量递进**：固定rate调并发，或固定并发调rate，不同时大幅调两个变量
- **低起步摸高**：从低并发起步逐步增加，避免高并发过载导致TTFT/e2e膨胀污染结果
- **小步递进**：并发步长20~30，二分搜索式逼近，而非大跳（如100）
- **每组3次**：同一配置至少跑3次，区分系统波动与参数效应
- **动态调整**：根据前一轮结果决定下一轮方向，不盲目按预设梯度

### 天花板探测修正方法（2026-05-12实测验证）

**⚠️ 原方法缺陷**：Phase 1用"c递增+r递增"双变量同时增长探测天花板（如c=400→600→800→1000, r=20→30→40→50），观察到QPS下降就判定天花板。**实际原因**：QPS下降不是系统处理能力到顶，而是过度并发导致排队恶化（TTFT飙升），反而降低有效吞吐。

**修正方法**：
1. **固定中等并发**（如c=270~300），单变量递增r（r=20→30→40→50→60），直到QPS不再增长
2. **或用多组(c,r)网格**：在c=200~400范围、r=20~60范围做稀疏网格采样，找全局QPS最大值
3. **关键原则**：并发和QPS不是单调关系——超过最优并发后，更多并发反而降低QPS。双变量同时递增无法区分"系统处理能力到顶"和"并发过载导致排队"

**实测案例（DSV3-0324 S3 shared, 2026-05-12）**：
- 错误路径：c=600/r=30→QPS=27.07，判定天花板≈27 → 实际c=270/r=45→QPS=31.04，天花板≈31
- 根因：c=800/r=40时TTFT=8.14s（过载排队），误判为系统处理能力上限

### 并发公式是起点而非终点（2026-05-12实测验证）

**c=ceil(target_e2e×peak_QPS)给出初始值，但实际最优c可能更低，需双向验证**：

- **公式高估案例**：e2e≤6s, peak_QPS=31 → c=ceil(6×31)=186, 但实测c=162/r=27(QPS=24.34)优于c=186/r=27(QPS=21.35, E2E=8.1s过载)
- **公式低估案例**：e2e≤11s, peak_QPS=31 → c=ceil(11×31)=341, 实测c=341/r=50(QPS=32.12)确实优于c=297/r=50(QPS=31.06)
- **推荐做法**：用公式算c作为起始点，同时测c×0.8和c×1.2两个邻近点，取三者中最优

### 爬坡甜点摸排方法论（climb模式寻优）

**适用场景**：已知大致并发范围，用climb模式快速精确定位e2e约束下的最优并发和步长。

**核心发现（LongCat varlen-3838-600, 缓存命中, 2026-05-13实测）**：

1. **并发c是e2e主导因素**：c每+1，e2e约+0.015~0.05s（非线性加速，越接近系统容量边际效应越大）
2. **growth-rate对e2e影响次之**：同c下步长变化对e2e影响约0.04s/10步长，但过大步长引入瞬态冲击
3. **growth-rate安全上限**：建议 `growth-rate ≤ ceil(c × 0.4)`，超过后瞬态效应明显（e2e微超目标）
4. **growth-interval=1000ms**：快速爬坡，大部分请求在稳态执行

**调参规律**：
- **先定c**：根据e2e目标，用公式 `c = ceil(target_e2e × peak_QPS)` 计算初始c，再微调±5验证
- **再定growth-rate**：`growth-rate ≤ ceil(c × 0.4)`，兼顾爬坡速度和稳态精度
- **c与e2e非线性**：c接近系统容量时，每+1并发e2e增量加速（边际递增），甜点附近需小步验证

**实测数据（LongCat varlen-3838-600, 缓存命中, e2e≤6s）**：
- c=160, g=50: QPS=25.42, E2E=5.889s ✅
- c=165, g=60: QPS=25.97, E2E=5.964s ✅ ← 甜点
- c=165, g=70: QPS=25.55, E2E=6.006s ⚠️微超（步长过大瞬态冲击）
- c=168, g=30: QPS=25.40, E2E=6.119s ❌（并发过高）

**多档甜点汇总（LongCat varlen-3838-600, 缓存命中, 2026-05-13实测）**：

| e2e目标 | c | growth-rate | QPS | AVG_E2E | 达标 |
|---------|---|------------|-----|---------|------|
| ≤4s | — | — | — | — | ❌不可达（单请求e2e≈4.3s） |
| ≤4.5s | 68 | 27 | 14.73 | 4.469s | ✅ |
| ≤5s | 100 | 40 | 19.56 | 4.903s | ✅ |
| ≤6s | 165 | 60 | 25.97 | 5.964s | ✅ |
| ≤7s | 208 | 83 | 27.34 | 6.990s | ✅ |
| ≤8s | 251 | 100 | 29.32 | 7.844s | ✅ |
| ≤10s | 310 | 124 | 29.86 | 9.239s | ✅ |
| ≤11s | 341 | 136 | 31.48 | 9.808s | ✅ |

- **QPS天花板**: ≈31.5
- **1700RPM最低时延配置**: c=230/growth-rate=92 → QPS≈28.51, RPM≈1711, E2E=7.499s ✅（比旧配置c=270/r=35的E2E=8.021s降低0.52s）
- **1700RPM稳定配置**: c=235/growth-rate=94 → QPS≈28.47, RPM≈1708, E2E=7.620s ✅（余量更大）
- **1700RPM旧配置**: c=270/r=35 → QPS=30.37, RPM=1822, E2E=8.021s ✅（吞吐余量大但时延高）
- **e2e≤4s不可达原因**: avg输入11944 token导致单请求e2e≈4.3s，即使c=1也无法满足4s约束
- **1700RPM验证**: c=270/r=35 → QPS=30.37, RPM=1822 ✅
- **e2e≤4s不可达原因**: avg输入11944 token导致单请求e2e≈4.3s，即使c=1也无法满足4s约束
- **e2e-c斜率(中档位)**: c=200~252区间≈0.018s/c，可用于快速估算邻近档位甜点

**命令模板**：
```bash
# 爬坡甜点摸排：c=165, growth-rate=60, 不限request-rate
bash scripts/run_scenario.sh \
  -s <scenario> -c 165 \
  --climb --climb-mode linear \
  --growth-rate 60 --growth-interval 1000 \
  --init-concurrency 1 \
  -l ramp-sweet
```

**⚠️ 爬坡模式禁止同时指定`-r`**（growth-rate与request-rate互斥，详见互斥规则）

### 典型错误模式
- **错误**: 双变量同时大跳 → **后果**: 无法归因 → **正确做法**: 固定一个，调另一个
- **错误**: 高并发起步 → **后果**: 过载污染结果 → **正确做法**: 低起步逐步摸高
- **错误**: 每组只跑1次 → **后果**: 无法区分波动 → **正确做法**: 至少3次观察方差
- **错误**: 步长过大 → **后果**: 错过甜点 → **正确做法**: 小步20~30递进
- **错误**: 不算concurrency_min就设并发（天花板探测阶段） → **后果**: 并发不足卡QPS，浪费测试轮次 → **正确做法**: **天花板探测阶段：每轮算concurrency_min = ceil(request_rate × target_e2e)，c必须>此值；甜点摸高阶段：concurrency = ceil(target_e2e × peak_QPS)，固定后仅调r**
- **错误**: 不加载skill就调参 → **后果**: 重复犯错 → **正确做法**: 每轮调参前重新加载skill确保遵循规范
- **错误**: **不计算concurrency_min就调参（天花板探测阶段）** → **后果**: **并发不足致QPS被卡，浪费轮次** → **正确做法**: **天花板探测阶段：每轮前算ceil(request_rate×target_e2e)，c必须>此值**
- **错误**: **用估算QPS代替request_rate** → **后果**: **引入不确定性，多次导致并发不足** → **正确做法**: **request_rate=命令-r参数值，直接取，不估算**
- **错误**: **用全局最宽松e2e目标算concurrency** → **后果**: **过度并发→TTFT/e2e膨胀，或并发不足→QPS被卡** → **正确做法**: **甜点摸高阶段：concurrency=ceil(target_e2e×peak_QPS)，target_e2e=当前档位自身目标，peak_QPS=天花板探测实测值；并发固定后仅调request_rate。天花板探测阶段：concurrency_min=ceil(request_rate×target_e2e)**
- **错误**: **不加载skill就调参** → **后果**: **重复犯已纠正的错误** → **正确做法**: **每轮前skill_view加载最新规范**
- **错误**: **场景数据集与实际命令不一致** → **后果**: **结果归因混乱，shuffled vs 保序性能不可比** → **正确做法**: **每轮执行前核对数据集名与场景YAML一致，日志文件名即数据集名**
- **错误**: **并发c≥1200触发客户端fd耗尽** → **后果**: **"Too many open files"错误，Fail Rate飙至7%+，数据不可靠** → **正确做法**: **执行 `ulimit -n 65535` 提升fd限制（⚠️ 当前默认值1024，需每次session前执行或配置永久生效）+`/etc/security/limits.conf`加`* soft/hard nofile 65535`永久生效；解除后c可达1500+，QPS天花板显著提升**
- **错误**: **QPS天花板附近r↑但QPS反降** → **后果**: **过载致排队恶化，实际吞吐反降（如r=49 QPS=33.98 > r=50 QPS=33.55）** → **正确做法**: **QPS不再随r单调递增即确认天花板；天花板附近应以最高QPS的r为甜点基准**
- **错误**: **天花板探测双变量递增（c↑+r↑同时）** → **后果**: **过度并发致QPS下降，误判天花板偏低（实测27 vs 实际31，偏差15%）** → **正确做法**: **固定中等并发，单变量递增r探测天花板；或用(c,r)网格采样找全局QPS最大值**
- **错误**: **盲信并发公式c=ceil(e2e×peak_QPS)不验证** → **后果**: **c高估致过载（c=186/r=27→E2E=8.1s，而c=162/r=27→E2E=5.82s）** → **正确做法**: **公式算c为起始点，同时测c×0.8和c×1.2，取三者最优**
- **错误**: **为跑坡测试修改acs-bench源码** → **后果**: **过度工程化，原生`--use-climb`已完整支持并发爬坡** → **正确做法**: **直接使用`--use-climb --climb-mode linear --growth-rate N --init-concurrency N --growth-interval N`，run_scenario.sh已透传这些参数（`--climb --growth-rate N --growth-interval N --init-concurrency N`）**
- **错误**: **摸高完成后不自动执行报告输出与回传（7步法步骤6~7）** → **后果**: **用户需手动要求输出报告，流程不闭环** → **正确做法**: **甜点确认后必须自动执行：CSV校验→summary_csv.py生成报告→输出5项分析结论→飞书回传→实录存references**
- **错误**: **跑坡模式同时指定--growth-rate和--request-rate** → **后果**: **request-rate限速卡死QPS增长，并发增长无法推高负载，跑坡曲线失真** → **正确做法**: **跑坡找上限→用climb增长并发，省略-r不限速；验证目标RPM→用固定并发+request_rate，不用climb**
- **错误**: **跑坡-c设过大（如c=300）而非用公式计算** → **后果**: **并发远超需要，e2e膨胀，跑坡大部分时间在过载区间** → **正确做法**: **`-c = ceil(target_e2e × target_QPS)`，如e2e≤6s×QPS≈28→c=168**
- **错误**: **run_scenario.sh用字符串拼接构建CMD（旧版）** → **后果**: **空变量时`--request-rate`仍泄漏到命令中（引号转义`\"--request-rate \"`被eval为单参数），acs-bench报错`No such option: --request-rate`** → **正确做法**: **已改用bash数组（`CMD_ARGS=()` + `CMD_ARGS+=()`），禁止回退到字符串拼接**

## 输出格式与命令确认

- **⚠️ 飞书输出避免Markdown表格**（列宽不自适应），改用列表/键值对格式（`- **key**: value`），紧凑且适配手机端。适用于测试方案、摸高路径、结果报告等所有飞书消息
- 首轮测试用例需与用户对齐确认，采用编号简选（1确认/2调整）
- 用户确认后立即执行，不加多余文字
- **摸高过程中后续轮次需与用户确认是否继续执行**
- 多组测试用例可一次性确认

## 模型特定调参经验

### LongCat-Flash-Chat (varlen-10k-600)
- **QPS瓶颈在并发c，不在速率r**: c固定时r变化对QPS影响<3%，应优先调c
- **E2E对r敏感度极低**: c=174下r=42~383，e2e仅波动0.06s，摸高时r可大胆放宽
- **并发天花板c≈174**: c>200后QPS反降、e2e飙升，安全上限留10% margin
- **推荐策略**: 先固定r=70(中等速率)，单变量递进调c逼近e2e目标，c确定后再微调r

### 执行方式与校验

- 执行方式（后台/串行/残留检查）→ 以 acs-bench-benchmark 执行规范为准
- 甜点校验：找到最优QPS后必须执行 `c_verify = ceil(target_e2e × (optimal_QPS + 1))` 校验（详见并发设置策略→甜点校验）
- 稳定性验证：甜点确认后，每个配置至少跑3次，QPS期望波动 < ±5%，e2e期望波动 < ±10%

## KV缓存效应（关键陷阱）

**⚠️ 复用同一注入数据集会导致e2e虚低0.1~0.2s，所有摸高结果不可信！**

- **根因**：acs-bench复用相同prompt时，服务端KV Cache命中，prefill阶段跳过已缓存token，TTFT/e2e显著降低
- **实测**：c=144,r=33用r01数据E2E=4.40s → 换r03新数据E2E=4.60s（差0.2s）
- **正确流程**：每轮压测必须注入新round数据（`inject_round_identifier.py --round N`），压测后删除注入数据
- **完整轮次**：注入r{N} → 压测(用r{N}数据) → CSV校验 → 结果解析 → 删除r{N}数据 → 分析下一轮
- **禁止**：跨轮复用同一r{N}数据集
- **⚠️ 缓存命中场景也必须注入数据**：用`--prefix-mode shared`（同前缀，KV Cache命中前缀部分），而非不注入。不注入会导致跨轮KV Cache效应（前轮数据缓存残留），指标虚高。缓存命中vs不命中的区别仅在`--prefix-mode shared/unique`，两种场景每轮都必须重新注入

**execute_code超时应对**：多轮压测脚本在execute_code中会超时(300s限制)。必须逐轮执行，每轮一个execute_code调用。禁止在单个execute_code中循环多轮（inject→bench→parse→delete），即使逻辑上是一个完整流程。正确模式：每轮一个execute_code，内含inject→bench→wait→parse→delete，单轮耗时约150~180s在300s限制内。

**报告生成必须用skill脚本**：摸高完成后输出报告时，禁止手动逐行解析CSV。必须使用：
1. `python3 scripts/validate_csv_report.py --raw --today` 校验CSV完整性
2. `python3 scripts/parse_benchmark_results.py --today --stage-map <json>` 结构化解析（需构建stage_map映射CSV时间戳→阶段信息）
3. 报告文件写入 `result/report/` 目录

手动解析CSV的问题：遗漏字段、格式不一致、无法复用校验逻辑。脚本已处理74列原始CSV的所有边界情况。

## 注意事项

- ⚠️ **甜点校验不可遗漏**：找到最优QPS后，必须执行 `c_verify = ceil(target_e2e × (optimal_QPS + 1))` 校验并发是否为瓶颈，通过后才进入稳定性验证。历史教训：甜点校验定义在并发设置策略中，容易在7步法/评估矩阵/动态调整等流程引用处遗漏，每次审查摸高流程必须显式检查
- ⚠️ **每次压测必须使用新注入数据集**：即使同一轮次内多次压测（如摸高调参），每次压测前都必须重新注入数据，禁止复用已压测过的注入数据集（KV Cache效应会导致e2e虚低0.1~0.2s）
- ⚠️ 关注Fail_Rate，>0说明服务端限流或过载
- ⚠️ request-rate不宜远超系统QPS上限，否则请求堆积导致TTFT飙升
- ⚠️ 摸高过程是"二分搜索"思路：根据每轮结果调整方向，不必严格按梯度
- ⚠️ **不同模型摸高起点不同**：Qwen3-32b历史最优不能直接用于DSV3-0324。DSV3是671B MoE模型，单token延迟和吞吐特征与32B dense模型差异大，必须从冒烟测试重新建立基线
- ⚠️ **服务端rate limit是首要瓶颈**：ModelArts MaaS有RPM/TPM限流，高并发+高rate（如c=300/r=30）会触发大量"Request timeout: Request is limited by rate limit"。摸高前必须先探测rate limit边界，再在限流内调参
- ⚠️ **e2e目标需基于冒烟基线校准**：若冒烟测试单请求e2e已超目标（如DSV3单请求14s > 目标10s），目标不可达，必须先调整目标再摸高
- ⚠️ **e2e目标可行性验证（冒烟测试后必做）**：冒烟测试完成后，必须验证e2e目标是否可达。计算：`单请求e2e ≈ TTFT + output_tokens × TPOT`。若单请求e2e已超过目标上限，说明目标不可达，需调整（提高e2e上限 / 缩短output / 改为纯QPS天花板模式）。**不要在不可达目标上浪费摸高轮次**
- ⚠️ **ignore_eos=False时不可用理论公式估算e2e**：`e2e ≈ TTFT + output_length × 0.9 × TPOT` 仅适用于ignore_eos=True（模型生成到max_tokens）。ignore_eos=False时，模型自然命中EOS停止，实际completion_tokens远小于output_length（实测DSV3 output=600时avg仅128 tokens，而非540），必须跑冒烟获取实际基线后再估算e2e。**不要用output_length×0.9估算ignore_eos=False场景的e2e，会严重高估**
- ⚠️ **跨场景e2e目标不可直接套用**：e2e与output_length强相关（e2e ≈ TTFT + output_tokens × TPOT），不同output_length的场景单请求e2e差异巨大。例如：output=400时e2e≈14s，output=600时e2e≈19.3s。**切换场景时必须重新冒烟建立基线，再基于新基线设定e2e目标**，绝不可将A场景的e2e目标直接用于B场景

## 多数据集对比测试

当需要对比不同数据集（如保序 vs shuffled）的性能差异时：

1. **先用一份数据集跑完所有配置**，记录结果
2. **切换数据集，基于已有结果设计寻优矩阵**，而非从零开始
3. **寻优矩阵设计原则**：
   - 在已发现最优点附近做 2×2 交叉（并发×速率）
   - 加1~2组边界上探（如更高并发），验证是否有收益
   - 总用例数控制在5组左右，避免过度测试
4. **示例矩阵**（基于保序数据集最优 c=350/r=20）：
   - ① c=350/r=20（基准复现）
   - ② c=350/r=22（同并发提速率）
   - ③ c=400/r=20（提并发保速率）
   - ④ c=400/r=22（双高压力）
   - ⑤ c=450/r=20（并发上探）
5. **多组用例一次性确认后串行执行，每组完成后汇报结果**

## 缓存命中/不命中双场景设计

当设计摸高测试方案时，默认应包含 shared（缓存命中）和 unique（缓存不命中）两种前缀模式：

1. **shared（缓存命中）**：prefix-mode=shared，所有数据共用相同前缀，KV Cache可命中，TTFT更低
2. **unique（缓存不命中）**：prefix-mode=unique，每条数据不同前缀（含data_id），KV Cache无法命中，反映真实无缓存性能

**方案设计要点**：
- shared有已有基线时，直接复用，仅探测未完成档位
- unique需完整流程（冒烟→天花板→摸高），可参考同模型其他场景unique/shared比值估算天花板
- 执行顺序：shared → unique，串行执行
- 两场景e2e目标相同，但QPS天花板和最优并发差异显著（unique天花板通常为shared的55~60%，最优并发更低）

## 合规性

> ⚠️ 2026-05-12审计发现17项不合规+2项内部矛盾，详见 `acs-bench-workflow → references/skill-compliance-audit-20260512.md`
> 主要问题：4处Markdown表格、实战案例属历史数据应迁references、auto-decision逻辑、memory规范错放skill、target_e2e表述内部矛盾

## 关联 Skills

- `acs-bench-workflow` — 全流程编排：数据注入（2b步）、CSV校验（4c步）、结果解析（4步）、报告输出（5步）
- `acs-bench-benchmark` — 压测执行：命令模板、脚本清单、参数说明、CSV校验脚本

> 摸高过程中的数据注入/结果解析/报告输出规范，以 `acs-bench-workflow` skill 为准

## 支撑文件

- `references/peak-finding-session-20260430.md` — 摸高实录：目标1700RPM测试数据、QPS天花板发现、最优甜点定位
- `references/benchmark-results-20260501.md` — 多配置对比测试结果：保序数据集3组 + shuffled数据集寻优矩阵
- `references/dsv3-varlen-peak-session-20260508.md` — DSV3-0324混长场景摸高实录：ignore_eos=False的completion_tokens发现、QPS天花板探测数据
- `references/dsv3-s3s2-peak-session-20260508b.md` — DSV3-0324 S3(3838保序)+S2(2699保序)重新摸高实录：完整S3四档甜点、S2进行中
- `references/dsv3-s2-keepmaxn-peak-plan-20260508c.md` — DSV3-0324 S2 keep_max_n保序数据集摸高方案：shared+unique双场景、≤18s/≤19s待探测、服务实例过期暂停
- `references/longcat-varlen-peak-session-20260512.md` — LongCat varlen-10k-600摸高实录：4档甜点、KV缓存效应发现、r对QPS影响有限
- `references/dsv3-s3-shared-peak-session-20260512c.md` — DSV3-0324 S3 shared摸高修正版：天花板探测修正(27→31)、并发公式高估发现、6档甜点+1700RPM验证
- `references/rate-growth-ramp-up-design.md` — ⚠️ **已过时/被替代**：该文档设计`--rate-growth`参数修改源码方案，但acs-bench原生`--use-climb`已完整支持并发爬坡（`--use-climb --climb-mode linear --growth-rate N --init-concurrency N --growth-interval N`），无需修改源码。保留仅供参考，新项目应使用原生climb模式
- `references/longcat-varlen3838-climb-session-20260513.md` — LongCat varlen-3838-600爬坡甜点摸排实录：c/growth-rate调参规律、4组数据
- `references/longcat-varlen3838-benchmark-plan-20260513.md` — LongCat varlen-3838-600完整压测方案：6档e2e+1700RPM+流程闭环
- `references/longcat-varlen3838-peak-results-20260513.md` — LongCat varlen-3838-600完整压测结果：5档甜点+天花板+1700RPM验证

## 报告Check规范

摸高完成后或中途暂停时，必须对报告输出做完整性检查：

- **残留压测进程**: `ps aux | grep acs-bench | grep -v grep` → 期望：无
- **CSV UTF-8 BOM**: `head -c 3 <file> | xxd` 首三字节=efbbbf → 期望：✅ 有BOM
- **CSV行数**: `wc -l` → 期望：与压测轮数匹配
- **甜点行完整性**: 检查summary CSV中甜点行是否填满 → 期望：已探测档有值，未探测档标⚠️
- **旧格式残留**: `ls *.txt *.md` in report dir → 期望：应清理（已有CSV替代）
- **注入数据残留**: `ls *_r*.json` in dataset dir → 期望：压测后应删除

**⚠️ BOM缺失是常见问题**：`summary_csv.py`已用`utf-8-sig`写入，但手动生成的报告（如Markdown格式）可能缺BOM。修复：
```python
with open(f,'r',encoding='utf-8') as fh: content=fh.read()
with open(f,'w',encoding='utf-8-sig') as fh: fh.write(content)
```