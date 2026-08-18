# 压测输出数据列规格

## 一、原始CSV（acs-bench 直接输出）— 共 74 列

按类别分组：

| 类别 | 列名 | 列数 |
|------|------|------|
| **基础信息** | Execution_Time, Num_Requests, Epochs, Input_Length, Output_Length, Concurrency | 6 |
| **吞吐量** | Total_Token_Throughput, Output_Token_Throughput | 2 |
| **TTFT（首token延迟）** | TP75/TP90/TP95/TP99/MAX/AVG_TTFT | 6 |
| **TPOT（每token延迟）** | TP75/TP90/TP95/TP99/MAX/AVG_TPOT | 6 |
| **Server TTFT** | TP75/TP90/TP95/TP99/MAX/AVG_Server_TTFT | 6 |
| **Server TPOT** | TP75/TP90/TP95/TP99/MAX/AVG_Server_TPOT | 6 |
| **TPOT_SEC（含推理段）** | TP90/TP95/TP99/MAX/AVG_TPOT_SEC | 5 |
| **首→次token间隔** | TP90/TP95/TP99/MIN/MAX/AVG_TIME_BETWEEN_FIRST_AND_SECOND_TOKEN | 6 |
| **E2E（端到端延迟）** | TP75/TP90/TP95/TP99/MAX/AVG_E2E | 6 |
| **Server E2E** | TP75/TP90/TP95/TP99/MAX/AVG_Server_E2E | 6 |
| **运行统计** | Total_Time, QPS, Fail_Rate, Timeout_Fail_Rate | 4 |
| **采样参数** | Control_Method, Growth_Rate, Backend, Provider, Top-K, Top-P, Temperature | 7 |
| **Completion Tokens** | TP90/TP95/TP99/MIN/AVG_COMPLETION_TOKENS | 5 |
| **Token细分** | AVG_REASONING_TOKENS, AVG_CONTENT_TOKENS, AVG_PROMPT_TOKENS | 3 |

### 完整列索引表

| 索引 | 列名 |
|------|------|
| 0 | Execution_Time |
| 1 | Num_Requests |
| 2 | Epochs |
| 3 | Input_Length |
| 4 | Output_Length |
| 5 | Concurrency |
| 6 | Total_Token_Throughput(tokens/s) |
| 7 | Output_Token_Throughput(tokens/s) |
| 8 | TP75_TTFT(s) |
| 9 | TP90_TTFT(s) |
| 10 | TP95_TTFT(s) |
| 11 | TP99_TTFT(s) |
| 12 | MAX_TTFT(s) |
| 13 | AVG_TTFT(s) |
| 14 | TP75_TPOT(s) |
| 15 | TP90_TPOT(s) |
| 16 | TP95_TPOT(s) |
| 17 | TP99_TPOT(s) |
| 18 | MAX_TPOT(s) |
| 19 | AVG_TPOT(s) |
| 20 | TP75_Server_TTFT(s) |
| 21 | TP90_Server_TTFT(s) |
| 22 | TP95_Server_TTFT(s) |
| 23 | TP99_Server_TTFT(s) |
| 24 | MAX_Server_TTFT(s) |
| 25 | AVG_Server_TTFT(s) |
| 26 | TP75_Server_TPOT(s) |
| 27 | TP90_Server_TPOT(s) |
| 28 | TP95_Server_TPOT(s) |
| 29 | TP99_Server_TPOT(s) |
| 30 | MAX_Server_TPOT(s) |
| 31 | AVG_Server_TPOT(s) |
| 32 | TP90_TPOT_SEC(s) |
| 33 | TP95_TPOT_SEC(s) |
| 34 | TP99_TPOT_SEC(s) |
| 35 | MAX_TPOT_SEC(s) |
| 36 | AVG_TPOT_SEC(s) |
| 37 | TP90_TIME_BETWEEN_FIRST_AND_SECOND_TOKEN(s) |
| 38 | TP95_TIME_BETWEEN_FIRST_AND_SECOND_TOKEN(s) |
| 39 | TP99_TIME_BETWEEN_FIRST_AND_SECOND_TOKEN(s) |
| 40 | MIN_TIME_BETWEEN_FIRST_AND_SECOND_TOKEN(s) |
| 41 | MAX_TIME_BETWEEN_FIRST_AND_SECOND_TOKEN(s) |
| 42 | AVG_TIME_BETWEEN_FIRST_AND_SECOND_TOKEN(s) |
| 43 | TP75_E2E(s) |
| 44 | TP90_E2E(s) |
| 45 | TP95_E2E(s) |
| 46 | TP99_E2E(s) |
| 47 | MAX_E2E(s) |
| 48 | AVG_E2E(s) |
| 49 | TP75_Server_E2E(s) |
| 50 | TP90_Server_E2E(s) |
| 51 | TP95_Server_E2E(s) |
| 52 | TP99_Server_E2E(s) |
| 53 | MAX_Server_E2E(s) |
| 54 | AVG_Server_E2E(s) |
| 55 | Total_Time(s) |
| 56 | QPS |
| 57 | Fail_Rate |
| 58 | Timeout_Fail_Rate |
| 59 | Control_Method |
| 60 | Growth_Rate |
| 61 | Backend |
| 62 | Provider |
| 63 | Top-K |
| 64 | Top-P |
| 65 | Temperature |
| 66 | TP90_COMPLETION_TOKENS |
| 67 | TP95_COMPLETION_TOKENS |
| 68 | TP99_COMPLETION_TOKENS |
| 69 | MIN_COMPLETION_TOKENS |
| 70 | AVG_COMPLETION_TOKENS |
| 71 | AVG_REASONING_TOKENS |
| 72 | AVG_CONTENT_TOKENS |
| 73 | AVG_PROMPT_TOKENS |

---

## 二、parse_result.py 提取的 16 个关键字段

| 列索引 | 字段名 | 显示名 | 格式 |
|--------|--------|--------|------|
| 0 | Concurrency | Concurrency | 整数 |
| 1 | Request_Rate | Request Rate | 浮点2位 |
| 5 | Num_Requests | Num Requests | 整数 |
| 13 | AVG_TTFT | AVG TTFT (ms) | 浮点2位 |
| 14 | TP90_TTFT | TP90 TTFT (ms) | 浮点2位 |
| 15 | TP95_TTFT | TP95 TTFT (ms) | 浮点2位 |
| 16 | TP99_TTFT | TP99 TTFT (ms) | 浮点2位 |
| 17 | AVG_TPOT | AVG TPOT (ms) | 浮点2位 |
| 20 | AVG_E2E | AVG E2E (ms) | 浮点2位 |
| 21 | TP90_E2E | TP90 E2E (ms) | 浮点2位 |
| 22 | TP95_E2E | TP95 E2E (ms) | 浮点2位 |
| 23 | TP99_E2E | TP99 E2E (ms) | 浮点2位 |
| 24 | Output_Token_Throughput | Output Throughput | 浮点2位 |
| 25 | Total_Token_Throughput | Total Throughput | 浮点2位 |
| -5 | QPS | QPS | 浮点2位 |
| -4 | Fail_Rate | Fail Rate | 百分比 |

⚠️ **兼容性警告**：parse_result.py 的列索引基于旧版CSV（~26列），与当前74列格式不一致。如需用 parse_result.py 解析新CSV，需更新其列索引定义。

---

## 三、summary_csv.py 汇总CSV — 19 列

| # | 列名 | 来源 | 说明 |
|---|------|------|------|
| 1 | 执行时间 | 原始CSV col0 (Execution_Time) | 压测完成时间 |
| 2 | 场景 | 推断 | 日志名含`in{n}_n{m}`→定长，否则→混长 |
| 3 | 请求数 | 原始CSV col1 (Num_Requests) | |
| 4 | 输入长度 | 原始CSV col3 (Input_Length)，空则用col73 AVG_PROMPT_TOKENS | |
| 5 | 输出长度 | 原始CSV col4 (Output_Length) | |
| 6 | 最大并发 | 原始CSV col5 (Concurrency) | |
| 7 | 压测QPS | 日志文件名 `_r{rate}_` 匹配 → bash history匹配 → 空 | ⚠️ 实际存的是request-rate（压测参数），非实际QPS |
| 8 | AVG_TTFT(s) | 原始CSV col13 | 平均首token延迟 |
| 9 | TP90_TTFT(s) | 原始CSV col9 | P90首token延迟 |
| 10 | AVG_TPOT(s) | 原始CSV col19 | 平均每token延迟 |
| 11 | TP90_TPOT(s) | 原始CSV col15 | P90每token延迟 |
| 12 | AVG_E2E(s) | 原始CSV col48 | 平均端到端延迟 |
| 13 | TP90_E2E(s) | 原始CSV col44 | P90端到端延迟 |
| 14 | 输入TPS | 计算值：总TPS - 输出TPS | 输入token吞吐量 |
| 15 | 输出TPS | 原始CSV col7 (Output_Token_Throughput) | 输出token吞吐量 |
| 16 | 总TPS | 原始CSV col6 (Total_Token_Throughput) | 总token吞吐量 |
| 17 | total_time | 原始CSV col55 (Total_Time) | 压测总耗时(秒) |
| 18 | 实际QPS | 原始CSV col56 (QPS) | 实际测得QPS |
| 19 | RPM | 计算值：实际QPS × 60 | 每分钟请求数 |

**3个计算列**：输入TPS、RPM 为衍生计算；场景、压测QPS 为推断/匹配。

---

## 三b、summary_csv.py 汇总CSV — 20 列（含前缀模式）

与19列格式相同，增加第3列"前缀模式"（shared/unique），其余列顺延。

| # | 列名 | 说明 |
|---|------|------|
| 1~2 | 执行时间, 场景 | 同19列 |
| 3 | **前缀模式** | shared(前缀匹配) / unique(前缀不匹配) |
| 4~20 | 请求数~RPM | 同19列第3~19列 |

### 前缀模式推断逻辑（2026-05-08增强）

**优先级链**：`DEFAULT_STAGE_MAP`显式映射 > 日志文件名推断 > 空

| 推断源 | 规则 | 可靠性 | 适用场景 |
|--------|------|--------|---------|
| `DEFAULT_STAGE_MAP` | CSV时间戳→(阶段,前缀模式,c,r) | 最高 | 已知阶段的压测结果 |
| 日志文件名 | 文件名含`_uid` → `unique(前缀不匹配)`，否则 → `shared(前缀匹配)` | 高 | 新增压测结果不在stage_map中 |
| 空 | 无法推断 | — | 无日志文件且不在stage_map |

**`match_log_file`返回值**：`(request_rate, is_fixedlen, prefix_mode)` 三元组

**调用链**：
1. `parse_csv_file()` 调用 `match_log_file()` 获取 `(rate, fixedlen, log_prefix_mode)`
2. 将 `log_prefix_mode` 写入结果字典的"前缀模式"字段
3. `main()` 中 `match_prefix_mode()` 检查 stage_map，若命中则覆盖 log 推断值

---

## 三c、摸高简报CSV — 13 列（parse_benchmark_results.py 生成）

| # | 列名 | 说明 |
|---|------|------|
| 1 | 场景 | S2/S3等 |
| 2 | 阶段 | P0冒烟/P1天花板/P2摸高等 |
| 3 | 轮次 | 测试轮次编号 |
| 4 | c | 并发数 |
| 5 | r | 请求速率 |
| 6 | QPS | 实际QPS |
| 7 | AVG_E2E(s) | 平均端到端延迟 |
| 8 | AVG_TTFT(s) | 平均首token延迟 |
| 9 | TPOT(s/token) | 每token延迟 |
| 10 | AVG_COMPLETION | 平均输出token数 |
| 11 | Fail_Rate(%) | 失败率(百分比) |
| 12 | concurrency_min | 并发下限 |
| 13 | 备注 | 衡量说明 |

> ⚠️ 摸高简报含分隔行/汇总行，部分行关键列可能为空

---

## 常用指标速查

| 指标 | 原始CSV列 | 汇总CSV列 | 含义 |
|------|-----------|-----------|------|
| QPS | col56 | 第18列(实际QPS) | 每秒完成请求数 |
| AVG_E2E | col48 | 第12列 | 平均端到端延迟 |
| TP99_E2E | col46 | — | P99端到端延迟（汇总CSV未含） |
| AVG_TTFT | col13 | 第8列 | 平均首token延迟 |
| AVG_TPOT | col19 | 第10列 | 平均每token延迟 |
| 输出吞吐量 | col7 | 第15列(输出TPS) | 输出token/s |
| 总吞吐量 | col6 | 第16列(总TPS) | 总token/s |
| Fail_Rate | col57 | — | 失败率 |
| AVG_COMPLETION_TOKENS | col70 | — | 平均输出token数 |
| AVG_PROMPT_TOKENS | col73 | — | 平均输入token数 |
