# Summary CSV Field Mapping Reference

## 汇总CSV输出列（20列，含前缀模式）

| 列 | 来源 | 转换规则 |
|---|---|---|
| 执行时间 | Execution_Time (col 0) | 原值 |
| 场景 | 推断 | 数据集输入长度一致→定长，不一致→混长 |
| 前缀模式 | 推断 | 日志文件名含`_uid`→unique(前缀不匹配)，否则→shared(前缀匹配) |
| 请求数 | Num_Requests (col 1) | 原值，整数 |
| 输入长度 | Input_Length (col 3) | 空时从AVG_PROMPT_TOKENS(col 73)取，整数 |
| **AVG_CONTENT_TOKENS** | AVG_CONTENT_TOKENS (col 72) | 原值取整，替代原"输出长度"列 |
| 最大并发 | Concurrency (col 5) | 原值，整数 |
| **压测QPS** | request-rate | 提取优先级：①日志文件名 ②bash history ③空 |
| AVG_TTFT(s) | AVG_TTFT (col 13) | 原值（秒），3位小数 |
| TP90_TTFT(s) | TP90_TTFT (col 9) | 原值，3位小数 |
| AVG_TPOT(s) | AVG_TPOT (col 19) | 原值，3位小数 |
| TP90_TPOT(s) | TP90_TPOT (col 15) | 原值，3位小数 |
| AVG_E2E(s) | AVG_E2E (col 48) | 原值（秒），3位小数 |
| TP90_E2E(s) | TP90_E2E (col 44) | 原值，3位小数 |
| **输入TPS** | 计算 | 总TPS - 输出TPS，整数 |
| **输出TPS** | Output_Token_Throughput (col 7) | 原值，整数 |
| **总TPS** | Total_Token_Throughput (col 6) | 原值，整数 |
| total_time | Total_Time (col 55) | 原值，整数 |
| **实际QPS** | QPS (col 56) | 结果CSV中的QPS，3位小数 |
| **RPM** | 计算 | 实际QPS × 60，整数 |

## 列变更历史

- **2026-05-15**: `输出长度`(col 4, Output_Length) → `AVG_CONTENT_TOKENS`(col 72, 实际平均content tokens)
  - 原因：客服评分类任务实际输出远小于output_length(600)，AVG_CONTENT_TOKENS(~235)更反映真实输出
  - 取整方式：`int(round(safe_float(val)))`

## 数值格式规范

- **小数列**：`format_val()` 使用 `f"{v:.3f}"`，保留3位小数
- **整数列**（索引0-based）：请求数(3), 输入长度(4), AVG_CONTENT_TOKENS(5), 最大并发(6), 压测QPS(7), 输入TPS(14), 输出TPS(15), 总TPS(16), total_time(17), RPM(19)
  - 写入CSV时去除 `.000` 后缀，保持整数格式

## 编码规范

- **写入**：`encoding='utf-8-sig'`（自动添加BOM头 EF BB BF）
- **原因**：Excel默认用GBK读CSV，无BOM则中文乱码
- **读取**：Python用 `encoding='utf-8-sig'` 自动去BOM，无影响
- **⚠️ 禁止改为 `utf-8`（无BOM）**：会导致Excel中文乱码

## 关键区分：压测QPS vs 实际QPS

- **压测QPS** = 压测命令中的 `--request-rate` 参数值（用户设定的发送速率）
- **实际QPS** = 结果CSV中的QPS字段（服务端实际处理的查询速率）
- 两者差异反映系统压力情况：实际QPS < 压测QPS 说明请求排队或服务端限流

## 压测QPS提取三级回退

| 优先级 | 来源 | 匹配方式 | 典型匹配率 |
|--------|------|---------|-----------|
| ① | 日志文件名 | `_r{rate}_` 正则提取 + 时间窗口模糊匹配 | ~33% |
| ② | bash history | 按(concurrency, num_requests, output_length)组合匹配 | ~40% |
| ③ | 空值 | 无法匹配的历史数据 | ~27% |

### 日志匹配细节
- CSV Execution_Time = 压测**完成时间**
- 日志文件名时间 = 压测**启动时间**
- 匹配窗口：0~120分钟（日志时间必须早于CSV时间）
- 并发数匹配给-100分bonus，不匹配给+200分penalty
- 选score最低（最匹配）的候选

### History匹配细节
- 从 `~/.bash_history` 解析含 `acs-bench prof` + `--request-rate` 的命令
- 提取 concurrency、request-rate、num-requests、output-length
- 按(concurrency, num_requests, output_length)三元组匹配
- 多个匹配取出现次数最多的request-rate（Counter.most_common）

## 场景推断规则

| 条件 | 场景 |
|------|------|
| 日志文件名含 `in{n}_n{m}` 模式 | 定长 |
| CSV Input_Length列非空 | 定长 |
| 其他 | 混长 |

> 注意：acs-bench CSV中Input_Length列在定长场景下也常为空，因此日志文件名推断是主要手段

## DeepSeek V3 vs LongCat Tokenizer差异

| 项目 | LongCat-Flash-Chat | DeepSeek-V3 |
|------|-------------------|-------------|
|| 路径 | $WORK_ROOT/LongCat-Flash-Chat | $WORK_ROOT/DeepSeek-V3 |
| vocab_size | ~128K | 129,280 (config) / 128,000 (tokenizer) |
| model_max_length | - | 131,072 |
| tokenizer_class | - | LlamaTokenizerFast |
| 需trust_remote_code | True | True |

> ⚠️ 不同tokenizer生成的定长数据集**不可混用**，tokenization结果不同
> DS-V3数据集目录建议加 `_dsv3` 后缀区分，如 `in10240_n10000_dsv3/`
