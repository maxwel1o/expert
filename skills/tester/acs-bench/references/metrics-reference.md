# ACS-Bench Result CSV Fields

## Key Metrics

| Column | Unit | Description |
|--------|------|-------------|
| `Total_Token_Throughput(tokens/s)` | tok/s | 总 token 吞吐量（输入+输出） |
| `Output_Token_Throughput(tokens/s)` | tok/s | 输出 token 吞吐量 |
| `AVG_TTFT(s)` | 秒 | 平均首 token 延迟 (Time To First Token) |
| `TP90_TTFT(s)` | 秒 | P90 首 token 延迟 |
| `TP99_TTFT(s)` | 秒 | P99 首 token 延迟 |
| `AVG_TPOT(s)` | 秒 | 平均每 token 输出时间 (Time Per Output Token) |
| `AVG_E2E(s)` | 秒 | 平均端到端延迟 |
| `TP90_E2E(s)` | 秒 | P90 端到端延迟 |
| `QPS` | req/s | 每秒查询数 |
| `Fail_Rate` | 0~1 | 失败率 |
| `Timeout_Fail_Rate` | 0~1 | 超时失败率 |
| `Total_Time(s)` | 秒 | 总耗时 |
| `AVG_COMPLETION_TOKENS` | tokens | 平均输出 token 数 |
| `AVG_PROMPT_TOKENS` | tokens | 平均输入 token 数 |

## Interpreting Results

- **TTFT** 反映 Prefill 速度，与上下文长度正相关
- **TPOT** 反映 Decode 速度，与上下文长度基本无关
- **E2E = TTFT + TPOT × output_tokens**
- **Fail_Rate > 0** 通常意味着触发了服务端限流
- **Server TTFT/TPOT = -1** 表示服务端不返回这些指标
