# 常见陷阱速查

| 陷阱 | 正确做法 |
|------|----------|
| 压测结果仅输出文本不生成CSV文件 | 必须按workflow第5步用summary_csv.py生成CSV落盘到result/report/，禁止仅输出到对话 |
| summary_csv.py输出含全量历史数据 | 生成后必须按场景grep过滤，仅保留header+目标场景行 |
| 不同tokenizer数据集混用 | 目录加`_dsv30324`后缀区分 |
| run_benchmark.sh硬编码LongCat tokenizer | DSV3场景必须`-p`覆盖provider+修改TOKENIZER |
| acs-bench output-path双重嵌套 | 设计-o路径时考虑实际路径=output-path/filename/filename |
| trans_to_json.py -n命名重复拼接 | -n只写基础名，不含裁剪信息 |
| DeepSeek V4 Flash reasoning tokens | AVG_Completion_Tokens大部分是reasoning tokens，实际content远少 |
| ignore_eos=False不可用output_length×0.9估算e2e | 必须跑冒烟获取实际completion_tokens |
| 跨场景e2e目标直接套用 | 切换场景必须重新冒烟建立基线 |
| 摸高完成后不自动执行报告输出与回传 | 甜点确认后必须自动执行步骤6~7 |
| run_scenario.sh字符串拼接构建CMD | 已改用bash数组，禁止回退 |
| climb末尾it/s下降误判为服务饱和 | 实为请求数耗尽，固定并发递进才能准确判断拐点 |
| 纯吞吐摸高用climb模式 | 应用固定并发递进，climb混合多并发段无法隔离各段QPS |
| output_length远大于实际completion | 客服质检/评分类任务实际输出很短(avg 200~300tok)，output_length=600仅是上限；吞吐计算以实际completion为准，不可用output_length估算 |
| 报告CSV输出长度列用Output_Length | 已改为AVG_CONTENT_TOKENS(col 72)，反映实际输出而非上限 |
| 报告CSV小数位数不一致 | format_val统一.3f(3位小数)，整数列去.000后缀 |
| 报告CSV无BOM致Excel中文乱码 | 必须用utf-8-sig编码写入，禁止改为utf-8 |
| 90k+长上下文nr=5触发MaaS排队超时 | 减nr=3 + timeout=900s + warmup=0 + epochs=1；MaaS对长上下文请求排队，过多inflight触发服务端丢弃 |
| 长上下文并发爬坡用nr=5+ | 长上下文(90k+)必须用nr=3，nr≥5时Fail_Rate飙至70%+（MaaS排队），与timeout无关 |
