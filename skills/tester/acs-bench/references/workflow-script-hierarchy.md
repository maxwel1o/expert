# 压测结果解析脚本层级关系

## 三层架构

```
parse_result.py (底层)  →  summary_csv.py (汇总层)  →  parse_benchmark_results.py (业务层)
     377行                      458行                         243行
```

## 各层职责

| 层级 | 脚本 | 输入 | 输出 | 核心能力 |
|------|------|------|------|---------|
| 底层 | `parse_result.py` | 单个/目录CSV | TXT人读表 / JSON | 16关键字段提取、排序、JSON输出 |
| 汇总层 | `summary_csv.py` | 目录CSV + 日志 | CSV(19列) | 关联日志推断场景/压测QPS、结构化汇总 |
| 业务层 | `parse_benchmark_results.py` | 目录CSV + 阶段映射 | TXT人读表 | 阶段自动映射、异常过滤(Fail>5%)、前缀模式识别(shared/unique)、A vs B对比 |

## 使用场景映射

| 场景 | 用哪个脚本 | 命令 |
|------|-----------|------|
| 查看单个CSV结果 | parse_result.py | `python3 scripts/parse_result.py result/csv/xxx.csv` |
| 浏览目录所有结果 | parse_result.py | `python3 scripts/parse_result.py --dir result/csv/ --sort qps` |
| 生成结构化报告CSV | summary_csv.py | `python3 scripts/summary_csv.py --dir result/csv/ -o result/report/xxx.csv` |
| 摸高过程看阶段结果 | parse_benchmark_results.py | `python3 scripts/parse_benchmark_results.py --today --max-fail 0.05` |
| A vs B前缀对比 | parse_benchmark_results.py | `python3 scripts/parse_benchmark_results.py --today --compare` |

## 报告输出规范

- **CSV报告**：`summary_csv.py` 生成，落盘到 `result/report/`
- **TXT报告**：`parse_result.py --dir` 生成，落盘到 `result/report/`
- **禁止使用md格式**：已有规范是CSV+TXT，不可自作主张发明新格式

## 前缀模式列（待实现）

当前 `summary_csv.py` 输出19列不含"前缀模式"列。需要新增：
- 列名：`前缀模式`
- 值：`shared(前缀匹配)` / `unique(前缀不匹配)` / 空（历史数据无注入）
- 推断方式：通过数据集文件名含`_uid`→unique，含`_r{N}.json`→shared，否则→空

## 各skill挂钩情况

| 脚本 | workflow skill | benchmark skill | peak-finding skill | task.md |
|------|---------------|----------------|-------------------|---------|
| parse_result.py | ✅ 4b节 | ✅ 脚本表 | ❌ | ❌ |
| summary_csv.py | ✅ 4b节+汇总CSV脚本节 | ✅ 脚本表 | ❌ | ❌ |
| parse_benchmark_results.py | ✅ 4a节 | ✅ (刚补) | ❌ | ❌ |
