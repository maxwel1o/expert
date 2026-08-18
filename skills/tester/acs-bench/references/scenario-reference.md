# 场景参考与QPS天花板

## 场景列表

### DSV3场景
- **`fixedlen-10k-400-deepseek-v3`** | 定长 | 数据集=in10240_n10000_dsv30324 | n=10000 | output=400
- **`varlen-10k-600-deepseek-v3`** | 混长 | 数据集=data_shuffled_cut_keep_max_n_n2699_avg10240 | n=2699 | output=600
- **`varlen-3838-600-deepseek-v3`** | 混长(保序) | 数据集=data_n3838_avg11944 | n=3838 | output=600

### LongCat场景
- **`varlen-10k-600-longcat`** | 混长(保序) | 数据集=data_n3838_avg11944_cut_keep_max_n_n2699_avg10240 | n=2699 | output=600
- **`varlen-3838-600-longcat`** | 混长(保序) | 数据集=data_n3838_avg11944 | n=3838 | output=600 | 缓存命中场景
- **`varlen-17829-600-longcat`** | 混长(客服质检) | 数据集=data_n17829_avg3894 | n=17829 | avg_input=2621tok | output=600 | 来源:prompt_long_utf8_dihuawei CSV
- **`varlen-17829-600-longcat`** | 混长(保序) | 数据集=data_n17829_avg3894 | n=17829 | avg 2580 input token | output=600 | 客服质检长文本场景

## QPS天花板参考

**变长场景（output=600）：**
- **10240 tokens** | QPS天花板≈18.5 | ≈RPM≈1110 | 最优c=350/r=20, E2E=5.72s
- **LongCat varlen-3838-600 (缓存命中)** | QPS天花板≈31.5 | e2e≤6s甜点c=165/QPS=25.97

**定长场景（input=10240, output=400）：**
- **c=400, r=26** | QPS≈25.4 | AVG_E2E≈8.9s | e2e≤10s最优

## 数据提取与格式转换（CSV→acs-bench格式）

**场景**: 原始数据为CSV文件，`prompt`列内嵌JSON（含`model`/`messages`/`stream`等字段），需提取`messages`转为acs-bench标准格式。

**输入CSV结构**:
```
prompt
"{"model": "longcat-flash-chat", "messages": [{"role": "user", "content": "..."}], "stream": false}"
```

**输出格式**: `[{id: int, input: str}]`，其中`input`是messages数组的JSON字符串（与3838数据集格式一致）。

**转换脚本**:
```python
import csv, json

result = []
with open('input.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for i, row in enumerate(reader):
        data = json.loads(row[0])        # 解析prompt列的JSON
        messages = data.get('messages', [])
        result.append({"id": i, "input": json.dumps(messages, ensure_ascii=False)})

with open('output.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False)
```

**注意事项**:
- CSV中JSON的双引号被转义，csv.reader自动处理
- 提取后仅保留`messages`，`model`/`stream`等字段由provider配置指定，不写入数据集
- 命名规范: `data_n{条数}_avg{平均input字符数}.json`
- ⚠️ `input`字段必须是`str`类型（messages数组的JSON序列化），非`list`；acs-bench内部会`json.loads(input)`解析
- acs-bench只消费JSON数组`[{id,input}]`格式

## Tokenizer下载指南

不同模型版本需对应tokenizer，不可混用：

```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
  deepseek-ai/<REPO_NAME> \
  tokenizer.json tokenizer_config.json special_tokens_map.json \
  --local-dir $PROF_ROOT/models/<MODEL_NAME> \
  --local-dir-use-symlinks False
```

⚠️ 必须用HF镜像；不同模型版本tokenizer不可混用；只需下载tokenizer文件，无需模型权重
