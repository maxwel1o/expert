# 数据准备参考 (trans_to_json.py)

## 数据源

- 原始CSV: `$WORK_ROOT/prof/mt_dataset/mt_dataset.csv`
- CSV格式: 单列 `prompt`，每行一个JSON对象含 `messages` 字段
- CSV共3838条，提取messages后生成基准数据集

## 基准数据集

- `data_n3838_avg11944.json`: 3838条，avg 11944 tokens，**= CSV原始顺序**
- 存在重复内容: 24组重复，共139条重复项（3838条中仅3723条唯一）
- ID为0~3837顺序编号

## trans_to_json.py 脚本流程

```
1. load_data → 加载数据（支持txt/jsonl/json格式）
2. compute_token_lengths → 用tokenizer计算每条token长度
3. cut (可选) → 裁剪到目标平均长度
4. shuffle (默认开启) → 随机打乱顺序
5. save → 保存JSON + 分布统计
```

## 裁剪模式

### keep_max_n
- 按token长度排序（短→长），保留前best_k个最短数据
- **按原始顺序重建**（遍历0~n，保留keep_indices_set中的项）
- 逻辑上保持原始顺序 ✅，但shuffle步骤会打乱 ❌

### keep_error
- 按长度降序排列，删除最长的，使均值落入误差范围
- 同样按原始顺序重建

## 顺序验证方法

由于基准数据有重复项，直接逐条对比索引会因重复映射到同一索引产生断点。
正确做法：**去重后验证唯一项的基准索引是否单调递增**。

```python
# 去重后顺序验证
baseline_first_map = {}
for i, item in enumerate(baseline):
    if item["input"] not in baseline_first_map:
        baseline_first_map[item["input"]] = i

unique_indices = []
seen = set()
for item in new_data:
    inp = item["input"]
    if inp in baseline_first_map and inp not in seen:
        unique_indices.append(baseline_first_map[inp])
        seen.add(inp)

is_ordered = all(unique_indices[i] < unique_indices[i+1] 
                  for i in range(len(unique_indices)-1))
```

## 关键参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-cm` | 裁剪模式: keep_max_n / keep_error | keep_max_n |
| `-ta` | 目标平均token长度 | None |
| `--no_shuffle` | 不打乱顺序（保序） | 默认shuffle=True |
| `-f json --json_field input` | 从JSON提取input字段 | -f txt |
| `-s 1000` | 分布统计步长 | None |

## 数据集JSON Schema（acs-bench消费格式）

**⚠️ 这是acs-bench唯一接受的格式，不可用JSONL或其他格式替代！**

```json
[
  {"id": 0, "input": "[{\"role\": \"user\", \"content\": \"...\"}]"},
  {"id": 1, "input": "[{\"role\": \"user\", \"content\": \"...\"}]"},
  ...
]
```

- **顶层**: JSON数组，非JSONL（每行一个JSON对象）
- **每条**: `{"id": int, "input": str}`
- **`id`**: 整数，从0开始顺序编号
- **`input`**: **字符串类型**，内容是messages数组的JSON序列化（`json.dumps(messages)`）
- **messages数组**: `[{"role": "user", "content": "..."}]`，标准OpenAI ChatCompletion格式

### 常见格式错误

| 错误格式 | 正确格式 | 说明 |
|----------|----------|------|
| JSONL: 每行`{"messages": [...]}` | JSON数组: `[{"id":0,"input":"..."}]` | acs-bench不消费JSONL |
| `input`为list/dict | `input`为str | input必须是JSON字符串，不是对象 |
| `id`为str | `id`为int | id必须是整数 |

### 从CSV提取messages字段构造数据集

当原始数据为CSV（单列`prompt`，每行是JSON含`messages`字段）时：

```python
import csv, json

result = []
with open(input_csv, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for i, row in enumerate(reader):
        data = json.loads(row[0])
        messages = data.get('messages', [])
        result.append({
            "id": i,
            "input": json.dumps(messages, ensure_ascii=False)
        })

with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False)
```

## 文件名约定

`{name}{_shuffled}{_cut_{mode}}_n{count}_avg{avg}.json`

- 有`shuffled` = 经过随机打乱
- 有`cut_keep_max_n` = keep_max_n裁剪
- 有`cut_keep_error_e2.0` = keep_error裁剪，误差2.0%
- `avg`为字符长度均值（非token长度），如`data_n17829_avg3894.json`
