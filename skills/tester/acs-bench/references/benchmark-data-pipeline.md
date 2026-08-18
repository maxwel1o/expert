# 数据管道：CSV → JSON 数据集

## 数据流

```
mt_dataset/mt_dataset.csv
  ↓ trans_to_json.py (load_data, input_format=txt)
prof-test/dataset/mt_dataset/data_n3838_avg11944.json  (3838条, avg 11944 tokens)
  ↓ cut_keep_max_n (target_avg=10240)
  ↓ shuffle (默认开启)
prof-test/dataset/mt_dataset/data_shuffled_cut_keep_max_n_n2699_avg10240.json
```

## trans_to_json.py 关键逻辑

### 脚本流程
1. `load_data` — 从 CSV/JSON 加载，提取 `messages` 字段
2. `compute_token_lengths` — 用 tokenizer 计算每条 token 长度
3. `cut_keep_max_n` / `cut_keep_error` — 裁剪到目标平均长度
4. `shuffle_data` — 随机打乱（**默认开启**，`--no-shuffle` 关闭）
5. `save_json` — 输出 `[{id, input}, ...]` 格式

### cut_keep_max_n 保序性
- 按 token 长度排序（短→长），保留前 best_k 个最短数据
- **按原始顺序重建**（遍历 0~n，保留在 keep_indices_set 中的）
- 裁剪步骤本身保持原始顺序 ✅
- 但后续 shuffle 步骤会打乱 ❌

### cut_keep_error 逻辑
- 按 token 长度降序排列，删除最长的数据直到平均长度落入误差范围
- 同样按原始顺序重建

## 现有数据集清单与顺序审计

基准：`data_n3838_avg11944.json`（3838条，= CSV 原始顺序）

| 文件 | 条数 | 与基准重叠 | 保持基准顺序 | 说明 |
|------|------|-----------|-------------|------|
| data_n3838_avg11944.json | 3838 | — | ✅ 基准 | = CSV 原始顺序 |
| data_n3838_avg11944_cut_n2820_avg10443.json | 2820 | 2820 | ❌ | 生成时 shuffle=True |
| data_n3838_avg11944_cut_n2699_avg10240.json | 2699 | 2699 | ❌ | 生成时 shuffle=True |
| data_n3838_avg11944_cut_keep_error_e2.0_shuffled_n2820_avg10443.json | 2820 | 2820 | ❌ | 文件名标明 shuffled |
| data_n3838_avg11944_cut_keep_max_n_shuffled_n2699_avg10240.json | 2699 | 2699 | ❌ | 文件名标明 shuffled |
| data_shuffled_n3838_avg11944.json | 3838 | 3838 | ❌ | 文件名标明 shuffled |
| data_shuffled_cut_keep_error_e2.0_n2820_avg10443.json | 2820 | 2820 | ❌ | 文件名标明 shuffled |
| data_shuffled_cut_keep_max_n_n2699_avg10240.json | 2699 | 2699 | ❌ | 压测实际使用 |

**结论**：所有裁剪子集都不保持原始顺序。如需保序，用 `--no-shuffle` 重新生成。

## 去重与清理
- 3838 条数据中有 3723 条唯一内容（115 条重复）
- keep_max_n 2699 条中有 2631 条唯一内容
- **已删除** `data_n2820_avg10443.json`（与 `data_n3838_avg11944_cut_n2820_avg10443.json` 内容完全相同，保留文件名信息更全的）
- **已删除** `data_n3839avg11941.json`（早期生成，非标准命名）

## 脚本版本差异
- `.bk` 版本：`items = sorted(zip(lengths, data_list, range(n)))` 排序
- 当前版本：`indexed_items = [(i, lengths[i], data_list[i])]; indexed_items.sort(key=lambda x: x[1])`
- 逻辑等价，均保持原始顺序重建

## 常用命令

```bash
# 生成 3838 全量数据集（不 shuffle）
python trans_to_json.py \
  -i $WORK_ROOT/prof/mt_dataset/mt_dataset.csv \
  -o ./dataset/mt_dataset/ \
  -n data \
  -f txt \
  -t $WORK_ROOT/LongCat-Flash-Chat \
  --no-shuffle

# keep_max_n 裁剪 + 不 shuffle（保序）
python trans_to_json.py \
  -i $WORK_ROOT/prof/mt_dataset/mt_dataset.csv \
  -o ./dataset/mt_dataset/ \
  -n data \
  -f txt \
  -t $WORK_ROOT/LongCat-Flash-Chat \
  -ta 10240 \
  -cm keep_max_n \
  --no-shuffle

# keep_max_n 裁剪 + shuffle（默认行为）
python trans_to_json.py \
  -i $WORK_ROOT/prof/mt_dataset/mt_dataset.csv \
  -o ./dataset/mt_dataset/ \
  -n data \
  -f txt \
  -t $WORK_ROOT/LongCat-Flash-Chat \
  -ta 10240 \
  -cm keep_max_n
```
