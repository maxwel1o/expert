# 数据集顺序分析

## 结论

以 `data_n3838_avg11944.json` 为基准（= CSV 原始顺序），所有子集/裁剪数据集的顺序均不一致。

## 原因

`trans_to_json.py` 脚本流程：
1. `load_data` → 从 CSV 加载，保持 CSV 顺序
2. `compute_token_lengths` → 计算 token 长度
3. `cut_keep_max_n` / `cut_keep_error` → **裁剪保持原始顺序**（按原始索引 0~n 遍历重建）
4. `shuffle_data` → **默认 shuffle=True（L42），打乱顺序**
5. `save` → 保存

关键：裁剪步骤本身保序，但 shuffle 步骤破坏顺序。

## 各数据集与基准对比

| 文件名 | 条数 | 与3838重叠 | 保持顺序 |
|--------|------|-----------|---------|
| data_n3838_avg11944 (基准) | 3838 | — | ✅ |
| data_n3838_avg11944_cut_n2820_avg10443 | 2820 | 2820 | ❌ |
| data_n3838_avg11944_cut_n2699_avg10240 | 2699 | 2699 | ❌ |
| data_n3838_avg11944_cut_keep_error_e2.0_shuffled_n2820_avg10443 | 2820 | 2820 | ❌ |
| data_n3838_avg11944_cut_keep_max_n_shuffled_n2699_avg10240 | 2699 | 2699 | ❌ |
| data_shuffled_n3838_avg11944 | 3838 | 3838 | ❌ |
| data_shuffled_cut_keep_error_e2.0_n2820_avg10443 | 2820 | 2820 | ❌ |
| data_shuffled_cut_keep_max_n_n2699_avg10240 | 2699 | 2699 | ❌ |

## removed_indices_keep_max_n.json 确认

```json
{
  "shuffled": true,
  "shuffle_seed": null,
  "cut_mode": "keep_max_n",
  "original_count": 3838,
  "removed_count": 1139,
  "kept_count": 2699,
  "target_avg_length": 10240,
  "original_avg_length": 11944.91,
  "new_avg_length": 10240.18
}
```

## trans_to_json.py 版本差异

`.bk` 备份与当前版本的核心差异仅在 `cut_keep_max_n` 的排序写法：
- 旧版：`items = sorted(zip(lengths, data_list, range(n)))` — 按 lengths 排序
- 新版：`indexed_items = [(i, lengths[i], data_list[i]) for i in range(n)]` + `indexed_items.sort(key=lambda x: x[1])`

两者逻辑等价，均按原始顺序重建结果。

## 如何生成保序数据集

```bash
cd $WORK_ROOT/prof/prof-test

python trans_to_json.py \
  -i ../mt_dataset/mt_dataset.csv \
  -o ./dataset/mt_dataset \
  -n data \
  -t $WORK_ROOT/LongCat-Flash-Chat \
  -cm keep_max_n \
  -ta 10240 \
  --no-shuffle
```

## 已清理的重复文件

- `data_n2820_avg10443.json` — 与 `data_n3838_avg11944_cut_n2820_avg10443.json` 内容完全相同，已删除
- `data_n3839avg11941.json` — 早期生成，条数不一致，已删除
