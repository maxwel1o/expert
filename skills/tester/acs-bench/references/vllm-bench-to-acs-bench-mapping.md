# vllm bench serve → acs-bench prof 参数映射与适配指南

## 适用场景

用户已有 `vllm bench serve` 脚本，但环境中无vllm CLI（或需统一到acs-bench体系），需将参数映射到 `acs-bench prof`。

## 参数映射表

| vllm bench serve | acs-bench prof | 说明 |
|------------------|----------------|------|
| `--dataset-name random` | `--dataset-type custom --input-length N` | acs-bench用custom类型+input-length生成随机数据 |
| `--random-input-len N` | `--input-length N` | 随机输入token长度 |
| `--random-output-len N` | `--output-length N` | 随机输出token长度 |
| `--request-rate R` | `--request-rate R` | 直接对应 |
| `--num-prompts N` | `--num-requests N` | 总请求数 |
| `--max-concurrency C` | `--concurrency C` | 最大并发 |
| `--ignore-eos` | `--ignore-eos True` | acs-bench需显式True |
| `--temperature T` | `--temperature T` | 直接对应 |
| `--endpoint /v1/chat/completions` | `--backend openai-chat --provider xxx.yaml` | acs-bench通过provider配置 |
| `--base-url http://IP:PORT` | provider.yaml中 `base_url` | |
| `--model NAME` | provider.yaml中 `model_name` | |
| `--tokenizer PATH` | `--tokenizer PATH` | 直接对应 |
| `--trust-remote-code` | `--trust-remote-code` | 直接对应 |
| `--metric-percentiles "50,90,99"` | 无直接对应 | acs-bench默认输出P50/P90/P99 |

## vLLM服务特有注意事项

1. **model_name通常是`auto`**：vLLM注册的model id可能是`auto`而非实际模型名。冒烟404时先 `GET /v1/models` 确认id字段值
2. **api_key可填`EMPTY`**：无鉴权的vLLM服务不需要真实api_key
3. **base_url用`http://`**：本地/内网vLLM通常非HTTPS
4. **max_model_len**：`GET /v1/models` 返回的max_model_len可用于校验input+output是否超限

## 典型适配流程

1. 确认vLLM服务端点：`curl http://IP:PORT/v1/models`
2. 创建provider YAML（参考 `templates/provider_vllm_remote.yaml`）
3. 下载对应tokenizer（`HF_ENDPOINT=https://hf-mirror.com huggingface-cli download ...`）
4. 按映射表转换参数，构建acs-bench prof命令
5. 冒烟验证：c=1, n=5, 确认返回正常

## 随机数据集 vs 真实数据集

- **随机数据集**（`--dataset-type custom --input-length`）：每轮随机生成prompt，无KV缓存效应，无需inject_round_identifier.py注入，适合并发扫描/吞吐探测
- **真实数据集**（`--input-path xxx.json`）：需每轮注入新round数据避免KV缓存命中，适合摸高寻优/甜点校验

## 并发扫描脚本模板

```bash
#!/bin/bash
set -euo pipefail
source $WORK_ROOT/$CONDA_ENV/bin/activate
ulimit -n 1048576

CONCURRENCY_ARRAY=(1 8 24 48 72 96 120 144 168)
RATE=5
TOKENIZER=$PROF_ROOT/models/<MODEL>
PROVIDER=$PROF_ROOT/conf/provider_vllm_remote.yaml
SAVE_PATH=$PROF_ROOT/result/csv/

for c in "${CONCURRENCY_ARRAY[@]}"; do
  n=$((c * 25))
  [[ $n -lt 100 ]] && n=100
  echo "=== Round: c=$c, r=$RATE, n=$n ==="
  
  acs-bench prof \
    --tokenizer $TOKENIZER --trust-remote-code \
    --dataset-type custom --input-length 4096 --output-length 1536 \
    --ignore-eos True --temperature 0.6 \
    --concurrency $c --request-rate $RATE --num-requests $n \
    --concurrency-backend threading-pool --backend openai-chat \
    --provider $PROVIDER \
    --benchmark-save-path $SAVE_PATH \
    --epochs 1 --warmup 0 -D
  
  echo "=== c=$c done, cooling 30s ==="
  sleep 30
done
echo "=== All rounds complete ==="
```
