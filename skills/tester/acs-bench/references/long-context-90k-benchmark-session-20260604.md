# Long-Context 90k Benchmark Session — GLM-5.1 on MaaS

**Date**: 2026-06-04
**Model**: glm-5.1 @ ModelArts MaaS (api.modelarts-maas.com)
**Objective**: Find peak throughput and optimal concurrency for 90k-token context

## Environment

- **Workdir**: `/tmp/acs-bench-peak/`
- **Provider**: `providers.yaml` with maas-glm51 (base_url: https://api.modelarts-maas.com/openai/v1)
- **Dataset**: `dataset_90k/90000.json` (4.3MB, generated with GLM-4-9b-chat tokenizer)
- **Tokenizer**: HF cache snapshot `THUDM/glm-4-9b-chat` (local, no network needed)

## Parameter Evolution

| Attempt | nr | cc | timeout | warmup | epochs | Fail_Rate | Notes |
|---------|----|----|---------|--------|--------|-----------|-------|
| 1 | 5 | 1 | 300s | 1 | 2 | 70% | TimeoutError on 7/10 requests |
| 2 | 5 | 1 | 600s | 1 | 2 | 70% | Same — not a timeout issue, MaaS queueing |
| 3 | 3 | 1 | 900s | 0 | 1 | 0% | All 3 requests succeed |

**Root cause**: MaaS server-side **rate limiting (HTTP 429)**. With 5 inflight 90k requests, the server returns 429 for excess requests. acs-bench catches the 429, retries 3× with backoff (`Rate limited. Waiting Xs (attempt N/3)...`), then gives up and **misreports it as TimeoutError**. Increasing timeout (300→600→900s) does NOT fix it — the issue is request volume, not latency. Only reducing nr eliminates the 429s.

## Concurrency Climb Results

| cc | AVG_TTFT(s) | TP90_TTFT(s) | AVG_TPOT(ms) | AVG_E2E(s) | Output_Throughput(tok/s) | Total_Throughput(tok/s) | QPS | Fail_Rate |
|----|-------------|-------------|-------------|------------|------------------------|----------------------|-----|-----------|
| 1 | 8.42 | 9.30 | 61 | 16.18 | 7.9 | 7.8 | 0.062 | 0% |
| 2 | 5.67 | 6.37 | 59 | 13.15 | 15.1 | 15.0 | 0.118 | 0% |
| 4 | 7.68 | 9.15 | 60 | 15.27 | 22.5 | 22.4 | 0.176 | 0% |
| 8 | 7.05 | 7.19 | 59 | 14.53 | 25.4 | 25.2 | 0.198 | 0% |
| 16 | 6.01 | 6.47 | 59 | 13.55 | 27.1 | 26.9 | 0.212 | 0% |

## Marginal Throughput Gain

| cc transition | Absolute gain (tok/s) | Relative gain |
|---------------|----------------------|---------------|
| 1→2 | +7.2 | +93% |
| 2→4 | +7.4 | +49% |
| 4→8 | +2.8 | +12.7% |
| 8→16 | +1.7 | +6.6% |

**Inflection point**: cc=8 (marginal gain drops below 10% after this)

## Key Observations

1. **TPOT is concurrency-independent**: ~60ms at all cc levels. Decode speed doesn't change with load.
2. **TTFT decreases with concurrency**: cc=1→8, TTFT drops from 8.4s to 7.0s. Likely MaaS server-side batch prefill.
3. **Throughput follows diminishing returns**: Classic Amdahl's law pattern — service has inherent parallelism limit.
4. **90k timeout sensitivity**: nr=5 causes 70% failures regardless of timeout (300s or 600s). Only nr reduction fixes it.

## Recommended Configuration

- **Optimal concurrency**: cc=8
- **Peak throughput**: ~25 tok/s total
- **Expected TTFT**: ~7s
- **Expected E2E**: ~14.5s
- **Timeout**: 900s
- **Max inflight requests**: 3 per batch to avoid MaaS queueing

## Comparison with Short-Context Results

| Context | cc=1 Throughput | cc=1 TTFT | cc=1 TPOT |
|---------|----------------|-----------|-----------|
| 128 | 24.0 tok/s | 3.0s | 60ms |
| 512 | 57.0 tok/s | 3.7s | 59ms |
| 1,024 | 96.7 tok/s | 4.2s | 60ms |
| 90,000 | 7.8 tok/s | 8.4s | 61ms |

Short-context throughput is 3-12x higher due to much faster prefill. TPOT is consistent (~60ms) across all context lengths.
