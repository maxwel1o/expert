# DeepSeek 官方 API Provider 配置

## 概述

DeepSeek官方API与ModelArts MaaS是两种不同的推理服务提供方，provider配置有显著差异。

## 配置对比

| 项目 | DeepSeek官方API | ModelArts MaaS |
|------|----------------|----------------|
| base_url | `https://api.deepseek.com/v1` | `https://api.modelarts-maas.com/v1` |
| api_key格式 | `sk-...` | 平台分配的key |
| model_name | 模型公开名（如`deepseek-v4-flash`） | 资源UUID（如`f17067bd-...`） |
| 认证方式 | Bearer Token（标准OpenAI兼容） | Bearer Token |

## Provider YAML模板

```yaml
# DeepSeek 官方 API
providers:
  - id: 'mt_test'
    name: 'deepseek-v4-flash'
    api_key: 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    base_url: 'https://api.deepseek.com/v1'
    model_name: 'deepseek-v4-flash'
```

## 已知模型列表

| model_name | 说明 |
|------------|------|
| `deepseek-v4-flash` | DeepSeek V4 Flash（快速推理） |
| `deepseek-chat` | DeepSeek V3 对话模型 |
| `deepseek-reasoner` | DeepSeek R1 推理模型 |

## Tokenizer要求

- DeepSeek官方API压测仍需本地tokenizer计算输入token数
- 不同模型版本需下载对应tokenizer（V4 Flash不能用V3-0324的tokenizer）
- 下载方式：`HF_ENDPOINT=https://hf-mirror.com huggingface-cli download deepseek-ai/<REPO> ...`

## 注意事项

- DeepSeek官方API可能有独立的rate limit策略（与MaaS不同），冒烟测试时关注429错误
- 官方API的模型名直接使用公开名称，无需UUID
- 压测命令中provider参数指向对应的YAML文件：`--provider ./conf/provider_deepseek_v4_flash.yaml`
