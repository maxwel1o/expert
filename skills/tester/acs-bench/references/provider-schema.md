# providers.yaml Schema

## Format

```yaml
providers:
  - id: '<unique-id>'           # 用于结果文件命名
    name: '<unique-id>'           # 同 id
    api_key: '<api-key>'          # API 密钥
    base_url: '<base-url>'        # OpenAI 兼容的 base URL
    model_name: '<model-name>'    # 模型名称
    model_category: '<category>'  # 模型分类（用于 tokenizer 匹配）
```

## 常见服务配置示例

### ModelArts MaaS
```yaml
providers:
  - id: 'maas-glm51'
    name: 'maas-glm51'
    api_key: '<your-key>'
    base_url: 'https://api.modelarts-maas.com/openai/v1'
    model_name: 'glm-5.1'
    model_category: 'glm-5.1'
```

### OpenAI
```yaml
providers:
  - id: 'openai-gpt4'
    name: 'openai-gpt4'
    api_key: '<your-key>'
    base_url: 'https://api.openai.com/v1'
    model_name: 'gpt-4'
    model_category: 'gpt-4'
```

### vLLM 本地
```yaml
providers:
  - id: 'vllm-local'
    name: 'vllm-local'
    api_key: 'EMPTY'
    base_url: 'http://localhost:8000/v1'
    model_name: '<model-name>'
    model_category: '<model-category>'
```
