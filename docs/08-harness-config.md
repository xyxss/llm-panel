# Local VLM endpoint — harness configuration

llama.cpp server (Qwen3.8-27B VLM, UD-IQ3_XXS) — OpenAI-compatible.

## Connection values

- Base URL (local): `http://localhost:8000/v1`
- Base URL (LAN):   `http://<box>:8000/v1`
- API key: REQUIRED — stored in ~/.vlm_api_key, chmod 600; serve-vlm.sh reads it automatically.
  To rotate: write a new key into that file and restart the server.
- Model id: `qwen3-vl`
- Capabilities: chat/completions, vision (image_url), tool/function calling (--jinja)

## Recommended: point at the ROUTER, switch models from the web panel

Instead of the direct URL below, point harnesses at the **router** and pick the
model (local Qwen, or a cloud provider) live in the panel at http://<box>:8080 —
no harness reconfig when you switch.

- Base URL: `http://<box>:8001/v1` (localhost on the box: `http://localhost:8001/v1`)
- API key: same as ~/.vlm_api_key (the router requires it, then injects the real
  upstream key server-side).
- Model id: send anything — the router rewrites it to the active endpoint's model.

The direct :8000/v1 below still works and always hits the local model only.

## Universal env vars (works for most CLI harnesses)

```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="sk-YOUR-PANEL-KEY"
export OPENAI_MODEL="qwen3-vl"
# some tools use these names instead:
export OPENAI_API_BASE="http://localhost:8000/v1"
```

## OpenCode (~/.config/opencode/opencode.json)

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llamacpp": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local llama.cpp",
      "options": { "baseURL": "http://localhost:8000/v1" },
      "models": { "qwen3-vl": { "name": "Qwen3.8-27B VLM" } }
    }
  }
}
```

Then pick the model `llamacpp/qwen3-vl` (Tab / model switcher).

## Aider

```bash
aider --openai-api-base http://localhost:8000/v1 \
      --openai-api-key sk-YOUR-PANEL-KEY \
      --model openai/qwen3-vl
```

## Generic "custom OpenAI provider" template

For pi / deepseek-harness / continue.dev / any other tool, find its
"custom provider" or "OpenAI-compatible" section and map:

```
baseURL / api_base  =  http://localhost:8000/v1
apiKey  / api_key   =  sk-YOUR-PANEL-KEY
model               =  qwen3-vl
```

## Continue.dev (~/.continue/config.json → "models")

```json
{
  "title": "Qwen3-VL (local)",
  "provider": "openai",
  "model": "qwen3-vl",
  "apiBase": "http://localhost:8000/v1",
  "apiKey": "sk-YOUR-PANEL-KEY"
}
```

## Quick tests

Text:

```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-vl","messages":[{"role":"user","content":"hello"}]}'
```

Vision (image URL or base64 data URI):

```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"qwen3-vl",
  "messages":[{"role":"user","content":[
    {"type":"text","text":"What is in this image?"},
    {"type":"image_url","image_url":{"url":"https://raw.githubusercontent.com/ggml-org/llama.cpp/master/examples/llava/dog.jpg"}}
  ]}]}'
```

## Notes

- This is a REASONING model: responses include `reasoning_content`. Some
  harnesses show the thinking separately; give it enough max_tokens.
- Tool calling needs `--jinja` (already enabled).
- Only ONE GPU server at a time on 16 GB (llama.cpp OR vLLM).
