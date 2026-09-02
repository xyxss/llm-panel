# llm-stack

A self-hosted **local + cloud LLM operations stack** for a single-GPU box
(RTX 5080, 16 GB). One web panel and one stable OpenAI-compatible URL; flip
between a local model (llama.cpp / vLLM / Ollama) and cloud providers
(OpenAI, OpenRouter, DeepSeek, Groq, Together, …) without reconfiguring any tool.

This repo is a working demo of that pattern — everything (serving scripts, the
control panel + router, setup scripts, systemd units, config, docs) lives here so
it can be read, changed, and re-deployed as one unit.

```
        your tools / harnesses
                 │  point once at the router:
                 ▼   http://<box>:8001/v1
        ┌──────────────────┐
        │  llm-stack panel │  ← pick the backend in the web UI (:8080)
        │   + router       │
        └───────┬──────────┘
      ┌─────────┴───────────┐
      ▼                     ▼
  LOCAL engine         CLOUD provider
  llama.cpp / vLLM     OpenAI / OpenRouter / DeepSeek / Groq / Together
  / Ollama (GGUF)      (any OpenAI-compatible base URL + key)
```

The router is a minimal, dependency-free version of what
[LiteLLM](https://docs.litellm.ai/docs/) does at scale — a single OpenAI-compatible
endpoint that forwards to local or cloud backends and applies your generation
defaults. Swap in LiteLLM later if you want spend tracking / load balancing.

## Layout
```
llm-stack/
├── bin/llm            # one CLI for everything (status, switch, panel, …)
├── Makefile           # `make help` — same tasks as targets
├── panel/             # panel.py (web UI + router) and index.html
├── serve/             # serve-vlm.sh (llama.cpp), serve-vllm.sh (vLLM), start/restart helpers
├── setup/             # one-time install scripts (GPU, llama.cpp, vLLM)
├── systemd/           # boot-autostart units (panel + model)
├── scripts/           # kvsweep.sh / kvtest.sh — VRAM & context benchmarks
├── config/
│   ├── presets.json          # THE config: presets, models, engines, endpoints, runtime, settings
│   └── secrets.example.json  # copy to secrets.json (git-ignored) and add cloud keys
└── docs/              # models.md, harness-config.md, panel.md, setup.md
```
Large / secret / machine-specific things stay **outside** the repo (git-ignored):
`~/models` (GGUF weights), `~/llama.cpp` (built engine), `~/.vlm_api_key`,
`config/secrets.json`, `logs/`.

## Quickstart
```bash
cd ~/llm-stack
cp config/secrets.example.json config/secrets.json   # then add any cloud keys
./bin/llm panel start                                 # web UI :8080, router :8001
./bin/llm status                                      # check GPU + model
```
Open `http://<box>:8080`, enter the API key from `~/.vlm_api_key` (top-right), and
switch models / presets / engines from there. Point your harnesses at the router:
`http://<box>:8001/v1` (any model id; the router rewrites it to the active model).

## Everyday commands
| Command | Does |
|---|---|
| `make status` / `./bin/llm status` | GPU, running model, engine, health |
| `make switch PRESET=q8-96k` | restart local model into a preset |
| `make engine ENGINE=vllm` | switch inference engine |
| `make models` | list models found under `~/models` |
| `make presets` | list configured presets |
| `make restart` / `stop` / `logs` | panel lifecycle + model log |
| `make install-service` | boot-autostart via systemd (sudo) |
| `make firewall` | print the ufw commands for LAN access |
| `make kvsweep` | benchmark max context per KV quant |

## Choosing / adding a model
See **[docs/models.md](docs/models.md)** — short version: drop a new GGUF folder into
`~/models/<name>/`, then in the panel open a preset's ✏️ editor and pick it from the
**model** dropdown (a newly-downloaded model auto-registers on first use). Tune with
`llama-bench` / `make kvsweep` before committing a preset (see docs).

## Setup from scratch
See **[docs/setup.md](docs/setup.md)** and the `setup/` scripts. Only ONE GPU engine
runs at a time on 16 GB.
