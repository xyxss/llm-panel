# llm-stack — Overview

A self-hosted **local + cloud LLM operations stack** for a single-GPU box
(RTX 5080, 16 GB). One web panel and one stable OpenAI-compatible URL; flip
between a local model (llama.cpp / vLLM / Ollama) and cloud providers
(OpenAI, OpenRouter, DeepSeek, Groq, Together, …) without reconfiguring any tool.

Everything in this repo is **measured, not guessed**: every preset carries a
measured VRAM figure, every benchmark number was run on this exact box,
and the config that produced them lives in `config/presets.json`.

## What it does

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

- **One stable endpoint** (`:8001`) for all your tools. Switch backends from the panel — no reconfig.
- **Web panel** (`:8080`) — manage models, presets, engines, cloud endpoints, benchmarks, GPU stats, power.
- **CLI** (`bin/llm`) — every panel action also available from the terminal.
- **Router** — rewrites `model` field, injects upstream API keys server-side, applies generation defaults.

## The REPL (interactive CLI)

`./bin/llm` is a full interactive shell. When you run it with no arguments,
you get a prompt where you can type commands like:

```
> status              # GPU / running model / engine health
> switch rco2-vision  # restart the local model into that preset
> models              # list all discovered models
> presets             # list configured presets
> logs                # tail the model server log
> bench <model> ...   # run a benchmark sweep
> panel start         # start the web UI + router
> url               # print the router URL for harnesses
> health            # quick router health check
```

The REPL is useful when you want to drive the stack from a terminal without
opening the browser. Every command maps 1:1 to an HTTP API call on `:8080`.

## What's important for the panel (what people should use)

**The panel is the primary interface.** It gives you:

1. **One-click model/preset switching** — pick a preset card, click Apply, done.
2. **Live GPU monitoring** — VRAM, util, power, temp, tokens/sec, all polled every 2s.
3. **Benchmarking** — run sweeps and MTP comparisons directly from the UI.
4. **Cloud endpoint management** — add/test/activate cloud providers without touching config files.
5. **Model downloads** — paste a HuggingFace repo name, watch progress live.
6. **Engine switching** — llama.cpp / vLLM / Ollama with install/reinstall buttons.
7. **Power management** — GPU power limit slider, shutdown/reboot with grace period.

**The router is the thing your tools talk to.** Point everything at
`http://<box>:8001/v1` once. The panel decides what actually serves.

## Documentation map

| File | What it covers |
|---|---|
| `docs/00-overview.md` | This file — summary, REPL, panel essentials |
| `docs/01-setup.md` | Step-by-step setup from scratch (GPU, llama.cpp, models, keys, start) |
| `docs/02-models.md` | Model anatomy (GGUF blocks, MTP head, MoE streaming), model table |
| `docs/03-panel.md` | Panel views, HTTP API, config, troubleshooting |
| `docs/04-runtime-params.md` | Every runtime parameter with meaning and measured notes |
| `docs/05-vram-estimator.md` | The VRAM fit formula, measured coefficients, caveats |
| `docs/06-presets.md` | Shipped presets, benchmark results, auto-switch map, profiles |
| `docs/07-benchmarking.md` | Benchmark methodology, traps, all measured numbers |
| `docs/08-harness-config.md` | How to point tools at the router (env vars, OpenCode, Aider, etc.) |

## Quick start (5 minutes)

```bash
# 1. Clone
git clone <your-fork> ~/llm-stack && cd ~/llm-stack

# 2. GPU + llama.cpp (one-time)
bash setup/setup_phase1_gpu.sh
cmake -S ~/llama.cpp-sm120 -B ~/llama.cpp-sm120/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build ~/llama.cpp-sm120/build -j

# 3. Model (one-time)
hf download <repo> --local-dir ~/models/<name>

# 4. Key
python3 -c "import secrets;print('sk-'+secrets.token_hex(20))" > ~/.vlm_api_key && chmod 600 ~/.vlm_api_key

# 5. Go
./bin/llm panel start
# → open http://localhost:8080, paste key, pick a preset, Apply
```

## Repo layout

```
llm-stack/
├── bin/llm            # CLI / REPL — everything via HTTP API
├── Makefile           # make help — same tasks as targets
├── panel/             # panel.py (web UI + router) and index.html
├── serve/
│   ├── serve-vlm.sh   # the llama.cpp launcher — every runtime knob lives here
│   ├── bench.sh       # llama-bench wrapper for setting sweeps
│   └── serve-vllm.sh  # vLLM launcher
├── setup/             # one-time install scripts (GPU, llama.cpp, vLLM)
├── systemd/           # boot-autostart units (panel + model)
├── scripts/           # kvsweep.sh / kvtest.sh — VRAM & context benchmarks
├── config/
│   ├── presets.json          # THE config: presets, models, engines, endpoints, runtime, benchmarks
│   └── secrets.example.json  # copy to secrets.json (git-ignored) and add cloud keys
└── docs/              # This documentation set
```

Large / secret / machine-specific things stay **outside** the repo (git-ignored):
`~/models` (GGUF weights), `~/llama.cpp` + `~/llama.cpp-sm120` (built engines),
`~/.vlm_api_key`, `config/secrets.json`, `logs/`.
