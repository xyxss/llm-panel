# llm-stack

A self-hosted **local + cloud LLM operations stack** for a single-GPU box
(RTX 5080, 16 GB). One web panel and one stable OpenAI-compatible URL; flip
between a local model (llama.cpp / vLLM / Ollama) and cloud providers
(OpenAI, OpenRouter, DeepSeek, Groq, Together, …) without reconfiguring any tool.

Everything in this repo is **measured, not guessed**: every preset carries a
measured VRAM figure, every benchmark number was run on this exact box,
and the config that produced them lives in `config/presets.json`.

```
        your tools / harnesses
                 |  point once at the router:
                 v   http://<box>:8001/v1
        +------------------+
        |  llm-stack panel |  <- pick the backend in the web UI (:8080)
        |   + router       |
        +-------+----------+
      /         \
     /           \
LOCAL engine    CLOUD provider
llama.cpp /    OpenAI / OpenRouter / DeepSeek / Groq / Together
vLLM / Ollama  (any OpenAI-compatible base URL + key)
```

## Documentation

| File | What it covers |
|---|---|
| [docs/00-overview.md](docs/00-overview.md) | Summary, REPL, panel essentials, quick start |
| [docs/01-setup.md](docs/01-setup.md) | Step-by-step setup from scratch (GPU, llama.cpp, models, keys, autostart) |
| [docs/02-models.md](docs/02-models.md) | Model anatomy: GGUF blocks, MTP head, MoE streaming, adding models |
| [docs/03-panel.md](docs/03-panel.md) | Panel views, HTTP API, config, troubleshooting |
| [docs/04-runtime-params.md](docs/04-runtime-params.md) | Every runtime parameter with meaning and measured notes |
| [docs/05-vram-estimator.md](docs/05-vram-estimator.md) | VRAM fit formula, measured coefficients, caveats |
| [docs/06-presets.md](docs/06-presets.md) | Shipped presets, benchmark results, auto-switch map, engines |
| [docs/07-benchmarking.md](docs/07-benchmarking.md) | Benchmark methodology, traps, all measured numbers |
| [docs/08-harness-config.md](docs/08-harness-config.md) | Point tools at the router (env vars, OpenCode, Aider, etc.) |

## Quick start

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
# -> open http://localhost:8080, paste key, pick a preset, Apply
```

## Repo layout

```
llm-stack/
+-- bin/llm            # CLI / REPL - everything via HTTP API
+-- Makefile           # make help - same tasks as targets
+-- panel/             # panel.py (web UI + router) and index.html
+-- serve/
|   +-- serve-vlm.sh   # the llama.cpp launcher - every runtime knob lives here
|   +-- bench.sh       # llama-bench wrapper for setting sweeps
|   +-- serve-vllm.sh  # vLLM launcher
+-- setup/             # one-time install scripts (GPU, llama.cpp, vLLM)
+-- systemd/           # boot-autostart units (panel + model)
+-- scripts/           # kvsweep.sh / kvtest.sh - VRAM & context benchmarks
+-- config/
|   +-- presets.json          # THE config: presets, models, engines, endpoints, runtime, benchmarks
|   +-- secrets.example.json  # copy to secrets.json (git-ignored) and add cloud keys
+-- docs/              # This documentation set
```

Large / secret / machine-specific things stay **outside** the repo (git-ignored):
`~/models` (GGUF weights), `~/llama.cpp` + `~/llama.cpp-sm120` (built engines),
`~/.vlm_api_key`, `config/secrets.json`, `logs/`.

## The router is a minimal LiteLLM

The router is a dependency-free version of what [LiteLLM](https://docs.litellm.ai/docs/)
does at scale — a single OpenAI-compatible endpoint that forwards to local or cloud
backends and applies your generation defaults. Swap in LiteLLM later if you want
spend tracking / load balancing.
