# LLM Control Panel + Router

A tiny, dependency-free (Python stdlib only) web console to operate your local
llama.cpp model **and** cloud models from one page — and one stable endpoint your
tools point at forever.

```
┌─────────── your tools / harnesses ───────────┐
│  point them all at the ROUTER, one time:      │
│     http://<box>:8001/v1               │
└───────────────────────┬───────────────────────┘
                        │  (you flip the switch in the web UI)
        ┌───────────────┴───────────────┐
        ▼                               ▼
  LOCAL llama.cpp                  CLOUD provider
  Qwen3.8-27B on the RTX 5080      OpenAI / OpenRouter / DeepSeek / Groq / …
  (context + KV + parallel          (any OpenAI-compatible base URL + key)
   preset, restarts the server)
```

## URLs
| What | URL |
|---|---|
| **Web control panel** | http://<box>:8080  (or http://localhost:8080 on the box) |
| **Router** (point harnesses here) | http://<box>:8001/v1 |
| Legacy direct local model | http://<box>:8000/v1  (local model only, no switching) |

Auth for everything = the key in `~/.vlm_api_key`
(`sk-YOUR-PANEL-KEY`). Enter it once, top-right of the
panel (stored only in your browser). Harnesses send it as the usual
`Authorization: Bearer <key>` / OpenAI API key.

## What you can change in the UI
- **Models & endpoints** — one click to route every tool to the local GPU model
  or to a cloud provider. Add your own cloud/custom endpoint (base URL + model +
  key) from the panel; keys are stored server-side in `secrets.json` (chmod 600),
  never sent back to the browser.
- **Local presets** — context window, KV-cache quant (f16 / q8_0 / q4_0),
  parallel slots. Applying one **restarts the local llama.cpp server** with it.
  Live **~VRAM estimate** (a linear fit of your own measured presets) with a
  warning when a choice would exceed the 16 GB budget.
- **Default vs Testing** — mark any preset as your ⭐ stable default or 🧪 testing
  profile; one button applies it.
- **Custom preset** — slider for context + KV + parallel with a live VRAM meter.
- **Operating settings** — generation defaults (temperature, top_p, top_k, min_p,
  max_tokens, repeat_penalty, presence/frequency penalty, optional system prompt).
  The router applies them to whichever backend is active. By default they only
  fill in values a client omits; flip "override client" to force them.
- **Inference engine** — switch the local backend between **llama.cpp**, **vLLM**,
  and **Ollama** (only one holds the 16 GB GPU at a time). Not-installed engines
  show their install command. llama.cpp is the only one that runs your GGUF vision
  model; vLLM (`~/serve-vllm.sh`) is for GPU-native AWQ/FP8 models.
- **Thinking & context handling** (llama.cpp runtime) —
  - **Thinking**: `on` (always reason) / `off` (never — fast tool-calling agents) /
    `auto` (chat template decides), plus reasoning effort. Maps to
    `--reasoning on|off|auto` + `--reasoning-effort`. serve-vlm.sh auto-applies
    Unsloth's thinking-aware sampling (thinking: temp 1.0/top-p 0.95/presence 0;
    non-thinking: temp 0.7/top-p 0.8/presence 1.5).
  - **Context shift / compaction**: `off` = `--no-context-shift` (server errors
    when full so your harness can compact — recommended for agents); `on` =
    `--context-shift` (drops oldest tokens, corrupts agent state).
  - **cache-reuse** (KV prefix reuse), **image-min-tokens** (Qwen-VL grounding ≥1024),
    and an **extra-flags** passthrough. Save applies on next restart, or "Save &
    restart now".
- **Operating settings** — generation defaults (temperature, top_p, top_k, min_p,
  max_tokens, repeat_penalty, presence/frequency penalty, optional system prompt).
  The router applies them to whichever backend is active. By default they only
  fill in values a client omits; flip "override client" to force them.
- **Live status** — GPU VRAM / util / temp, current served config (engine,
  thinking, context-shift), tokens/sec, and a live tail of `~/server.log`.

The preset list also includes task-shaped presets: `agent` (128k, thinking off),
`deep` (thinking on, high effort), and `vision` (max image headroom). Settings
follow Unsloth's Qwen3-VL guide, validated against your llama-server build.

## Run it
```bash
# manual (foreground)
python3 ~/llm-panel/panel.py

# background
setsid nohup python3 ~/llm-panel/panel.py </dev/null >~/llm-panel/panel.log 2>&1 &
```

### Autostart on boot (systemd)
```bash
sudo cp ~/llm-panel/llm-panel.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llm-panel.service
systemctl status llm-panel        # check
journalctl -u llm-panel -f        # logs
```

### LAN access (open the two ports, like you did for :8000)
```bash
sudo ufw allow from <your-subnet> to any port 8080 proto tcp
sudo ufw allow from <your-subnet> to any port 8001 proto tcp
sudo ufw reload
```

## Add a cloud provider
Either use the **“＋ Add endpoint”** form in the UI, or edit files:
1. Put your key in `~/llm-panel/secrets.json` under the endpoint's `key_ref`
   (e.g. `"openrouter": "sk-or-..."`).
2. The endpoint is already listed in `presets.json` (OpenAI, OpenRouter, DeepSeek,
   Groq, Together are seeded). Add more by copying one block and changing
   `base_url` / `model` / `key_ref`.
3. Reload the panel page → the endpoint shows **key ✓** → click **Activate**.

Then any tool on the router gets that model. The client's `model` field is
ignored and rewritten to the active endpoint's model, so you never reconfigure
the harness — just flip it here.

## Important: don't double-manage the local model
The panel starts/stops the local model by launching `~/serve-vlm.sh` directly
(as user `liuyang`, no sudo). If you *also* enable the `llama-vlm.service`
systemd unit, the two will fight (systemd restarts what the panel stops). Pick
one: **either** use this panel to run the local model, **or** the systemd unit —
not both at once.

## Files
- `panel.py`     — the panel + router (edit only to change behavior)
- `presets.json` — all config: presets, endpoints, engines, runtime, profiles, settings
- `secrets.json` — cloud API keys (chmod 600, git-ignore it)
- `index.html`   — the web UI
- `llm-panel.service` — systemd unit
- `panel.log`    — panel's own stdout/stderr
- `~/serve-vlm.sh`  — llama.cpp launcher (all Unsloth-aligned knobs; env-driven)
- `~/serve-vllm.sh` — vLLM launcher (used when you switch engine to vLLM)

## API (for scripting, all POSTs need `Authorization: Bearer <key>`)
| Method | Path | Body | Does |
|---|---|---|---|
| GET  | `/api/status` | — | GPU, running config, engine, health, tps |
| GET  | `/presets.json` | — | full config + computed VRAM + router URL |
| GET  | `/api/log?n=40` | — | server.log tail |
| POST | `/api/switch` | `{"preset":"q4-96k"}` or `{"custom":{"ctx":98304,"kv":"q4_0","parallel":1}}` | restart local model |
| POST | `/api/engine` | `{"engine":"vllm"}` | switch inference engine (llamacpp/vllm/ollama) |
| POST | `/api/runtime` | `{"thinking":"off","context_shift":"off","apply":true}` | set thinking/context knobs (+optional restart) |
| POST | `/api/activate` | `{"endpoint":"openrouter"}` | route tools to that backend |
| POST | `/api/gen` | `{"temperature":0.6,...}` | set operating defaults |
| POST | `/api/profile` | `{"slot":"default","preset":"q4-96k"}` | set ⭐/🧪 |
| POST | `/api/endpoint` | `{"id":"x","base_url":"...","model":"...","api_key":"..."}` | add/update cloud endpoint |
| POST | `/api/stop` | — | stop the local model, free the GPU |
