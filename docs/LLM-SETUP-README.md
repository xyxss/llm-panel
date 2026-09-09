# Local VLM server — how to run & continue work

Qwen3.8-27B vision-LM (GGUF UD-IQ3_XXS) served by llama.cpp, OpenAI-compatible.

## Connection (for deepseek-harness / any harness)
- Base URL (same box): `http://localhost:8000/v1`
- Base URL (LAN):       `http://<box>:8000/v1`   (plain http, NOT https)
- API key: in `~/.vlm_api_key`  (currently `sk-YOUR-PANEL-KEY`)
- Model id: `qwen3-vl`
- Full harness configs: see ~/harness-config.md

## One-time: make it permanent (auto-start on every boot)
```bash
pkill -f 'build/bin/llama-server'; sleep 2          # stop the manual instance
sudo cp ~/llama-vlm.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llama-vlm.service        # start now + on every boot
sudo ufw allow from <your-subnet> to any port 8000 proto tcp   # LAN access
sudo ufw reload
```

## Daily use (once the service is installed)
| Action | Command |
|---|---|
| Is it running? | `systemctl status llama-vlm` |
| Live logs | `journalctl -u llama-vlm -f` |
| Restart (after editing serve-vlm.sh) | `sudo systemctl restart llama-vlm` |
| Stop | `sudo systemctl stop llama-vlm` |
| Start | `sudo systemctl start llama-vlm` |
| Don't auto-start anymore | `sudo systemctl disable llama-vlm` |
| Quick health check | `curl -s localhost:8000/v1/models -H "Authorization: Bearer $(cat ~/.vlm_api_key)"` |

## Run manually instead (no service)
```bash
bash ~/serve-vlm.sh                      # base default = q8_0 @ 96k
CTX=131072 KV_QUANT=q4_0 bash ~/serve-vlm.sh   # override on the fly
```

## All runtime options (in ~/serve-vlm.sh, override inline as env vars)

### Thinking / reasoning
| Var | Values | Meaning |
|---|---|---|
| `THINKING` | `on` / `off` / `auto` | `off` = no reasoning tokens → **much faster agent loops**. `on` = max quality. `auto` = template decides. |
| `REASON_EFFORT` | `default`/`low`/`medium`/`high` | effort passed to the template when thinking |
| `REASON_FORMAT` | `deepseek` / `deepseek-legacy` | `deepseek` = thoughts in separate `reasoning_content` (best for harnesses) |

### Context handling / compaction
| Var | Values | Meaning |
|---|---|---|
| `CONTEXT_SHIFT` | `off` (default) / `on` | **off = `--no-context-shift`**: server errors when full instead of silently dropping oldest tokens. **Keep off for agents** — `on` corrupts agent state (drops system prompt / early tool results). |
| `CACHE_REUSE` | `256` (default), 0=off | reuse KV chunks via shifting — big speedup for agent loops that compact/edit history |

**Compaction is the HARNESS's job** (llama.cpp does not compact). With
`CONTEXT_SHIFT=off` the server errors at the limit, so configure your harness:
- strategy: **summarize**, not truncate
- trigger at **~70%** of the slot context (`CTX/PARALLEL`)
- reserve **~30k tokens** for reasoning output + next tool result

### Sampling (auto-set from THINKING, per Unsloth's Qwen3-VL guide)
| Mode | temp | top_p | top_k | min_p | presence | repeat |
|---|---|---|---|---|---|---|
| thinking (`on`/`auto`) | 1.0 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| non-thinking (`off`) | 0.7 | 0.8 | 20 | 0.0 | 1.5 | 1.0 |

Override any: `TEMP`, `TOP_P`, `TOP_K`, `MIN_P`, `PRESENCE`, `REPEAT_PEN`.
(Community tip: temp `0.6` in thinking mode curbs over-long reasoning.)

### Vision
| Var | Meaning |
|---|---|
| `IMAGE_MIN_TOKENS` | default 1024 — Qwen-VL needs ≥1024 for accurate grounding |
| `IMAGE_MAX_TOKENS` | cap huge images (e.g. 2048) to save VRAM |

### Performance
`BATCH` (2048), `UBATCH` (512 — raise to 1024 for faster prompt eval, more VRAM),
`NGL` (99), `THREADS` (-1 auto).

## Engines (only ONE can hold the GPU at a time)
| Engine | Status | Formats | Use when |
|---|---|---|---|
| **llama.cpp** | installed | GGUF incl. UD-IQ3_XXS + vision mmproj | current setup; only engine that runs your 27B VLM |
| **vLLM** | `bash ~/setup_vllm_optional.sh` | BF16/FP8/AWQ (**no** GGUF IQ-quants, no multimodal GGUF) | many concurrent users; needs a 7B-class GPU-native model |
| **Ollama** | `curl -fsSL https://ollama.com/install.sh \| sh` | GGUF (llama.cpp under the hood) | simplest management; fewer exposed flags |

## Switch "model" presets (context window + KV quant)
Same Qwen3.8-27B model, model id stays `qwen3-vl` — only ctx/KV change.
Parallel slots default to 1; pass a 2nd arg to change it.

> **The `start-model.sh` commands below are BROKEN — use the panel or `llm switch`.**
> That script carries its own hardcoded preset table (unrelated to `config/presets.json`)
> and never sets `MDIR`, so `serve-vlm.sh` falls back to `~/models/Qwen3.8-27B`, which
> does not exist — every one of these commands exits with "no model .gguf in …".
> `restart-server.sh` is also stale: it kills the *old* `~/llama.cpp` binary path, so it
> cannot stop a server the panel started from the sm120 build. Working equivalents:
> `llm switch <preset>` or the Presets view in the panel. Kept here only so the drift is
> visible until they are rewritten or deleted.
```bash
bash ~/start-model.sh              # list all presets
bash ~/start-model.sh q4-96k       # full 96k, safe with images  [recommended]
bash ~/start-model.sh q8-96k       # best KV quality, TIGHT (OOM risk on big images)
bash ~/start-model.sh q8-64k       # best KV quality, safe headroom
bash ~/start-model.sh q4-128k      # 128k context
bash ~/start-model.sh q4-160k      # 160k context, tight
bash ~/start-model.sh q4-128k 2    # ...with 2 parallel slots (each gets ctx/2)
bash ~/start-model.sh agent        # 128k, THINKING OFF — fast agent loops
bash ~/start-model.sh deep         # 96k, THINKING ON (high) — max reasoning
bash ~/start-model.sh vision       # 64k + image headroom for vision work
bash ~/start-model.sh q4-96k 1 off # any preset, force thinking off
```
Each call stops the running server and starts the chosen preset.
Usage: `bash ~/start-model.sh <preset> [parallel] [thinking]`

## Tuning (edit the vars at the top of ~/serve-vlm.sh)
- CTX       context window     (q8_0 max ~96k, q4_0 max ~192k on this 16GB GPU)
- KV_QUANT  q8_0 (quality) / q4_0 (more context)
- PARALLEL  concurrent slots (each gets CTX/PARALLEL tokens)
- NGL       GPU layers (99 = all)
After editing: `sudo systemctl restart llama-vlm` (or re-run serve-vlm.sh).

## Files
- ~/serve-vlm.sh        launcher (all tuning knobs)
- ~/llama-vlm.service   systemd unit (boot autostart)
- ~/.vlm_api_key        the API key (chmod 600)
- ~/harness-config.md   ready configs for OpenCode / Aider / Continue / generic
- ~/models/Qwen3.8-27B/ the model + vision projector
- ~/llama.cpp/          the built engine
- ~/setup_vllm_optional.sh  add vLLM later (run ONE GPU server at a time)

## Switch engines
`~/start-llm.sh` lets you pick llama.cpp or vLLM interactively.
Only ONE GPU server at a time (16 GB VRAM).
