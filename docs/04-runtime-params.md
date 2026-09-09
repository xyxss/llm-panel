# Every runtime parameter — what each one does

`serve-vlm.sh` exposes every llama.cpp knob as an environment variable. A preset
in `presets.json` can override any of them; absent values fall back to the
`runtime{}` block, then to the script's own default.

## Core serving

| Variable | Flag | Meaning / measured notes |
|---|---|---|
| `CTX` | `-c` | Total KV tokens. Bigger = more VRAM (see 05-vram-estimator.md). |
| `PARALLEL` | `--parallel` | Server slots. Empty = auto, llama.cpp forces 4 slots + kv_unified. A number reserves CTX/PARALLEL per slot. Flash-Next must stay at 1. |
| `NGL` | `-ngl` | Layers on GPU. 99 = all; auto = omit so --fit decides. Dense models: always 99. MoE-stream: always auto. |
| `MDRAFT` | `-md` | External draft model. Empty for Qwen3.8-27B (head is embedded); Flash-Next would need a separate non-shared mtp-*.gguf. |
| `KV_QUANT` | `--cache-type-k/-v` | KV cache precision: f16 (best, 2x mem) / q8_0 (recommended) / q4_0 (quarter mem, max ctx). |
| `PORT` | — | Server port (8000). |
| `API_KEY` | — | Read from ~/.vlm_api_key. |

## Thinking / reasoning

| Variable | Flag | Meaning |
|---|---|---|
| `THINKING` | `--reasoning` | on = always reason (best quality, burns output tokens); off = never (fast, best for agent tool loops); auto = chat template decides. |
| `REASON_EFFORT` | `--reasoning-effort` | default / low / medium / high (template-dependent). |
| `REASON_FORMAT` | `--reasoning-format` | deepseek = separate reasoning_content field (best for harnesses); deepseek-legacy = think tags inline. |

## Sampling (Unsloth Qwen3-VL)

Auto-selected from THINKING; override any with the env var.

| Variable | Flag | non-thinking | thinking |
|---|---|---|---|
| `TEMP` | `-temp` | 0.7 | 1.0 |
| `TOP_P` | `-top-p` | 0.8 | 0.95 |
| `TOP_K` | `-top-k` | 20 | 20 |
| `MIN_P` | `-min-p` | 0.0 | 0.0 |
| `PRESENCE` | `-ppl` | 1.5 | 0.0 |
| `REPEAT_PEN` | `-rep` | 1.0 (disabled) | 1.0 (disabled) |

The router's `gen_defaults` are all null by default — the server's profile wins.
Set a value in gen_defaults only to force it for every client.

## Context handling / compaction

| Variable | Flag | Meaning |
|---|---|---|
| `CONTEXT_SHIFT` | `--context-shift` | **off (recommended for agents)**: server errors at the limit, harness compacts. on: silently drops oldest tokens — corrupts agent state. |
| `CACHE_REUSE` | `--cache-reuse` | Min chunk size (tokens) to reuse from KV cache via shifting. 256 = good default; big speedup for agent loops that compact/edit history. 0 disables. |

### Compaction guidance (for the harness, not llama.cpp)

With context_shift off, the server **errors** at the limit — the harness must
summarize/compact before then. Recommended:

- Trigger compaction at ~70% of the slot's context (ctx/parallel)
- Reserve ~30k tokens for reasoning output + next tool result
- Use **summarization, not hard truncation** — truncation drops the system prompt and early tool results
- If parallel > 1, each slot only gets ctx/parallel, so scale the trigger down

## Vision

| Variable | Flag | Meaning |
|---|---|---|
| `IMAGE_MIN_TOKENS` | `--image-min-tokens` | Qwen-VL needs at least 1024 image tokens for accurate grounding. Higher = more detail, more VRAM/time. |
| `IMAGE_MAX_TOKENS` | `--image-max-tokens` | Cap huge images (e.g. 2048). |

## Speculative decoding (MTP)

| Variable | Flag | Meaning / measured notes |
|---|---|---|
| `SPEC_TYPE` | `--spec-type` | draft-mtp = use the embedded NextN head (no separate draft model); none = plain decoding. Requires sm120 build. Measured: off → 53 tok/s; n=2 → 76 prose / 88 code; n=3 → 70 prose / 94 code. |
| `SPEC_N_MAX` | `--spec-draft-n-max` | Draft depth. **2 = best all-round** (77 prose / 89 code); 3 favours code (70 prose / 94 code); 4 was worse on both. |
| `SPEC_N_MIN` | `--spec-draft-n-min` | Floor on tokens drafted per step. |
| `SPEC_DRAFT_KV` | `-ctkd/-ctvd` | KV type for the DRAFT context. Quantizing it shrinks the per-ctx half of MTP's cost without touching target quality. |

## Performance

| Variable | Flag | Meaning |
|---|---|---|
| `BATCH` | `-b` | Logical batch (prompt processing throughput). 2048 default. |
| `UBATCH` | `-ub` | Physical batch. 512 default; raising to 1024 can speed prompt eval (more VRAM). |
| `THREADS` | `-t` | CPU threads, -1 = auto. |
| `THREADS_BATCH` | `-tb` | Threads for prompt processing. Matters once experts live on CPU (prefill becomes CPU-bound). |

## Fit / placement

| Variable | Flag | Meaning |
|---|---|---|
| `FIT` | `--fit` | Let llama.cpp auto-shrink UNSET args to fit VRAM. Empty = don't pass it. Only meaningful with NGL=auto. |
| `FIT_TARGET` | `--fit-target` | MiB of VRAM margin --fit leaves free per device (default 1024). Raise when the fit lands too close to the edge and long prompts OOM at peak. |
| `FIT_CTX` | `--fit-ctx` | Floor on the context --fit may choose. |

## MoE / streaming (Flash-Next only)

| Variable | Flag | Meaning |
|---|---|---|
| `NCMOE` | `--n-cpu-moe N` | Keep expert weights of first N blocks on CPU. 40 = floor that loads; 36 = OOM. |
| `CPU_MOE` | `--cpu-moe` | Keep ALL MoE expert weights on CPU (all-or-nothing form of NCMOE). Set one or the other, never both. |
| `LAZY_MODE` | `--lazy-mode` | on/auto/off: read rows of oversized tensors from disk on demand. llama.cpp default auto = on for tensors >4 GiB (covers Flash-Next's 26.8 GB engram). |
| `LOAD_MODE` | `--load-mode` | auto / none / mmap / mlock / mmap+mlock / dio. RAM-vs-disk residency. Replaces deprecated --mmap/--no-mmap/--mlock. mmap = page in from file (streams off NVMe); mlock = pin in RAM; dio = bypass page cache. |
| `NUMA` | `--numa` | distribute/isolate/numactl. Only helps multi-socket. |
| `OVERRIDE_TENSOR` | `-ot` | Hand-place tensors, e.g. .ffn_.*_exps.=CPU. Finer than NCMOE. |
| `NO_HOST` | `--no-host` | Bypass host buffer so extra buffer types can be used. Only if you know you want it. |

## KV pool / attention

| Variable | Flag | Meaning |
|---|---|---|
| `KV_UNIFIED` | `--kv-unified` / `--no-kv-unified` | on = all slots SHARE one KV pool (one request can use the whole context); off = each slot reserves ctx/parallel. |
| `KV_UNIFIED_PER_SLOT` | `--kv-unified-per-slot` | Per-slot cap inside a shared pool. |
| `FLASH_ATTN` | `-fa` | on/off/auto. FlashAttention fuses attention kernels — faster prompt eval + less VRAM. **Required for quantized KV** (q8_0/q4_0). |

## Misc

| Variable | Flag | Meaning |
|---|---|---|
| `EXTRA_FLAGS` | appended verbatim | Last resort for anything above. |
