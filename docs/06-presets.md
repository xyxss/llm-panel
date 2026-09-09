# Presets — the shipped configurations

Every preset in `presets.json` with its measured VRAM and benchmark result:

| Preset | Model | ctx | KV | ngl | MTP | slots | Measured VRAM (GB) | tg256 decode (tok/s) | pp prefill (tok/s) |
|---|---|---|---|---|---|---|---|---|---|
| `iq4xs-q8-60k` | UD-IQ4_XS | 62 464 | q8_0 | 99 | off | 1 | **15.42** (measured) | 48.07 | 209.8 |
| `iq4xs-q4-48k` | UD-IQ4_XS | 49 152 | q4_0 | 99 | off | 1 | **15.43** (measured) | 47.92 | 222.4 |
| `flashnext-speed` | Flash-Next IQ3_XXS | 102 400 | q8_0 | auto | off | auto | **14.9** (measured) | 13.03 | 17.9 |
| `flashnext-max` | Flash-Next IQ3_XXS | 262 144 | q8_0 | auto | off | 1 | **10.82** (measured) | 11.83 | 16.6 |
| `rco2-mtp-q8` | GSQ-RCO IQ2_S-mtp | 134 144 | q8_0 | 99 | draft-mtp n=2 | 1 | — (estimated) | — | — |
| `rco2-vision` | GSQ-RCO IQ2_S-mtp | 102 400 | q8_0 | 99 | draft-mtp n=2 | 1 | — (estimated) | — | — |
| `rco2-mtp` | GSQ-RCO IQ2_S-mtp | 215 040 | q4_0 | 99 | draft-mtp n=2 | 1 | — (estimated) | — | — |

## Additional variants benchmarked (not all shipped as presets)

| Variant | ctx | KV | MTP | tg256 (tok/s) | pp (tok/s) |
|---|---|---|---|---|---|
| `iq3UDxxs-q8-112k-mtp2` | 92 160 | q8_0 | n=2 | **86.71** | 146.3 |
| `iq3-q4-112k-mtp2` | 114 688 | q4_0 | n=2 | 84.97 | 142.5 |
| `iq3-q4-164k` | 167 936 | q4_0 | off | 54.12 | 167.2 |
| `rco-mtp-q8-110k` | 113 664 | q8_0 | n=2 | **72.13** | 128.9 |
| `rco-vision-q4-244k` | 249 856 | q4_0 | off | 60.04 | 134.5 |
| `rq3s-mtp-q2-80k` | 83 968 | q8_0 | n=2 | 48.31 | 187.4 |

## What each preset is for

| Preset | Role | Why these numbers |
|---|---|---|
| `iq4xs-q8-60k` | balanced text | IQ4_XS quality, q8_0 KV, 60k ctx — fits at 15.42 GB measured |
| `iq4xs-q4-48k` | tight / vision | q4_0 KV frees VRAM for the projector; 48k ctx |
| `flashnext-speed` | fast long-context MoE | 98k ctx, n_cpu_moe=40 (floor), mmap+lazy — 13 tok/s but 98k context |
| `flashnext-max` | extreme context MoE | 262k ctx, n_cpu_moe=48 — 11.8 tok/s but the longest window on this box |
| `rco2-mtp-q8` | speed (text) | IQ2_S + MTP n=2, q8_0 KV, 134k ctx — fastest dense decode |
| `rco2-vision` | speed (vision) | same as above but 102k ctx to leave room for the projector |
| `rco2-mtp` | speed (long text) | IQ2_S + MTP n=2, q4_0 KV, 215k ctx — longest dense window with MTP |

## Auto-switch map

`auto_switch.map` in `presets.json` maps client model names to presets. When a
client sends `model=<alias>`, the router loads that preset (its GGUF/quant, ctx,
KV) if it differs from what is serving; same preset = no reload. Unmapped names
are served by whatever is loaded.

```json
{
  "blance-text":   "iq4xs-q8-60k",
  "blance-vision": "iq4xs-q4-48k",
  "quality-blance":"flashnext-speed",
  "quality-maxtext":"flashnext-max",
  "speed-text":    "rco2-mtp-q8",
  "speed-vision":  "rco2-vision",
  "speed-longtext":"rco2-mtp"
}
```

## Profiles

| Profile | Preset | Purpose |
|---|---|---|
| `default` | rco2-vision | what runs on boot |
| `testing` | (none) | — |
| `extreme` | (none) | — |
| `tight` | iq4xs-q4-48k | minimum-VRAM config |

## Engines

Only ONE can hold the GPU at a time (16 GB).

| Engine | Port | Formats | Use when |
|---|---|---|---|
| **llama.cpp** | 8000 | GGUF incl. UD-IQ3_XXS + vision mmproj | current setup; only engine that runs your 27B VLM |
| **vLLM** | 8000 | BF16/FP16, FP8, AWQ/GPTQ 4-bit (NOT GGUF IQ-quants, no multimodal GGUF) | high-throughput concurrent serving; needs a GPU-native model |
| **Ollama** | 11434 | GGUF (llama.cpp engine underneath) | simplest setup, auto model management |

vLLM notes for this box: requires flashinfer-jit-cache 0.6.18+cu130,
apache-tvm-ffi==0.1.11 (0.1.12+ double-registers TVM FFI), no flashinfer-cubin,
VLLM_USE_FLASHINFER_SAMPLER=0, FLASHINFER_DISABLE_VERSION_CHECK=1, cu13 toolchain.
`serve-vllm.sh` sets these.
