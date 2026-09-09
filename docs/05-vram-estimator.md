# VRAM estimator — the measured fit

`estimator{}` in `presets.json` is a linear fit, refit **2026-09-04** against the
sm120 build from clean measurements:

```
VRAM_gb(ctx, kv) = base_gb + per_1k_gb * ctx/1000
                  + mtp_overhead_gb (when MTP on)
                  + (model file size - base_weight_gb)
```

## Coefficients

| KV type | intercept (GB) | slope (GB/1k tokens) | accuracy |
|---|---|---|---|
| f16 | 11.117 | 0.0677 | **derived** from q4_0/q8_0 slopes (f16 OOMs past ~32k anyway) |
| q8_0 | 11.117 | 0.0372 | exact to ~0.01 GB over 32k–196k |
| q4_0 | 11.117 | 0.02194 | exact to ~0.01 GB over 32k–196k |

## Additional measured terms

| Term | Value | How it was measured |
|---|---|---|
| `mtp_overhead_gb` | **0.75 + 0.0039 * ctx/1000** | (MTP on minus MTP off) at IQ3_XXS, identical for q4_0 and q8_0 KV: +900 MiB at 32k, +1030 MiB at 64k. Bundles the head weights with the draft KV; the estimator removes one copy of mtp_head_gb before charging the model's own. |
| `mtp_head_gb` | **0.35** | Read from the tensor table: 0.351 GB in UD-IQ3_XXS, 0.351 in UD-IQ4_XS, 0.348 in GSQ-RCO-IQ3_XXS-mtp. The head keeps its own precision whatever the trunk quant is. Loaded **only** when --spec-type draft-mtp is passed. |
| `slot_overhead_gb` | **0.4385 * (slots-1)** | CONSTANT, not ctx-scaled: 1346 MiB for 3 extra slots at BOTH 32k and 96k (IQ3_XXS/q4_0/MTP2): 32k 13020 to 14366 MiB, 96k 13646 to 14992 MiB. |
| `vision_gb` | **1.11** (subtracted when vision off) | MEASURED, not file size: mmproj-F16.gguf is 885 MiB but dropping it saved **1136 MiB** (96k/q4_0/MTP2/1 slot: 14782 to 13646 MiB) because its own compute buffers go too. |
| `warn_gb` / `cap_gb` | 15.0 / 15.5 | The panel warns at 15.0 GB and refuses at 15.5. |

## Critical caveat

The fit models VRAM **at load**. Real failures happen at **peak during a long
prompt**, which grows with context: q8_0/80k/MTP loads at 15.24 GB and dies, while
Q4_K_S/8k loads at 15.31 GB and is fine. No single cap separates those — so shipped
presets carry **measured** `vram_gb` and the fit is only used for the custom-preset
builder.

## Slot / unified-KV cost beyond VRAM

llama.cpp's `--parallel -1` (auto) forces `n_parallel=4 + kv_unified=true`.
Unified KV shares one pool of `-c` tokens, so the KV term is **not** multiplied —
only the per-slot constant applies. **Cost beyond VRAM:** unified KV is ~3x slower
per stream. At 96k: 1 slot decodes 73–82 tok/s; auto/4-slot decodes 25.4 tok/s
single-stream and ~7.5 x 4 = 30 tok/s aggregate under 4 concurrent requests.
Auto only pays off when several clients must be served at once, and even then total
throughput is barely above one 1-slot stream.
