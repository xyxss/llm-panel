# Model anatomy — the GGUF is a stack of numbered blocks

`blk.0` … `blk.N-1` are the transformer layers. Around them sit `token_embd`
(the input embedding table) and `output` / `output_norm` (the head that turns the
last hidden state back into token logits). `-ngl N` says how many of those blocks
go on the GPU — `-ngl 99` means "all of them", and the embedding/output tensors
count as one more layer on top.

Two things in a model directory are **not** part of that stack:

| file | what it is | flag |
|---|---|---|
| `mmproj-*.gguf` | vision projector — a separate CLIP-style encoder that turns image patches into tokens the LM can read | `--mmproj` |
| `mtp-*.gguf` | a standalone speculative draft head | `-md` |

Neither is detected by filename. `general.architecture = clip` (or
`general.type = mmproj`) identifies a projector; draft heads are identified by
architecture (`eagle3`, `dflash`, `dspark`) or by carrying `blk.<n>.nextn.*`
tensors. Rename them however you like — rescan reads the header.

## Model 1 — Qwen3.8-27B: dense hybrid, everything in VRAM

```
arch qwen35 · 65 blocks · embd 5120 · 27.3 B params · attn_interval 4

  blk.0 … blk.63   the trunk. NOT uniform: some blocks are attention, the rest are
                   SSM / linear-attention (gated delta-net) blocks with ssm_conv1d,
                   ssm_a, ssm_alpha/beta, ssm_out instead of a KV-cached attention.
                   That hybrid layout is why KV cost per 1k tokens is so low here.
  blk.64           the MTP / NextN speculative head — a FULL extra block (attn + FFN)
                   plus the nextn projections. ~0.35 GB.
  token_embd, output, output_norm
```

KV bytes per token (measured from the GGUF header):

| KV type | bytes/token | per 1k tokens |
|---|---|---|
| f16 | 66 560 | 66.56 MB |
| q8_0 | 35 360 | 35.36 MB |
| q4_0 | 17 680 | 17.68 MB |

This one fits entirely on the GPU (`-ngl 99`), so the only questions are context
length and KV precision. **Block 64 is the interesting part.** It is only loaded
when you ask for speculative decoding: `common.cpp` sets `mparams.load_mtp` from
the `--spec-type`, and `models/qwen35.cpp` marks every one of block 64's 15
tensors `TENSOR_SKIP` when that is false. So the same GGUF costs ~0.35 GB more
with MTP on than with it off, on top of the draft context's own KV cache.

**The same quant ships both with and without block 64**, with nothing in the
filename to say which — `GSQ-RCO-IQ3_XXS` has 64 blocks, `GSQ-RCO-IQ3_XXS-mtp`
has 65, and they sit in the same folder. Asking a 64-block build for
`--spec-type draft-mtp` is a **hard load failure**, not a slow path — which is why
the panel reads the tensor table instead of guessing.

## Model 2 — Qwen3.8-Flash-Next: sparse MoE, mostly *not* in VRAM

```
arch qwen4exp · 48 blocks · embd 2560 · 512 experts, 10 used per token · 82 GB on disk

  per block:  attention  ~small, always on GPU
              router     ~tiny, picks 10 of 512 experts
              512 experts  ~huge, and only 10 of them run per token
  plus:       per_layer_token_embd — a ~27 GB engram table
```

82 GB of weights against 16 GB of VRAM **and 30 GB of RAM**. It runs because the
parts are used unevenly:

- **`--n-cpu-moe N`** keeps the expert weights of the first N blocks in system RAM,
  leaving attention and the router on the GPU. Only 10 of 512 experts fire per
  token, so the CPU does far less work than the size suggests. **40 is the floor
  that loads on this box; 36 hard-OOMs.**
- **`--lazy-mode on`** reads rows of oversized tensors from disk on demand instead
  of keeping them resident — the only reason the 27 GB engram table survives.
- **`--load-mode mmap`** pages weights in from the file rather than reading them up
  front. Replaces the deprecated `--mmap` / `--no-mmap` / `--mlock`.
- **`-ngl auto`** (omit `-ngl`) so `--fit` sizes the split. Pinning `-ngl` on a
  model this size makes llama.cpp abort the fit and then OOM on compute buffers.

Two consequences:

- **Effectively single-context.** Concurrent decoding against a streamed model
  shares one expert cache and can corrupt output, so slots should be 1. The
  launcher warns when they are not.
- **Speed depends on the page cache**, which makes short benchmarks lie (see §7).

## Adding a new model (3 steps)

1. **Download** a GGUF into its own folder:
   ```bash
   hf download unsloth/Qwen3-VL-8B-Instruct-GGUF \
       --include "*UD-Q4_K_XL*" "mmproj-F16.gguf" \
       --local-dir ~/models/Qwen3-VL-8B
   ```
2. **Rescan**: `./bin/llm models` (or just reload the panel page) — the new folder
   shows up, with its size and whether it has a vision projector.
3. **Use it**: in the panel, open any preset's **✏️ Edit** (or **＋ New preset**),
   pick the model from the **model** dropdown, set context / KV / thinking, **Save**,
   then **Apply**. A model that isn't registered yet auto-registers the first time a
   preset uses it.

The served model id stays whatever the endpoint says (default `qwen3-vl`), so tools
on the router never need reconfiguring when you change the underlying weights.
