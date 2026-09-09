# Choosing & adding models

Models live in `~/models/<name>/` — **not** in this repo (they're large and
git-ignored). A folder is "servable" when it contains a `*.gguf` (plus an optional
`mmproj-*.gguf` for vision). The panel scans `~/models` and lets any preset point at
any model.

## Add a newly downloaded model (3 steps)
1. **Download** a GGUF into its own folder, e.g. with the Hugging Face CLI:
   ```bash
   # example: an 8B vision model in Unsloth dynamic quant
   hf download unsloth/Qwen3-VL-8B-Instruct-GGUF \
       --include "*UD-Q4_K_XL*" "mmproj-F16.gguf" \
       --local-dir ~/models/Qwen3-VL-8B
   ```
   Any GGUF works (chat, coder, vision). Put each model in its **own** subfolder.
2. **Rescan**: `./bin/llm models` (or just reload the panel page) — the new folder
   shows up, with its size and whether it has a vision projector.
3. **Use it**: in the panel, open any preset's **✏️ Edit** (or **＋ New preset**),
   pick the model from the **model** dropdown, set context / KV / thinking, **Save**,
   then **Apply**. A model that isn't registered yet auto-registers the first time a
   preset uses it.

The served model id stays whatever the endpoint says (default `qwen3-vl`), so tools
on the router never need reconfiguring when you change the underlying weights.

## Never hand-edit model facts — rescan

`llm rescan` (or **Rescan models** at the top of the panel's Models view) re-reads every
directory and writes back what it finds: the main GGUF, whether it embeds a NextN/MTP head,
whether a projector is present (which is what makes the vision toggle live), any
`mtp-*.gguf` drafts, and the on-disk size. `llm rescan-dry` previews. New folders are
registered automatically, one entry per build — two builds in one folder become two models
sharing a `dir` and differing by `gguf`. The only field rescan will not touch is
`supports_mtp`, which is the deliberate override.

## MTP: the same quant ships with and without the head

Speculative decoding here needs a NextN/MTP draft head. For Qwen3.8-27B that head lives
**inside** the main GGUF as one extra block (`blk.<n_layer>.nextn.*`), and whether a given
download has it is *not* visible in the file name:

| GGUF | blocks | head | vision | MTP |
|---|---|---|---|---|
| `Qwen3.8-27B-UD-IQ3_XXS` | 0–64 | 0.351 GB | mmproj | yes |
| `Qwen3.8-27B-UD-IQ4_XS` | 0–64 | 0.351 GB | mmproj | yes |
| `Qwen3.8-27B-GSQ-RCO-IQ3_XXS` | 0–63 | — | mmproj | **no** |
| `Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp` | 0–64 | 0.348 GB | — | yes |

Asking a head-less GGUF for `--spec-type draft-mtp` is a hard load failure, not a slow
path, so nothing guesses from the name: `serve-vlm.sh` and the panel both read the tensor
table out of the GGUF header (`gguf_has_nextn`). A model with no head has its MTP selector
greyed out in the preset editor and gets `spec_type: none` forced at launch; the launcher
degrades to plain decoding with a warning rather than dying.

**The head is only resident when MTP is on.** `common.cpp` sets `mparams.load_mtp` from the
spec type and `models/qwen35.cpp` marks every nextn tensor `TENSOR_SKIP` when it is false.
So an `-mtp` build costs exactly what its head-less sibling costs while MTP is off, and
~0.35 GB of weights *plus* the draft KV (`estimator.mtp_overhead_gb`) the moment you turn
it on. The preset editor spells that number out under the VRAM bar as you toggle MTP.

Note `estimator.mtp_overhead_gb` was measured as (MTP on − MTP off) on a GGUF that has the
head, so it already contains one copy of it; the estimator swaps that for the head of
whichever model you picked instead of counting both.

## Which quant / context fits 16 GB?
Rough rule for a dense model on this RTX 5080 (measure to be sure):
- **Weights**: a 4-bit (Q4/IQ4) quant of an *N*B model ≈ *N* × ~0.6 GB.
- **KV cache** scales with context and KV precision (`q4_0` ≈ ¼, `q8_0` ≈ ½ of f16).
  The panel shows a live `~VRAM` estimate per preset from your measured points.

**Measure, don't guess** — use `llama-bench` for speed and `make kvsweep` for the
max context that loads:
```bash
# generation / prompt throughput for a model
~/llama.cpp/build/bin/llama-bench -m ~/models/<name>/<file>.gguf -ngl 99

# max context that loads per KV quant, on this GPU
make kvsweep
```
Run `llama-bench` before/after changing flags (KV quant, `-ngl`, batch) to see the
real impact rather than guessing. `pp` (prompt) throughput matters for long-context /
RAG; `tg` (generation) is what you feel while chatting.

## Engines and model formats
| Engine | Status on this box | Formats | Model chosen via |
|---|---|---|---|
| **llama.cpp** | ✅ primary (running) | GGUF (Unsloth IQ/K-quants) + `mmproj` vision | preset ✏️ editor → **model** dropdown (scans `~/models`) |
| **Ollama** | ✅ installed + verified | GGUF (llama.cpp underneath), port 11434 | `ollama pull <tag>` → set the tag in Ollama **⚙ Options** |
| **vLLM** | ✅ installed + verified | AWQ/GPTQ 4-bit, FP8, BF16 (no GGUF) | HF repo id in vLLM **⚙ Options** |

Switch engines in the panel (Inference engine → Activate). Only ONE holds the GPU at a
time; the panel stops the others first. The router rewrites the client's model id to
the active engine's configured model, so tools never need reconfiguring.

### vLLM on the RTX 5080 (Blackwell / sm_120) — solved
vLLM 0.28 **serves on this GPU** (verified with `Qwen/Qwen2.5-3B-Instruct-AWQ`). Getting
there needed a specific fix chain (all baked into `setup/setup_vllm_optional.sh` +
`serve/serve-vllm.sh`):
1. **CUDA 13.3 toolchain** — FlashInfer JIT needs CUDA ≥ 12.9 but the system `nvcc` is
   12.4; `serve-vllm.sh` points `CUDA_HOME` at the cu13 toolkit bundled with torch.
2. **`flashinfer-jit-cache 0.6.18+cu130`** (from `https://flashinfer.ai/whl/cu130`) —
   prebuilt sm_120 kernels; the PyPI `flashinfer-cubin` has none for consumer Blackwell.
3. **`apache-tvm-ffi==0.1.11`** — 0.1.12+ double-registers a TVM FFI type and aborts.
4. **no `flashinfer-cubin`** (its 0.6.13 clashes with flashinfer 0.6.18).
5. env: `VLLM_USE_FLASHINFER_SAMPLER=0`, `FLASHINFER_DISABLE_VERSION_CHECK=1`,
   `FLASHINFER_CUDA_ARCH_LIST=12.0f`, `--enforce-eager`.

vLLM needs a GPU-native model (AWQ/FP8/GPTQ), not GGUF — set it in vLLM **⚙ Options**.
