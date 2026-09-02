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
| Engine | Formats | Notes |
|---|---|---|
| **llama.cpp** | GGUF (incl. Unsloth IQ/K-quants) + `mmproj` vision | only engine that runs the 27B UD-IQ3_XXS vision model here |
| **vLLM** | BF16/FP16, FP8, AWQ/GPTQ 4-bit | GPU-native, high throughput; **no** GGUF IQ-quants. Configure model in the vLLM ⚙ Options |
| **Ollama** | GGUF (llama.cpp under the hood) | simplest; pull the model first (`ollama pull <tag>`) |

vLLM and Ollama each have their own **⚙ Options** editor in the panel (model, context,
quantization, etc.), saved even before they're installed and used when you switch to them.
