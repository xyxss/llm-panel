# Setup — every step, in order

Target: Ubuntu, one NVIDIA GPU (this box: RTX 5080, 16 GB). Only ONE GPU inference
engine runs at a time on 16 GB.

## Step 1 — GPU driver + CUDA

```bash
bash setup/setup_phase1_gpu.sh
```

Installs/validates the NVIDIA driver + CUDA 12.9. On Secure Boot machines the
driver's kernel module must be MOK-enrolled (reboot + enroll key) or the GPU won't
be visible to `nvidia-smi`. Verified on this box: `nvidia-smi` reports
**NVIDIA GeForce RTX 5080, 16303 MiB VRAM**.

## Step 2 — llama.cpp (two builds)

Two builds are kept because the old PTX-JIT build **asserts in `graph_mtp` and dies**
with speculative decoding:

```bash
# Build A — native sm_120 (Blackwell kernels, MTP works). Preferred.
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp-sm120
cmake -S ~/llama.cpp-sm120 -B ~/llama.cpp-sm120/build -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build ~/llama.cpp-sm120/build -j --config Release

# Build B — fallback (PTX-JIT), used only if A is missing.
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DGGML_CUDA=ON
cmake --build ~/llama.cpp/build -j --config Release
```

`serve-vlm.sh` prefers `~/llama.cpp-sm120/build/bin/llama-server` and falls back to
`~/llama.cpp/build/bin/llama-server`. It also exports
`LD_LIBRARY_PATH=$HOME/cuda129/lib` (rootless CUDA toolkit).

## Step 3 — Models on disk

Each model lives in its own folder under `~/models/<name>/`:

| Folder | Model | Size | Blocks | Notes |
|---|---|---|---|---|
| `Qwen3.8-27B-GGUF-UD-IQ4_XS/` | Qwen3.8-27B UD-IQ4_XS + mmproj-BF16 | 14.25 GB | 65 (incl. MTP head) | dense-vl, MTP embedded |
| `Qwen3.8-27B-GGUF-UD-IQ3_XXS/` | Qwen3.8-27B UD-IQ3_XXS + mmproj-BF16 | 10.93 GB | 65 (incl. MTP head) | dense-vl, MTP embedded |
| `Qwen3.8-Flash-Next-UD-IQ3_XXS/` | Qwen3.8-Flash-Next 125B-A6B UD-IQ3_XXS | 81.96 GB | 48 MoE blocks, 512 experts × 10 used/token | moe-stream; + ~27 GB engram table |
| `Qwen3.8-27B-GSQ-RCO-GGUF-IQ3_XXS/` | GSQ-RCO IQ3_XXS **plain** (64 blocks) and **-mtp** (65 blocks) in one folder | 10.09 / 10.44 GB | 64 / 65 | rescan registers both as separate models |
| `Qwen3.8-27B-GSQ-RCO-GGUF-IQ3_S/` | GSQ-RCO IQ3_S + MTP head | 12.12 GB | 65 | dense-vl |
| `Qwen3.8-27B-GSQ-RCO-GGUF-IQ2_S/` | GSQ-RCO IQ2_S + MTP head | 9.61 GB | 65 | dense-vl |

Download example (any GGUF works):

```bash
hf download unsloth/Qwen3-VL-8B-Instruct-GGUF \
    --include "*UD-Q4_K_XL*" "mmproj-F16.gguf" \
    --local-dir ~/models/Qwen3-VL-8B
```

After dropping a folder in, run `./bin/llm rescan` — it reads the GGUF header and
writes back: main build, MTP head presence, projector presence, on-disk size.
**Never hand-edit model facts.** The only field rescan never touches is
`supports_mtp` (the deliberate override).

## Step 4 — API key

```bash
python3 -c "import secrets;print('sk-'+secrets.token_hex(20))" > ~/.vlm_api_key
chmod 600 ~/.vlm_api_key
```

The panel requires this key for every mutating call; without it the panel is read-only.
`serve-vlm.sh` reads the same file and passes it to `llama-server`.

## Step 5 — Cloud keys (optional)

```bash
cp config/secrets.example.json config/secrets.json   # git-ignored
# then add any cloud keys: "openai", "openrouter", "deepseek", "groq", …
```

## Step 6 — Start the stack

```bash
cd ~/llm-stack
./bin/llm panel start        # web UI :8080, router :8001, model server :8000
./bin/llm rescan             # re-read every model folder off disk
./bin/llm status             # GPU + running model + engine health
```

Open `http://<box>:8080`, enter the API key (top-right), and switch models /
presets / engines from there. Point harnesses at `http://<box>:8001/v1`.

## Step 7 — Autostart at login + boot (rootless, no sudo)

```bash
./bin/llm autostart      # systemd USER service + linger: panel starts at login and boot
```

This installs `~/.config/systemd/user/llm-panel.service`, enables it, and turns on
linger so it also starts at boot without a login. The panel restarts on failure.
`./bin/llm autostart-off` disables it. Manage it with
`systemctl --user status|restart llm-panel` and `journalctl --user -u llm-panel -f`.

With `server.autostart_local: true` in `config/presets.json` (the default), the panel
also brings up the default local model on boot — guarded so a panel restart never
disturbs a model that's already running. Set it `false` to leave the GPU idle until you
pick a preset. (A system-wide unit needs sudo: `./bin/llm install-service`.)

Recommended alongside it — the panel spawns `llama-server` and download jobs as
children, so a default unit stop would kill them too:

```bash
mkdir -p ~/.config/systemd/user/llm-panel.service.d
printf '[Service]\nKillMode=process\n' > ~/.config/systemd/user/llm-panel.service.d/override.conf
systemctl --user daemon-reload
```

## Step 8 — Firewall (optional, LAN access)

```bash
./bin/llm firewall            # prints the ufw commands to open :8080 and :8001
```

If you use the systemd unit `systemd/llama-vlm.service` to run the model, do NOT also
drive the model from the panel — pick one, or they fight over the GPU process.

## Optional: vLLM

```bash
bash setup/setup_vllm_optional.sh    # creates ~/vllm-env
```

Then switch to it in the panel (Inference engine → vLLM → ⚙ Options for the model).
vLLM needs a GPU-native model (AWQ/FP8/GPTQ), not a GGUF IQ-quant.

## Optional: Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull hf.co/unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q4_K_XL
```

Same engine as llama.cpp, port 11434, fewer exposed flags; bundles its own CUDA.
