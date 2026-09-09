# Setup from scratch

Target: Ubuntu, one NVIDIA GPU (this box: RTX 5080, 16 GB). Only ONE GPU inference
engine runs at a time on 16 GB.

## 1. GPU driver + CUDA
```bash
bash setup/setup_phase1_gpu.sh
```
Installs/validates the NVIDIA driver + CUDA. On Secure Boot machines the driver's
kernel module must be MOK-enrolled (reboot + enroll key) or the GPU won't be visible
to `nvidia-smi`.

## 2. llama.cpp (CUDA build)
Build llama.cpp with CUDA into `~/llama.cpp` (the panel expects
`~/llama.cpp/build/bin/llama-server`):
```bash
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DGGML_CUDA=ON
cmake --build ~/llama.cpp/build -j --config Release
```

## 3. A model
Put a GGUF under `~/models/<name>/` (see [models.md](models.md)). The current default
model is `~/models/Qwen3.8-27B/` (Unsloth UD-IQ3_XXS + `mmproj-F16.gguf`).

## 4. API key
```bash
python3 -c "import secrets;print('sk-'+secrets.token_hex(20))" > ~/.vlm_api_key
chmod 600 ~/.vlm_api_key
```

## 5. Start the stack
```bash
cd ~/llm-stack
cp config/secrets.example.json config/secrets.json    # add cloud keys if any
./bin/llm panel start
```

## 6. Autostart at login + boot (rootless — no sudo)
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

LAN access (open the ports, needs sudo):
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
