#!/usr/bin/env bash
# =============================================================================
# OPTIONAL — vLLM alongside llama.cpp (run whenever; AFTER the GPU driver works)
# Coexists with llama.cpp. NOTE: on 16 GB VRAM, run only ONE server at a time.
# vLLM needs a GPU-native format (not GGUF/IQ3). A 7B VLM is the sweet spot here.
# Run:  bash ~/setup_vllm_optional.sh
# =============================================================================
set -euo pipefail

command -v nvidia-smi >/dev/null 2>&1 || { echo "GPU driver not active (nvidia-smi missing). Do Phase 1 + reboot first."; exit 1; }

echo "==> Creating vLLM venv at ~/vllm-env (separate from llama.cpp)"
python3 -m venv "$HOME/vllm-env"
# shellcheck disable=SC1091
source "$HOME/vllm-env/bin/activate"
python -m pip install --upgrade pip wheel setuptools
echo "==> Installing vLLM (bundles a Blackwell-compatible torch)"
pip install --upgrade vllm

cat <<'EOF'

vLLM ready. Activate + serve a 7B vision model that fits 16 GB:

  source ~/vllm-env/bin/activate

  # AWQ 4-bit (recommended headroom on 16 GB):
  vllm serve Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
      --quantization awq_marlin \
      --gpu-memory-utilization 0.90 \
      --max-model-len 16384

OpenAI-compatible at http://localhost:8000/v1
REMINDER: stop llama.cpp's server first — one GPU server at a time on 16 GB.
EOF
