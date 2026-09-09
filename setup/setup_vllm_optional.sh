#!/usr/bin/env bash
# =============================================================================
# OPTIONAL — vLLM alongside llama.cpp. Rootless, no sudo.
# On 16 GB VRAM run only ONE GPU server at a time. vLLM needs a GPU-native
# format (AWQ/GPTQ/FP8), NOT GGUF IQ-quants.
#
# IMPORTANT: this box's system python is 3.14, which vLLM/torch have no wheels
# for. So we use `uv` to make a dedicated Python 3.12 venv at ~/vllm-env and let
# uv pick the CUDA build of torch that matches this GPU (RTX 5080 / sm_120).
# =============================================================================
set -euo pipefail
command -v nvidia-smi >/dev/null 2>&1 || { echo "GPU driver not active. Do the GPU setup first."; exit 1; }

# 1) uv (rootless python/venv manager)
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  echo "==> installing uv (rootless)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# 2) Python 3.12 + venv
echo "==> creating ~/vllm-env with Python 3.12"
uv python install 3.12
uv venv --python 3.12 "$HOME/vllm-env"

# 3) vLLM + a GPU-matched torch (--torch-backend=auto detects sm_120/CUDA)
echo "==> installing vLLM (this pulls a Blackwell-compatible torch, ~7 GB)"
uv pip install --python "$HOME/vllm-env" vllm --torch-backend=auto

# 4) THE RTX 5080 / sm_120 FIX (confirmed working with vLLM 0.28):
#    a. prebuilt FlashInfer kernels for CUDA 13 (has sm_120; the pip 'lite' wheel does not)
uv pip install --python "$HOME/vllm-env" flashinfer-jit-cache \
    --index-url https://flashinfer.ai/whl/cu130
#    b. pin apache-tvm-ffi: 0.1.12+ double-registers a TVM FFI type and aborts
#       ("TypeAttr __ffi_repr__ is already registered"). 0.1.11 is clean.
uv pip install --python "$HOME/vllm-env" "apache-tvm-ffi==0.1.11"
#    c. do NOT install flashinfer-cubin — PyPI only has 0.6.13 (no sm_120) and it
#       version-clashes with flashinfer 0.6.18; jit-cache supplies the kernels.
uv pip uninstall --python "$HOME/vllm-env" flashinfer-cubin 2>/dev/null || true
#    (serve-vllm.sh also sets CUDA_HOME=<bundled cu13 toolkit>, FLASHINFER_CUDA_ARCH_LIST=12.0f,
#     VLLM_USE_FLASHINFER_SAMPLER=0, FLASHINFER_DISABLE_VERSION_CHECK=1, --enforce-eager.)

echo
"$HOME/vllm-env/bin/vllm" --version
cat <<'EOF'

vLLM is CONFIRMED WORKING on the RTX 5080 (sm_120) with the fix above (verified with
Qwen/Qwen2.5-3B-Instruct-AWQ). ~/llm-stack/serve/serve-vllm.sh is what the panel
launches when you switch the engine to vLLM. Pick the model + options in the panel
(Inference engine -> vLLM -> Options), e.g.:
  Qwen/Qwen2.5-3B-Instruct-AWQ    (small, fast, text)
  Qwen/Qwen2.5-VL-7B-Instruct-AWQ (vision, ~7 GB)
REMINDER: one GPU server at a time on 16 GB — the panel stops llama.cpp/Ollama first.
EOF
