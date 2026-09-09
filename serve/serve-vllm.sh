#!/usr/bin/env bash
# =============================================================================
# serve-vllm.sh — start a vLLM OpenAI-compatible server on :8000 (or $PORT).
# Only ONE GPU server at a time on the 16 GB RTX 5080 (llama.cpp OR vLLM).
#
# vLLM CANNOT run the current 27B UD-IQ3_XXS GGUF (no GGUF IQ-quant / mmproj
# support). Point VLLM_MODEL at a GPU-native model that fits 16 GB, e.g. a 7B
# AWQ VLM (Qwen2.5-VL-7B-Instruct-AWQ) or an FP8/GPTQ 4-bit model.
#
# All knobs are env vars (the control panel sets them):
#   VLLM_MODEL MAX_MODEL_LEN GPU_UTIL QUANT KV_CACHE_DTYPE MAX_NUM_SEQS
#   PREFIX_CACHE REASONING_PARSER PORT
# =============================================================================
set -euo pipefail

VENV="${VLLM_VENV:-$HOME/vllm-env}"
[ -d "$VENV" ] || { echo "vLLM not installed at $VENV — run ~/setup_vllm_optional.sh first"; exit 1; }
# shellcheck disable=SC1091
source "$VENV/bin/activate"

MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct-AWQ}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"     # context window (vLLM preallocates KV)
GPU_UTIL="${GPU_UTIL:-0.90}"                # fraction of VRAM vLLM may use
QUANT="${QUANT:-awq_marlin}"               # awq_marlin | gptq_marlin | fp8 | (empty for bf16)
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"   # auto | fp8 (fp8 ~halves KV, needs support)
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"         # max concurrent sequences (throughput)
PREFIX_CACHE="${PREFIX_CACHE:-on}"         # on = --enable-prefix-caching (agent speedup)
REASONING_PARSER="${REASONING_PARSER:-}"   # e.g. qwen3 / deepseek_r1 to split reasoning
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-$(cat "$HOME/.vlm_api_key" 2>/dev/null)}"
# --- RTX 5080 / Blackwell (sm_120) FlashInfer fix -------------------------------
# FlashInfer JIT-compiles kernels and needs CUDA toolkit >= 12.9 for sm_120, but the
# system nvcc is 12.4. torch's cu13 wheel ships a full CUDA 13.3 toolkit — point
# FlashInfer at it. FLASHINFER_CUDA_ARCH_LIST=12.0f targets Blackwell explicitly.
_CU13="$HOME/vllm-env/lib/python3.12/site-packages/nvidia/cu13"
if [ -x "$_CU13/bin/nvcc" ]; then
    export CUDA_HOME="$_CU13"
    export PATH="$_CU13/bin:$PATH"
fi
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.0f}"
# The "FlashInfer requires sm75+" error is actually the FlashInfer top-k/top-p SAMPLER
# JIT mis-parsing the sm_120 arch token. Fall back to the torch-native sampler.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
# flashinfer-cubin (PyPI) is 0.6.13 while flashinfer/jit-cache are 0.6.18 — no matching
# cubin exists, but jit-cache 0.6.18+cu130 supplies the real sm_120 kernels, so bypass
# the version guard.
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
# Attention backend: leave UNSET so vLLM auto-selects (FlashInfer here). Only export
# if the caller sets a valid V1 backend (e.g. FLASHINFER / FLASH_ATTN / TRITON_ATTN).
[ -n "${VLLM_ATTENTION_BACKEND:-}" ] && export VLLM_ATTENTION_BACKEND
# enforce-eager skips CUDA-graph capture: faster startup, more robust on new GPUs.
ENFORCE_EAGER="${ENFORCE_EAGER:-on}"
EAGER_FLAG=""; [[ "$ENFORCE_EAGER" == "on" ]] && EAGER_FLAG="--enforce-eager"

echo "vLLM model : $MODEL"
echo "max_len    : $MAX_MODEL_LEN   gpu_util: $GPU_UTIL   quant: ${QUANT:-none}   kv: $KV_CACHE_DTYPE"
echo "attn       : ${VLLM_ATTENTION_BACKEND:-auto}   eager: $ENFORCE_EAGER"
echo "max_seqs   : $MAX_NUM_SEQS   prefix_cache: $PREFIX_CACHE   reasoning: ${REASONING_PARSER:-off}"

PREFIX_FLAG=""; [[ "$PREFIX_CACHE" == "on" ]] && PREFIX_FLAG="--enable-prefix-caching"

exec vllm serve "$MODEL" \
    ${QUANT:+--quantization "$QUANT"} \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    $PREFIX_FLAG \
    $EAGER_FLAG \
    ${REASONING_PARSER:+--reasoning-parser "$REASONING_PARSER"} \
    ${API_KEY:+--api-key "$API_KEY"} \
    --host 0.0.0.0 --port "$PORT"
