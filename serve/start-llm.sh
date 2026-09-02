#!/usr/bin/env bash
# =============================================================================
# start-llm.sh — choose which local inference engine to start.
# Only ONE runs at a time (16 GB VRAM). Both are OpenAI-compatible on :8000.
# =============================================================================
set -euo pipefail

PORT=8000

# Refuse to start a second server on the same GPU/port.
if command -v ss >/dev/null && ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    echo "A server is already listening on :$PORT. Stop it first (Ctrl-C in its terminal)."
    exit 1
fi

echo "Which engine do you want to start?"
echo "  1) llama.cpp  — Qwen3.8-27B VLM (GGUF UD-IQ3_XXS)   [big vision model]"
echo "  2) vLLM       — Qwen2.5-VL-7B-AWQ                    [smaller, high-throughput]"
read -rp "Choice [1/2]: " choice

case "${choice:-}" in
  1)
    HERE="$(cd "$(dirname "$0")" && pwd)"
    [ -x "$HERE/serve-vlm.sh" ] || { echo "serve-vlm.sh not found next to start-llm.sh"; exit 1; }
    echo "==> Starting llama.cpp VLM server on :$PORT ..."
    exec "$HERE/serve-vlm.sh"
    ;;
  2)
    [ -d "$HOME/vllm-env" ] || { echo "vLLM not installed. Run ~/setup_vllm_optional.sh first."; exit 1; }
    # shellcheck disable=SC1091
    source "$HOME/vllm-env/bin/activate"
    echo "==> Starting vLLM server on :$PORT ..."
    exec vllm serve Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
        --quantization awq_marlin \
        --gpu-memory-utilization 0.90 \
        --max-model-len 16384 \
        --port "$PORT"
    ;;
  *)
    echo "No valid choice; nothing started."
    exit 1
    ;;
esac
