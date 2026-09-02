#!/usr/bin/env bash
# =============================================================================
# Phase 2 — llama.cpp (CUDA) + Qwen3.8-27B VLM (UD-IQ3_XXS)   run AFTER reboot
# Model: unsloth/Qwen3.8-27B-GGUF  quant: UD-IQ3_XXS (~10.9 GB) — a vision LM
# Runtime: llama.cpp (GGUF IQ3_XXS + vision cannot run on vLLM; this is the fit)
# Run:  bash ~/setup_phase2_vllm.sh
# Prereq: `nvidia-smi` works (Phase 1 done + MOK enrolled at reboot).
# =============================================================================
set -euo pipefail

MODELS_DIR="$HOME/models"
LLAMA_DIR="$HOME/llama.cpp"
HF_REPO="unsloth/Qwen3.8-27B-GGUF"

echo "=============================================================="
echo " Phase 2: verify GPU -> build llama.cpp (CUDA) -> get VLM"
echo "=============================================================="

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found — the NVIDIA driver is not active."
    echo "  -> Did you complete the blue 'Enroll MOK' screen at reboot?"
    echo "  -> Check:  dpkg -l | grep nvidia-driver ; sudo dmesg | grep -i nvidia"
    exit 1
fi
echo "==> GPU detected:"; nvidia-smi
echo "==> Enabling persistence mode"; sudo nvidia-smi -pm 1 || true

echo "==> Installing CUDA toolkit + build deps (sudo password needed)"
sudo apt-get update
sudo apt-get install -y nvidia-cuda-toolkit libcurl4-openssl-dev
echo "    nvcc version:"; nvcc --version | tail -2 || true

echo "==> Cloning + building llama.cpp with CUDA (Blackwell sm_120)"
if [ ! -d "$LLAMA_DIR" ]; then
    git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
else
    git -C "$LLAMA_DIR" pull --ff-only || true
fi
# Pick a CUDA arch nvcc can actually compile. sm_120 (Blackwell) needs CUDA>=12.8.
# If nvcc is older, build compute_90 PTX and let the driver JIT it to sm_120.
CUDA_ARCH=120
if ! nvcc --list-gpu-arch 2>/dev/null | grep -q compute_120; then
    echo "    nvcc too old for sm_120 -> using '90-virtual' (PTX JITs to Blackwell at runtime)"
    CUDA_ARCH="90-virtual"
fi
rm -rf "$LLAMA_DIR/build"   # clear any stale/failed cmake cache
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DLLAMA_CURL=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA_DIR/build" --config Release -j"$(nproc)"
echo "    Built binaries in $LLAMA_DIR/build/bin"

echo "==> Downloading model + vision projector (~11-13 GB) into $MODELS_DIR"
python3 -m pip install --user --upgrade "huggingface_hub[cli]"
mkdir -p "$MODELS_DIR"
# Grab only the UD-IQ3_XXS weights + the mmproj (vision) file, by pattern.
python3 -m huggingface_hub download "$HF_REPO" \
    --include "*UD-IQ3_XXS*" "*mmproj*" \
    --local-dir "$MODELS_DIR/Qwen3.8-27B" 2>/dev/null \
  || huggingface-cli download "$HF_REPO" \
        --include "*UD-IQ3_XXS*" "*mmproj*" \
        --local-dir "$MODELS_DIR/Qwen3.8-27B"

MODEL=$(find "$MODELS_DIR/Qwen3.8-27B" -name '*UD-IQ3_XXS*.gguf' | sort | head -1)
MMPROJ=$(find "$MODELS_DIR/Qwen3.8-27B" -iname 'mmproj*.gguf' | sort | head -1)
echo "    MODEL  = ${MODEL:-NOT FOUND}"
echo "    MMPROJ = ${MMPROJ:-NOT FOUND (vision may be unavailable)}"

# Write a convenience launcher tuned for 16 GB VRAM.
cat > "$HOME/serve-vlm.sh" <<EOF
#!/usr/bin/env bash
# OpenAI-compatible VLM server at http://localhost:8000/v1
exec "$LLAMA_DIR/build/bin/llama-server" \\
    -m "$MODEL" \\
    --mmproj "$MMPROJ" \\
    -ngl 99 \\
    -c 16384 \\
    -fa on \\
    --host 0.0.0.0 --port 8000
EOF
chmod +x "$HOME/serve-vlm.sh"

echo
echo "=============================================================="
echo " Done. Start the vision-LM server with:"
echo "     bash ~/serve-vlm.sh"
echo " Then it's OpenAI-compatible at http://localhost:8000/v1"
echo "   curl http://localhost:8000/v1/models"
echo
echo " Notes for 16 GB:"
echo "  -ngl 99 puts all layers on the GPU (10.9GB model + mmproj fit)."
echo "  If you hit OOM: lower -c (context) to 8192, or drop -ngl a bit."
echo "  Raise -c for longer context if VRAM headroom allows."
echo "=============================================================="

# ----------------------------------------------------------------------------
# ALTERNATIVE — Ollama (simplest, bundles its own CUDA). If you prefer this
# over building llama.cpp, you don't need any of the above:
#   curl -fsSL https://ollama.com/install.sh | sh
#   ollama run hf.co/unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS
# ----------------------------------------------------------------------------
