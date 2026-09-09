#!/usr/bin/env bash
# =============================================================================
# OPTIONAL — Ollama, rootless (no sudo). Installs to ~/.local, serves on :11434.
# Same llama.cpp engine underneath; simplest model management (ollama pull).
# On 16 GB run only ONE GPU server at a time — the panel stops the others first.
# =============================================================================
set -euo pipefail

VER_URL="https://github.com/ollama/ollama/releases/latest/download"
ASSET="ollama-linux-amd64.tar.zst"   # note: .tar.zst (zstd), not .tgz
TMP="$(mktemp -d)"

echo "==> downloading $ASSET (~1.4 GB)"
# resolve the 'latest' redirect to a concrete release, then grab the asset
REL="$(curl -fsSLI -o /dev/null -w '%{url_effective}' "$VER_URL/$ASSET" | sed 's#/download/[^/]*/.*#/download#')" || true
curl -fL --retry 3 "$VER_URL/$ASSET" -o "$TMP/$ASSET"

echo "==> extracting to ~/.local"
mkdir -p "$HOME/.local"
tar --zstd -xf "$TMP/$ASSET" -C "$HOME/.local"
rm -rf "$TMP"

# put ~/.local/bin on PATH for future shells
grep -q '.local/bin' "$HOME/.bashrc" 2>/dev/null || \
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
export PATH="$HOME/.local/bin:$PATH"

echo "==> installed:"; "$HOME/.local/bin/ollama" --version | head -1 || true
cat <<'EOF'

Ollama installed (rootless). Start the daemon (the panel also does this when you
switch the engine to Ollama):

  OLLAMA_HOST=0.0.0.0:11434 ~/.local/bin/ollama serve &

Pull a model, then select its tag in the panel (Inference engine -> Ollama -> Options):

  ollama pull qwen2.5:0.5b        # tiny smoke-test
  ollama pull qwen2.5:7b          # a real 7B
EOF
