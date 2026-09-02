#!/usr/bin/env bash
# =============================================================================
# Phase 1 — NVIDIA driver + build tools + headless target
# Target: Ubuntu 26.04, RTX 5080 (Blackwell), Secure Boot ENABLED
# Run:  bash ~/setup_phase1_gpu.sh
# You will be prompted for: (1) your sudo password, (2) a NEW "MOK password"
# during the driver install — WRITE IT DOWN, you need it at the reboot screen.
# =============================================================================
set -euo pipefail

echo "=============================================================="
echo " Phase 1: GPU driver + essential build tools + headless mode"
echo "=============================================================="
echo "This will:"
echo "  1. Install build-essential, git, cmake, python venv tools, headers"
echo "  2. Install nvidia-driver-595-open (recommended for RTX 5080/Blackwell)"
echo "  3. Set the system to boot WITHOUT the desktop GUI (multi-user.target)"
echo "     -> frees GPU VRAM for LLM/VLM work. Revert with:"
echo "        sudo systemctl set-default graphical.target"
echo "  4. Offer to reboot."
echo
echo "IMPORTANT (Secure Boot is ON): during step 2 you'll be asked to SET a"
echo "MOK password. Remember it. On reboot a blue 'MOK Management' screen shows:"
echo "  Enroll MOK -> Continue -> Yes -> enter that password -> Reboot"
echo "If you skip that, the GPU driver will NOT load."
echo
read -rp "Continue? [y/N] " ans
[[ "${ans:-}" == [yY] ]] || { echo "Aborted."; exit 1; }

echo "==> [1/4] apt update + essential packages"
sudo apt-get update
sudo apt-get install -y \
    build-essential git cmake ninja-build pkg-config \
    curl wget ca-certificates gnupg \
    python3-pip python3-venv python3-dev \
    dkms "linux-headers-$(uname -r)" \
    htop nvtop lm-sensors

echo "==> [2/4] NVIDIA driver (nvidia-driver-595-open)"
echo "    (you may be prompted to SET a MOK password now — remember it!)"
sudo apt-get install -y nvidia-driver-595-open

echo "==> [3/4] Set default boot target to multi-user (no desktop GUI)"
sudo systemctl set-default multi-user.target

echo "==> [4/4] Done with Phase 1. (NOT rebooting — reboot yourself when ready.)"
echo
echo "WHEN YOU CHOOSE TO REBOOT ( sudo reboot ):"
echo "  - Complete the blue MOK 'Enroll MOK' screen (password you just set)."
echo "  - Log in at the text console."
echo "  - Verify the GPU:   nvidia-smi"
echo "  - Then run:         bash ~/setup_phase2_vllm.sh"
echo
echo "The NVIDIA driver only becomes active AFTER a reboot + MOK enrollment."
