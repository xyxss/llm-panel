#!/usr/bin/env bash
# =============================================================================
# serve-vlm.sh — Qwen3.8-27B vision-LM (GGUF) via llama.cpp
# OpenAI-compatible server at http://localhost:8000/v1
#
# >>> ALL TUNING KNOBS ARE THE VARIABLES BELOW — edit or override inline. <<<
# e.g.  THINKING=off CTX=65536 KV_QUANT=q4_0 bash serve-vlm.sh
#
# Sampling defaults follow Unsloth's Qwen3-VL guide (they differ for
# thinking vs non-thinking — this script picks the right set automatically).
# =============================================================================
set -euo pipefail

# ---- paths ----------------------------------------------------------------
LLAMA="$HOME/llama.cpp/build/bin/llama-server"
MDIR="${MDIR:-$HOME/models/Qwen3.8-27B}"   # override with MDIR=... for another model dir

# ---- CORE TUNING ----------------------------------------------------------
CTX="${CTX:-98304}"        # context window (total KV tokens). Bigger = more VRAM.
PARALLEL="${PARALLEL:-1}"  # concurrent slots. Each slot gets CTX/PARALLEL tokens.
NGL="${NGL:-99}"           # layers on GPU (99 = all). Lower only on VRAM OOM.
KV_QUANT="${KV_QUANT:-q8_0}" # KV cache: f16 (best, 2x mem) | q8_0 (recommended)
                             #           | q4_0 (quarter mem, max context)
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-$(cat "$HOME/.vlm_api_key" 2>/dev/null)}"

# ---- THINKING / REASONING -------------------------------------------------
# on   = always think (best quality, slower, burns output tokens)
# off  = never think (fast, good for tool-calling agents & simple tasks)
# auto = let the chat template decide (default)
THINKING="${THINKING:-auto}"
# Reasoning effort passed to the template when thinking is active:
#   default | low | medium | high   (template-dependent)
REASON_EFFORT="${REASON_EFFORT:-default}"
# How thoughts are returned: deepseek = separate `reasoning_content` field
# (best for harnesses), deepseek-legacy = <think> tags inline in content.
REASON_FORMAT="${REASON_FORMAT:-deepseek}"

# ---- SAMPLING (Unsloth Qwen3-VL recommendations) --------------------------
# Thinking and non-thinking modes want DIFFERENT values. Auto-selected below;
# set any var explicitly to override.
if [[ "$THINKING" == "off" ]]; then
    # Instruct / non-thinking profile
    TEMP="${TEMP:-0.7}"; TOP_P="${TOP_P:-0.8}"; PRESENCE="${PRESENCE:-1.5}"
else
    # Thinking profile
    TEMP="${TEMP:-1.0}"; TOP_P="${TOP_P:-0.95}"; PRESENCE="${PRESENCE:-0.0}"
fi
TOP_K="${TOP_K:-20}"          # Unsloth: 20 for both modes
MIN_P="${MIN_P:-0.0}"         # Unsloth: 0.0 for both modes
REPEAT_PEN="${REPEAT_PEN:-1.0}"  # Unsloth: 1.0 (disabled). >1.0 can hurt quality.

# ---- CONTEXT HANDLING / COMPACTION ----------------------------------------
# CONTEXT_SHIFT: what happens when the window fills up.
#   off (DEFAULT, recommended for agents) — server returns an error instead of
#        silently dropping your oldest tokens. Your harness then compacts.
#   on  — llama.cpp discards oldest tokens to keep generating. Fine for endless
#        chat, CORRUPTING for agents (system prompt / early tool results vanish).
CONTEXT_SHIFT="${CONTEXT_SHIFT:-off}"
# CACHE_REUSE: reuse cached KV chunks even when the prefix isn't identical.
# Big speedup for agent loops that edit history / compact. 256 is a safe start;
# 0 disables. Requires prompt caching (on by default).
CACHE_REUSE="${CACHE_REUSE:-256}"

# ---- VISION ---------------------------------------------------------------
# Qwen-VL warns it needs >=1024 image tokens for accurate grounding tasks.
# Higher = better image detail but more VRAM+time per image. Empty = model default.
IMAGE_MIN_TOKENS="${IMAGE_MIN_TOKENS:-1024}"
IMAGE_MAX_TOKENS="${IMAGE_MAX_TOKENS:-}"   # e.g. 2048 to cap huge images

# ---- PERFORMANCE ----------------------------------------------------------
BATCH="${BATCH:-2048}"     # logical batch (prompt processing throughput)
UBATCH="${UBATCH:-512}"    # physical batch. Raising to 1024 can speed prompt
                           #   eval (more VRAM). Lower if tight.
THREADS="${THREADS:--1}"   # CPU threads, -1 = auto
FIT="${FIT:-}"             # --fit on|off: let llama.cpp auto-shrink UNSET args to
                           #   fit VRAM. Empty = don't pass it (use explicit ctx).
EXTRA_FLAGS="${EXTRA_FLAGS:-}"  # any extra llama-server flags, appended verbatim.
                           #   e.g. EXTRA_FLAGS='--slots --metrics -ot ".ffn_.*_exps.=CPU"'
# ---------------------------------------------------------------------------

MODEL="$(find "$MDIR" -iname '*.gguf' ! -iname 'mmproj*' 2>/dev/null | sort | head -1)"
MMPROJ="$(find "$MDIR" -iname 'mmproj*.gguf' 2>/dev/null | sort | head -1)"

[ -x "$LLAMA" ] || { echo "llama-server not built at $LLAMA"; exit 1; }
[ -n "$MODEL" ] || { echo "no model .gguf in $MDIR"; exit 1; }

echo "Model    : $MODEL"
echo "MMProj   : ${MMPROJ:-<none — vision disabled>}"
echo "Context  : $CTX   Parallel: $PARALLEL   GPU layers: $NGL   KV: $KV_QUANT"
echo "Thinking : $THINKING (effort=$REASON_EFFORT, format=$REASON_FORMAT)"
echo "Sampling : temp=$TEMP top_p=$TOP_P top_k=$TOP_K min_p=$MIN_P presence=$PRESENCE repeat=$REPEAT_PEN"
echo "Context  : shift=$CONTEXT_SHIFT cache_reuse=$CACHE_REUSE"

# Build optional flags conditionally
SHIFT_FLAG="--no-context-shift"; [[ "$CONTEXT_SHIFT" == "on" ]] && SHIFT_FLAG="--context-shift"

exec "$LLAMA" \
    -m "$MODEL" \
    ${MMPROJ:+--mmproj "$MMPROJ"} \
    --alias "qwen3-vl" \
    --jinja \
    --reasoning "$THINKING" \
    --reasoning-effort "$REASON_EFFORT" \
    --reasoning-format "$REASON_FORMAT" \
    -ngl "$NGL" \
    -c "$CTX" \
    --parallel "$PARALLEL" \
    -fa on \
    --cache-type-k "$KV_QUANT" \
    --cache-type-v "$KV_QUANT" \
    $SHIFT_FLAG \
    --cache-reuse "$CACHE_REUSE" \
    --temp "$TEMP" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --min-p "$MIN_P" \
    --presence-penalty "$PRESENCE" \
    --repeat-penalty "$REPEAT_PEN" \
    ${IMAGE_MIN_TOKENS:+--image-min-tokens "$IMAGE_MIN_TOKENS"} \
    ${IMAGE_MAX_TOKENS:+--image-max-tokens "$IMAGE_MAX_TOKENS"} \
    -b "$BATCH" \
    -ub "$UBATCH" \
    -t "$THREADS" \
    ${FIT:+--fit "$FIT"} \
    ${API_KEY:+--api-key "$API_KEY"} \
    --host 0.0.0.0 --port "$PORT" \
    $EXTRA_FLAGS
