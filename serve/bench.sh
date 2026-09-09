#!/usr/bin/env bash
# =============================================================================
# bench.sh — sweep one or more llama.cpp settings and print JSON results.
#
# Wraps llama-bench, which takes a LIST for most flags and runs the cross
# product: `-ncmoe 40,48` benchmarks both and reports each separately. That is
# the whole point — "is 40 or 48 faster on this box" is one command, not two
# server restarts and a stopwatch.
#
#   MDIR=~/models/X MODEL_FILE=x.gguf NCMOE=40,48 bash bench.sh
#
# NOTE llama-bench cannot measure MTP: it has no --spec-type/-md. Speculative
# decoding has to be measured against the running server instead, which is what
# the panel's "bench running preset" path does.
# =============================================================================
set -euo pipefail

LLAMA="${LLAMA_BENCH:-$([ -x "$HOME/llama.cpp-sm120/build/bin/llama-bench" ] \
        && echo "$HOME/llama.cpp-sm120/build/bin/llama-bench" \
        || echo "$HOME/llama.cpp/build/bin/llama-bench")}"
export LD_LIBRARY_PATH="$(dirname "$LLAMA"):$HOME/cuda129/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

MDIR="${MDIR:?MDIR is required}"
MODEL_FILE="${MODEL_FILE:-}"
if [ -n "$MODEL_FILE" ]; then
    case "$MODEL_FILE" in /*) MODEL="$MODEL_FILE";; *) MODEL="$MDIR/$MODEL_FILE";; esac
else
    MODEL="$(find "$MDIR" -iname '*.gguf' ! -iname 'mmproj*' ! -iname 'mtp-*' 2>/dev/null \
             | LC_ALL=C sort | head -1)"
fi
[ -f "$MODEL" ] || { echo "no model at $MODEL" >&2; exit 1; }

# Each of these takes a comma-separated LIST; llama-bench runs every combination.
PROMPT="${PROMPT:-512}"       # -p  prompt tokens (prefill)
GEN="${GEN:-128}"             # -n  generated tokens (decode)
REPS="${REPS:-3}"             # -r  repetitions per combination
NGL="${NGL:-}"                # empty or 'auto' = do NOT pass -ngl. Required for any model
                              #   that relies on fitting: pinning -ngl on a model bigger
                              #   than VRAM just fails the load outright.
FIT_TARGET="${FIT_TARGET:-}"  # -fitt MiB. llama-bench does not fit unless this is set
                              #   (unlike llama-server, where --fit defaults to on), so a
                              #   disk-streaming model needs it or it will not load.
FIT_CTX="${FIT_CTX:-}"        # -fitc: floor on the ctx the fit may choose
NCMOE="${NCMOE:-}"            # the flagship sweep: experts kept on CPU
LOAD_MODE="${LOAD_MODE:-}"    # mmap | mlock | mmap+mlock | dio | none
LAZY_MODE="${LAZY_MODE:-}"
KV="${KV:-}"                  # -ctk/-ctv together
FLASH_ATTN="${FLASH_ATTN:-on}"
UBATCH="${UBATCH:-}"
BATCH="${BATCH:-}"
THREADS="${THREADS:-}"
OVERRIDE_TENSOR="${OVERRIDE_TENSOR:-}"
EXTRA="${EXTRA:-}"

NGL_ARG=""
case "$NGL" in ""|auto) ;; *) NGL_ARG="-ngl $NGL";; esac

exec "$LLAMA" -m "$MODEL" -p "$PROMPT" -n "$GEN" -r "$REPS" $NGL_ARG \
    -fa "$FLASH_ATTN" \
    ${FIT_TARGET:+-fitt "$FIT_TARGET"} \
    ${FIT_CTX:+-fitc "$FIT_CTX"} \
    ${NCMOE:+-ncmoe "$NCMOE"} \
    ${LOAD_MODE:+-lm "$LOAD_MODE"} \
    ${LAZY_MODE:+-lzm "$LAZY_MODE"} \
    ${KV:+-ctk "$KV" -ctv "$KV"} \
    ${UBATCH:+-ub "$UBATCH"} \
    ${BATCH:+-b "$BATCH"} \
    ${THREADS:+-t "$THREADS"} \
    ${OVERRIDE_TENSOR:+-ot "$OVERRIDE_TENSOR"} \
    -o json $EXTRA
