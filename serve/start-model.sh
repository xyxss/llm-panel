#!/usr/bin/env bash
# =============================================================================
# start-model.sh — launch a named PRESET ("model") of the local stack.
#
# Usage:  bash ~/start-model.sh <preset> [parallel] [thinking]
#   <preset>    name from the table below
#   [parallel]  concurrent slots (DEFAULT 1). Each slot gets CTX/parallel tokens.
#   [thinking]  on | off | auto   (DEFAULT: the preset's own setting)
#
# Model id stays "qwen3-vl" so harness config never changes between presets.
# =============================================================================
set -euo pipefail

#            preset      KV     CTX     THINK  note
PRESETS="
q8-64k      q8_0   65536   auto  Best KV quality, safe image headroom (~13.9GB)
q8-96k      q8_0   98304   auto  Best KV quality; TIGHT, can OOM on big images (~15.4GB)
q4-96k      q4_0   98304   auto  Full context + safe headroom (~13.6GB) [recommended]
q4-128k     q4_0  131072   auto  More context, still comfortable (~14.3GB)
q4-160k     q4_0  163840   auto  Most context; TIGHT (~15.1GB)
agent       q4_0  131072   off   128k, THINKING OFF — fast tool-calling agent loops
deep        q4_0   98304   on    96k, THINKING ON — max reasoning quality
vision      q4_0   65536   auto  64k + big image headroom, for heavy image work
"

P="${1:-}"; PAR="${2:-1}"; TH="${3:-}"

show_help() {
  echo "Usage: bash ~/start-model.sh <preset> [parallel] [thinking]"
  echo
  printf "  %-11s %-6s %-8s %-6s %s\n" preset KV ctx think note
  echo "$PRESETS" | sed '/^$/d' | while read -r id kv ctx th note; do
    printf "  %-11s %-6s %-8s %-6s %s\n" "$id" "$kv" "$ctx" "$th" "$note"
  done
  echo
  echo "  parallel defaults to 1.  thinking: on|off|auto (overrides preset)"
  echo "  examples:"
  echo "    bash ~/start-model.sh q4-96k"
  echo "    bash ~/start-model.sh agent          # fast, no thinking"
  echo "    bash ~/start-model.sh q4-128k 2      # 2 parallel slots"
  echo "    bash ~/start-model.sh q4-96k 1 off   # force thinking off"
  echo
  echo "  Other engines:  bash ~/start-llm.sh   (choose llama.cpp or vLLM)"
}

[[ -z "$P" || "$P" == "-h" || "$P" == "--help" ]] && { show_help; exit 1; }

LINE=$(echo "$PRESETS" | awk -v p="$P" '$1==p {print; exit}')
[[ -z "$LINE" ]] && { echo "Unknown preset: $P"; echo; show_help; exit 1; }

read -r _id KV CTXV THDEF _rest <<<"$LINE"
export KV_QUANT="$KV"
export CTX="$CTXV"
export PARALLEL="$PAR"
export THINKING="${TH:-$THDEF}"

echo "==> preset '$P' -> CTX=$CTX KV=$KV_QUANT PARALLEL=$PARALLEL THINKING=$THINKING"
exec bash "$(cd "$(dirname "$0")" && pwd)/restart-server.sh"
