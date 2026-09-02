#!/usr/bin/env bash
BIN="$HOME/llama.cpp/build/bin/llama-server"

run_test() {
  local ctx=$1 kv=$2 log=$3
  pkill -f "$BIN" 2>/dev/null; sleep 3
  CTX=$ctx KV_QUANT=$kv setsid bash "$HOME/serve-vlm.sh" > "$log" 2>&1 &
  local r="timeout"
  for i in $(seq 1 40); do
    if grep -qiE 'listening on|srv.*model loaded' "$log"; then r="LOADED_OK"; break; fi
    if grep -qiE 'out of memory|failed to allocate|cudaMalloc|error loading model|terminate|GGML_ASSERT|std::bad_alloc' "$log"; then r="OOM_ERROR"; break; fi
    sleep 4
  done
  local vram
  vram=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | tr -d ' ')
  pkill -f "$BIN" 2>/dev/null; sleep 3
  echo "=================================================="
  echo "  CTX=$ctx  KV=$kv  ->  $r"
  echo "  VRAM peak used/total (MiB): $vram"
  grep -iE 'KV cache|kv_cache|kv self size|out of memory|failed to alloc|n_ctx' "$log" | tail -5
}

run_test 65536 q8_0 /home/liuyang/test-q8.log
run_test 65536 q4_0 /home/liuyang/test-q4.log
echo "=================================================="
echo "done"
