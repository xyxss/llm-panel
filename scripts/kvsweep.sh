#!/usr/bin/env bash
# Find max context that LOADS for a given KV quant on this GPU.
BIN="$HOME/llama.cpp/build/bin/llama-server"

test_one() {  # ctx kv -> prints result line, returns 0 if loaded, 1 if OOM
  local ctx=$1 kv=$2 log=/tmp/kvsweep.log
  pkill -f "$BIN" 2>/dev/null; sleep 3
  CTX=$ctx KV_QUANT=$kv setsid bash "$HOME/serve-vlm.sh" > "$log" 2>&1 &
  local r=1 status="OOM/timeout"
  for i in $(seq 1 45); do
    if grep -qiE 'listening on|srv.*model loaded' "$log"; then r=0; status="LOADED"; break; fi
    if grep -qiE 'out of memory|failed to allocate|cudaMalloc|error loading model|GGML_ASSERT|std::bad_alloc|terminate called' "$log"; then r=1; status="OOM"; break; fi
    sleep 4
  done
  local vram; vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  pkill -f "$BIN" 2>/dev/null; sleep 2
  printf "  ctx=%-7s %-5s -> %-8s  VRAM used: %s MiB / 16303\n" "$ctx" "$kv" "$status" "$vram"
  return $r
}

for kv in q8_0 q4_0; do
  echo "===== KV = $kv ====="
  for ctx in 98304 131072 163840 196608 262144; do
    test_one "$ctx" "$kv" || { echo "  -> stop (first failure at ctx=$ctx)"; break; }
  done
done
pkill -f "$BIN" 2>/dev/null
echo "done"
