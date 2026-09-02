#!/usr/bin/env bash
# restart the VLM server cleanly (safe: matches the binary path, not this script)
HERE="$(cd "$(dirname "$0")" && pwd)"
BIN="$HOME/llama.cpp/build/bin/llama-server"
pkill -f "$BIN" 2>/dev/null; sleep 3
setsid bash "$HERE/serve-vlm.sh" > "$HOME/server.log" 2>&1 &
for i in $(seq 1 40); do
  grep -qiE 'listening on|srv.*model loaded' "$HOME/server.log" && { echo "READY"; break; }
  grep -qiE 'out of memory|failed to alloc|error loading' "$HOME/server.log" && { echo "ERROR"; break; }
  sleep 3
done
tail -3 "$HOME/server.log"
