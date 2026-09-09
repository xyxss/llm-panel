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
# Binary: prefer the native-sm_120 build (Blackwell kernels + working MTP) when
# it exists; fall back to the original PTX-JIT build. Override with LLAMA=...
LLAMA="${LLAMA:-$([ -x "$HOME/llama.cpp-sm120/build/bin/llama-server" ] \
        && echo "$HOME/llama.cpp-sm120/build/bin/llama-server" \
        || echo "$HOME/llama.cpp/build/bin/llama-server")}"
# That build has no $ORIGIN rpath and links the rootless CUDA 12.9 toolkit.
export LD_LIBRARY_PATH="$(dirname "$LLAMA"):$HOME/cuda129/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
MDIR="${MDIR:-$HOME/models/Qwen3.8-27B}"   # override with MDIR=... for another model dir

# ---- CORE TUNING ----------------------------------------------------------
CTX="${CTX:-98304}"        # context window (total KV tokens). Bigger = more VRAM.
PARALLEL="${PARALLEL:-}"   # server slots. EMPTY = auto (llama.cpp picks, and enables a
                           #   shared KV pool). A number reserves CTX/PARALLEL per slot.
NGL="${NGL:-99}"           # layers on GPU (99 = all, 'auto' = let --fit decide).
                           #   Lower only on VRAM OOM. Use 'auto' for models that
                           #   need --fit (see NGL_ARG below).
ALIAS="${ALIAS:-qwen3-vl}" # model name the server advertises (router rewrites to it)
MDRAFT="${MDRAFT:-}"       # external draft model for --spec-type draft-mtp. Empty for
                           #   Qwen3.8-27B (its NextN head is inside the main GGUF);
                           #   Qwen3.8-Flash-Next needs a separate non-shared mtp-*.gguf.
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

# ---- SPECULATIVE DECODING (MTP) -------------------------------------------
# Qwen3.8-27B ships a NextN/MTP draft head inside every GGUF (blk.64.nextn.*),
# so no separate draft model is needed. Measured on the RTX 5080 @ IQ3_XXS:
#   off -> 53 tok/s | n-max 2 -> 76 prose / 88 code | n-max 3 -> 70 prose / 94 code
# Requires the sm120 build; the old b10736 build ASSERTS in graph_mtp and dies.
# Set SPEC_TYPE=none to disable. n-max 2 = best all-round, 3 = better for code.
SPEC_TYPE="${SPEC_TYPE:-draft-mtp}"
SPEC_N_MAX="${SPEC_N_MAX:-2}"

# ---- PERFORMANCE ----------------------------------------------------------
BATCH="${BATCH:-2048}"     # logical batch (prompt processing throughput)
UBATCH="${UBATCH:-512}"    # physical batch. Raising to 1024 can speed prompt
                           #   eval (more VRAM). Lower if tight.
THREADS="${THREADS:--1}"   # CPU threads, -1 = auto
FIT="${FIT:-}"             # --fit on|off: let llama.cpp auto-shrink UNSET args to
                           #   fit VRAM. Empty = don't pass it (use explicit ctx).
# --- disk-streaming models (Qwen3.8-Flash-Next) --------------------------------
NCMOE="${NCMOE:-}"         # --n-cpu-moe N: keep the MoE expert weights of the first
                           #   N layers on CPU. Empty = don't pass it (dense models,
                           #   or MoE models small enough to sit entirely in VRAM).
LAZY_MODE="${LAZY_MODE:-}" # --lazy-mode on|auto|off: read rows of oversized tensors
                           #   from disk on demand instead of keeping them resident.
                           #   llama.cpp's own default is 'auto' = on for tensors
                           #   >4 GiB, which already covers Flash-Next's 26.8 GiB
                           #   per_layer_token_embd. Empty = leave at that default.
CPU_MOE="${CPU_MOE:-}"     # --cpu-moe: keep ALL MoE expert weights on CPU. The all-or-
                           #   nothing form of NCMOE; set one or the other, never both.
LOAD_MODE="${LOAD_MODE:-}" # --load-mode: auto | none | mmap | mlock | mmap+mlock | dio.
                           #   This is where RAM-vs-disk residency is decided, and it
                           #   REPLACES --mmap/--no-mmap/--mlock (both now deprecated):
                           #     mmap  = page the weights in from the file on demand, so a
                           #             model bigger than RAM streams off NVMe (the only
                           #             way Flash-Next's 76 GiB runs on a 64 GB box)
                           #     mlock = pin in RAM, never swap. Needs the model to FIT.
                           #     dio   = DirectIO, bypasses the page cache.
                           #   Empty = llama.cpp's 'auto' (mmap where supported).
FIT_TARGET="${FIT_TARGET:-}"  # --fit-target MiB: VRAM margin --fit leaves free per device
                           #   (llama.cpp default 1024). Raise it when the fit lands too
                           #   close to the edge and long prompts OOM at peak.
FIT_CTX="${FIT_CTX:-}"     # --fit-ctx N: floor on the context --fit is allowed to pick.
NUMA="${NUMA:-}"           # --numa distribute|isolate|numactl. Only helps multi-socket.
OVERRIDE_TENSOR="${OVERRIDE_TENSOR:-}"  # -ot '<pattern>=<buffer>': hand-place tensors,
                           #   e.g. '.ffn_.*_exps.=CPU'. Finer than NCMOE.
NO_HOST="${NO_HOST:-}"     # --no-host: bypass the host buffer so extra buffer types can
                           #   be used. Only set if you know you want it.
THREADS_BATCH="${THREADS_BATCH:-}"  # -tb: threads for prompt processing. Matters once the
                           #   experts live on CPU, where prefill is CPU-bound.
SPEC_DRAFT_KV="${SPEC_DRAFT_KV:-}"  # -ctkd/-ctvd: KV cache type for the DRAFT context.
                           #   The MTP draft carries its own KV; quantizing it shrinks the
                           #   per-ctx half of MTP's cost without touching target quality.
SPEC_N_MIN="${SPEC_N_MIN:-}"        # --spec-draft-n-min: floor on tokens drafted per step.
EXTRA_FLAGS="${EXTRA_FLAGS:-}"  # any extra llama-server flags, appended verbatim.
                           #   e.g. EXTRA_FLAGS='--slots --metrics -ot ".ffn_.*_exps.=CPU"'
KV_UNIFIED="${KV_UNIFIED:-}"    # on = all slots SHARE one KV pool (one request can use the
                           #      whole context); off = each slot reserves ctx/parallel.
                           #      empty = do not pass the flag.
KV_UNIFIED_PER_SLOT="${KV_UNIFIED_PER_SLOT:-}"  # optional per-slot cap inside a shared pool
FLASH_ATTN="${FLASH_ATTN:-on}"  # -fa: on | off | auto. FlashAttention fuses the attention
                           #   kernels — faster prompt eval + less VRAM. Required for
                           #   quantized KV cache (q8_0/q4_0). Turn off only to compare.
# ---------------------------------------------------------------------------

# MODEL_FILE names WHICH gguf in MDIR to load. It is not optional bookkeeping: a single
# repo can ship several, and ISTA-DASLab's GSQ-RCO dirs hold the plain build and the
# -mtp build side by side. Picking "the first one" then depends on collation — shell sort
# and Python's sorted() disagree on "..._S.gguf" vs "..._S-mtp.gguf" — so the panel and
# this script could load DIFFERENT files while both believed they agreed. Always pass it.
MODEL_FILE="${MODEL_FILE:-}"
if [ -n "$MODEL_FILE" ]; then
    case "$MODEL_FILE" in /*) MODEL="$MODEL_FILE";; *) MODEL="$MDIR/$MODEL_FILE";; esac
    [ -f "$MODEL" ] || { echo "MODEL_FILE '$MODEL_FILE' not found in $MDIR"; exit 1; }
else
    # NOTE: mtp* must be excluded here. Shell `sort` is locale-collated and
    # case-insensitive, so "mtp-Q8_0.gguf" sorts BEFORE "Qwen3.8-...gguf" and the
    # draft head would be loaded as the main model (Qwen3.8-Flash-Next ships both
    # in one directory). For split models this correctly picks -00001-of-000NN,
    # which is what llama.cpp wants -- it loads the rest itself. LC_ALL=C makes the
    # order reproducible instead of locale-dependent.
    MODEL="$(find "$MDIR" -iname '*.gguf' ! -iname 'mmproj*' ! -iname 'mtp-*' 2>/dev/null \
             | LC_ALL=C sort | head -1)"
    N_CAND="$(find "$MDIR" -iname '*.gguf' ! -iname 'mmproj*' ! -iname 'mtp-*' \
                           ! -iname '*-of-*' 2>/dev/null | wc -l)"
    if [ "$N_CAND" -gt 1 ]; then
        echo "WARN: $MDIR holds $N_CAND candidate GGUFs and MODEL_FILE is unset —" \
             "picking $(basename "$MODEL") by name order. Set MODEL_FILE (or the" \
             "preset's 'gguf' key) to say which build you meant." >&2
    fi
fi
VISION="${VISION:-on}"   # off = skip the mmproj projector entirely. The F16 projector
                         #   costs ~0.9 GB of VRAM; extreme-context presets cannot use
                         #   vision anyway, so dropping it buys back real headroom.
# MMPROJ_FILE names the projector explicitly. The panel fills it in from the GGUF header
# (general.architecture=clip / general.type=mmproj), so a projector still counts as one
# after being renamed or downloaded separately -- the name glob below is only a fallback.
MMPROJ_FILE="${MMPROJ_FILE:-}"
if [ -n "$MMPROJ_FILE" ]; then
    case "$MMPROJ_FILE" in /*) MMPROJ="$MMPROJ_FILE";; *) MMPROJ="$MDIR/$MMPROJ_FILE";; esac
    [ -f "$MMPROJ" ] || { echo "WARN: MMPROJ_FILE '$MMPROJ_FILE' not found — vision off." >&2; MMPROJ=""; }
else
    MMPROJ="$(find "$MDIR" -iname 'mmproj*.gguf' 2>/dev/null | LC_ALL=C sort | head -1)"
fi
[ "$VISION" = "off" ] && MMPROJ=""

[ -x "$LLAMA" ] || { echo "llama-server not built at $LLAMA"; exit 1; }
[ -n "$MODEL" ] || { echo "no model .gguf in $MDIR"; exit 1; }

# With no external draft, MTP can only come from a NextN head EMBEDDED in the main GGUF
# (blk.<n>.nextn.*). Do not guess that from the file name: the same quant of Qwen3.8-27B
# ships both ways -- GSQ-RCO-IQ3_XXS has no head, GSQ-RCO-IQ3_XXS-mtp does, and the
# Unsloth UD builds carry one with nothing in the name to say so. Asking for
# --spec-type draft-mtp without a head is a hard load failure, so read the tensor table.
# Bounded on purpose: the tensor table always precedes the tensor DATA, so 256 MB covers
# any plausible metadata block (the biggest tokenizer here lands well inside 20 MB) while
# a plain grep over a head-less 10 GB file would read all of it before answering no. tr
# breaks the binary into short lines so grep never buffers a huge one.
gguf_has_nextn() {
    # grep -q exits on the first match, which SIGPIPEs head and tr. Under this script's
    # `set -o pipefail` that 141 becomes the pipeline's status and a successful detection
    # would read as a failure, so scope pipefail off for the duration.
    local rc
    set +o pipefail
    head -c 268435456 "$1" 2>/dev/null \
        | LC_ALL=C tr -c '[:print:]' '\n' \
        | LC_ALL=C grep -qa '\.nextn\.'
    rc=$?
    set -o pipefail
    return $rc
}
# Where the draft head comes from. MTP_HEAD picks it explicitly; MDRAFT is the resolved
# path that reaches -md.
#   embedded  — use the NextN head inside the main GGUF, never a file (default for the 27B)
#   auto      — embedded if the GGUF has one, else the best mtp-*.gguf beside it
#   <name>    — a file name inside MDIR, or an absolute path, used verbatim
#   none      — no external head even if one is lying in the directory
MTP_HEAD="${MTP_HEAD:-auto}"
if [ "${SPEC_TYPE:-none}" != "none" ] && [ -z "$MDRAFT" ]; then
    case "$MTP_HEAD" in
        none|embedded) ;;                      # never look for a file
        auto|"")
            # IMPORTANT: only the NON-shared heads work. mtp-shared-*.gguf omits
            # token_embd.weight by design and the draft loader rejects it with
            # "check_tensor_dims: tensor 'token_embd.weight' not found". Prefer the
            # embedded head when the GGUF has one -- it is smaller and always matches.
            if ! gguf_has_nextn "$MODEL"; then
                for c in "$MDIR"/mtp-Q8_0.gguf "$MDIR"/mtp-Q4_K_M.gguf; do
                    [ -f "$c" ] && { MDRAFT="$c"; break; }
                done
                if [ -z "$MDRAFT" ]; then
                    MDRAFT="$(find "$MDIR" -maxdepth 1 -iname 'mtp-*.gguf' \
                                           ! -iname 'mtp-shared-*' 2>/dev/null | sort | head -1)"
                fi
            fi
            ;;
        /*) MDRAFT="$MTP_HEAD" ;;              # absolute path
        *)  MDRAFT="$MDIR/$MTP_HEAD" ;;        # a file name inside the model dir
    esac
fi
# Degrade gracefully rather than dying: a preset may ask for MTP before the draft head has
# finished downloading, and a headless box should still come up serving. The floor is 64 MB
# only to catch a truncated download -- do NOT raise it to the size of the heads that
# happen to live here, because a legitimate small draft model is a perfectly valid -md.
if [ "${SPEC_TYPE:-none}" != "none" ] && [ -n "$MDRAFT" ]; then
    if [ ! -f "$MDRAFT" ] || [ "$(stat -c %s "$MDRAFT" 2>/dev/null || echo 0)" -lt 67108864 ]; then
        echo "WARN: MTP draft '$MDRAFT' missing or incomplete" \
             "${MTP_HEAD:+(MTP_HEAD=$MTP_HEAD)} — falling back to the embedded head." >&2
        MDRAFT=""
    fi
fi
if [ "${SPEC_TYPE:-none}" != "none" ] && [ -z "$MDRAFT" ]; then
    if gguf_has_nextn "$MODEL"; then
        MTP_SOURCE="embedded NextN head"
    else
        echo "WARN: $(basename "$MODEL") has no embedded NextN head and no mtp-*.gguf" \
             "draft beside it — starting WITHOUT MTP." >&2
        SPEC_TYPE="none"
    fi
fi

# A disk-streamed model keeps ONE shared expert cache; decoding several contexts against
# it concurrently can corrupt output (llama.cpp discussion #25294). Slots are only safe
# here when the weights are resident.
if [ "${LAZY_MODE:-}" = "on" ] || [ -n "${NCMOE:-}" ] || [ "${CPU_MOE:-}" = "on" ]; then
    if [ -z "$PARALLEL" ] || [ "$PARALLEL" -gt 1 ] 2>/dev/null; then
        echo "WARN: this model streams experts (lazy-mode/n-cpu-moe) and PARALLEL is" \
             "'${PARALLEL:-auto (=4 slots)}'. Concurrent contexts share one expert cache" \
             "and can corrupt output — use PARALLEL=1 unless you have verified otherwise." >&2
    fi
fi

echo "Model    : $MODEL"
echo "MMProj   : ${MMPROJ:-<none — vision disabled>}"
if [ "${SPEC_TYPE:-none}" != "none" ]; then
    echo "MTP      : $SPEC_TYPE n-max=$SPEC_N_MAX (${MDRAFT:-${MTP_SOURCE:-embedded NextN head}})"
    echo "           the head is loaded ONLY because MTP is on — it costs ~0.35 GB of"
    echo "           weights on top of its draft KV. SPEC_TYPE=none frees both."
else
    echo "MTP      : off (any NextN head in the GGUF is skipped at load)"
fi
echo "Context  : $CTX   Parallel: $PARALLEL   GPU layers: $NGL   KV: $KV_QUANT"
echo "Residency: load-mode=${LOAD_MODE:-auto} lazy=${LAZY_MODE:-auto} ncmoe=${NCMOE:-none}" \
     "cpu-moe=${CPU_MOE:-off} fit=${FIT:-default}${FIT_TARGET:+ target=${FIT_TARGET}MiB}" \
     "threads=$THREADS${THREADS_BATCH:+/$THREADS_BATCH}"
echo "Thinking : $THINKING (effort=$REASON_EFFORT, format=$REASON_FORMAT)"
echo "Sampling : temp=$TEMP top_p=$TOP_P top_k=$TOP_K min_p=$MIN_P presence=$PRESENCE repeat=$REPEAT_PEN"
echo "Context  : shift=$CONTEXT_SHIFT cache_reuse=$CACHE_REUSE"

# Build optional flags conditionally
SHIFT_FLAG="--no-context-shift"; [[ "$CONTEXT_SHIFT" == "on" ]] && SHIFT_FLAG="--context-shift"

# NGL=auto omits -ngl entirely. Required for models that rely on --fit: setting
# n_gpu_layers explicitly makes llama.cpp ABORT the fit ("n_gpu_layers already
# set by user"), after which the compute buffers OOM on a 16 GB card.
NGL_ARG="-ngl $NGL"; [[ "$NGL" == "auto" ]] && NGL_ARG=""

exec "$LLAMA" \
    -m "$MODEL" \
    ${MMPROJ:+--mmproj "$MMPROJ"} \
    --alias "$ALIAS" \
    --jinja \
    --reasoning "$THINKING" \
    --reasoning-effort "$REASON_EFFORT" \
    --reasoning-format "$REASON_FORMAT" \
    $NGL_ARG \
    -c "$CTX" \
    ${PARALLEL:+--parallel "$PARALLEL"} \
    -fa "$FLASH_ATTN" \
    --cache-type-k "$KV_QUANT" \
    --cache-type-v "$KV_QUANT" \
    $SHIFT_FLAG \
    ${KV_UNIFIED:+$([ "$KV_UNIFIED" = "on" ] && echo --kv-unified || echo --no-kv-unified)} \
    ${KV_UNIFIED_PER_SLOT:+--kv-unified-per-slot "$KV_UNIFIED_PER_SLOT"} \
    --cache-reuse "$CACHE_REUSE" \
    --temp "$TEMP" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --min-p "$MIN_P" \
    --presence-penalty "$PRESENCE" \
    --repeat-penalty "$REPEAT_PEN" \
    ${IMAGE_MIN_TOKENS:+--image-min-tokens "$IMAGE_MIN_TOKENS"} \
    ${IMAGE_MAX_TOKENS:+--image-max-tokens "$IMAGE_MAX_TOKENS"} \
    ${NCMOE:+--n-cpu-moe "$NCMOE"} \
    ${CPU_MOE:+$([ "$CPU_MOE" = "on" ] && echo --cpu-moe)} \
    ${LAZY_MODE:+--lazy-mode "$LAZY_MODE"} \
    ${LOAD_MODE:+--load-mode "$LOAD_MODE"} \
    ${OVERRIDE_TENSOR:+-ot "$OVERRIDE_TENSOR"} \
    ${NUMA:+--numa "$NUMA"} \
    ${NO_HOST:+$([ "$NO_HOST" = "on" ] && echo --no-host)} \
    ${MDRAFT:+$([ "${SPEC_TYPE:-none}" != "none" ] && echo -md "$MDRAFT")} \
    ${SPEC_TYPE:+$([ "$SPEC_TYPE" != "none" ] && echo --spec-type "$SPEC_TYPE" --spec-draft-n-max "$SPEC_N_MAX")} \
    ${SPEC_N_MIN:+$([ "${SPEC_TYPE:-none}" != "none" ] && echo --spec-draft-n-min "$SPEC_N_MIN")} \
    ${SPEC_DRAFT_KV:+$([ "${SPEC_TYPE:-none}" != "none" ] && echo --spec-draft-type-k "$SPEC_DRAFT_KV" --spec-draft-type-v "$SPEC_DRAFT_KV")} \
    -b "$BATCH" \
    -ub "$UBATCH" \
    -t "$THREADS" \
    ${THREADS_BATCH:+-tb "$THREADS_BATCH"} \
    ${FIT:+--fit "$FIT"} \
    ${FIT_TARGET:+--fit-target "$FIT_TARGET"} \
    ${FIT_CTX:+--fit-ctx "$FIT_CTX"} \
    ${API_KEY:+--api-key "$API_KEY"} \
    --host 0.0.0.0 --port "$PORT" \
    $EXTRA_FLAGS
