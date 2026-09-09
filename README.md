# llm-stack

A self-hosted **local + cloud LLM operations stack** for a single-GPU box
(RTX 5080, 16 GB). One web panel and one stable OpenAI-compatible URL; flip
between a local model (llama.cpp / vLLM / Ollama) and cloud providers
(OpenAI, OpenRouter, DeepSeek, Groq, Together, …) without reconfiguring any tool.

This repo is a working demo of that pattern — everything (serving scripts, the
control panel + router, setup scripts, systemd units, config, docs) lives here so
it can be read, changed, and re-deployed as one unit.

```
        your tools / harnesses
                 │  point once at the router:
                 ▼   http://<box>:8001/v1
        ┌──────────────────┐
        │  llm-stack panel │  ← pick the backend in the web UI (:8080)
        │   + router       │
        └───────┬──────────┘
      ┌─────────┴───────────┐
      ▼                     ▼
  LOCAL engine         CLOUD provider
  llama.cpp / vLLM     OpenAI / OpenRouter / DeepSeek / Groq / Together
  / Ollama (GGUF)      (any OpenAI-compatible base URL + key)
```

The router is a minimal, dependency-free version of what
[LiteLLM](https://docs.litellm.ai/docs/) does at scale — a single OpenAI-compatible
endpoint that forwards to local or cloud backends and applies your generation
defaults. Swap in LiteLLM later if you want spend tracking / load balancing.

## Layout
```
llm-stack/
├── bin/llm            # one CLI for everything (status, switch, rescan, bench, panel, …)
├── Makefile           # `make help` — same tasks as targets
├── panel/             # panel.py (web UI + router) and index.html
├── serve/
│   ├── serve-vlm.sh   # the llama.cpp launcher — every runtime knob lives here
│   ├── bench.sh       # llama-bench wrapper for setting sweeps
│   └── serve-vllm.sh  # vLLM launcher
├── setup/             # one-time install scripts (GPU, llama.cpp, vLLM)
├── systemd/           # boot-autostart units (panel + model)
├── scripts/           # kvsweep.sh / kvtest.sh — VRAM & context benchmarks
├── config/
│   ├── presets.json          # THE config: presets, models, engines, endpoints, runtime, benchmarks
│   └── secrets.example.json  # copy to secrets.json (git-ignored) and add cloud keys
└── docs/              # models.md, benchmarking.md, harness-config.md, panel.md, setup.md
```
Large / secret / machine-specific things stay **outside** the repo (git-ignored):
`~/models` (GGUF weights), `~/llama.cpp` (built engine), `~/.vlm_api_key`,
`config/secrets.json`, `logs/`.

## Quickstart
```bash
cd ~/llm-stack
cp config/secrets.example.json config/secrets.json   # then add any cloud keys
./bin/llm panel start                                 # web UI :8080, router :8001
./bin/llm rescan                                      # read every model folder off disk
./bin/llm status                                      # check GPU + model
```
Open `http://<box>:8080`, enter the API key from `~/.vlm_api_key` (top-right), and
switch models / presets / engines from there. Point your harnesses at the router:
`http://<box>:8001/v1` (any model id; the router rewrites it to the active model).

---

# Where the layers actually live

A 16 GB card forces a decision that most inference guides skip: **which layers sit in
VRAM, which sit in system RAM, and which stay on disk.** Every preset in this repo is a
different answer to that, and the two model families here answer it in completely
different ways.

## The GGUF is a stack of numbered blocks

`blk.0` … `blk.N-1` are the transformer layers. Around them sit `token_embd` (the input
embedding table) and `output` / `output_norm` (the head that turns the last hidden state
back into token logits). `-ngl N` says how many of those blocks go on the GPU — `-ngl 99`
means "all of them", and the embedding/output tensors count as one more layer on top.

Two things in a model directory are **not** part of that stack:

| file | what it is | flag |
|---|---|---|
| `mmproj-*.gguf` | vision projector — a separate CLIP-style encoder that turns image patches into tokens the LM can read | `--mmproj` |
| `mtp-*.gguf` | a standalone speculative draft head | `-md` |

Neither is detected by filename in this stack. `general.architecture = clip` (or
`general.type = mmproj`) identifies a projector; the draft heads are identified by
architecture (`eagle3`, `dflash`, `dspark`) or by carrying `blk.<n>.nextn.*` tensors.
Rename them however you like — rescan reads the header.

## Model 1 — Qwen3.8-27B: dense hybrid, everything in VRAM

```
arch qwen35 · 65 blocks · embd 5120

  blk.0 … blk.63   the trunk. NOT uniform: some blocks are attention, the rest are
                   SSM / linear-attention (gated delta-net) blocks with ssm_conv1d,
                   ssm_a, ssm_alpha/beta, ssm_out instead of a KV-cached attention.
                   That hybrid layout is why KV cost per 1k tokens is so low here.
  blk.64           the MTP / NextN speculative head — a FULL extra block (attn + FFN)
                   plus the nextn projections. ~0.35 GB.
  token_embd, output, output_norm
```

This one fits entirely on the GPU (`-ngl 99`), so the only questions are context length
and KV precision. **Block 64 is the interesting part.** It is only loaded when you ask
for speculative decoding: `common.cpp` sets `mparams.load_mtp` from the `--spec-type`,
and `models/qwen35.cpp` then marks every one of block 64's 15 tensors `TENSOR_SKIP` when
that is false. So the same GGUF costs ~0.35 GB more with MTP on than with it off, on top
of the draft context's own KV cache.

And **the same quant ships both with and without block 64**, with nothing in the filename
to say which — `GSQ-RCO-IQ3_XXS` has 64 blocks, `GSQ-RCO-IQ3_XXS-mtp` has 65, and they
can sit in the same folder. Asking a 64-block build for `--spec-type draft-mtp` is a hard
load failure, not a slow path, which is why the panel reads the tensor table instead of
guessing.

## Model 2 — Qwen3.8-Flash-Next: sparse MoE, mostly *not* in VRAM

```
arch qwen4exp · 48 blocks · embd 2560 · 512 experts, 10 used per token · 82 GB on disk

  per block:  attention  ~small, always on GPU
              router     ~tiny, picks 10 of 512 experts
              512 experts  ~huge, and only 10 of them run per token
  plus:       per_layer_token_embd — a ~27 GB engram table
```

82 GB of weights against 16 GB of VRAM **and 30 GB of RAM**. It runs anyway because the
parts are used unevenly:

- **`--n-cpu-moe N`** keeps the expert weights of the first N blocks in system RAM,
  leaving attention and the router on the GPU. Since only 10 of 512 experts fire per
  token, the CPU does far less work than the size suggests. This is *the* tuning dial:
  40 is the floor that loads on this box, 36 hard-OOMs.
- **`--lazy-mode on`** reads rows of oversized tensors from disk on demand instead of
  keeping them resident — which is the only reason the 27 GB engram table is survivable.
- **`--load-mode mmap`** pages weights in from the file rather than reading them up front.
  This replaces `--mmap` / `--no-mmap` / `--mlock`, all now deprecated.
- **`-ngl auto`** (i.e. omit `-ngl`) so `--fit` sizes the split. Pinning `-ngl` on a model
  this size makes llama.cpp abort the fit and then OOM on the compute buffers.

Two consequences worth knowing before you tune it:

- **It is effectively single-context.** Concurrent decoding against a streamed model
  shares one expert cache and can corrupt output, so slots should be 1. The launcher
  warns when they are not.
- **Its speed depends on the page cache**, which makes short benchmarks lie — see below.

## What that means for a preset

| | 27B dense | Flash-Next MoE |
|---|---|---|
| `ngl` | `99` — all blocks on GPU | `auto` — let `--fit` decide |
| the dial that matters | context × KV precision | `n_cpu_moe` |
| MTP | block 64, in the file, +0.35 GB | needs a separate draft file (none load on this build) |
| vision | `mmproj` if the folder has one | none |
| slots | safe above 1 | keep at 1 |
| bottleneck | VRAM bandwidth | disk + CPU |

The panel refuses to hand one family's knobs to the other: `n_cpu_moe` / `lazy_mode` are
dropped for dense models, `--fit` flags are dropped when `ngl` is pinned, and `spec_type`
is forced off for any GGUF without a head.

---

# Benchmarking

There are **two** benchmark engines in the panel, because one tool cannot answer both
questions. Full detail in **[docs/benchmarking.md](docs/benchmarking.md)**.

## Sweep — `llama-bench`, for everything except MTP

**Benchmark → Run sweep**, or:

```bash
llm bench Qwen3.8-Flash-Next-UD-IQ3_XXS NCMOE=40,48 LOAD_MODE=mmap LAZY_MODE=on \
    NGL=auto FIT_TARGET=1536 PROMPT=512 GEN=64
```

Most `llama-bench` flags take a **list** and it runs the cross product, so settling
"40 or 48 experts on CPU" is one job with two rows instead of two server restarts and a
stopwatch. Results are stored in `presets.json` and the table hides every column that
does not vary.

Two traps, both of which bit while this was being built:

- **`llama-bench` does not fit by default.** `llama-server` defaults to `--fit on`;
  `llama-bench` defaults `--fit-target` to *off*. A model bigger than VRAM therefore
  fails to load outright unless you pass `FIT_TARGET` **and** leave `NGL` at `auto`.
- **`KV=q4_0,q8_0` is four combinations, not two.** It sets `-ctk` and `-ctv`
  independently and llama-bench crosses them, so the table shows both columns.

## Server benchmark — the only one that sees MTP

**Benchmark → Run server benchmark**

`llama-bench` has no `--spec-type` and no `-md`, so it is blind to speculative decoding —
the single biggest performance lever on this box. This mode starts a real preset, waits
for it to load, sends real completions, reads llama.cpp's own `timings`, then moves on.
With no variants given it compares the preset at **MTP off / n=2 / n=3**.

Measured on `rco-mtp-q8-110k` (GSQ-RCO IQ3_XXS+mtp, 113k ctx, q8_0 KV), code prompt,
256 generated tokens:

| variant | decode | prefill |
|---|---|---|
| MTP off | 56.9 tok/s | 174.2 tok/s |
| MTP n=2 | 73.2 | 154.9 |
| **MTP n=3** | **74.9** | 155.8 |

MTP is worth **+32%** on decode here and costs ~11% of prefill.

## The trap that makes short benchmarks lie

For a model larger than RAM, a short run measures **the page cache, not the setting** —
and whichever variant runs first is cold and loses. Both engines were pointed at the same
question on Flash-Next (82 GB against 30 GB of RAM):

| | n-cpu-moe 40 | n-cpu-moe 48 |
|---|---|---|
| `llama-bench`, 1 repetition | 7.22 tok/s | **7.39** |
| server benchmark, real requests | **13.00** | 12.06 |

Opposite rankings. The server result is the correct one and matches earlier measurements
(16.9 vs 11.2 warm). The sweep was not wrong about its own numbers — it was answering a
different question. The panel now detects this case (model > 80% of RAM) and prints the
warning above the results by itself.

## Run for real

**Benchmark → Run for real →** takes the first value of each list in the sweep form and
starts the actual `llama-server` with it, then jumps to **Command & log**. `llama-bench`
is a different binary with a different loader; a combination is not proven until the thing
that actually serves requests has run it.

---

## Keeping the config honest: rescan

Nothing about a model directory should be typed by hand — projectors get dropped in later,
a second build lands beside the first, folders get moved.

**Models → Rescan models**, or `llm rescan` (`llm rescan-dry` to preview).

It re-reads every folder and writes back: which GGUF is the main build, whether it embeds
an MTP head, whether a projector is present (which is what makes the vision toggle live),
any usable draft heads — and any *unusable* ones, with the reason. New folders are
registered automatically, **one entry per build**, so a plain and a `-mtp` GGUF in one
folder become two models sharing a `dir` and differing by a `gguf` key.

The one field rescan never touches is `supports_mtp`: that is the deliberate override.

## Panel views

| view | what it is for |
|---|---|
| Overview | what is serving, GPU/CPU load, tok/s |
| Models & endpoints | rescan, local models, cloud endpoints |
| Presets | context / KV / slots / MTP per preset, with a VRAM estimate |
| Runtime | global llama.cpp knobs and router generation defaults |
| Server log | live tail |
| **Command & log** | the running process's real argv from `/proc`, one flag per line, plus the launcher environment — where a preset card and reality get compared |
| **Benchmark** | both benchmark engines, and run-for-real |
| Power | uptime, GPU power limit, shutdown/restart |
| **CC Switch** | provider profiles for claude / codex / gemini — add, switch, delete, usage logs (data from `~/.cc-switch/cc-switch.db`) |
| **CC Connect** | chat-bridge service status, web-admin link, restart, journal log (systemd user unit + its local API on :9820) |

> The panel reuses two existing tools rather than reimplementing them:
> [CC Switch](https://github.com/farion1231/cc-switch) (provider profiles, stored in its SQLite DB)
> and [cc-connect](https://github.com/chenhg5/cc-connect) (the Telegram/chat bridge daemon).
> The panel only adds a thin read/write layer over their state — no new ports or services.

## Everyday commands
| Command | Does |
|---|---|
| `make status` / `./bin/llm status` | GPU, running model, engine, health |
| `./bin/llm rescan` / `rescan-dry` | re-detect model folders (heads, projectors, sizes) |
| `./bin/llm bench <model> K=V` | llama-bench sweep |
| `./bin/llm bench-status` | running sweep + last results |
| `make switch PRESET=…` | restart local model into a preset |
| `make engine ENGINE=vllm` | switch inference engine |
| `make models` / `make presets` | list what is configured |
| `make restart` / `stop` / `logs` | panel lifecycle + model log |
| `make install-service` | boot-autostart via systemd (sudo) |
| `make firewall` | print the ufw commands for LAN access |
| `make kvsweep` | benchmark max context per KV quant |

## Choosing / adding a model
Drop a new GGUF folder into `~/models/<name>/`, press **Rescan models**, then pick it in a
preset's ✏️ editor. See **[docs/models.md](docs/models.md)** for the quant/context
trade-offs on 16 GB and **[docs/benchmarking.md](docs/benchmarking.md)** for how to settle
them with numbers instead of guesses.

## Setup from scratch
See **[docs/setup.md](docs/setup.md)** and the `setup/` scripts. Only ONE GPU engine
runs at a time on 16 GB.
