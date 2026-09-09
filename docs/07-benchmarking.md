# Benchmarking — every test, with numbers

There are two benchmark engines in the panel because there are two questions, and one
tool cannot answer both.

## 1. Sweep (llama-bench) — everything except MTP

**Benchmark → Run sweep**, or:

```bash
llm bench <model-id> KEY=VALUE ...
```

Most `llama-bench` flags take a **list** and it runs the cross product, so settling
"40 or 48 experts on CPU" is one job with two rows, not two server restarts and a
stopwatch:

```bash
llm bench Qwen3.8-Flash-Next-UD-IQ3_XXS NCMOE=40,48 LOAD_MODE=mmap LAZY_MODE=on \
    NGL=auto FIT_TARGET=1536 PROMPT=512 GEN=64
```

Two traps, both of which bit during setup:

- **llama-bench does not fit by default.** llama-server has `--fit on` as its default;
  llama-bench has `--fit-target off`. A model bigger than VRAM therefore fails to load
  outright unless you pass FIT_TARGET **and** leave NGL at auto. Pinning `-ngl 99`
  on an 82 GB model just errors.
- **KV type is a cross product.** KV=q4_0,q8_0 sets both -ctk and -ctv to that list,
  which is *four* combinations, not two. The results table shows type_k and type_v as
  separate columns so the rows are not mistaken for duplicates.

EXTRA is appended verbatim for anything the form does not name.

## 2. Server benchmark — the MTP one

**Benchmark → Run server benchmark**

llama-bench has no --spec-type and no -md, so it is blind to speculative decoding —
the single biggest performance lever on this box. This mode starts a real preset, waits
for it to load, sends real completions, reads llama.cpp's own timings, then moves to the
next variant. With no variants given it compares the preset at **MTP off / n=2 / n=3**.

Measured on `rco-mtp-q8-110k` (GSQ-RCO IQ3_XXS+mtp, 113k ctx, q8_0 KV), code prompt,
256 generated tokens:

| variant | decode tok/s | prefill tok/s |
|---|---|---|
| MTP off | 56.9 | 174.2 |
| MTP n=2 | 73.2 | 154.9 |
| MTP n=3 | **74.9** | 155.8 |

So MTP is worth **+32%** on decode here and costs ~11% of prefill — and n=3 edges n=2 on
code, matching what the 27B UD build showed.

## 3. Run for real

**Benchmark → Run for real** takes the first value of each list in the sweep form and
starts the actual llama-server with it, then jumps to **Command & log**.

This exists because llama-bench is a *different binary with a different loader*. A
combination that benches well is not proven until the thing that actually serves requests
has run it. The Command & log view shows the process's real argv straight from /proc,
one flag per line, plus the launcher environment — which is where any discrepancy between
what a preset card says and what the server is doing will always show up.

## 4. KV sweep (scripts/kvsweep.sh)

Finds the maximum context per KV quant on this GPU by binary-searching the context
window until OOM:

```bash
bash scripts/kvsweep.sh
```

This produced the measured coefficients in 05-vram-estimator.md.

## 5. All-preset sweep (stored in config/presets.json → benchmarks)

The full all-presets-sweep results are stored in `config/presets.json` under
`benchmarks.all-presets-sweep`. Each entry records: preset id, ctx, kv, mtp, tg256
tok/s, pp tok/s, and the date it was run. Use these as ground truth when comparing
new runs.

## 6. MTP depth sweep (stored in config/presets.json → benchmarks)

The MTP depth comparison (n=0, n=1, n=2, n=3, n=4) on both prose and code prompts
is stored under `benchmarks.mtp-depth`. Key finding: **n=2 is best all-round**;
n=3 favours code; n=4 was worse on both.

## 7. KV type smoke test (stored in config/presets.json → benchmarks)

A quick f16 / q8_0 / q4_0 comparison at a fixed context to verify the KV slope
coefficients. Confirmed: q8_0 is ~1.7x cheaper than f16 per token, q4_0 is ~2x
cheaper than q8_0, with no visible quality loss for agent workloads.
