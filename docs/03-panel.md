# LLM Control Panel + Router

A dependency-free (Python stdlib only) web console to operate a local llama.cpp
model **and** cloud models from one page — behind one stable endpoint your tools
point at forever.

One process (`panel/panel.py`) serves both:

| port   | what                                                                      |
| ------ | ------------------------------------------------------------------------- |
| `8080` | the web panel                                                             |
| `8001` | the OpenAI-compatible router — rewrites `model` and injects the upstream key |

## Install (rootless, no sudo)

```bash
git clone <your-fork> ~/llm-stack
cd ~/llm-stack
```

**1. Prerequisites.** Python 3.9+, and a built `llama-server` (path set in
`config/presets.json` → `server.llama_bin`). For GPU monitoring, `nvidia-smi`.

**2. Set the API key.** Every mutating call requires it; without it the panel is
read-only.

```bash
head -c 24 /dev/urandom | xxd -p -c 24 | sed 's/^/sk-/' > ~/.vlm_api_key
chmod 600 ~/.vlm_api_key
```

**3. Point the config at your models.** Edit `config/presets.json`:

- `models` — id → `{dir, label, vision}`. `dir` may use `~` or the `$ROOT` token.
- `server.llama_bin`, `server.serve_script`, `server.server_log`
- `estimator.coeffs_gb` — the VRAM fit for **your** GPU (see 05-vram-estimator.md)

**4. Start it.**

```bash
bin/llm panel start          # or: make start
bin/llm panel status|logs|stop|restart
```

Open `http://<box>:8080`, paste the key from `~/.vlm_api_key` into the sidebar
field, press **save**. It is stored only in that browser.

**5. Autostart at login + boot** (rootless user service):

```bash
bin/llm autostart
```

**6. Optional — allow the panel to power the box off.** Rootless by default means
it cannot; the Power view shows the exact rule and a Copy button. It grants only
power-off and reboot to one user:

```bash
sudo tee /etc/polkit-1/rules.d/49-llm-panel-power.rules >/dev/null <<'EOF'
polkit.addRule(function(action, subject) {
  if ((action.id == "org.freedesktop.login1.power-off" ||
       action.id == "org.freedesktop.login1.reboot") &&
      subject.user == "YOUR_USER") {
    return polkit.Result.YES;
  }
});
EOF
```

**7. Optional — open the ports on the LAN** (sudo, and only if you want other
machines to reach it):

```bash
sudo ufw allow from 192.168.0.0/24 to any port 8080 proto tcp
sudo ufw allow from 192.168.0.0/24 to any port 8001 proto tcp
```

> The API key is the only thing standing between the LAN and full control of this
> box — switching models, stopping the server, running installs, and (if you added
> the polkit rule) powering it off. Treat it like a password, and never commit it.

## The panel, view by view

A sidebar with five destinations. Light and dark follow the OS, with a toggle in
the top bar that overrides and persists per browser.

### Overview

- **Currently serving** — model, preset, engine, KV type, context, slots,
  thinking, context-shift, estimated VRAM, PID. Context shows the per-request
  figure when slots are split (see Context vs slots).
- **GPU memory** — used / total with a bar that turns amber past 90% and red past 96%.
- **System monitor** — GPU util, VRAM, power, temp, clock, fan, CPU util, load,
  RAM, tokens/sec. Polled every 2s.
- **GPU power limit** — slider + Apply. Needs `sudo nvidia-smi -pl`; if it isn't
  permitted the panel hands you the exact command instead of failing silently.
- **Router** — the URL to point every tool at, a copy button, and which backend is
  currently answering.

### Models and endpoints

- **Endpoints** — one card per backend: the local model plus any OpenAI-compatible
  cloud provider. **Activate** switches which one the router forwards to; cloud
  activation warns that requests will leave the machine. `test` checks
  reachability. Keys live in `config/secrets.json` (git-ignored, chmod 600) and are
  never sent to the browser — cards show only `key set` / `no key`.
- **Add endpoint** — id, label, base URL, model id, key, vision flag.
- **Inference engine** — llama.cpp / vLLM / Ollama. Only one may hold the GPU;
  switching stops the other. Installed engines are green; ones with a configured
  setup script also offer **Reinstall**, and uninstalled ones offer **Install**,
  which runs the configured command server-side (the browser only ever sends an
  engine name) in the background, logging to `logs/install-<engine>.log`.
  Reinstalling the engine that is currently serving is refused.
- **Download a model** — paste `org/name`, a full URL, or `org/name-GGUF:QUANT`
  (the `:QUANT` becomes an `--include` glob and pulls any `mmproj*` too). Runs in
  the background with live progress. Job state is persisted, so a panel restart
  re-adopts running downloads instead of orphaning them, and still auto-registers
  the model when it lands.

### Presets

- **Profiles** — your **Default** preset (select + Apply) beside a live **Active**
  card showing what is serving right now, its model, KV/context/slots, measured
  VRAM, and a **Reapply** button.
- **All presets** — one card each: model, note, KV, context, slots, estimated
  VRAM with a colour-coded bar. The running preset is ringed in green with a
  **LIVE** badge and swaps its estimate for *measured* VRAM, refreshed every 2s.
  A star marks your Default. **Drag a card by its grip to reorder**; the order is
  saved.
- **Editing** — pencil opens the editor: id (renameable — profiles follow the
  rename), model, KV, context, slots, GPU layers, thinking, reasoning effort,
  context-shift, cache-reuse, image-min-tokens, tag, note. The VRAM line updates
  as you change model or context, and says plainly whether it fits.
- **Custom preset** — a slider for context, KV precision and slots, with a live
  estimate before you commit.

### Runtime and thinking

llama.cpp knobs, applied as env to `serve-vlm.sh`. They take effect on the next
restart — **Save** stores them, **Save & restart model now** applies immediately.

- **Thinking** auto / on / off and **reasoning effort**
- **Context shift** — off means the server errors when full (correct for agents
  that compact); on silently drops the oldest tokens and corrupts agent state
- **Flash attention** — required for quantized KV
- **KV cache across slots** — shared (one pool, a single request may use the
  whole context) or split (each slot reserves context divided by slots)
- **cache-reuse**, **image-min-tokens**, **extra llama-server flags**
- **Operating settings** — generation defaults the router injects when a client
  omits them (temperature, top_p, top_k, min_p, max_tokens, penalties), an optional
  default system prompt, and an "override client" switch. Leave them blank to let
  the server's own thinking-aware sampling profile win.

### Server log

Live tail of the llama.cpp log, **newest line first**, pinned to the top.

### Power

Host uptime, model-server state, and **Shut down** / **Restart** — which stop the
model first, then run after a 60-second grace period with a Cancel banner visible
from any view. If the panel lacks permission it says so and shows the one-time
polkit rule rather than failing later.

## Two things that bite

### Context vs slots

`-c` in llama.cpp is the **total** context, divided across server slots:

```
-c 98304 --parallel 2   ->   n_ctx_slot = 49152     # 48K per request, not 96K
```

Set your client's context window to the **per-slot** figure, not the total. The
panel shows both wherever slots > 1.

Leaving slots **blank = auto** is the default: `--parallel` isn't passed at all,
llama.cpp picks, and it enables a shared KV pool — so a single client can use the
whole context while concurrency still works. Set an explicit number only when you
want each slot's context *reserved*.

### VRAM estimates

`estimator.coeffs_gb` is a linear fit of measured runs on one base model:

```
VRAM_gb = base_gb + per_1k_gb * ctx/1000
```

Because it only knows context and KV type, a preset pointing at a *different*
model is corrected by the difference in on-disk weight size (`base_weight_gb` is
the fit's reference). If neither size can be resolved the figure is marked `?`
rather than quietly reported as if it were exact. Treat all of it as a guide and
watch the log for OOM — on a 16 GB card the fit runs ~0.3–0.5 GB conservative.

## HTTP API

Everything mutating needs `Authorization: Bearer $(cat ~/.vlm_api_key)`.

| method | path | what |
| --- | --- | --- |
| GET | `/presets.json` | full config + per-preset VRAM estimate |
| GET | `/api/status` | GPU/CPU, running model, health, engines, uptime, power capability |
| GET | `/api/models` | models found under the models root, with weight sizes |
| GET | `/api/downloads` | download jobs and progress |
| GET | `/api/log?n=N` | last N log lines |
| POST | `/api/switch` | preset or custom — restarts the server |
| POST | `/api/stop` | stop the model, free the GPU |
| POST | `/api/preset` | upsert / delete / reorder — upsert accepts orig_id to rename |
| POST | `/api/profile` | slot, preset |
| POST | `/api/activate` | endpoint — which backend the router forwards to |
| POST | `/api/endpoint` | add / delete a cloud endpoint |
| POST | `/api/check` | reachability test for an endpoint |
| POST | `/api/gen` | generation defaults |
| POST | `/api/runtime` | llama.cpp runtime knobs (apply: true restarts) |
| POST | `/api/engine` | switch inference engine |
| POST | `/api/engine-config` | per-engine options |
| POST | `/api/engine-install` | engine, force — runs the configured install script |
| POST | `/api/download` | repo, dest, name, include |
| POST | `/api/models` | register a downloaded model |
| POST | `/api/power` | GPU power limit (watts) |
| POST | `/api/system` | op: poweroff/reboot/cancel, delay_s, stop_model |

## Config

`config/presets.json` — presets, models, endpoints, engines, profiles, runtime,
estimator, gen defaults. `config/secrets.json` — cloud API keys, git-ignored,
chmod 600, referenced by each endpoint's `key_ref`.

## Troubleshooting

- **Client rejected above N tokens** — slots are splitting the context. See
  Context vs slots.
- **Model won't load / OOM** — check the estimate against the card and watch the
  Server log; the estimator cannot see other processes' VRAM.
- **Panel says a change needs a restart** — runtime knobs only apply to a freshly
  launched server; use Save & restart model now.
- **Power buttons missing** — the polkit rule isn't installed; the Power view
  shows the exact command.
- **A panel restart killed the model** — add the KillMode=process override above.
