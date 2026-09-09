#!/usr/bin/env python3
"""
llm-panel — dependency-free web console to operate local + cloud LLMs on this box.

Two servers run from this one process:

  1. PANEL  (default :8080)  — the web UI + control API.
  2. ROUTER (default :8001)  — an OpenAI-compatible endpoint. Point ANY harness at
     http://<box>:8001/v1 with any model id; it forwards to whichever endpoint is
     "active" (the local llama.cpp model OR a cloud provider) and applies your
     generation defaults. Flip the active endpoint in the UI -> every tool that
     uses the router instantly follows, no reconfig.

Local model switching restarts ~/serve-vlm.sh with the chosen context / KV quant /
parallel. Cloud endpoints just change the router's forward target.

Standard library only. Runs as user 'liuyang' (no sudo needed to stop/start the
local server). Config: presets.json. Remote keys: secrets.json (chmod 600, never
sent to the browser).
"""
import http.client
import json
import os
import re
import shlex
import shutil
import socket
import ssl
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))     # …/llm-stack/panel
ROOT = os.path.dirname(HERE)                           # …/llm-stack (repo root)


def _find_config():
    for p in (os.environ.get("LLM_STACK_CONFIG"),
              os.path.join(ROOT, "config", "presets.json"),
              os.path.join(HERE, "presets.json")):     # fallback: legacy layout
        if p and os.path.exists(p):
            return p
    return os.path.join(ROOT, "config", "presets.json")


CONFIG_PATH = _find_config()


def expand(p):
    """Expand ~ and the $ROOT token (repo root) so config is relocatable."""
    if not p:
        return p
    p = p.replace("${ROOT}", ROOT).replace("$ROOT", ROOT)
    return os.path.expanduser(p)


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def load_secrets(cfg):
    try:
        with open(expand(cfg["server"].get("secrets_file", ""))) as f:
            return json.load(f)
    except Exception:
        return {}


def read_api_key(cfg):
    try:
        with open(expand(cfg["server"].get("api_key_file", ""))) as f:
            return f.read().strip()
    except OSError:
        return ""


def save_config(mutate):
    """Load, mutate(dict) in place, write back atomically."""
    data = load_config()
    mutate(data)
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        # ensure_ascii=False + trailing newline: this file is in git, and escaping
        # every non-ASCII character churned the whole thing on each settings save.
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, CONFIG_PATH)
    return data


# ---------------------------------------------------------------- models / version
MODELS_ROOT = os.path.expanduser("~/models")


def purge_model_dir(cfg, mid):
    """Delete a model's files from disk. Returns (ok, message, freed_gb).

    Deliberately paranoid: this is the only call in the panel that destroys data.
    Refuses anything outside MODELS_ROOT, the model currently holding the GPU, and
    any directory that does not actually look like a model folder."""
    entry = (cfg.get("models") or {}).get(mid) or {}
    d = expand(entry.get("dir") or os.path.join(MODELS_ROOT, mid))
    real = os.path.realpath(d)
    root = os.path.realpath(MODELS_ROOT)
    if not os.path.isdir(real):
        return False, f"no such directory: {d}", 0.0
    # must live under ~/models — blocks traversal and absolute paths elsewhere
    if real != root and not real.startswith(root + os.sep):
        return False, f"refusing to delete outside {MODELS_ROOT}: {real}", 0.0
    if real == root:
        return False, "refusing to delete the models root itself", 0.0
    # must look like a model folder, not some unrelated directory
    try:
        files = os.listdir(real)
    except OSError as e:
        return False, f"cannot read {real}: {e}", 0.0
    if not any(f.endswith(".gguf") for f in files):
        return False, f"{real} holds no .gguf — refusing to delete it", 0.0
    # never delete what is currently loaded on the GPU
    run = running_server(cfg)
    if run:
        live = run.get("model_file")
        if live and live in files:
            return False, f"{mid} is the model currently loaded — stop it first", 0.0
    freed = 0
    for dirpath, _, fs in os.walk(real):
        for f in fs:
            try:
                freed += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    shutil.rmtree(real)
    return True, f"deleted {real}", round(freed / 2**30, 2)


def presets_using_model(cfg, mid):
    return [p["id"] for p in cfg.get("presets", []) if p.get("model") == mid]


def scan_models(cfg):
    """Discover model folders under ~/models that hold a servable .gguf (plus an
    optional mmproj for vision), and merge with the models declared in config."""
    found = {}
    roots = {MODELS_ROOT}
    for m in cfg.get("models", {}).values():
        d = expand(m.get("dir", ""))
        if d:
            roots.add(os.path.dirname(d.rstrip("/")))
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if not os.path.isdir(d):
                continue
            try:
                files = os.listdir(d)
            except OSError:
                continue
            # main-weights candidates only: no projector, no mtp-* draft head. Including
            # the drafts made a Flash-Next directory read as ~89 GB of "model".
            ggufs = sorted(f for f in files if f.endswith(".gguf")
                           and not f.lower().startswith("mmproj")
                           and not f.lower().startswith("mtp-"))
            if not ggufs:
                continue
            mmproj = [f for f in files if f.lower().startswith("mmproj") and f.endswith(".gguf")]
            pick = ggufs[0]
            try:
                # only the selected build and its shards, not every build in the directory
                size = sum(os.path.getsize(os.path.join(d, f))
                           for f in gguf_shard_set(ggufs, pick))
            except OSError:
                size = 0
            main = os.path.join(d, pick)
            embedded = bool(gguf_has_nextn(main))
            # separate draft heads sitting next to the model. mtp-shared-* is excluded: it
            # omits token_embd.weight by design and the draft loader rejects it outright.
            drafts = sorted(f for f in files
                            if f.lower().startswith("mtp-") and f.endswith(".gguf")
                            and not f.lower().startswith("mtp-shared-"))
            found[name] = {"id": name, "dir": d, "gguf": pick,
                           "ggufs": ggufs,
                           "vision": bool(mmproj), "size_gb": round(size / 1e9, 1),
                           # Read out of the GGUF header, not the file name: this is what
                           # decides whether --spec-type draft-mtp can work at all, and
                           # how much extra VRAM turning it on costs.
                           "mtp_embedded": embedded,
                           "mtp_head_gb": (cfg.get("estimator") or {}).get(
                               "mtp_head_gb", 0.35) if embedded else 0.0,
                           "mtp_draft_files": drafts,
                           "configured": False}
    # Every CONFIGURED model gets its own row, keyed by its config id -- never by its
    # directory. Two entries can legitimately share a folder (the plain and the -mtp build
    # of one quant), and keying by folder silently dropped the second, so it never appeared
    # in the model picker at all.
    out = {}
    for mid, m in cfg.get("models", {}).items():
        d = expand(m.get("dir", ""))
        gg = preset_gguf(cfg, {"model": mid})
        row = {"id": mid, "dir": d, "config_id": mid, "configured": True,
               "label": m.get("label", mid), "gguf": gg,
               # missing means the WEIGHTS are gone, not that the folder is empty: a
               # directory holding only an mmproj (weights deleted by hand) still shows
               # as installed, which made the card's Remove button refuse to act.
               "missing": bool(m.get("missing")) or not gg}
        if d and os.path.isdir(d):
            facts = detect_model_facts(cfg, d, m.get("gguf")) or {}
            _, cands = model_gguf_candidates(cfg, mid)
            row.update(vision=facts.get("has_mmproj", False),
                       ggufs=cands,
                       size_gb=facts.get("weight_gb"),
                       arch=facts.get("arch"),
                       mmproj_file=facts.get("mmproj_file"),
                       spec_types=facts.get("spec_types", []),
                       mtp_draft_rejected=facts.get("mtp_draft_rejected", {}),
                       mtp_embedded=facts.get("mtp_embedded", False),
                       mtp_draft_files=facts.get("mtp_draft_files", []))
            # a configured per-model head size wins over the family constant, so the
            # browser's estimate and the panel's agree to the last 10 MB
            row["mtp_head_gb"] = mtp_head_gb(cfg, mid)
        else:
            row.update(vision=m.get("vision", False), size_gb=m.get("weight_gb"),
                       mtp_embedded=bool(m.get("mtp_embedded")),
                       mtp_head_gb=mtp_head_gb(cfg, mid),
                       mtp_draft_files=m.get("mtp_draft_files", []))
        out[mid] = row
    # then any scanned directory no configured model already covers
    covered = {(os.path.realpath(r["dir"]), r.get("gguf")) for r in out.values() if r["dir"]}
    for r in found.values():
        if (os.path.realpath(r["dir"]), r.get("gguf")) not in covered:
            out.setdefault(r["id"], r)
    return list(out.values())


# ---------------------------------------------------------------- model downloads
DOWNLOADS = {}
DL_LOCK = threading.Lock()


def hf_bin():
    for p in (os.path.expanduser("~/.hf-venv/bin/hf"), os.path.expanduser("~/.hf-venv/bin/huggingface-cli"),
              shutil.which("hf"), shutil.which("huggingface-cli")):
        if p and os.path.exists(p):
            return p
    return None


def _repo_from(s):
    """Accept a full huggingface.co URL or a plain org/name."""
    s = (s or "").strip()
    m = re.search(r"huggingface\.co/([^/\s]+/[^/\s?#]+)", s)
    return (m.group(1) if m else s).rstrip("/")


def _hf_repo_size_gb(repo, include=None):
    """Total download size in GB. If `include` globs are given, only count matching files."""
    import fnmatch
    try:
        with urllib.request.urlopen(
                f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true", timeout=12) as r:
            data = json.load(r)
        files = [f for f in data if f.get("type") == "file"]
        if include:
            files = [f for f in files
                     if any(fnmatch.fnmatch(f["path"], g) or fnmatch.fnmatch(os.path.basename(f["path"]), g)
                            for g in include)]
        return round(sum(f.get("size", 0) for f in files) / 1e9, 2)
    except Exception:
        return None


def _dir_size_gb(d):
    total = 0
    for root, _, files in os.walk(d):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return round(total / 1e9, 2)


def start_download(cfg, repo, dest, name, include=None):
    hb = hf_bin()
    if not hb:
        return None, "hf CLI not found (expected ~/.hf-venv/bin/hf)"
    repo = _repo_from(repo)
    # Accept the llama.cpp/Ollama shorthand 'org/name-GGUF:QUANT' — the ':QUANT' is a
    # file selector, not part of the repo id, so split it off into an --include glob.
    quant = None
    if ":" in repo.split("/")[-1]:
        repo, quant = repo.rsplit(":", 1)
    if "/" not in repo:
        return None, "model must be 'org/name', a huggingface.co URL, or 'org/name:QUANT'"
    if quant and not include:
        include = [f"*{quant}*", "mmproj*"]  # the chosen quant + any vision projector
    name = (name or (repo.split("/")[-1] + (("-" + quant) if quant else ""))).strip()
    name = re.sub(r"[^\w.\-]", "-", name)  # safe folder name (no ':' etc.)
    dest = expand(dest) if dest else os.path.join(MODELS_ROOT, name)
    try:
        os.makedirs(dest, exist_ok=True)
    except OSError as e:
        return None, f"cannot create {dest}: {e}"
    jid = str(int(time.time() * 1000))
    logp = os.path.join(ROOT, "logs", f"download-{jid}.log")
    os.makedirs(os.path.dirname(logp), exist_ok=True)
    total = _hf_repo_size_gb(repo, include=include)
    env = os.environ.copy()
    # use hf_transfer (Rust, multi-connection) for faster large downloads if installed
    import glob as _glob
    if _glob.glob(os.path.expanduser("~/.hf-venv/lib/python*/site-packages/hf_transfer")):
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    cmd = [hb, "download", repo, "--local-dir", dest]
    if include:
        for pat in include:
            cmd += ["--include", pat]
    logf = open(logp, "wb")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                            start_new_session=True, env=env, cwd=os.path.expanduser("~"))
    with DL_LOCK:
        DOWNLOADS[jid] = {"id": jid, "repo": repo, "dest": dest, "name": name,
                          "phase": "downloading", "total_gb": total, "log": logp,
                          "pid": proc.pid, "started": time.time(), "error": ""}
    _save_downloads()
    threading.Thread(target=_watch_download, args=(jid, proc, name, dest), daemon=True).start()
    return jid, "started"


DL_STATE = os.path.join(ROOT, "logs", "downloads.json")


def _save_downloads():
    """Persist job state so a panel restart doesn't orphan a running download."""
    try:
        os.makedirs(os.path.dirname(DL_STATE), exist_ok=True)
        with DL_LOCK:
            data = list(DOWNLOADS.values())
        tmp = DL_STATE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, DL_STATE)
    except Exception:
        pass


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _dl_looks_complete(j):
    """No exit code for an adopted process — judge by bytes on disk."""
    tot = j.get("total_gb") or 0
    got = _dir_size_gb(j["dest"]) if os.path.isdir(j["dest"]) else 0
    return bool(tot) and got >= tot * 0.98


def _finish_download(jid, ok, name, dest, err=""):
    try:
        files = os.listdir(dest)
    except OSError:
        files = []
    has_mmproj = any(f.lower().startswith("mmproj") and f.endswith(".gguf") for f in files)
    if ok:
        # auto-register the downloaded model so it shows up in the model pickers
        save_config(lambda d: d.setdefault("models", {}).__setitem__(
            name, {"dir": dest, "label": name, "vision": has_mmproj, "note": "downloaded"}))
        with DL_LOCK:
            DOWNLOADS[jid]["phase"] = "done"
    else:
        with DL_LOCK:
            if not err:
                err = " ".join(tail(DOWNLOADS[jid].get("log", ""), 3))[-300:]
            DOWNLOADS[jid].update(phase="error", error=err)
    _save_downloads()


def _watch_download(jid, proc, name, dest):
    rc = proc.wait()
    _finish_download(jid, rc == 0, name, dest)


def _watch_adopted(jid, pid, name, dest):
    """Re-attached after a restart: we are not the parent, so poll for liveness."""
    while _pid_alive(pid):
        time.sleep(3)
    with DL_LOCK:
        j = dict(DOWNLOADS.get(jid, {}))
    ok = _dl_looks_complete(j)
    _finish_download(jid, ok, name, dest,
                     err="" if ok else "download stopped before finishing (panel restarted)")


def adopt_downloads():
    """On startup, pick up downloads that were running when the panel last stopped."""
    try:
        with open(DL_STATE) as f:
            jobs = json.load(f)
    except Exception:
        return
    for j in jobs:
        if not j.get("id"):
            continue
        with DL_LOCK:
            DOWNLOADS[j["id"]] = j
        if j.get("phase") != "downloading":
            continue
        if j.get("pid") and _pid_alive(j["pid"]):
            threading.Thread(target=_watch_adopted,
                             args=(j["id"], j["pid"], j["name"], j["dest"]), daemon=True).start()
        else:
            _finish_download(j["id"], _dl_looks_complete(j), j["name"], j["dest"],
                             err="" if _dl_looks_complete(j) else "ended while the panel was down")


def downloads_view():
    with DL_LOCK:
        jobs = list(DOWNLOADS.values())
    out = []
    for j in jobs:
        got = _dir_size_gb(j["dest"]) if os.path.isdir(j["dest"]) else 0
        pct = round(100 * got / j["total_gb"]) if j.get("total_gb") else None
        out.append({k: j.get(k) for k in ("id", "repo", "dest", "name", "phase", "total_gb", "error", "started")}
                   | {"got_gb": got, "pct": pct})
    return sorted(out, key=lambda x: -x["started"])


def panel_version():
    """Changes whenever panel.py or index.html changes — drives the UI refresh banner."""
    t = 0
    for f in (os.path.join(HERE, "panel.py"), os.path.join(HERE, "index.html")):
        try:
            t = max(t, int(os.path.getmtime(f)))
        except OSError:
            pass
    return str(t)


# ---------------------------------------------------------------- GPU / CPU stats
_GPU_FIELDS = ["memory.used", "memory.total", "memory.free", "utilization.gpu",
               "temperature.gpu", "power.draw", "power.limit", "power.max_limit",
               "power.min_limit", "clocks.current.graphics", "clocks.max.graphics",
               "clocks.current.memory", "fan.speed"]


def _num(x):
    x = (x or "").strip()
    if x in ("", "[N/A]", "N/A", "[Not Supported]"):
        return None
    try:
        return float(x) if "." in x else int(x)
    except ValueError:
        return None


def gpu_stats():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=" + ",".join(_GPU_FIELDS), "--format=csv,noheader,nounits"],
            text=True, timeout=5).strip().splitlines()[0]
        vals = [v.strip() for v in out.split(",")]
        d = dict(zip(_GPU_FIELDS, vals))
        return {"used_mib": _num(d["memory.used"]), "total_mib": _num(d["memory.total"]),
                "free_mib": _num(d["memory.free"]), "util_pct": _num(d["utilization.gpu"]),
                "temp_c": _num(d["temperature.gpu"]), "power_w": _num(d["power.draw"]),
                "power_limit_w": _num(d["power.limit"]), "power_max_w": _num(d["power.max_limit"]),
                "power_min_w": _num(d["power.min_limit"]),
                "clock_mhz": _num(d["clocks.current.graphics"]), "clock_max_mhz": _num(d["clocks.max.graphics"]),
                "mem_clock_mhz": _num(d["clocks.current.memory"]), "fan_pct": _num(d["fan.speed"])}
    except Exception as e:
        return {"error": str(e)}


_CPU_PREV = {"total": 0, "idle": 0}


def cpu_stats():
    """CPU utilization (delta since last call), load average, and RAM — stdlib /proc."""
    out = {}
    try:
        with open("/proc/stat") as f:
            parts = f.readline().split()[1:]
        nums = [int(x) for x in parts]
        idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
        total = sum(nums)
        dt = total - _CPU_PREV["total"]
        di = idle - _CPU_PREV["idle"]
        if _CPU_PREV["total"] and dt > 0:
            out["util_pct"] = round(100.0 * (dt - di) / dt, 1)
        _CPU_PREV["total"], _CPU_PREV["idle"] = total, idle
    except Exception:
        pass
    try:
        out["cores"] = os.cpu_count()
        with open("/proc/loadavg") as f:
            out["load1"] = float(f.read().split()[0])
    except Exception:
        pass
    try:
        mt = ma = None
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mt = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    ma = int(line.split()[1])
                if mt and ma:
                    break
        if mt and ma:
            out["ram_total_gb"] = round(mt / 1e6, 1)
            out["ram_used_gb"] = round((mt - ma) / 1e6, 1)
    except Exception:
        pass
    return out


def set_power_limit(watts):
    """Set the GPU power limit. Needs sudo — try non-interactive; report if unavailable."""
    try:
        r = subprocess.run(["sudo", "-n", "nvidia-smi", "-pl", str(int(watts))],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            return {"ok": True, "message": r.stdout.strip() or f"power limit set to {watts}W"}
        return {"ok": False, "needs_sudo": True,
                "hint": f"run: sudo nvidia-smi -pl {int(watts)}", "err": (r.stderr or "").strip()}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------- host power
# Powering the box off needs root. This process is rootless, so we detect which
# non-interactive route (if any) is open and otherwise hand back the one-time
# setup — same contract as set_power_limit() above.
PENDING_POWER = {"op": None, "at": 0, "timer": None}
_POWER_CAP = {"at": 0, "val": None}


def _panel_user():
    try:
        import pwd
        return pwd.getpwuid(os.getuid()).pw_name
    except Exception:
        return os.environ.get("USER", "the panel user")


def polkit_rule_text():
    """A polkit rule granting ONLY poweroff/reboot to the user running the panel."""
    return (
        'polkit.addRule(function(action, subject) {\n'
        '  if ((action.id == "org.freedesktop.login1.power-off" ||\n'
        '       action.id == "org.freedesktop.login1.reboot") &&\n'
        '      subject.user == "%s") {\n'
        '    return polkit.Result.YES;\n'
        '  }\n'
        '});\n' % _panel_user()
    )


def power_capability(force=False):
    """Can we power the host off without a password, and if not, what enables it?"""
    now = time.time()
    if not force and _POWER_CAP["val"] and now - _POWER_CAP["at"] < 300:
        return _POWER_CAP["val"]
    val = None
    try:
        if shutil.which("pkcheck"):
            r = subprocess.run(["pkcheck", "--action-id", "org.freedesktop.login1.power-off",
                                "--process", str(os.getpid())],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                val = {"can": True, "method": "polkit"}
        if val is None and shutil.which("sudo"):
            r = subprocess.run(["sudo", "-n", "true"], capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                val = {"can": True, "method": "sudo"}
    except Exception:
        val = None
    if val is None:
        path = "/etc/polkit-1/rules.d/49-llm-panel-power.rules"
        val = {
            "can": False, "method": None,
            "hint": "One-time root step: allow %s to power this box off." % _panel_user(),
            "polkit_path": path,
            "polkit_rule": polkit_rule_text(),
            "install_cmd": "sudo tee %s >/dev/null <<'EOF'\n%sEOF" % (path, polkit_rule_text()),
        }
    _POWER_CAP.update(at=now, val=val)
    return val


def host_uptime_s():
    try:
        with open("/proc/uptime") as f:
            return float(f.read().split()[0])
    except Exception:
        return None


def pending_power():
    op = PENDING_POWER.get("op")
    return {"op": op, "at": PENDING_POWER.get("at", 0)} if op else None


def cancel_power():
    t = PENDING_POWER.get("timer")
    if t:
        try:
            t.cancel()
        except Exception:
            pass
    PENDING_POWER.update(op=None, at=0, timer=None)
    return {"ok": True, "cancelled": True}


def _scrub(text, secret):
    """Never let a password reach a response body or a log line."""
    if secret and text:
        text = text.replace(secret, "********")
    return text


def run_power(op, password=None):
    """Actually power off / reboot. Returns a dict; on success the box is going down.

    `password` is used ONCE, over stdin to `sudo -S`, and is never stored, cached or
    logged. It exists because this box's polkit requires admin auth for power-off and the
    panel runs rootless: without it the only route is the one-time polkit rule. Prefer
    that rule -- the panel speaks plain HTTP on the LAN, so a password typed into it
    crosses the network in the clear."""
    action = "poweroff" if op == "poweroff" else "reboot"
    cap = power_capability(force=True)
    if cap.get("can"):
        cmd = (["sudo", "-n", "systemctl", action] if cap["method"] == "sudo"
               else ["systemctl", action])
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
            if r.returncode == 0:
                return {"ok": True}
            return {"ok": False, "needs_privilege": True,
                    "err": (r.stderr or r.stdout or "").strip(), "hint": " ".join(cmd)}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    if not password:
        return dict({"ok": False, "needs_privilege": True}, **cap)
    try:
        # -S read the password from stdin, -k ignore any cached ticket so a wrong
        # password fails immediately instead of silently reusing someone else's session
        r = subprocess.run(["sudo", "-S", "-k", "-p", "", "systemctl", action],
                           input=password + "\n", capture_output=True, text=True, timeout=25)
        if r.returncode == 0:
            return {"ok": True, "method": "password"}
        err = _scrub((r.stderr or r.stdout or "").strip(), password)
        bad = "incorrect password" in err.lower() or "sorry, try again" in err.lower()
        return {"ok": False, "needs_privilege": True, "bad_password": bad,
                "err": err or "sudo refused"}
    except Exception as e:
        return {"ok": False, "error": _scrub(str(e), password)}
    finally:
        password = None


def schedule_power(op, delay_s, cfg_for_stop=None, password=None):
    """Schedule op after delay_s, cancellable until it fires. Stops the model first."""
    cancel_power()
    at = time.time() + delay_s

    def fire():
        PENDING_POWER.update(op=None, at=0, timer=None)
        if cfg_for_stop is not None:
            try:
                _kill_server(cfg_for_stop)
            except Exception:
                pass
        run_power(op, password)

    t = threading.Timer(delay_s, fire)
    t.daemon = True
    PENDING_POWER.update(op=op, at=at, timer=t)
    t.start()
    return {"ok": True, "scheduled": True, "op": op, "at": at, "delay_s": delay_s}


_ARG_RE = {
    "ctx": re.compile(r"(?:-c|--ctx-size)\s+(\d+)"),
    "parallel": re.compile(r"(?:--parallel|-np)\s+(\d+)"),
    "kv": re.compile(r"--cache-type-k\s+(\S+)"),
    "ngl": re.compile(r"(?:-ngl|--n-gpu-layers|--gpu-layers)\s+(\d+)"),
    "model": re.compile(r"(?:-m|--model)\s+(\S+)"),
    "port": re.compile(r"--port\s+(\d+)"),
    "alias": re.compile(r"--alias\s+(\S+)"),
    "thinking": re.compile(r"(?:--reasoning|-rea)\s+(on|off|auto)"),
    "flash_attn": re.compile(r"-fa\s+(on|off|auto)"),
    "reason_effort": re.compile(r"--reasoning-effort\s+(\S+)"),
    "cache_reuse": re.compile(r"--cache-reuse\s+(\d+)"),
    "spec_type": re.compile(r"--spec-type\s+(\S+)"),
    "spec_n_max": re.compile(r"--spec-draft-n-max\s+(\d+)"),
    "max_model_len": re.compile(r"--max-model-len\s+(\d+)"),
    "mmproj": re.compile(r"--mmproj\s+(\S+)"),
}


# ---------------------------------------------------------------- engines
def ollama_bin():
    """Resolve the ollama binary, incl. the rootless ~/.local/bin install."""
    for p in (shutil.which("ollama"), os.path.expanduser("~/.local/bin/ollama"),
              "/usr/local/bin/ollama", "/usr/bin/ollama"):
        if p and os.path.exists(p):
            return p
    return None


def vllm_bin():
    p = os.path.expanduser("~/vllm-env/bin/vllm")
    return p if os.path.exists(p) else (shutil.which("vllm") or None)


def engine_installed(cfg, eng):
    """Is a given engine actually available on this box? (checks the real binary)."""
    if eng == "llamacpp":
        return os.path.exists(expand(cfg["server"]["llama_bin"]))
    if eng == "vllm":
        return bool(vllm_bin())          # venv dir alone isn't enough — need the vllm entrypoint
    if eng == "ollama":
        return bool(ollama_bin())
    return False


INSTALLS = {}
INSTALL_LOCK = threading.Lock()


def install_state(engine=None):
    with INSTALL_LOCK:
        if engine:
            j = INSTALLS.get(engine)
            return {k: j[k] for k in ("phase", "started", "error")} if j else None
        return {k: {kk: v[kk] for kk in ("phase", "started", "error")} for k, v in INSTALLS.items()}


def _watch_install(engine, proc):
    rc = proc.wait()
    with INSTALL_LOCK:
        j = INSTALLS.get(engine)
        if not j:
            return
        if rc == 0:
            j.update(phase="done", error="")
        else:
            j.update(phase="error",
                     error=(" ".join(tail(j.get("log", ""), 4))[-300:] or f"exit code {rc}"))


def start_install(cfg, engine):
    """Run the install command CONFIGURED for this engine. The client only ever
    names the engine — it can never supply a command of its own."""
    spec = (cfg.get("engines") or {}).get(engine) or {}
    cmd = (spec.get("install") or "").strip()
    if not cmd:
        return None, f"no install command is configured for {engine}"
    with INSTALL_LOCK:
        cur = INSTALLS.get(engine)
        if cur and cur.get("phase") == "running":
            return None, f"{engine} is already installing"
    logp = os.path.join(ROOT, "logs", f"install-{engine}.log")
    os.makedirs(os.path.dirname(logp), exist_ok=True)
    logf = open(logp, "wb")
    proc = subprocess.Popen(["bash", "-lc", expand(cmd)], stdout=logf,
                            stderr=subprocess.STDOUT, start_new_session=True,
                            cwd=os.path.expanduser("~"))
    with INSTALL_LOCK:
        INSTALLS[engine] = {"engine": engine, "phase": "running", "started": time.time(),
                            "log": logp, "pid": proc.pid, "error": ""}
    threading.Thread(target=_watch_install, args=(engine, proc), daemon=True).start()
    return logp, "started"


def engine_port(cfg, eng=None):
    eng = eng or cfg.get("active_engine", "llamacpp")
    engines = cfg.get("engines", {})
    if eng == "llamacpp":
        return int(engines.get("llamacpp", {}).get("port", cfg["server"].get("vlm_port", 8000)))
    if eng == "vllm":
        return int(engines.get("vllm", {}).get("port", cfg["server"].get("vlm_port", 8000)))
    if eng == "ollama":
        return int(engines.get("ollama", {}).get("port", 11434))
    return cfg["server"].get("vlm_port", 8000)


def detect_engine(cfg):
    """Which engine process currently holds the GPU? Returns (engine, pid, args) or (None, None, '')."""
    llama_bin = expand(cfg["server"]["llama_bin"])
    try:
        out = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True, timeout=5)
    except Exception:
        return None, None, ""
    for line in out.splitlines():
        line = line.strip()
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        pid, args = parts[0], parts[1]
        if llama_bin in args:
            return "llamacpp", int(pid), args
        if re.search(r"(^|/)vllm\b|vllm\.entrypoints|-m vllm", args):
            return "vllm", int(pid), args
        if re.search(r"(^|/)ollama\b.*\bserve\b|ollama runner", args):
            return "ollama", int(pid), args
    return None, None, ""


def running_server(cfg):
    """Parse the currently running local engine's process into a config dict."""
    eng, pid, args = detect_engine(cfg)
    if not eng:
        return None
    info = {"pid": pid, "engine": eng}
    for k, rx in _ARG_RE.items():
        m = rx.search(args)
        if m:
            info[k] = m.group(1)
    for k in ("ctx", "parallel", "ngl", "port", "cache_reuse", "max_model_len", "spec_n_max"):
        if k in info:
            info[k] = int(info[k])
    # context shift is a bare boolean flag
    if "--no-context-shift" in args:
        info["context_shift"] = "off"
    elif "--context-shift" in args:
        info["context_shift"] = "on"
    if "--no-kv-unified" in args:
        info["kv_unified"] = "off"
    elif "--kv-unified" in args:
        info["kv_unified"] = "on"
    info["vision"] = "on" if info.pop("mmproj", None) else "off"
    if "model" in info:
        info["model_file"] = os.path.basename(info.pop("model"))
    return info


def running_cmdline(cfg):
    """The EXACT argv of the running engine, plus where it came from.

    The server log says what llama.cpp thinks; this says what it was actually asked. When
    a preset does not behave the way its card reads, the difference is always here -- a
    flag that was dropped, a path that resolved elsewhere, an env override that won."""
    eng, pid, args = detect_engine(cfg)
    if not eng or not pid:
        return {"running": False}
    argv = []
    try:
        with open("/proc/%s/cmdline" % pid, "rb") as f:
            argv = [a for a in f.read().split(b"\x00") if a]
        argv = [a.decode("utf-8", "replace") for a in argv]
    except OSError:
        argv = args.split()
    env = {}
    try:
        with open("/proc/%s/environ" % pid, "rb") as f:
            for kv in f.read().split(b"\x00"):
                if not kv or b"=" not in kv:
                    continue
                k, v = kv.decode("utf-8", "replace").split("=", 1)
                # only the knobs this stack sets, never the user's whole environment
                if k in ("MDIR", "MODEL_FILE", "MMPROJ_FILE", "CTX", "KV_QUANT", "NGL",
                         "PARALLEL", "SPEC_TYPE", "SPEC_N_MAX", "MTP_HEAD", "VISION",
                         "NCMOE", "CPU_MOE", "LAZY_MODE", "LOAD_MODE", "FIT", "FIT_TARGET",
                         "THINKING", "FLASH_ATTN", "THREADS", "EXTRA_FLAGS"):
                    env[k] = v
    except OSError:
        pass
    started = None
    try:
        started = os.stat("/proc/%s" % pid).st_mtime
    except OSError:
        pass
    # mask the api key: this endpoint is for reading, not for leaking
    safe = []
    skip = False
    for a in argv:
        if skip:
            safe.append("********")
            skip = False
            continue
        safe.append(a)
        if a == "--api-key":
            skip = True
    return {"running": True, "engine": eng, "pid": pid, "argv": safe,
            "cmd": " ".join(shlex.quote(a) for a in safe), "env": env, "started": started,
            "matched_preset": match_preset(cfg, running_server(cfg))}


def local_health(cfg):
    eng = cfg.get("active_engine", "llamacpp")
    port = engine_port(cfg, eng)
    # ollama exposes /api/tags; llama.cpp & vLLM expose /health
    paths = ["/api/tags"] if eng == "ollama" else ["/health"]
    for p in paths:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{p}", timeout=2) as r:
                if r.status == 200:
                    return "ready"
        except Exception:
            pass
    return "loading" if running_server(cfg) else "down"


def tail(path, n):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            data = b""
            while size > 0 and data.count(b"\n") <= n:
                step = min(8192, size)
                size -= step
                f.seek(size)
                data = f.read(step) + data
        return data.decode("utf-8", "replace").splitlines()[-n:]
    except OSError:
        return []


def last_tps(cfg):
    for line in reversed(tail(expand(cfg["server"]["server_log"]), 200)):
        m = re.search(r"tg\s*=\s*([\d.]+)\s*t/s", line)
        if m:
            return float(m.group(1))
    return None


def model_weight_gb(cfg, model_id):
    """On-disk size of the weights this model would actually LOAD, or None.

    Only the selected GGUF and its shards -- NOT every gguf in the directory. Summing the
    directory double-counted any repo that ships two builds of the same model (the GSQ-RCO
    dirs hold the plain and the -mtp GGUF together, which read as a 24 GB model)."""
    d, g = model_gguf_candidates(cfg, model_id)
    if not g:
        return None
    main = preset_gguf(cfg, {"model": model_id})
    if not main:
        return None
    try:
        tot = sum(os.path.getsize(os.path.join(d, f)) for f in gguf_shard_set(g, main))
    except OSError:
        return None
    return round(tot / 1e9, 2) if tot else None


# ---------------------------------------------------------------------------
# MTP / NextN head detection.
#
# Qwen3.8-27B ships in TWO flavours of the same quant: one whose GGUF carries the
# NextN/MTP draft head as an extra block (blk.<n_layer>.nextn.*) and one that does
# not. Only the first can run --spec-type draft-mtp; asking the second for MTP is a
# hard load failure, not a slow path. The distinction is NOT visible in the file
# name (ISTA-DASLab marks it with a "-mtp" suffix, Unsloth does not mark it at all),
# so detect it from the tensor names in the GGUF header instead of trusting config.
#
# llama.cpp only ALLOCATES the head when MTP is requested: common.cpp sets
# mparams.load_mtp from the presence of COMMON_SPECULATIVE_TYPE_DRAFT_MTP, and
# models/qwen35.cpp then marks every nextn tensor TENSOR_SKIP when it is false. So a
# "-mtp" GGUF costs exactly the same VRAM as the plain one while MTP is off, and
# ~0.35 GB more the moment it is turned on -- on top of the draft KV/compute buffers.
_NEXTN_CACHE = {}


_GGUF_KV_SKIP = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
# KV values we actually want out of the header. Everything else is skipped by width.
_GGUF_WANT = {"general.architecture": "arch", "general.type": "type",
              "general.name": "name"}


def _gguf_read_str(f, unpack):
    n = unpack("<Q", f.read(8))[0]
    if n > (1 << 20):
        raise ValueError("absurd string length")
    return f.read(n)


_GGUF_FMT = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4),
             5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8),
             12: ("<d", 8)}

# Numeric metadata worth keeping. These are what decide whether a model can run at all on
# a given box and how much KV each token costs -- the things that used to be a per-model
# hand-fit and are simply written down in the header.
_GGUF_NUM_SUFFIXES = (
    "block_count", "context_length", "embedding_length", "feed_forward_length",
    "attention.head_count", "attention.head_count_kv",
    "attention.key_length", "attention.value_length",
    "expert_count", "expert_used_count", "expert_feed_forward_length",
    "expert_shared_feed_forward_length", "nextn_predict_layers", "vocab_size",
    "rope.freq_base", "rope.scaling.factor",
    "ssm.state_size", "ssm.conv_kernel", "ssm.inner_size", "ssm.time_step_rank",
    "ssm.group_count", "full_attention_interval", "attention.sliding_window",
)


def _gguf_read_value(f, unpack, t):
    """Read a scalar, or an array of scalars (kept, capped) -- head_count_kv is often a
    per-layer array, and in a hybrid model its zeros are exactly the layers that have no
    KV cache at all."""
    if t == 8:
        return _gguf_read_str(f, unpack).decode("utf-8", "replace")
    if t == 9:
        et = unpack("<I", f.read(4))[0]
        n = unpack("<Q", f.read(8))[0]
        if et == 8:
            return [_gguf_read_str(f, unpack).decode("utf-8", "replace")
                    if i < 64 else (f.seek(unpack("<Q", f.read(8))[0], 1) or None)
                    for i in range(n)][:64]
        fmt = _GGUF_FMT.get(et)
        if not fmt or n > 100000:
            f.seek((fmt[1] if fmt else 4) * n, 1)
            return None
        raw = f.read(fmt[1] * n)
        from struct import unpack_from
        return [unpack_from(fmt[0], raw, i * fmt[1])[0] for i in range(n)]
    fmt = _GGUF_FMT.get(t)
    if not fmt:
        return None
    return unpack(fmt[0], f.read(fmt[1]))[0]


def _gguf_skip_value(f, unpack, t=None):
    """Skip one GGUF metadata value. Types: 8 = string, 9 = array, rest are fixed width."""
    if t is None:
        t = unpack("<I", f.read(4))[0]
    if t == 8:
        f.seek(unpack("<Q", f.read(8))[0], 1)
    elif t == 9:
        et = unpack("<I", f.read(4))[0]
        n = unpack("<Q", f.read(8))[0]
        if et == 8:
            for _ in range(n):
                f.seek(unpack("<Q", f.read(8))[0], 1)
        elif et == 9:
            for _ in range(n):
                _gguf_skip_value(f, unpack)   # nested arrays are legal, if rare
        else:
            f.seek(_GGUF_KV_SKIP.get(et, 4) * n, 1)
    else:
        f.seek(_GGUF_KV_SKIP.get(t, 4), 1)


def gguf_probe(path):
    """What this GGUF actually IS, read from its header. Returns a dict or None.

        arch            general.architecture -- 'clip' for a projector, 'eagle3'/'dflash'/
                        'dspark' for a standalone draft model, otherwise the model family
        type            general.type -- 'mmproj' marks a projector
        has_nextn       carries blk.<n>.nextn.* -- an embedded (or standalone) MTP head
        nextn_shared    carries <arch>.nextn_shared_target_tensors -- a SHARED head, which
                        omits token_embd.weight and the draft loader rejects outright
        n_tensors       tensor count

    Filenames are not evidence. The same repo ships `mmproj-*.gguf`, `mtp-*.gguf` and the
    main weights with no naming rule anyone enforces, folders get reorganised by hand, and
    a projector renamed on download would otherwise vanish. Everything the panel decides --
    is this the model, a projector, or a draft head; can it speculate; can it see images --
    comes from here. Cached on (size, mtime)."""
    if not path:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (path, st.st_size, int(st.st_mtime))
    hit = _NEXTN_CACHE.get(path)
    if hit and hit[0] == key:
        return hit[1]
    from struct import unpack
    out = {"arch": None, "type": None, "name": None, "has_nextn": False,
           "nextn_shared": False, "n_tensors": 0, "size": st.st_size, "hp": {},
           "max_tensor": 0}
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return None
            unpack("<I", f.read(4))                      # version
            n_tensors = unpack("<Q", f.read(8))[0]
            n_kv = unpack("<Q", f.read(8))[0]
            if n_tensors > (1 << 20) or n_kv > (1 << 20):
                return None
            out["n_tensors"] = n_tensors
            for _ in range(n_kv):
                k = _gguf_read_str(f, unpack).decode("utf-8", "replace")
                t = unpack("<I", f.read(4))[0]
                if k in _GGUF_WANT and t == 8:
                    out[_GGUF_WANT[k]] = _gguf_read_str(f, unpack).decode("utf-8", "replace")
                    continue
                if k.endswith(".nextn_shared_target_tensors"):
                    out["nextn_shared"] = True
                if k.split(".", 1)[-1] in _GGUF_NUM_SUFFIXES and t != 9 or (
                        t == 9 and k.split(".", 1)[-1] in _GGUF_NUM_SUFFIXES):
                    try:
                        out["hp"][k.split(".", 1)[-1]] = _gguf_read_value(f, unpack, t)
                        continue
                    except Exception:
                        pass
                _gguf_skip_value(f, unpack, t)
            # walk the whole tensor table: the largest single tensor is what decides
            # whether --lazy-mode is needed (Flash-Next's ~27 GB per-layer embedding
            # table cannot be resident anywhere on this box)
            prev_off, prev_name = None, None
            for _ in range(n_tensors):
                name = _gguf_read_str(f, unpack)
                if b".nextn." in name:
                    out["has_nextn"] = True
                n_dims = unpack("<I", f.read(4))[0]
                dims = [unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
                unpack("<I", f.read(4))                  # ggml type
                off = unpack("<Q", f.read(8))[0]
                if prev_off is not None:
                    out["max_tensor"] = max(out["max_tensor"], off - prev_off)
                prev_off, prev_name = off, name
                out["n_elements"] = out.get("n_elements", 0) + (
                    int(dims[0]) if n_dims == 1 else
                    int(dims[0]) * int(dims[1]) if n_dims == 2 else
                    int(dims[0]) * int(dims[1]) * int(dims[2]) if n_dims == 3 else 0)
    except Exception:      # a header we cannot parse is not a reason to 500 the panel
        return None
    _NEXTN_CACHE[path] = (key, out)
    return out


# Architectures that ARE a draft model in their own right, loaded with -md, and the
# --spec-type each one needs. An embedded NextN head needs none of these: it is inside the
# target GGUF and selected with draft-mtp.
_DRAFT_ARCH_SPEC = {"eagle3": "draft-eagle3", "dflash": "draft-dflash",
                    "dspark": "draft-dspark"}


def draft_arch_spec(cfg=None):
    """Architecture -> --spec-type, overridable from presets.json.

    llama.cpp keeps adding speculative-decoding families (this build already ships
    eagle3, dflash and dspark alongside MTP). When the next one lands, adding
    estimator.draft_archs = {"<arch>": "draft-<x>"} to the config teaches the panel to
    recognise it -- no code change, and rescan picks it up on the next run."""
    out = dict(_DRAFT_ARCH_SPEC)
    try:
        out.update((cfg.get("estimator") or {}).get("draft_archs") or {})
    except Exception:
        pass
    return out


def classify_gguf(path, cfg=None):
    """What role does this file play in a model directory?

    Returns (role, detail) where role is one of:
        'projector'  a vision/audio mmproj (arch 'clip' or type 'mmproj')
        'draft'      a standalone draft head -- detail['spec_type'] says which --spec-type
                     it needs and detail['usable'] is False for a shared NextN head
        'model'      main weights. detail['mtp_embedded'] says whether MTP is available
                     without any extra file."""
    pr = gguf_probe(path)
    if not pr:
        return None, {}
    arch = (pr.get("arch") or "").lower()
    if arch == "clip" or (pr.get("type") or "").lower() == "mmproj":
        return "projector", {"arch": arch}
    known = draft_arch_spec(cfg)
    if arch in known:
        return "draft", {"arch": arch, "spec_type": known[arch], "usable": True}
    if pr["has_nextn"] and pr["n_tensors"] < 200:
        # a NextN head shipped as its own file: a couple of dozen tensors, not a whole
        # model. The shared variant omits token_embd.weight by design and llama.cpp fails
        # it with "check_tensor_dims: tensor 'token_embd.weight' not found".
        return "draft", {"arch": arch, "spec_type": "draft-mtp",
                         "usable": not pr["nextn_shared"],
                         "why": "shared head: omits token_embd.weight" if pr["nextn_shared"] else ""}
    return "model", {"arch": arch, "mtp_embedded": pr["has_nextn"],
                     "n_tensors": pr["n_tensors"]}


def gguf_has_nextn(path):
    """Back-compat shim: does this file embed a NextN/MTP head?"""
    pr = gguf_probe(path)
    return None if pr is None else pr["has_nextn"]


def model_has_mtp_head(cfg, model_id):
    """Does this model's main GGUF embed the MTP head? Detected from the file; the
    config key 'mtp_embedded' is only an override for when the file is not readable."""
    m = (cfg.get("models") or {}).get(model_id or "", {})
    if not m:
        return None
    d, g = expand(m.get("dir", "")), preset_gguf(cfg, {"model": model_id})
    got = gguf_has_nextn(os.path.join(d, g) if d and g else None)
    if got is not None:
        return got
    return m.get("mtp_embedded")


def mtp_head_gb(cfg, model_id):
    """Weight cost of the embedded head, loaded ONLY when MTP is on. Measured from the
    tensor table at 0.351 GB for UD-IQ3_XXS, 0.351 for UD-IQ4_XS and 0.348 for
    GSQ-RCO-IQ3_XXS-mtp -- the head is stored at the same precision regardless of the
    trunk quant, so one constant covers the family. 0 for a head-less GGUF."""
    if not model_has_mtp_head(cfg, model_id):
        return 0.0
    m = (cfg.get("models") or {}).get(model_id or "", {})
    v = m.get("mtp_head_gb")
    if v is None:
        v = (cfg.get("estimator") or {}).get("mtp_head_gb", 0.35)
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _build_suffixes(stems):
    """Turn ['x-IQ3_S', 'x-IQ3_S-mtp'] into ['', '-mtp'] so two builds of one model in one
    directory get ids that differ by what actually differs about them."""
    if len(stems) < 2:
        return [""] * len(stems)
    pre = os.path.commonprefix(stems)
    out = [st[len(pre):] for st in stems]
    return out if all(out.count(x) == 1 for x in out) else ["-" + st for st in stems]


_KV_BYTES = {"f16": 2.0, "bf16": 2.0, "q8_0": 1.0625, "q5_1": 0.75, "q5_0": 0.6875,
             "q4_1": 0.5625, "q4_0": 0.53125}
# bytes per element INCLUDING the block scales: q8_0 is 34 bytes per 32 values,
# q4_0 is 17 per 32. Getting this wrong under-reports KV by ~6% at q8_0.


def kv_bytes_per_token(hp, kv_type="q8_0"):
    """KV cache cost of ONE token, computed from the header instead of curve-fitted.

    sum over layers of (n_head_kv * (key_length + value_length)) * bytes-per-element.
    head_count_kv is often a per-layer ARRAY, and in a hybrid model its zeros mark the
    recurrent layers, which keep no KV at all -- which is why Qwen3.8-27B's KV is so much
    cheaper per 1k tokens than its block count suggests. This is what lets the panel
    estimate a model it has never seen, instead of relying on a fit measured on one."""
    if not hp:
        return None
    n_layer = hp.get("block_count")
    hkv = hp.get("attention.head_count_kv")
    if n_layer is None or hkv is None:
        return None
    k_len = hp.get("attention.key_length")
    v_len = hp.get("attention.value_length")
    if k_len is None or v_len is None:
        embd, n_head = hp.get("embedding_length"), hp.get("attention.head_count")
        if isinstance(n_head, list):
            n_head = max(n_head) or 1
        if not embd or not n_head:
            return None
        k_len = v_len = embd // n_head
    per = _KV_BYTES.get(str(kv_type).lower(), 1.0625)
    heads = hkv if isinstance(hkv, list) else [hkv] * int(n_layer)
    total = sum(int(h) * (int(k_len) + int(v_len)) for h in heads[:int(n_layer)])
    # A hybrid model keeps a KV cache only on its full-attention layers; the rest are
    # SSM / linear-attention and cache nothing. Qwen3.8-27B says so with
    # full_attention_interval=4, and ignoring it overstates KV by 3-4x -- 0.071 GB per
    # 1k tokens at q4_0 against a measured 0.022.
    iv = hp.get("full_attention_interval")
    if iv and int(iv) > 1:
        total = total // int(iv)
    return total * per


def residency_plan(cfg, weight_gb, hp, max_tensor_b):
    """Can this model live in VRAM, in RAM, or only on disk -- and what does that imply?

    The single most consequential thing about a new model, and the one the old rescan
    said nothing about. A 10 GB dense model and an 82 GB MoE are not the same kind of
    object: one is a context-length problem, the other is a residency problem, and they
    want completely different flags."""
    vram = float((cfg.get("gpu") or {}).get("vram_total_gb") or 0)
    ram = host_ram_gb() or 0
    out = {"vram_gb": vram, "ram_gb": ram, "weight_gb": weight_gb}
    if not weight_gb:
        return out
    experts = (hp or {}).get("expert_count")
    # leave room for KV, compute buffers and the projector; 1.5 GB is the observed floor
    if weight_gb + 1.5 <= vram:
        out["residency"] = "vram"
        out["hint"] = "fits in VRAM: -ngl 99, no expert offload"
        out["suggest"] = {"ngl": 99}
    elif experts:
        out["residency"] = "disk" if weight_gb > ram else "ram"
        # each block's experts are roughly (weight - the non-expert remainder) / blocks;
        # start by offloading enough of them to get the rest under VRAM
        n_layer = int(hp.get("block_count") or 0)
        per_block = (weight_gb / n_layer) if n_layer else 0
        need = max(0.0, weight_gb + 1.5 - vram)
        floor = int(min(n_layer, round(need / per_block) + 1)) if per_block else None
        out["hint"] = ("%s of experts: start at --n-cpu-moe %s with -ngl auto and a "
                       "--fit-target, then sweep it" % (
                           "streams from disk" if out["residency"] == "disk" else
                           "lives in RAM", floor))
        out["suggest"] = {"ngl": "auto", "n_cpu_moe": floor, "fit_target": 1536,
                          "load_mode": "mmap", "parallel": 1}
        if out["residency"] == "disk":
            out["suggest"]["lazy_mode"] = "on"
    else:
        out["residency"] = "too-big"
        out["hint"] = ("dense and larger than VRAM: no expert offload is possible, so "
                       "only a lower -ngl (partial offload) or a smaller quant will run it")
        out["suggest"] = {"ngl": "auto", "fit_target": 1536}
    if max_tensor_b and max_tensor_b > 4 * (1 << 30):
        out["lazy_needed"] = round(max_tensor_b / 1e9, 1)
        out["suggest"] = dict(out.get("suggest") or {}, lazy_mode="on")
    return out


def scan_dir_roles(d, cfg=None):
    """Classify EVERY gguf in a directory by reading its header. Returns
    (models, projectors, drafts) where each entry is (filename, detail)."""
    try:
        files = sorted(f for f in os.listdir(d) if f.endswith(".gguf"))
    except OSError:
        return [], [], []
    models, projectors, drafts = [], [], []
    for f in files:
        # only the first shard carries the header; the rest are data
        mm = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", f)
        if mm and mm.group(2) != "00001":
            continue
        role, detail = classify_gguf(os.path.join(d, f), cfg)
        if role == "projector":
            projectors.append((f, detail))
        elif role == "draft":
            drafts.append((f, detail))
        elif role == "model":
            models.append((f, detail))
    return models, projectors, drafts


def detect_model_facts(cfg, d, gguf):
    """Everything about a model we can read off the disk instead of being told."""
    models, projectors, drafts = scan_dir_roles(d, cfg)
    names = [f for f, _ in models]
    if not names:
        return None
    gguf = gguf if gguf in names else names[0]
    detail = dict(models)[gguf]
    embedded = bool(detail.get("mtp_embedded"))
    pr = gguf_probe(os.path.join(d, gguf)) or {}
    hp = pr.get("hp") or {}
    usable_drafts = [f for f, dd in drafts if dd.get("usable")]
    facts = {
        "gguf": gguf,
        "arch": detail.get("arch"),
        "has_mmproj": bool(projectors),
        "vision": bool(projectors),
        "mmproj_file": projectors[0][0] if projectors else None,
        "mtp_embedded": embedded,
        "mtp_head_gb": (cfg.get("estimator") or {}).get("mtp_head_gb", 0.35)
                       if embedded else 0.0,
        "mtp_draft_files": usable_drafts,
        # rejected drafts are recorded, not hidden: "there is an mtp file here and it will
        # not load" is the single most confusing thing to discover at launch time
        "mtp_draft_rejected": {f: dd.get("why") or "unusable"
                               for f, dd in drafts if not dd.get("usable")},
        "mtp_source": "embedded" if embedded else ("external" if usable_drafts else "none"),
        "spec_types": sorted({dd["spec_type"] for _, dd in drafts
                              if dd.get("usable")} | ({"draft-mtp"} if embedded else set())),
        # --- read straight out of the header, not configured by hand ---
        "n_layer": hp.get("block_count"),
        "n_embd": hp.get("embedding_length"),
        "ctx_train": hp.get("context_length"),
        "n_expert": hp.get("expert_count"),
        "n_expert_used": hp.get("expert_used_count"),
        "max_tensor_gb": round((pr.get("max_tensor") or 0) / 1e9, 2) or None,
        "params_b": round((pr.get("n_elements") or 0) / 1e9, 1) or None,
        "attn_interval": hp.get("full_attention_interval"),
        # KV ONLY -- it excludes the compute buffers, which also grow with context. On
        # this box the measured total slope runs ~0.002-0.005 GB/1k above this.
        "kv_bytes_per_token": {q: round(kv_bytes_per_token(hp, q) or 0)
                               for q in ("f16", "q8_0", "q4_0")} if hp else None,
    }
    try:
        _, cands = None, [f for f, _ in models]
        all_g = sorted(f for f in os.listdir(d) if f.endswith(".gguf"))
        facts["weight_gb"] = round(sum(os.path.getsize(os.path.join(d, f))
                                       for f in gguf_shard_set(all_g, gguf)) / 1e9, 2)
    except OSError:
        pass
    facts["residency"] = residency_plan(cfg, facts.get("weight_gb"), hp,
                                        pr.get("max_tensor"))
    return facts


# Facts the rescan owns. Everything else in a model entry -- label, note, kind, and the
# supports_mtp OVERRIDE (Flash-Next sets it false on purpose, because its draft heads do
# not load on this build even though they exist) -- is the user's and is never touched.
_DETECTED_KEYS = ("gguf", "arch", "has_mmproj", "vision", "mmproj_file", "mtp_embedded",
                  "mtp_head_gb", "mtp_draft_files", "mtp_draft_rejected", "mtp_source",
                  "spec_types", "weight_gb", "n_layer", "n_embd", "ctx_train",
                  "n_expert", "n_expert_used", "max_tensor_gb", "params_b", "attn_interval",
                  "kv_bytes_per_token", "residency")


def model_roots(cfg):
    roots = {MODELS_ROOT}
    for m in (cfg.get("models") or {}).values():
        d = expand(m.get("dir", ""))
        if d:
            roots.add(os.path.dirname(d.rstrip("/")))
    return sorted(r for r in roots if os.path.isdir(r))


def find_gguf_everywhere(cfg, filename):
    """Where does this exact GGUF live now? Returns every directory holding it.

    Folders get reorganised -- two builds that shipped in separate directories get merged
    into one, a model gets moved to another disk. Marking the entry "missing" when the
    file is simply somewhere else is not good enough: the preset pointing at it would
    stay broken until someone noticed and retyped a path."""
    hits = []
    for root in model_roots(cfg):
        try:
            names = os.listdir(root)
        except OSError:
            continue
        if filename in names and os.path.isfile(os.path.join(root, filename)):
            hits.append(root)
        for name in names:
            d = os.path.join(root, name)
            if os.path.isdir(d) and os.path.isfile(os.path.join(d, filename)):
                hits.append(d)
    return sorted(set(hits))


def relocate_model(cfg, mid, m):
    """If a model's directory or file has moved, find it and return the new dir."""
    d = expand(m.get("dir", ""))
    gg = m.get("gguf")
    if d and os.path.isdir(d) and (not gg or os.path.isfile(os.path.join(d, gg))):
        return None
    if not gg:
        return None
    hits = [h for h in find_gguf_everywhere(cfg, gg) if os.path.realpath(h) != os.path.realpath(d or "")]
    if len(hits) == 1:
        return hits[0]
    return None


def resync_models(cfg, add_new=True):
    """Re-read every model directory and return (updates, changes).

    This is the whole point of the models config: nothing about a GGUF should have to be
    typed in by hand, because none of it is stable -- a projector gets dropped into a
    directory later, a second build lands beside the first, a folder gets moved. Read it
    all back off the disk and write it down."""
    updates, changes = {}, []
    for mid, m in (cfg.get("models") or {}).items():
        d = expand(m.get("dir", ""))
        moved = relocate_model(cfg, mid, m)
        if moved:
            changes.append({"model": mid, "field": "dir (moved)", "old": d, "new": moved})
            updates.setdefault(mid, {}).update(dir=moved, missing=False)
            d = moved
        if not d or not os.path.isdir(d):
            if not m.get("missing"):
                changes.append({"model": mid, "field": "dir", "old": d, "new": "MISSING"})
                updates.setdefault(mid, {})["missing"] = True
            continue
        facts = detect_model_facts(cfg, d, m.get("gguf"))
        if facts is None:
            changes.append({"model": mid, "field": "gguf", "old": m.get("gguf"),
                            "new": "UNREADABLE — no usable GGUF in this directory"})
            continue
        if m.get("missing"):
            updates.setdefault(mid, {})["missing"] = False
            changes.append({"model": mid, "field": "dir", "old": "MISSING", "new": d})
        for k in _DETECTED_KEYS:
            # never replace a measured head size with the family constant; only fill it in
            if k == "mtp_head_gb" and m.get(k) and facts.get(k):
                continue
            if k in facts and m.get(k) != facts[k]:
                changes.append({"model": mid, "field": k,
                                "old": m.get(k), "new": facts[k]})
                updates.setdefault(mid, {})[k] = facts[k]

    if not add_new:
        return updates, changes

    known = {os.path.realpath(expand(m.get("dir", "")))
             for m in (cfg.get("models") or {}).values() if m.get("dir")}
    roots = {MODELS_ROOT} | {os.path.dirname(expand(m["dir"]).rstrip("/"))
                             for m in (cfg.get("models") or {}).values() if m.get("dir")}
    for root in sorted(roots):
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if not os.path.isdir(d):
                continue
            try:
                cands = sorted(f for f in os.listdir(d) if f.endswith(".gguf")
                               and not f.lower().startswith("mmproj")
                               and not f.lower().startswith("mtp-"))
            except OSError:
                continue
            if not cands:
                continue
            # one entry per BUILD, not per directory: a repo can ship the plain and the
            # -mtp GGUF side by side and they are different models to every caller here
            builds = [c for c in cands if gguf_shard_set(cands, c) == [c]
                      or c.endswith("-00001-of-%s.gguf" % c.split("-of-")[-1])] or cands[:1]
            sufs = _build_suffixes([os.path.splitext(b)[0] for b in builds])
            for b, suf in zip(builds, sufs):
                mid = name + suf
                if mid in (cfg.get("models") or {}):
                    continue
                if os.path.realpath(d) in known and len(builds) == 1:
                    continue
                if any((mm.get("gguf") == b and
                        os.path.realpath(expand(mm.get("dir", ""))) == os.path.realpath(d))
                       for mm in (cfg.get("models") or {}).values()):
                    continue
                facts = detect_model_facts(cfg, d, b)
                if facts is None:
                    continue
                entry = dict(facts)
                entry.update(dir=d, label=mid, kind="dense-vl",
                             note="auto-registered by rescan")
                updates[mid] = entry
                changes.append({"model": mid, "field": "*new*", "old": None, "new": b})
    return updates, changes


def apply_model_resync(add_new=True):
    """Run the rescan and persist it. Returns the change list."""
    cfg = load_config()
    updates, changes = resync_models(cfg, add_new=add_new)
    if updates:
        def mut(d):
            ms = d.setdefault("models", {})
            for mid, fields in updates.items():
                ms.setdefault(mid, {}).update(fields)
        save_config(mut)
    return changes


def model_id_for_file(cfg, model_file):
    """Map a running server's .gguf back to the model id that owns it."""
    if not model_file:
        return None
    for mid in (cfg.get("models") or {}):
        if preset_gguf(cfg, {"model": mid}) == model_file:
            return mid
    return None


def base_weight_gb(cfg):
    """Weight size the ctx/KV fit was measured against. Prefer the live file, but
    fall back to the recorded constant so a renamed/moved model cannot silently
    turn every estimate back into a model-blind one."""
    est = cfg["estimator"]
    live = model_weight_gb(cfg, est.get("base_model", "qwen3-vl"))
    return live if live is not None else est.get("base_weight_gb")


def base_loaded_weight_gb(cfg):
    """What the fit ACTUALLY had resident. coeffs_gb were measured with MTP off, so the
    base model's own MTP head was skipped at load and must come off its file size --
    otherwise a head-less model looks 0.35 GB cheaper than it is."""
    b = base_weight_gb(cfg)
    if b is None:
        return None
    return b - mtp_head_gb(cfg, (cfg.get("estimator") or {}).get("base_model", "qwen3-vl"))


def effective_spec_type(cfg, spec_type):
    """Apply the global MTP master switch (see _launch): runtime 'none' wins."""
    if str((cfg.get("runtime") or {}).get("spec_type", "")).strip().lower() in ("none", "off"):
        return "none"
    return spec_type


def mtp_overhead_gb(cfg, ctx, spec_type):
    """VRAM the MTP draft head costs on top of the base fit. Measured on the RTX 5080
    (identical for q4_0 and q8_0 KV): 900 MiB @32k, 1030 MiB @64k -> 0.75 + 0.0039/1k."""
    spec_type = effective_spec_type(cfg, spec_type)
    if not spec_type or spec_type in ("none", "off"):
        return 0.0
    o = cfg["estimator"].get("mtp_overhead_gb")
    if not o:
        return 0.0
    return o[0] + o[1] * ctx / 1000.0


AUTO_SLOTS = 4   # llama.cpp: --parallel defaults to -1 = auto, and server.cpp then does
                 # `params.n_parallel = 4; params.kv_unified = true;`. So "auto" is not a
                 # memory-aware choice — it is always 4 slots sharing one KV pool.


def effective_slots(parallel):
    """How many server slots llama.cpp will actually open for this `parallel` value.
    None/''/0/'auto' -> AUTO_SLOTS, because omitting --parallel selects auto, not 1."""
    if parallel in (None, "", 0, "0", "auto"):
        return AUTO_SLOTS
    try:
        return max(1, int(parallel))
    except (TypeError, ValueError):
        return AUTO_SLOTS


def slot_overhead_gb(cfg, ctx, parallel):
    """VRAM the slots beyond the first cost. Unified KV does not multiply the pool, but
    the per-sequence compute buffers (KQ mask grows with n_kv) and output buffers do."""
    o = cfg["estimator"].get("slot_overhead_gb")
    if not o:
        return 0.0
    return max(0, effective_slots(parallel) - 1) * (o[0] + o[1] * ctx / 1000.0)


def vision_saving_gb(cfg, vision, model=None):
    """coeffs_gb were fit with the F16 projector loaded, so vision: off gives VRAM back.
    A model that ships no mmproj at all (the -mtp GGUFs are text-only) always gets it
    back, whatever the preset's vision field says -- there is nothing to load."""
    off = str(vision or "").strip().lower() in ("off", "no", "false", "0")
    if not off and model is not None and not model_caps(cfg, model)["mmproj"]:
        off = True
    if not off:
        return 0.0
    return float(cfg["estimator"].get("vision_gb") or 0.0)


def vram_estimate(cfg, ctx, kv, model=None, spec_type=None, parallel=1, vision=None):
    """Returns (gb, exact). exact=False means the model's own size could not be
    accounted for, so the number is only right for the fit's base model."""
    base, per1k = cfg["estimator"]["coeffs_gb"].get(kv, cfg["estimator"]["coeffs_gb"]["q4_0"])
    gb = (base + per1k * ctx / 1000.0 + mtp_overhead_gb(cfg, ctx, spec_type)
          + slot_overhead_gb(cfg, ctx, parallel) - vision_saving_gb(cfg, vision, model))
    if not model:
        return round(gb, 2), True
    a, b = model_weight_gb(cfg, model), base_loaded_weight_gb(cfg)
    if a is None or b is None:
        return round(gb, 2), False
    # mtp_overhead_gb was measured as (MTP on - MTP off) on the base model, so it already
    # contains ONE copy of the base model's head weights. Charge this model's own head
    # instead: subtract it from the file size when MTP is off (llama.cpp TENSOR_SKIPs it),
    # and take the base head back out of the measured overhead when it is on.
    head = mtp_head_gb(cfg, model)
    if str(effective_spec_type(cfg, spec_type) or "none").lower() in ("none", "off", ""):
        a -= head
    else:
        gb -= mtp_head_gb(cfg, (cfg.get("estimator") or {}).get("base_model", "qwen3-vl"))
    return round(gb + a - b, 2), True


def estimate_vram(cfg, ctx, kv, model=None, spec_type=None, parallel=1, vision=None):
    return vram_estimate(cfg, ctx, kv, model, spec_type, parallel, vision)[0]


def model_gguf_candidates(cfg, model_id):
    """Every GGUF in the model's directory that could be the MAIN weights: no projector,
    no mtp-* draft head. Ordered the same way serve-vlm.sh orders them (LC_ALL=C, i.e.
    plain byte order) so both sides pick the same file when nothing names one."""
    m = (cfg.get("models") or {}).get(model_id or "", {})
    d = expand(m.get("dir", "")) if m else ""
    if not d or not os.path.isdir(d):
        return d, []
    try:
        g = [f for f in os.listdir(d)
             if f.endswith(".gguf") and not f.lower().startswith("mmproj")
             and not f.lower().startswith("mtp-")]
    except OSError:
        return d, []
    return d, sorted(g)


def preset_gguf(cfg, p):
    """The weights file a preset would actually load, or None if unknown.

    Must agree with serve-vlm.sh exactly. A directory can hold more than one build --
    ISTA-DASLab ships GSQ-RCO's plain and -mtp GGUFs side by side -- and "..._S.gguf" vs
    "..._S-mtp.gguf" sort differently under shell collation than under Python's, so
    guessing here meant the panel could describe a different file than the one that
    loaded. The model's (or preset's) `gguf` key settles it; the sort is only a fallback."""
    m = (cfg.get("models") or {}).get(p.get("model") or "", {})
    want = p.get("gguf") or m.get("gguf")
    d, g = model_gguf_candidates(cfg, p.get("model"))
    if want:
        return want if (d and os.path.isfile(os.path.join(d, want))) else None
    return g[0] if g else None


def gguf_shard_set(files, main):
    """`main` plus its sibling shards. llama.cpp is handed -00001-of-000NN and loads the
    rest itself, so the on-disk size of a split model is the whole set, not one file."""
    mm = re.match(r"^(.*)-\d{5}-of-(\d{5})\.gguf$", main or "")
    if not mm:
        return [main] if main else []
    return sorted(f for f in files
                  if re.match(r"^%s-\d{5}-of-%s\.gguf$" % (re.escape(mm.group(1)),
                                                            re.escape(mm.group(2))), f))


def match_preset(cfg, run):
    if not run:
        return None
    cands = [p for p in cfg["presets"]
             if p["ctx"] == run.get("ctx") and p["kv"] == run.get("kv")
             # a preset with no slot count launches as 1 (resolve_launch's default), while a
             # server started without --parallel reports none at all, i.e. auto — keep both sides
             # using the same convention or nothing ever matches
             and slots_or_auto(p.get("parallel", 1)) == slots_or_auto(run.get("parallel"))]
    if not cands:
        return None
    # presets can now differ by model alone, so the weights file breaks the tie
    mf = run.get("model_file")
    if mf:
        cands = [p for p in cands if preset_gguf(cfg, p) == mf] or cands

    # ...and they can differ by draft depth alone (the text vs coding pair are the
    # same model/ctx/kv and vary only in --spec-draft-n-max), so match that too.
    def spec_key(spec_type, n_max):
        on = str(spec_type or "none").strip().lower() not in ("none", "off", "")
        return (on, int(n_max) if on and n_max else (2 if on else 0))
    run_key = spec_key(run.get("spec_type"), run.get("spec_n_max"))
    exact = [p for p in cands
             if spec_key(p.get("spec_type"), p.get("spec_n_max")) == run_key]
    if exact:
        return exact[0]["id"]
    return cands[0]["id"]


# ---------------------------------------------------------------- local switching
SWITCH_LOCK = threading.Lock()
SWITCH_STATE = {"phase": "idle", "target": None, "detail": "", "ts": 0}


def _set_switch(phase, target=None, detail=""):
    SWITCH_STATE.update(phase=phase, target=target, detail=detail, ts=time.time())


def _kill_server(cfg):
    bin_path = expand(cfg["server"]["llama_bin"])
    subprocess.run(["pkill", "-f", bin_path], check=False)
    for _ in range(25):
        if subprocess.run(["pgrep", "-f", bin_path], capture_output=True).returncode != 0:
            break
        time.sleep(1)
    time.sleep(2)


def _launch(cfg, launch):
    serve = expand(cfg["server"]["serve_script"])
    logp = expand(cfg["server"]["server_log"])
    env = os.environ.copy()
    env.update(CTX=str(launch["ctx"]), KV_QUANT=launch["kv"], NGL=str(launch["ngl"]),
               PORT=str(cfg["server"].get("vlm_port", 8000)))
    if launch.get("parallel"):          # absent -> serve-vlm.sh omits --parallel (auto)
        env["PARALLEL"] = str(launch["parallel"])
    if launch.get("model_dir"):
        env["MDIR"] = expand(launch["model_dir"])
    # Name the exact GGUF. Without it the launcher falls back to name order, which is not
    # the same order Python used to decide what the panel showed.
    gg = launch.get("gguf") or preset_gguf(cfg, {"model": launch.get("model")})
    if gg:
        env["MODEL_FILE"] = gg
    # and the projector, identified by its header rather than its name
    mj = (cfg.get("models") or {}).get(launch.get("model") or "", {}).get("mmproj_file")
    if mj:
        env["MMPROJ_FILE"] = mj
    # Optional runtime knobs — only set when specified, so serve-vlm.sh
    # defaults (incl. its thinking-aware sampling profile) still apply.
    rt = cfg.get("runtime", {})
    for key, envname in (("thinking", "THINKING"), ("context_shift", "CONTEXT_SHIFT"),
                         ("cache_reuse", "CACHE_REUSE"), ("reason_effort", "REASON_EFFORT"),
                         ("image_min_tokens", "IMAGE_MIN_TOKENS"),
                         ("image_max_tokens", "IMAGE_MAX_TOKENS"),
                         ("ubatch", "UBATCH"), ("batch", "BATCH"),
                         ("extra_flags", "EXTRA_FLAGS"), ("fit", "FIT"),
                         ("spec_type", "SPEC_TYPE"), ("spec_n_max", "SPEC_N_MAX"),
                         ("kv_unified", "KV_UNIFIED"),
                         ("kv_unified_per_slot", "KV_UNIFIED_PER_SLOT"),
                         ("flash_attn", "FLASH_ATTN"), ("vision", "VISION"),
                         ("n_cpu_moe", "NCMOE"), ("lazy_mode", "LAZY_MODE"),
                         # where the draft head comes from, and what the draft costs
                         ("mtp_head", "MTP_HEAD"), ("spec_draft_kv", "SPEC_DRAFT_KV"),
                         ("spec_n_min", "SPEC_N_MIN"),
                         # RAM / disk residency. These decide whether a model that does not
                         # fit in VRAM streams off NVMe, sits pinned in RAM, or refuses to
                         # load at all, so they belong in the preset like ctx and kv do.
                         ("load_mode", "LOAD_MODE"), ("cpu_moe", "CPU_MOE"),
                         ("fit_target", "FIT_TARGET"), ("fit_ctx", "FIT_CTX"),
                         ("override_tensor", "OVERRIDE_TENSOR"), ("numa", "NUMA"),
                         ("no_host", "NO_HOST"), ("threads", "THREADS"),
                         ("threads_batch", "THREADS_BATCH"),
                         ("reason_format", "REASON_FORMAT"),
                         # server-level sampling. Left unset by default so serve-vlm.sh's
                         # thinking-aware Unsloth profile still picks the right set.
                         ("temp", "TEMP"), ("top_p", "TOP_P"), ("top_k", "TOP_K"),
                         ("min_p", "MIN_P"), ("presence_penalty", "PRESENCE"),
                         ("repeat_penalty", "REPEAT_PEN")):
        val = launch.get(key, rt.get(key))
        if val not in (None, ""):
            env[envname] = str(val)

    # MTP master switch. Presets carry their own spec_type (several only fit with MTP
    # off), and preset values normally win over cfg["runtime"] — which meant the panel's
    # global MTP control did nothing once any preset was applied. Off wins in both
    # directions: global off disables MTP everywhere, and a preset asking for off stays
    # off even when the global switch is on.
    if str(rt.get("spec_type", "")).strip().lower() in ("none", "off"):
        env["SPEC_TYPE"] = "none"
        env.pop("SPEC_N_MAX", None)

    logf = open(logp, "wb")
    subprocess.Popen(["bash", serve], env=env, stdout=logf, stderr=subprocess.STDOUT,
                     start_new_session=True, cwd=os.path.expanduser("~"))


def _switch_worker(cfg, launch, label):
    try:
        _set_switch("stopping", label, "stopping current server")
        _kill_server(cfg)
        _set_switch("starting", label, "launching preset")
        _launch(cfg, launch)
        logp = expand(cfg["server"]["server_log"])
        for _ in range(90):
            time.sleep(2)
            low = "\n".join(tail(logp, 60)).lower()
            if "listening on" in low or "model loaded" in low:
                _set_switch("ready", label, "server is up")
                return
            if any(k in low for k in ("out of memory", "failed to allocate", "cudamalloc",
                                      "error loading model", "ggml_assert", "std::bad_alloc",
                                      "terminate called")):
                _set_switch("error", label, "load failed (likely OOM) — pick a smaller preset")
                return
        _set_switch("error", label, "timed out waiting for server")
    except Exception as e:
        _set_switch("error", label, str(e))


def start_switch(cfg, launch, label):
    with SWITCH_LOCK:
        if SWITCH_STATE["phase"] in ("stopping", "starting"):
            return False, "a switch is already in progress"
        _set_switch("stopping", label, "queued")
    threading.Thread(target=_switch_worker, args=(cfg, launch, label), daemon=True).start()
    return True, "switching"


# ---------------------------------------------------------------- engine switching
def _wait_gpu_free(timeout=25):
    for _ in range(timeout):
        eng, _, _ = detect_engine(load_config())
        if eng is None:
            break
        time.sleep(1)
    time.sleep(2)


def stop_all_engines(cfg):
    """Free the GPU by stopping whichever local engine is running (only ONE fits 16GB).
    'vllm-env' catches vLLM's EngineCore child procs, not just the `vllm serve` parent."""
    for pat in (expand(cfg["server"]["llama_bin"]), "vllm-env", "vllm serve", "ollama serve"):
        subprocess.run(["pkill", "-f", pat], check=False)
    _wait_gpu_free()


# ------------------------------------------------------------------ benchmarks
# llama-bench answers "which of these settings is faster on THIS box" directly: most of
# its flags take a list and it runs the cross product, so `-ncmoe 40,48` is one job with
# two rows, not two server restarts. It cannot measure MTP (no --spec-type/-md), so
# speculative decoding still has to be compared by running the server -- that is what the
# panel's measured_tps on a preset is for.
BENCH_STATE = {"running": False, "started": 0, "label": "", "cmd": "", "log": [],
               "rows": [], "error": None, "done": 0}
_BENCH_LOCK = threading.Lock()

_BENCH_ENV_KEYS = ("PROMPT", "GEN", "REPS", "NGL", "NCMOE", "LOAD_MODE", "LAZY_MODE",
                   "KV", "FLASH_ATTN", "UBATCH", "BATCH", "THREADS", "OVERRIDE_TENSOR",
                   # llama-bench does NOT fit by default the way llama-server does, so a
                   # model bigger than VRAM needs -fitt or it simply fails to load
                   "FIT_TARGET", "FIT_CTX", "EXTRA")


def bench_rows_from_json(text):
    """llama-bench -o json prints one object per (model, settings) combination."""
    try:
        data = json.loads(text)
    except Exception:
        return []
    rows = []
    for r in data if isinstance(data, list) else [data]:
        rows.append({
            "model": r.get("model_filename") or r.get("model_type"),
            "n_cpu_moe": r.get("n_cpu_moe"), "ngl": r.get("n_gpu_layers"),
            "load_mode": r.get("load_mode"), "lazy_mode": r.get("lazy_mode"),
            "type_k": r.get("type_k"), "type_v": r.get("type_v"),
            "n_batch": r.get("n_batch"), "n_ubatch": r.get("n_ubatch"),
            "threads": r.get("n_threads"), "flash_attn": r.get("flash_attn"),
            "test": ("pp%s" % r["n_prompt"]) if r.get("n_prompt") else
                    ("tg%s" % r.get("n_gen")),
            "tps": r.get("avg_ts"), "tps_stddev": r.get("stddev_ts"),
        })
    return rows


def _bench_worker(cfg, env, label, cmd_str):
    proc = None
    try:
        serve = os.path.join(ROOT, "serve", "bench.sh")
        proc = subprocess.Popen(["bash", serve], env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True,
                                start_new_session=True, cwd=os.path.expanduser("~"))
        out, err = proc.communicate(timeout=3 * 3600)
        try:
            with open(os.path.join(ROOT, "logs", "bench.log"), "a") as lf:
                lf.write("\n===== %s =====\n%s\n%s\n%s\n"
                         % (time.strftime("%F %T"), cmd_str, err or "", out or ""))
        except OSError:
            pass
        rows = bench_rows_from_json(out)
        with _BENCH_LOCK:
            BENCH_STATE["rows"] = rows
            BENCH_STATE["log"] = (err or "").strip().splitlines()[-60:]
            if not rows:
                BENCH_STATE["error"] = (err or out or "llama-bench produced no rows")[-2000:]
    except Exception as e:
        with _BENCH_LOCK:
            BENCH_STATE["error"] = str(e)
        if proc:
            try:
                proc.kill()
            except Exception:
                pass
    finally:
        with _BENCH_LOCK:
            BENCH_STATE["running"] = False
            BENCH_STATE["done"] = time.time()
        # persist so the numbers outlive the panel process
        try:
            save_config(lambda d: d.setdefault("benchmarks", {}).__setitem__(
                label, {"at": time.time(), "cmd": cmd_str,
                        "rows": BENCH_STATE["rows"], "error": BENCH_STATE["error"]}))
        except Exception:
            pass


def _server_tps(cfg, prompt, gen_tokens, timeout=300):
    """One completion against the RUNNING server; returns llama.cpp's own timings."""
    port = engine_port(cfg, "llamacpp")
    key = read_api_key(cfg)
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/chat/completions" % port,
        data=json.dumps({"model": "qwen3-vl", "max_tokens": gen_tokens, "temperature": 0.7,
                         "messages": [{"role": "user", "content": prompt}]}).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": "Bearer " + key} if key else {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode())
    t = body.get("timings") or {}
    return {"tps": t.get("predicted_per_second"), "pp_tps": t.get("prompt_per_second"),
            "n_gen": t.get("predicted_n"), "n_prompt": t.get("prompt_n")}


def _wait_healthy(cfg, timeout=600):
    end = time.time() + timeout
    while time.time() < end:
        if local_health(cfg) == "ready":
            return True
        time.sleep(3)
    return False


def _server_bench_worker(cfg, variants, prompt, gen_tokens, reps, label):
    """Benchmark by actually RUNNING each variant.

    This is the half llama-bench cannot do: it has no --spec-type and no -md, so MTP --
    the single biggest performance lever on this box -- is invisible to it. Here each
    variant is a real preset (optionally with overrides such as spec_type), launched,
    waited for, measured with real completions, and torn down."""
    rows = []
    try:
        for v in variants:
            body = {"preset": v["preset"]}
            body.update(v.get("overrides") or {})
            launch, _ = resolve_launch(cfg, body)
            if not launch:
                rows.append({"variant": v.get("name") or v["preset"],
                             "error": "unknown preset"})
                continue
            msg = "starting %s…" % (v.get("name") or v["preset"])
            with _BENCH_LOCK:
                BENCH_STATE["log"] = (BENCH_STATE["log"] + [msg])[-60:]
            try:
                with open(os.path.join(ROOT, "logs", "bench.log"), "a") as lf:
                    lf.write("%s  %s\n" % (time.strftime("%F %T"), msg))
            except OSError:
                pass
            _kill_server(cfg)
            time.sleep(2)
            _launch(cfg, launch)
            if not _wait_healthy(cfg):
                rows.append({"variant": v.get("name") or v["preset"], "error": "never became ready"})
                continue
            got = []
            for _ in range(max(1, reps)):
                try:
                    got.append(_server_tps(cfg, prompt, gen_tokens))
                except Exception as e:
                    rows.append({"variant": v.get("name") or v["preset"], "error": str(e)[:200]})
                    break
            ok = [g for g in got if g.get("tps")]
            # a draft depth means nothing with speculation off; reporting the inherited
            # value there made the control row look like it had MTP on
            spec = launch.get("spec_type") or cfg.get("runtime", {}).get("spec_type")
            spec_off = str(spec or "none").lower() in ("none", "off", "")
            if ok:
                # last-run decode rate: the first request pays for a cold KV cache
                best = max(g["tps"] for g in ok)
                rows.append({
                    "variant": v.get("name") or v["preset"],
                    "spec_type": spec, "spec_n_max": None if spec_off else (
                        launch.get("spec_n_max") or cfg.get("runtime", {}).get("spec_n_max")),
                    "ctx": launch.get("ctx"), "type_k": launch.get("kv"),
                    "n_cpu_moe": launch.get("n_cpu_moe"), "load_mode": launch.get("load_mode"),
                    "test": "tg%d" % gen_tokens,
                    "tps": round(best, 2),
                    "pp_tps": round(max(g.get("pp_tps") or 0 for g in ok), 1),
                    "runs": len(ok),
                })
            with _BENCH_LOCK:
                BENCH_STATE["rows"] = list(rows)
    except Exception as e:
        with _BENCH_LOCK:
            BENCH_STATE["error"] = str(e)
    finally:
        _kill_server(cfg)
        with _BENCH_LOCK:
            BENCH_STATE["running"] = False
            BENCH_STATE["done"] = time.time()
            BENCH_STATE["rows"] = list(rows)
        try:
            save_config(lambda d: d.setdefault("benchmarks", {}).__setitem__(
                label, {"at": time.time(), "cmd": "server bench · %d variant(s) · tg%d ×%d"
                                                  % (len(variants), gen_tokens, reps),
                        "rows": rows, "error": BENCH_STATE["error"], "kind": "server"}))
        except Exception:
            pass


def run_bench_settings(cfg, body):
    """Launch a REAL llama-server using the benchmark form's settings.

    The sweep above measures with llama-bench, which is a different binary with a
    different loader. This closes that gap: take the combination the sweep picked, run it
    as the actual server the panel will serve from, and watch it in the log. A setting is
    not proven until the thing that serves it has run it."""
    mid = body.get("model")
    m = (cfg.get("models") or {}).get(mid or "")
    if not m:
        return False, "unknown model '%s'" % mid

    def one(v):
        """The bench form takes LISTS; a real server takes one value. Use the first."""
        if v in (None, ""):
            return None
        return str(v).split(",")[0].strip()

    launch = {"model": mid, "model_dir": m.get("dir"),
              "gguf": preset_gguf(cfg, {"model": mid}),
              "ctx": int(body.get("ctx") or 32768),
              "kv": one(body.get("kv")) or "q4_0",
              "parallel": 1,
              "ngl": ngl_or_auto(one(body.get("ngl")) or 99)}
    for src, dst in (("ncmoe", "n_cpu_moe"), ("load_mode", "load_mode"),
                     ("lazy_mode", "lazy_mode"), ("ubatch", "ubatch"),
                     ("batch", "batch"), ("threads", "threads"),
                     ("override_tensor", "override_tensor"),
                     ("spec_type", "spec_type"), ("spec_n_max", "spec_n_max"),
                     ("extra", "extra_flags")):
        v = one(body.get(src)) if src != "extra" else body.get(src)
        if v not in (None, ""):
            launch[dst] = v
    launch = drop_inapplicable(cfg, launch, mid)
    _kill_server(cfg)
    time.sleep(2)
    _launch(cfg, launch)
    return True, "started — watch it in Command & log"


def start_server_bench(cfg, body):
    """Compare real presets by running them. `variants` is a list of
    {name, preset, overrides:{spec_type,...}}; with none given it compares the named
    preset with MTP on and off, which is the comparison people actually want."""
    with _BENCH_LOCK:
        if BENCH_STATE["running"]:
            return False, "a benchmark is already running"
    variants = body.get("variants")
    if not variants:
        pid = body.get("preset")
        p = next((x for x in cfg["presets"] if x["id"] == pid), None)
        if not p:
            return False, "unknown preset '%s'" % pid
        caps = model_caps(cfg, p.get("model"))
        if not caps["mtp"]:
            return False, ("%s cannot run MTP (its GGUF has no NextN head), so there is "
                           "nothing to compare — give explicit variants instead" % pid)
        variants = [
            {"name": "%s · MTP off" % pid, "preset": pid,
             "overrides": {"custom": None, "spec_type": "none"}},
            {"name": "%s · MTP n=2" % pid, "preset": pid,
             "overrides": {"spec_type": "draft-mtp", "spec_n_max": 2}},
            {"name": "%s · MTP n=3" % pid, "preset": pid,
             "overrides": {"spec_type": "draft-mtp", "spec_n_max": 3}},
        ]
    for v in variants:
        (v.get("overrides") or {}).pop("custom", None)
    prompt = body.get("prompt_text") or (
        "Write a complete, well-structured Python module that implements an LRU cache "
        "with a decorator API, thread safety, and unit tests. Explain each design choice.")
    gen = int(body.get("gen") or 256)
    reps = int(body.get("reps") or 2)
    label = body.get("label") or "server bench %s" % time.strftime("%m-%d %H:%M")
    with _BENCH_LOCK:
        BENCH_STATE.update(running=True, started=time.time(), label=label,
                           cmd="server bench · %d variants" % len(variants),
                           rows=[], log=[], error=None, done=0)
    threading.Thread(target=_server_bench_worker,
                     args=(cfg, variants, prompt, gen, reps, label), daemon=True).start()
    return True, label


def host_ram_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / (1024 * 1024), 1)
    except OSError:
        pass
    return None


def bench_caveats(cfg, mid, body):
    """Warnings that decide whether a result means anything.

    The big one: a model larger than RAM is read from disk through the page cache, so a
    short llama-bench run measures how much of the file happened to be cached, not the
    setting under test -- and whichever configuration runs FIRST pays the cold-cache cost
    and loses. Measured here: llama-bench put n-cpu-moe 48 ahead of 40 (7.39 vs 7.22
    tok/s) while the running server, warm, had 40 ahead by a wide margin (16.9 vs 11.2).
    The sweep was not wrong about its own numbers; it was answering a different question."""
    out = []
    ram = host_ram_gb()
    w = model_weight_gb(cfg, mid)
    if ram and w and w > ram * 0.8:
        out.append(
            "%s is %.0f GB against %.0f GB of RAM, so it streams from disk. Short runs "
            "measure the page cache, not the setting: the variant that runs first is "
            "cold and will lose. Use the server benchmark for this model, or raise "
            "repetitions and re-run each variant on its own." % (mid, w, ram))
    try:
        if int(body.get("reps") or 3) < 3 and len(str(body.get("ncmoe") or "").split(",")) > 1:
            out.append("Fewer than 3 repetitions on a multi-variant sweep is noisy; "
                       "llama-bench defaults to 5 for a reason.")
    except (TypeError, ValueError):
        pass
    return out


def start_bench(cfg, body):
    """Kick off a sweep. Refuses while a model holds the GPU -- llama-bench wants it."""
    with _BENCH_LOCK:
        if BENCH_STATE["running"]:
            return False, "a benchmark is already running"
    run = running_server(cfg)
    if run and not body.get("force"):
        return False, ("stop the model first — llama-bench needs the GPU to itself "
                       "(or pass force to run anyway)")
    mid = body.get("model") or (cfg.get("profiles", {}) or {}).get("default")
    m = (cfg.get("models") or {}).get(mid or "")
    if not m:
        return False, "unknown model '%s'" % mid
    d = expand(m.get("dir", ""))
    if not d or not os.path.isdir(d):
        return False, "model directory is missing"
    env = os.environ.copy()
    env["MDIR"] = d
    gg = preset_gguf(cfg, {"model": mid})
    if gg:
        env["MODEL_FILE"] = gg
    for k in _BENCH_ENV_KEYS:
        v = body.get(k.lower())
        if v not in (None, ""):
            env[k] = str(v)
    label = body.get("label") or "%s %s" % (mid, time.strftime("%m-%d %H:%M"))
    cmd = " ".join("%s=%s" % (k, env[k]) for k in _BENCH_ENV_KEYS if k in env)
    with _BENCH_LOCK:
        BENCH_STATE.update(running=True, started=time.time(), label=label, cmd=cmd,
                           rows=[], log=bench_caveats(cfg, mid, body), error=None, done=0)
    threading.Thread(target=_bench_worker, args=(cfg, env, label, cmd), daemon=True).start()
    return True, label


def _launch_vllm(cfg, body):
    serve = expand(cfg.get("engines", {}).get("vllm", {}).get("serve_script", "~/serve-vllm.sh"))
    logp = expand(cfg["server"].get("vllm_log", "~/vllm-server.log"))
    v = cfg.get("engines", {}).get("vllm", {})
    env = os.environ.copy()
    env["PORT"] = str(engine_port(cfg, "vllm"))
    for key, envname in (("model", "VLLM_MODEL"), ("max_model_len", "MAX_MODEL_LEN"),
                         ("gpu_util", "GPU_UTIL"), ("quantization", "QUANT"),
                         ("kv_cache_dtype", "KV_CACHE_DTYPE"), ("max_num_seqs", "MAX_NUM_SEQS"),
                         ("reasoning_parser", "REASONING_PARSER"), ("enable_prefix_caching", "PREFIX_CACHE"),
                         ("attention_backend", "VLLM_ATTENTION_BACKEND"),
                         ("flash_attn_version", "VLLM_FLASH_ATTN_VERSION"),
                         ("enforce_eager", "ENFORCE_EAGER")):
        val = body.get(key, v.get(key))
        if val not in (None, ""):
            env[envname] = str(val)
    logf = open(logp, "wb")
    subprocess.Popen(["bash", serve], env=env, stdout=logf, stderr=subprocess.STDOUT,
                     start_new_session=True, cwd=os.path.expanduser("~"))


def _launch_ollama(cfg, body):
    # Start the ollama daemon (exposes an OpenAI-compatible API on :11434/v1).
    # The model is loaded lazily on first request; it must already be pulled.
    logp = expand(cfg["server"].get("ollama_log", "~/ollama-server.log"))
    ob = ollama_bin() or "ollama"
    env = os.environ.copy()
    env.setdefault("OLLAMA_HOST", "0.0.0.0:%d" % engine_port(cfg, "ollama"))
    env["PATH"] = os.path.expanduser("~/.local/bin") + ":" + env.get("PATH", "")
    logf = open(logp, "wb")
    subprocess.Popen([ob, "serve"], env=env, stdout=logf, stderr=subprocess.STDOUT,
                     start_new_session=True, cwd=os.path.expanduser("~"))


def start_engine(cfg, eng, body):
    with SWITCH_LOCK:
        if SWITCH_STATE["phase"] in ("stopping", "starting"):
            return False, "a switch is already in progress"
        _set_switch("stopping", eng, "switching engine")

    def worker():
        try:
            _set_switch("stopping", eng, "stopping current engine")
            stop_all_engines(cfg)
            _set_switch("starting", eng, f"launching {eng}")
            fresh = load_config()
            if eng == "llamacpp":
                pid = fresh.get("profiles", {}).get("default") or fresh["presets"][0]["id"]
                launch, _ = resolve_launch(fresh, {"preset": pid})
                _launch(fresh, launch)
            elif eng == "vllm":
                _launch_vllm(fresh, body)
            elif eng == "ollama":
                _launch_ollama(fresh, body)
            save_config(lambda d: d.update(active_engine=eng, active_endpoint="local"))
            port = engine_port(fresh, eng)
            hp = "/api/tags" if eng == "ollama" else "/health"
            for _ in range(120):
                time.sleep(2)
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}{hp}", timeout=2) as r:
                        if r.status == 200:
                            _set_switch("ready", eng, f"{eng} is up")
                            return
                except Exception:
                    pass
            _set_switch("error", eng, f"{eng} did not become ready — check its log")
        except Exception as e:
            _set_switch("error", eng, str(e))

    threading.Thread(target=worker, daemon=True).start()
    return True, "switching engine"


# runtime knobs a preset (or custom/body) may override; when unset here, _launch
# falls back to the global cfg["runtime"], then to serve-vlm.sh's own defaults.
_RT_OVERRIDE_KEYS = ("thinking", "context_shift", "cache_reuse", "reason_effort",
                     "image_min_tokens", "image_max_tokens", "batch", "ubatch",
                     "extra_flags", "fit", "flash_attn", "spec_type", "spec_n_max", "vision",
                     # which KV layout a multi-slot preset uses: shared pool vs ctx split
                     # per slot. Was global-only, so no preset could pick its own.
                     "kv_unified", "kv_unified_per_slot",
                     # disk-streaming models (Qwen3.8-Flash-Next): how many leading
                     # layers keep their experts on CPU, and on-demand reads of
                     # oversized tensors (the 26.8 GiB engram table).
                     "n_cpu_moe", "lazy_mode",
                     # RAM/disk residency and where a model that does not fit in VRAM
                     # actually lives. --load-mode is the one that decides mmap-from-NVMe
                     # vs pinned-in-RAM, and it supersedes --mmap/--mlock (both deprecated).
                     "load_mode", "cpu_moe", "fit_target", "fit_ctx",
                     "override_tensor", "numa", "no_host", "threads", "threads_batch",
                     # speculative decoding: which head, how deep, and what the draft's own
                     # KV costs. mtp_head is the explicit answer to "embedded or a file".
                     "mtp_head", "spec_n_min", "spec_draft_kv",
                     "reason_format",
                     # server-level sampling overrides (normally left unset)
                     "temp", "top_p", "top_k", "min_p", "presence_penalty", "repeat_penalty")


def slots_or_auto(v):
    """Normalise a slot count. None/blank/0/'auto' -> None, meaning: don't pass
    --parallel at all, so llama.cpp picks and shares one KV pool."""
    if v in (None, "", 0, "0", "auto"):
        return None
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def ngl_or_auto(v):
    """Normalise -ngl. 'auto' passes through so serve-vlm.sh omits the flag
    entirely and llama.cpp's --fit chooses the split. Required for models that
    depend on --fit: passing n_gpu_layers explicitly makes llama.cpp abort the
    fit, after which the compute buffers OOM on a 16 GB card."""
    if v in (None, "", "auto"):
        return "auto"
    try:
        return int(v)
    except (TypeError, ValueError):
        return "auto"


def model_caps(cfg, model_id):
    """What the model family actually supports. Qwen3.8-27B is dense + F16 projector;
    Qwen3.8-Flash-Next is a disk-streaming MoE with no projector and no usable MTP graph."""
    m = (cfg.get("models") or {}).get(model_id or "", {})
    kind = m.get("kind") or "dense-vl"
    embedded = model_has_mtp_head(cfg, model_id)
    # A dense GGUF can run MTP only if the head is IN the file; the streaming MoE needs a
    # separate mtp-*.gguf next to it and has no usable MTP graph here either way.
    mtp = m.get("supports_mtp", True) and kind != "moe-stream"
    if embedded is False:
        mtp = False
    d = expand(m.get("dir", ""))
    mmproj = m.get("has_mmproj", True) and kind != "moe-stream"
    if d and os.path.isdir(d):
        try:
            mmproj = any(f.lower().startswith("mmproj") and f.endswith(".gguf")
                         for f in os.listdir(d))
        except OSError:
            pass
    return {"kind": kind, "mtp": mtp, "mtp_embedded": bool(embedded),
            "mmproj": mmproj, "moe": kind == "moe-stream"}


def drop_inapplicable(cfg, launch, model_id):
    """Never pass one family's knobs to the other. The panel leaves stale fields in the
    preset on purpose (editing a preset must not rewrite them), so filter at launch:
    -ncmoe/--lazy-mode mean nothing to the dense 27B, and the MoE has no projector and
    no MTP graph — a stale spec_type there is a load failure, not a slow path."""
    caps = model_caps(cfg, model_id)
    if not caps["moe"]:
        for k in ("n_cpu_moe", "cpu_moe", "lazy_mode"):
            launch.pop(k, None)
    else:
        launch.pop("vision", None)
    # --fit only adjusts arguments that are UNSET. With an explicit -ngl llama.cpp aborts
    # the fit ("n_gpu_layers already set by user") and the compute buffers then OOM, so a
    # preset that pins ngl must not also ship fit knobs.
    if str(launch.get("ngl", "")).lower() != "auto":
        for k in ("fit", "fit_target", "fit_ctx"):
            launch.pop(k, None)
    # A stale spec_type on a model whose GGUF has no NextN head is a load failure, not a
    # slow path -- and that is not a MoE-only hazard: the GSQ-RCO IQ3_XXS build ships
    # head-less while its -mtp sibling does not. Check the capability, not the family.
    if not caps["mtp"]:
        launch["spec_type"] = "none"
        for k in ("spec_n_max", "spec_n_min", "mtp_head", "spec_draft_kv"):
            launch.pop(k, None)
    elif str(launch.get("spec_type", "")).lower() in ("none", "off"):
        for k in ("spec_n_max", "spec_n_min", "mtp_head", "spec_draft_kv"):
            launch.pop(k, None)
    if not caps["mmproj"]:
        launch["vision"] = "off"
    return launch


def resolve_launch(cfg, body):
    if body.get("preset"):
        p = next((x for x in cfg["presets"] if x["id"] == body["preset"]), None)
        if not p:
            return None, f"unknown preset '{body['preset']}'"
        model = cfg["models"].get(p.get("model", ""), {})
        launch = {"ctx": int(p["ctx"]), "kv": p["kv"],
                  "parallel": slots_or_auto(body.get("parallel", p.get("parallel", 1))),
                  "ngl": ngl_or_auto(p.get("ngl", 99)), "model_dir": model.get("dir"),
                  "model": p.get("model"),
                  "gguf": p.get("gguf") or model.get("gguf")}
        # request body wins over preset; only include keys actually specified so the
        # global runtime fallback in _launch still applies for the rest.
        for k in _RT_OVERRIDE_KEYS:
            if body.get(k) is not None:
                launch[k] = body[k]
            elif p.get(k) is not None:
                launch[k] = p[k]
        return drop_inapplicable(cfg, launch, p.get("model")), p["id"]
    c = body.get("custom")
    if c:
        if c.get("kv", "q4_0") not in ("f16", "q8_0", "q4_0"):
            return None, "kv must be f16, q8_0, or q4_0"
        model = cfg["models"].get(c.get("model", ""), {})
        launch = {"ctx": int(c["ctx"]), "kv": c.get("kv", "q4_0"),
                  "parallel": slots_or_auto(c.get("parallel", 1)), "ngl": ngl_or_auto(c.get("ngl", 99)),
                  "model_dir": model.get("dir"), "model": c.get("model"),
                  "gguf": c.get("gguf") or model.get("gguf")}
        for k in _RT_OVERRIDE_KEYS:
            if c.get(k) is not None:
                launch[k] = c[k]
        return drop_inapplicable(cfg, launch, c.get("model")), "custom"
    return None, "body needs 'preset' or 'custom'"


# ---------------------------------------------------------------- auto-switch
AUTO_SWITCH_LOCK = threading.Lock()

def _client_model(body):
    try:
        d = json.loads(body)
        return str(d.get("model") or "").strip()
    except Exception:
        return ""

def tag_alias(tag):
    """A preset's tag, normalized to a client-safe model name.
    'Quality-Blance' -> 'quality-blance'; '' -> None (preset is not mapped)."""
    if not tag:
        return None
    s = re.sub(r"[^a-z0-9]+", "-", str(tag).lower()).strip("-")
    return s or None

def sync_auto_switch_map(cfg):
    """Rebuild auto_switch.map from the presets' tags.

    The tag IS the client model name: creating/renaming/deleting a preset's tag
    adds/removes its entry automatically. Manual aliases (added via /api/auto_switch)
    are kept separately in cfg['auto_switch']['manual'] and merged into map.
    """
    as_cfg = cfg.setdefault("auto_switch", {"enabled": False, "map": {}, "manual": {}})
    from_tags = {}
    for p in cfg.get("presets", []):
        a = tag_alias(p.get("tag"))
        if a:
            from_tags[a] = p["id"]
    # Manual aliases (explicitly added via /api/auto_switch) survive only while
    # their target preset still exists and no tag claims that name.
    known_ids = {p["id"] for p in cfg.get("presets", [])}
    manual = as_cfg.get("manual") or {}
    manual = {k: v for k, v in manual.items() if v in known_ids and k not in from_tags}
    # Merge: tags win over manual aliases with the same name
    m = dict(manual)
    m.update(from_tags)
    as_cfg["map"] = m
    as_cfg["manual"] = manual
    return as_cfg

def _auto_switch_target(cfg, client_model):
    """The preset id the client's model name maps to, or None.

    The map is auto-derived from preset tags (plus any manual aliases). A client
    that names an alias gets that preset's GGUF (quantization), ctx and KV;
    naming the currently-loaded preset (or anything unmapped) means no switch."""
    if not client_model:
        return None
    sync_auto_switch_map(cfg)
    pid = (cfg.get("auto_switch") or {}).get("map", {}).get(client_model)
    if not pid:
        return None
    return next((p["id"] for p in cfg.get("presets", []) if p["id"] == pid), None)

def _wait_switch_ready(timeout=300):
    """Block until SWITCH_STATE says the new server is up (or it failed)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = SWITCH_STATE
        if st["phase"] == "ready":
            return True
        if st["phase"] == "error":
            return False
        time.sleep(1)
    return False

def _maybe_auto_switch(cfg, client_model):
    """If the client asked for a mapped model that differs from what is loaded,
    trigger the switch and wait for it. Returns (ok, message).

    ok=True  -> server is now serving the requested preset (or already was)
    ok=False -> message explains: busy / not running / load failed"""
    if not (cfg.get("auto_switch") or {}).get("enabled"):
        return True, "disabled"
    want = _auto_switch_target(cfg, client_model)
    if not want:
        return True, "unmapped — serving current model"
    run = running_server(cfg)
    if not run or run.get("engine") != "llamacpp":
        # nothing local is up: start the mapped preset (e.g. after a remote activate)
        launch, label = resolve_launch(cfg, {"preset": want})
        if not launch:
            return False, f"cannot load preset '{want}'"
        ok, _ = start_switch(cfg, launch, label)
        if not ok:
            return False, SWITCH_STATE.get("detail", "switch refused")
        if not _wait_switch_ready():
            return False, "timed out waiting for the model to load"
        return True, "started"
    cur = match_preset(cfg, run)
    if cur == want:
        return True, "already serving"
    ok, _ = start_switch(cfg, *resolve_launch(cfg, {"preset": want})[:2])
    if not ok:
        return False, SWITCH_STATE.get("detail", "switch refused")
    if not _wait_switch_ready():
        return False, "timed out waiting for the model to load"
    return True, "switched"

# ---------------------------------------------------------------- endpoints
def endpoint_by_id(cfg, eid):
    return next((e for e in cfg.get("endpoints", []) if e["id"] == eid), None)


def endpoint_public(cfg, e):
    """Endpoint dict safe to send to the browser (no keys). Adds key_set flag."""
    out = {k: v for k, v in e.items() if k not in ("api_key",)}
    if e.get("type") == "remote":
        secrets = load_secrets(cfg)
        keyed = bool(secrets.get(e.get("key_ref", "")) or os.environ.get(e.get("key_env", "")))
        out["key_set"] = keyed
    return out


def remote_target(cfg, e):
    """Return (base_url, api_key, model) for a remote endpoint."""
    secrets = load_secrets(cfg)
    key = secrets.get(e.get("key_ref", "")) or os.environ.get(e.get("key_env", ""), "")
    return e["base_url"].rstrip("/"), key, e.get("model", "")


def check_remote(cfg, e):
    """Lightweight reachability check: GET {base}/models with the key."""
    base, key, _ = remote_target(cfg, e)
    try:
        req = urllib.request.Request(base + "/models")
        if key:
            req.add_header("Authorization", "Bearer " + key)
        with urllib.request.urlopen(req, timeout=8) as r:
            return {"ok": r.status == 200, "status": r.status}
    except urllib.error.HTTPError as ex:
        return {"ok": ex.code in (200, 401, 403), "status": ex.code,
                "note": "reachable (auth error)" if ex.code in (401, 403) else str(ex)}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}


# ---------------------------------------------------------------- ROUTER (proxy)
class RouterHandler(BaseHTTPRequestHandler):
    server_version = "llm-router/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _err(self, code, msg):
        b = json.dumps({"error": {"message": msg, "type": "router_error"}}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        try:
            self.wfile.write(b)
        except Exception:
            pass

    def _handle(self):
        cfg = load_config()
        path = self.path
        if not path.startswith("/v1"):
            if path in ("/health", "/"):
                return self._err(200, "router up") if False else self._ok_json({"status": "ok",
                        "active": cfg.get("active_endpoint")})
            return self._err(404, "router only serves /v1/*")

        # auth gate: require the local API key (same one harnesses already send).
        # Prevents random LAN devices from spending your cloud credits via the router.
        gate = read_api_key(cfg)
        if gate:
            got = ""
            auth = self.headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                got = auth[7:].strip()
            got = got or self.headers.get("X-Api-Key", "").strip()
            import hmac
            if not hmac.compare_digest(got, gate):
                return self._err(401, "unauthorized — set your API key in the client")

        eid = cfg.get("active_endpoint", "local")
        ep = endpoint_by_id(cfg, eid)
        if not ep:
            return self._err(503, f"active endpoint '{eid}' not found")

        # target
        if ep.get("type") == "local":
            eng = cfg.get("active_engine", "llamacpp")
            base = f"http://127.0.0.1:{engine_port(cfg, eng)}/v1"  # follows the active engine
            key = read_api_key(cfg)
            # llama.cpp answers to its alias; ollama/vllm need their real model id,
            # so rewrite the client's model to the active engine's configured model.
            force_model = cfg.get("engines", {}).get(eng, {}).get("model") if eng in ("ollama", "vllm") else None
        else:
            base, key, model = remote_target(cfg, ep)
            if not key:
                return self._err(401, f"no API key set for '{eid}' (fill secrets.json)")
            force_model = model

        # read request body
        n = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(n) if n else b""

        # auto-switch: the client's model name selects a preset (quant + ctx + KV).
        # Only meaningful on the local llama.cpp endpoint — remote endpoints have
        # their own model and no reload to do.
        if ep.get("type") == "local" and cfg.get("active_engine", "llamacpp") == "llamacpp":
            cm = _client_model(body)
            if cm:
                with AUTO_SWITCH_LOCK:
                    ok, msg = _maybe_auto_switch(cfg, cm)
                if not ok:
                    return self._err(503, f"auto-switch: {msg}")

        # inject gen defaults + model on chat/completions & completions
        upstream_path = path[3:] or "/"  # strip /v1
        if body and upstream_path in ("/chat/completions", "/completions", "/responses"):
            body = self._apply_gen(cfg, body, force_model)
        elif force_model and body:
            body = self._maybe_set_model(body, force_model)

        target = base + upstream_path
        stream = b'"stream":true' in body.replace(b" ", b"")

        # build upstream request
        try:
            u = urllib.parse.urlparse(target)
            conn_cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
            ctx = ssl.create_default_context() if u.scheme == "https" else None
            timeout = cfg.get("router", {}).get("request_timeout_s", 600)
            conn = conn_cls(u.netloc, timeout=timeout, context=ctx) if ctx else conn_cls(u.netloc, timeout=timeout)
            headers = {"Content-Type": "application/json", "Accept": self.headers.get("Accept", "*/*")}
            if key:
                headers["Authorization"] = "Bearer " + key
            if ep.get("extra_headers"):
                headers.update(ep["extra_headers"])
            up_path = u.path + (("?" + u.query) if u.query else "")
            conn.request(self.command, up_path, body=body or None, headers=headers)
            resp = conn.getresponse()
        except Exception as ex:
            return self._err(502, f"upstream connect failed: {ex}")

        # relay status + headers
        self.send_response(resp.status)
        hop = {"connection", "keep-alive", "transfer-encoding", "content-encoding", "content-length"}
        for k, v in resp.getheaders():
            if k.lower() in hop:
                continue
            self.send_header(k, v)
        if stream:
            self.send_header("Content-Type", resp.getheader("Content-Type", "text/event-stream"))
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        # stream body as chunked
        try:
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                self.wfile.write(b"%X\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _ok_json(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _apply_gen(self, cfg, body, force_model):
        try:
            d = json.loads(body)
        except Exception:
            return body
        g = cfg.get("gen_defaults", {})
        override = g.get("override_client", False)
        for k in ("temperature", "top_p", "top_k", "min_p", "max_tokens",
                  "repeat_penalty", "presence_penalty", "frequency_penalty"):
            v = g.get(k, None)
            if v is None:
                continue
            if override or k not in d or d.get(k) is None:
                d[k] = v
        if force_model:
            d["model"] = force_model
        if g.get("apply_system") and g.get("system_prompt"):
            msgs = d.get("messages")
            if isinstance(msgs, list):
                if not (msgs and msgs[0].get("role") == "system"):
                    d["messages"] = [{"role": "system", "content": g["system_prompt"]}] + msgs
        try:
            return json.dumps(d).encode()
        except Exception:
            return body

    def _maybe_set_model(self, body, model):
        try:
            d = json.loads(body)
            d["model"] = model
            return json.dumps(d).encode()
        except Exception:
            return body

    def do_GET(self):
        self._handle()

    def do_POST(self):
        self._handle()


# ---------------------------------------------------------------- CC SWITCH / CC CONNECT
CCSW_DB = os.path.expanduser("~/.cc-switch/cc-switch.db")
CCSW_BACKUP_DIR = os.path.join(ROOT, "config", "ccswitch-backups")

import sqlite3 as _sqlite3

def _ccsw_db():
    if not os.path.exists(CCSW_DB):
        raise RuntimeError("CC Switch database not found at %s" % CCSW_DB)
    c = _sqlite3.connect(CCSW_DB, timeout=5)
    return c

def _ccsw_provider_row(c, pid):
    r = c.execute(
        "SELECT id, app_type, name, settings_config, is_current FROM providers WHERE id=?",
        (pid,)).fetchone()
    if not r:
        raise RuntimeError("provider '%s' not found" % pid)
    try:
        cfg = json.loads(r[3] or "{}")
    except Exception:
        cfg = {}
    return {"id": r[0], "app_type": r[1], "name": r[2],
            "config": cfg, "is_current": bool(r[4])}

def _ccsw_masked(p):
    out = dict(p)
    def mask(o):
        if isinstance(o, dict):
            return {k: ("•" * 8 + str(v)[-4:] if any(t in k.lower() for t in
                     ("key", "token", "secret")) and isinstance(v, str) and len(v) > 12
                     else mask(v)) for k, v in o.items()}
        if isinstance(o, list):
            return [mask(x) for x in o]
        return o
    out["config"] = mask(p["config"])
    return out

def ccswitch_view(path):
    """GET /api/ccswitch[/providers|/usage?days=N]"""
    sub = path.split("?", 1)[0][len("/api/ccswitch"):].strip("/")
    c = _ccsw_db()
    try:
        if sub == "providers":
            pass  # bare path: fall through to the provider list
        elif sub.startswith("providers/"):
            pid = sub.split("/")[1]  # ignore trailing segments (e.g. /switch)
            return {"provider": _ccsw_masked(_ccsw_provider_row(c, pid))}
        if sub == "usage":
            import urllib.parse as _up
            q = _up.parse_qs(path.split("?", 1)[1] if "?" in path else "")
            days = min(int(q.get("days", ["7"])[0]), 90)
            cutoff = time.time() - days * 86400
            rows = c.execute(
                "SELECT provider_id, app_type, COUNT(*), ROUND(SUM(total_cost_usd),4),"
                " SUM(input_tokens), SUM(output_tokens) FROM proxy_request_logs"
                " WHERE created_at > ? GROUP BY provider_id, app_type ORDER BY 3 DESC",
                (str(int(cutoff * 1000)),)).fetchall()
            return {"days": days,
                    "rows": [{"provider": r[0], "app": r[1], "requests": r[2],
                              "cost_usd": r[3] or 0, "in_tokens": r[4] or 0,
                              "out_tokens": r[5] or 0} for r in rows]}
        # default: provider list
        rows = c.execute(
            "SELECT id, app_type, name, is_current FROM providers ORDER BY app_type, id"
        ).fetchall()
        return {"providers": [{"id": r[0], "app_type": r[1], "name": r[2],
                               "is_current": bool(r[3])} for r in rows]}
    finally:
        c.close()

_CCADD_TPL = {
    "claude":  lambda n, k, u: {"env": dict({"ANTHROPIC_API_KEY": k},
                 **({"ANTHROPIC_BASE_URL": u} if u else {}))},
    "codex":   lambda n, k, u: dict({"auth": {"OPENAI_API_KEY": k}, "config": ""},
                 **({"base_url": u} if u else {})),
    "gemini":  lambda n, k, u: dict({"env": {"GEMINI_API_KEY": k}, "config": {}},
                 **({"baseUrl": u} if u else {})),
}

def ccswitch_mutate(path, body):
    """POST /api/ccswitch/providers|/providers/<id>/switch|/usage  -> (status, view)"""
    path = path.split("?", 1)[0]
    sub = path[len("/api/ccswitch"):].strip("/")
    if sub == "providers":
        kind = body.get("kind") or "claude"
        name = (body.get("name") or "").strip()
        keyv = (body.get("key") or "").strip()
        url = (body.get("base_url") or "").strip()
        if kind not in _CCADD_TPL or not name or not keyv:
            raise RuntimeError("need kind (claude|codex|gemini), name and key")
        import uuid
        pid = "%s-%s" % (kind, uuid.uuid4().hex[:8])
        c = _ccsw_db()
        try:
            c.execute(
                "INSERT INTO providers (id, app_type, name, settings_config, created_at)"
                " VALUES (?,?,?,?,?)",
                (pid, kind, name, json.dumps(_CCADD_TPL[kind](name, keyv, url)),
                 str(int(time.time() * 1000))))
            c.commit()
        finally:
            c.close()
        return 200
    m = re.match(r"providers/([^/]+)/switch$", sub)
    if m:
        pid = m.group(1)
        c = _ccsw_db()
        try:
            r = c.execute("SELECT id FROM providers WHERE id=?", (pid,)).fetchone()
            if not r:
                raise RuntimeError("provider '%s' not found" % pid)
            c.execute("UPDATE providers SET is_current=0")
            c.execute("UPDATE providers SET is_current=1 WHERE id=?", (pid,))
            c.commit()
        finally:
            c.close()
        return 200
    m = re.match(r"providers/([^/]+)$", sub)
    if m:
        pid = m.group(1)
        # delete: backup the row first
        c = _ccsw_db()
        try:
            r = c.execute("SELECT * FROM providers WHERE id=?", (pid,)).fetchone()
            if not r:
                raise RuntimeError("provider '%s' not found" % pid)
            os.makedirs(CCSW_BACKUP_DIR, exist_ok=True)
            with open(os.path.join(CCSW_BACKUP_DIR, "%s.json" % pid), "w") as f:
                json.dump({"id": r[0], "app_type": r[1], "name": r[2],
                            "settings_config": r[3]}, f)
            c.execute("DELETE FROM providers WHERE id=?", (pid,))
            c.commit()
        finally:
            c.close()
        return 200
    raise RuntimeError("unknown ccswitch endpoint: %s" % sub)

# ---------------------------------------------------------------- CC CONNECT
CCONNECT_WEB_PORT = 9820

def _ccconnect_journal(n=30):
    try:
        out = subprocess.run(
            ["journalctl", "--user", "-u", "cc-connect", "--no-pager", "-n", str(n)],
            capture_output=True, text=True, timeout=5).stdout
        return out.strip().splitlines()[-n:]
    except Exception:
        return []

def ccconnect_status():
    st = {"service": "unknown", "web": False, "auth_token": None,
          "info": {}, "log": []}
    try:
        out = subprocess.run(
            ["systemctl", "--user", "show", "cc-connect",
             "--property=ActiveState,SubState,MainPID"],
            capture_output=True, text=True, timeout=5).stdout
        props = dict(l.split("=", 1) for l in out.strip().splitlines() if "=" in l)
        st["service"] = (props.get("ActiveState") or "unknown").lower()
    except Exception:
        pass
    # management token from config file
    try:
        with open(os.path.expanduser("~/.cc-connect/config.toml")) as f:
            txt = f.read()
        m2 = re.search(r'\[management\][^\[]*?token\s*=\s*"?([^"\s\]]+)', txt, re.S)
        if m2:
            st["auth_token"] = m2.group(1)
    except OSError:
        pass
    # management API (needs the token)
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:%d/api/v1/status" % CCONNECT_WEB_PORT,
            headers={"Authorization": "Bearer " + (st.get("auth_token") or "")})
        with urllib.request.urlopen(req, timeout=3) as r:
            st["web"] = True
            j = json.loads(r.read().decode() or "{}")
            st["info"] = j.get("data", {})
    except Exception:
        try:
            with urllib.request.urlopen(
                    "http://127.0.0.1:%d/" % CCONNECT_WEB_PORT, timeout=3) as r:
                st["web"] = (r.status == 200)
        except Exception:
            st["web"] = False
    st["log"] = _ccconnect_journal(30)
    return st

def ccconnect_restart():
    try:
        p = subprocess.run(["systemctl", "--user", "restart", "cc-connect"],
                           capture_output=True, text=True, timeout=15)
        if p.returncode == 0:
            return True, "cc-connect restarted"
        return False, (p.stderr or p.stdout).strip()[:300]
    except Exception as e:
        return False, str(e)

# cc-connect management API proxy — lets the panel UI configure cc-connect
# without opening any new port: everything is forwarded to 127.0.0.1:9820.
_CC_API_ALLOW = {
    "GET": ("status", "agents", "cron", "projects", "providers",
            "providers/cc-switch", "providers/presets", "settings",
            "skills", "skills/presets"),
    "POST": ("providers", "providers/cc-switch", "reload", "restart",
             "cron"),
    "PATCH": ("settings", "projects/home"),
}


def _cc_api_token():
    try:
        with open(os.path.expanduser("~/.cc-connect/config.toml")) as f:
            txt = f.read()
        m2 = re.search(r'\[management\][^\[]*?token\s*=\s*"?([^"\s\]]+)', txt, re.S)
        if m2:
            return m2.group(1)
    except OSError:
        pass
    return ""


def ccconnect_proxy(method, sub, body=None):
    """Forward a management-API call to cc-connect's local web admin.
    Returns (status_code, parsed_json_or_text)."""
    sub = (sub or "").strip("/")
    allowed = _CC_API_ALLOW.get(method, ())
    if sub not in allowed:
        raise RuntimeError("cc-connect proxy: %s /%s not allowed" % (method, sub))
    req = urllib.request.Request(
        "http://127.0.0.1:%d/api/v1/%s" % (CCONNECT_WEB_PORT, sub),
        data=(json.dumps(body).encode() if method in ("POST", "PATCH") and body else None),
        headers={"Authorization": "Bearer " + _cc_api_token(),
                 **({"Content-Type": "application/json"} if method in ("POST", "PATCH") else {})},
        method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            raw = r.read().decode() or ""
            code = r.status
    except urllib.error.HTTPError as e:
        raw = e.read().decode() or ""
        code = e.code
    try:
        return code, json.loads(raw)
    except Exception:
        return code, {"raw": raw[:2000]}


# ---------------------------------------------------------------- PANEL
class PanelHandler(BaseHTTPRequestHandler):
    server_version = "llm-panel/1.0"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _authed(self, cfg):
        key = read_api_key(cfg)
        if not key:
            return True
        got = ""
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            got = auth[7:].strip()
        got = got or self.headers.get("X-Api-Key", "").strip()
        import hmac
        return hmac.compare_digest(got, key)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return {}

    def _lan_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        try:
            cfg = load_config()
        except Exception as e:
            return self._send(500, {"error": f"bad presets.json: {e}"})

        if path in ("/", "/index.html"):
            try:
                with open(os.path.join(HERE, "index.html"), "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            except OSError:
                return self._send(500, "index.html missing", "text/plain")

        if path == "/presets.json":
            view = json.loads(json.dumps(cfg))
            for p in view["presets"]:
                # A preset that carries a measured vram_gb keeps it — the linear fit
                # models VRAM at load, while the measured figure is the observed peak
                # under a long prompt, which is what actually decides OOM.
                if p.get("vram_gb") is None:
                    spec = p.get("spec_type") or cfg.get("runtime", {}).get("spec_type")
                    p["vram_gb"], p["vram_exact"] = vram_estimate(
                        cfg, p["ctx"], p["kv"], p.get("model"), spec,
                        p.get("parallel"), p.get("vision"))
                    p["vram_measured"] = False
                else:
                    p["vram_exact"], p["vram_measured"] = True, True
                # Switching a saved preset onto another model does not rewrite its other
                # knobs (on purpose -- see drop_inapplicable), so a preset can be left
                # asking for MTP that its GGUF cannot provide. The launcher drops it
                # safely, but silently; say so on the card instead.
                caps = model_caps(cfg, p.get("model"))
                p["mtp_unavailable"] = bool(
                    str(p.get("spec_type") or "").lower() not in ("", "none", "off")
                    and not caps["mtp"])
                # same for a projector the repo does not ship
                p["vision_unavailable"] = bool(
                    str(p.get("vision") or "").lower() == "on" and not caps["mmproj"])
            view["endpoints"] = [endpoint_public(cfg, e) for e in cfg.get("endpoints", [])]
            view["router_url"] = f"http://{self._lan_ip()}:{cfg['server'].get('router_port', 8001)}/v1"
            view["version"] = panel_version()
            return self._send(200, view)

        if path == "/api/models":
            return self._send(200, {"models": scan_models(cfg), "models_root": MODELS_ROOT})

        if path == "/api/command":
            q = urllib.parse.parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
            n = min(int(q.get("n", ["200"])[0]), 2000)
            which = (q.get("src", ["server"])[0] or "server").lower()
            logs = {"server": expand(cfg["server"]["server_log"]),
                    "panel": os.path.join(ROOT, "logs", "panel.log"),
                    "bench": os.path.join(ROOT, "logs", "bench.log")}
            with _BENCH_LOCK:
                bench = {"running": BENCH_STATE["running"], "label": BENCH_STATE["label"],
                         "cmd": BENCH_STATE["cmd"], "log": BENCH_STATE["log"][-40:]}
            return self._send(200, {
                "command": running_cmdline(cfg), "bench": bench,
                "sources": sorted(logs), "src": which,
                "lines": tail(logs.get(which, logs["server"]), n)})

        if path == "/api/bench":
            with _BENCH_LOCK:
                st = dict(BENCH_STATE)
            st["saved"] = cfg.get("benchmarks", {})
            return self._send(200, st)

        if path == "/api/rescan":
            # read-only preview: what WOULD change
            _, changes = resync_models(load_config())
            return self._send(200, {"changes": changes})

        if path == "/api/downloads":
            return self._send(200, {"downloads": downloads_view(), "models_root": MODELS_ROOT})

        if path == "/api/status":
            run = running_server(cfg)
            if run and "kv" in run:
                run["vram_gb_est"] = estimate_vram(cfg, run.get("ctx", 0), run["kv"],
                                                   model_id_for_file(cfg, run.get("model_file")),
                                                   run.get("spec_type"),
                                                   run.get("parallel"), run.get("vision"))
                run["slots"] = effective_slots(run.get("parallel"))
            eng_status = {e: {"installed": engine_installed(cfg, e), "port": engine_port(cfg, e),
                              "install": install_state(e),
                              "can_install": bool((cfg.get("engines", {}).get(e) or {}).get("install"))}
                          for e in ("llamacpp", "vllm", "ollama")}
            det_eng = run.get("engine") if run else None
            return self._send(200, {
                "gpu": gpu_stats(), "cpu": cpu_stats(), "running": run, "health": local_health(cfg),
                "matched_preset": match_preset(cfg, run), "profiles": cfg.get("profiles", {}),
                "tps": last_tps(cfg), "switch": SWITCH_STATE,
                "active_endpoint": cfg.get("active_endpoint"),
                "active_engine": cfg.get("active_engine", "llamacpp"),
                "running_engine": det_eng, "engines_status": eng_status,
                "runtime": cfg.get("runtime", {}), "version": panel_version(),
                "vram_total_gb": cfg["gpu"]["vram_total_gb"], "server_time": time.time(),
                "system": {"uptime_s": host_uptime_s(), "power": power_capability(),
                           "pending_power": pending_power()},
            })

        if path == "/api/log":
            q = urllib.parse.parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
            n = min(int(q.get("n", ["40"])[0]), 400)
            return self._send(200, {"lines": tail(expand(cfg["server"]["server_log"]), n)})

        if path == "/api/auto_switch":
            sync_auto_switch_map(cfg)
            return self._send(200, {"auto_switch": cfg.get("auto_switch")})

        if path.startswith("/api/ccswitch"):
            try:
                return self._send(200, ccswitch_view(path))
            except Exception as e:
                return self._send(500, {"error": str(e)})

        if path == "/api/ccconnect/status":
            try:
                return self._send(200, ccconnect_status())
            except Exception as e:
                return self._send(500, {"error": str(e)})

        if path.startswith("/api/ccconnect/api/"):
            sub = path[len("/api/ccconnect/api/"):]
            try:
                code, data = ccconnect_proxy("GET", sub)
            except Exception as e:
                return self._send(400, {"error": str(e)})
            return self._send(code, data)

        return self._send(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            cfg = load_config()
        except Exception as e:
            return self._send(500, {"error": f"bad presets.json: {e}"})
        if not self._authed(cfg):
            return self._send(401, {"error": "unauthorized — enter the API key in the panel"})
        body = self._body()

        if path.startswith("/api/ccswitch"):
            try:
                return self._send(ccswitch_mutate(path, body), ccswitch_view(path))
            except Exception as e:
                return self._send(400, {"error": str(e)})

        if path == "/api/ccconnect/restart":
            try:
                ok, msg = ccconnect_restart()
                return self._send(200 if ok else 500, {"ok": ok, "message": msg})
            except Exception as e:
                return self._send(500, {"error": str(e)})

        if path.startswith("/api/ccconnect/api/"):
            sub = path[len("/api/ccconnect/api/"):]
            try:
                code, data = ccconnect_proxy("POST", sub, body)
            except Exception as e:
                return self._send(400, {"error": str(e)})
            return self._send(code, data)

    def do_PATCH(self):
        path = self.path.split("?", 1)[0]
        try:
            cfg = load_config()
        except Exception as e:
            return self._send(500, {"error": f"bad presets.json: {e}"})
        if not self._authed(cfg):
            return self._send(401, {"error": "unauthorized — enter the API key in the panel"})
        body = self._body()
        if path.startswith("/api/ccconnect/api/"):
            sub = path[len("/api/ccconnect/api/"):]
            try:
                code, data = ccconnect_proxy("PATCH", sub, body)
            except Exception as e:
                return self._send(400, {"error": str(e)})
            return self._send(code, data)
        return self._send(404, {"error": "not found"})

        if path == "/api/switch":
            launch, label = resolve_launch(cfg, body)
            if not launch:
                return self._send(400, {"error": label})
            # a preset with a measured peak is trusted over the fit
            pmeas = next((x.get("vram_gb") for x in cfg.get("presets", [])
                          if x["id"] == label and x.get("vram_gb") is not None), None)
            est = pmeas if pmeas is not None else estimate_vram(
                cfg, launch["ctx"], launch["kv"], launch.get("model"),
                launch.get("spec_type", cfg.get("runtime", {}).get("spec_type")),
                launch.get("parallel"), launch.get("vision"))
            if est > cfg["estimator"]["cap_gb"] and not body.get("force"):
                return self._send(409, {"error": "over_vram_budget", "est_gb": est,
                                        "cap_gb": cfg["estimator"]["cap_gb"]})
            d = expand(launch.get("model_dir") or "")
            if not d or not os.path.isdir(d):
                return self._send(400, {"error": f"model directory missing ({d or 'unset'}) — "
                                                 "re-register the model under Models"})
            ok, msg = start_switch(cfg, launch, label)
            # applying a local preset also makes 'local' the active router endpoint
            if ok:
                save_config(lambda d: d.update(active_endpoint="local"))
            return self._send(202 if ok else 409, {"ok": ok, "message": msg, "target": label, "est_gb": est})

        if path == "/api/stop":
            _set_switch("stopping", None, "manual stop")
            _kill_server(cfg)
            _set_switch("idle", None, "stopped")
            return self._send(200, {"ok": True})

        if path == "/api/profile":
            slot, preset = body.get("slot"), body.get("preset")
            if slot not in ("default", "testing"):
                return self._send(400, {"error": "slot must be 'default' or 'testing'"})
            if not any(p["id"] == preset for p in cfg["presets"]):
                return self._send(400, {"error": f"unknown preset '{preset}'"})
            save_config(lambda d: d.setdefault("profiles", {}).__setitem__(slot, preset))
            return self._send(200, {"ok": True, "profiles": load_config().get("profiles")})

        if path == "/api/activate":
            eid = body.get("endpoint")
            ep = endpoint_by_id(cfg, eid)
            if not ep:
                return self._send(400, {"error": f"unknown endpoint '{eid}'"})
            if ep.get("type") == "remote":
                secrets = load_secrets(cfg)
                if not (secrets.get(ep.get("key_ref", "")) or os.environ.get(ep.get("key_env", ""))):
                    return self._send(400, {"error": "no_key", "hint": f"set '{ep.get('key_ref')}' in secrets.json"})
                save_config(lambda d: d.update(active_endpoint=eid))
                if cfg.get("router", {}).get("free_gpu_on_remote"):
                    _kill_server(cfg)
                return self._send(200, {"ok": True, "active_endpoint": eid, "reachability": check_remote(cfg, ep)})
            # local: activate = ensure a local server is running (apply default profile if down)
            save_config(lambda d: d.update(active_endpoint="local"))
            run = running_server(cfg)
            if not run:
                pid = cfg.get("profiles", {}).get("default") or cfg["presets"][0]["id"]
                launch, label = resolve_launch(cfg, {"preset": pid})
                start_switch(cfg, launch, label)
                return self._send(202, {"ok": True, "active_endpoint": "local", "started": label})
            return self._send(200, {"ok": True, "active_endpoint": "local"})

        if path == "/api/auto_switch":
            op = body.get("op", "upsert")
            alias = (body.get("alias") or "").strip()
            pid = (body.get("preset") or "").strip()
            if op == "delete":
                if not alias:
                    return self._send(400, {"error": "alias is required"})

                def mut(d):
                    as_cfg = d.setdefault("auto_switch", {"enabled": False, "map": {}, "manual": {}})
                    as_cfg["manual"].pop(alias, None)
                    sync_auto_switch_map(d)
                save_config(mut)
            elif op == "enable":
                save_config(lambda d: d.setdefault("auto_switch", {"enabled": False, "map": {}, "manual": {}}).update(enabled=True))
            elif op == "disable":
                save_config(lambda d: d.setdefault("auto_switch", {"enabled": False, "map": {}, "manual": {}}).update(enabled=False))
            else:
                if not re.fullmatch(r"[A-Za-z0-9][\w.\-]{0,63}", alias):
                    return self._send(400, {"error": "alias may use letters, digits, dot, dash and underscore (max 64 chars)"})
                if not any(p["id"] == pid for p in cfg.get("presets", [])):
                    return self._send(400, {"error": f"unknown preset '{pid}'"})

                def mut(d):
                    as_cfg = d.setdefault("auto_switch", {"enabled": False, "map": {}, "manual": {}})
                    as_cfg["manual"][alias] = pid
                    sync_auto_switch_map(d)
                save_config(mut)
            return self._send(200, {"ok": True, "auto_switch": load_config().get("auto_switch")})

        if path == "/api/gen":
            allowed = ("temperature", "top_p", "top_k", "min_p", "max_tokens", "repeat_penalty",
                       "presence_penalty", "frequency_penalty", "system_prompt", "apply_system",
                       "override_client")
            upd = {k: body[k] for k in allowed if k in body}
            save_config(lambda d: d.setdefault("gen_defaults", {}).update(upd))
            return self._send(200, {"ok": True, "gen_defaults": load_config().get("gen_defaults")})

        if path == "/api/endpoint":
            # add or update a remote endpoint; optionally store its key
            op = body.get("op", "upsert")
            eid = body.get("id")
            if not eid:
                return self._send(400, {"error": "id required"})
            if op == "delete":
                save_config(lambda d: d.update(endpoints=[e for e in d.get("endpoints", []) if e["id"] != eid]))
                return self._send(200, {"ok": True})
            ep = {"id": eid, "type": "remote", "label": body.get("label", eid),
                  "base_url": body.get("base_url", ""), "model": body.get("model", ""),
                  "key_ref": body.get("key_ref", eid), "vision": bool(body.get("vision", False)),
                  "note": body.get("note", "")}
            if not ep["base_url"]:
                return self._send(400, {"error": "base_url required"})

            def mut(d):
                eps = d.setdefault("endpoints", [])
                for i, x in enumerate(eps):
                    if x["id"] == eid:
                        eps[i] = {**x, **ep}
                        return
                eps.append(ep)
            save_config(mut)
            # store key in secrets.json if provided
            if body.get("api_key"):
                spath = expand(cfg["server"]["secrets_file"])
                sec = load_secrets(cfg)
                sec[ep["key_ref"]] = body["api_key"]
                tmp = spath + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(sec, f, indent=2)
                os.replace(tmp, spath)
                os.chmod(spath, 0o600)
            return self._send(200, {"ok": True})

        if path == "/api/check":
            ep = endpoint_by_id(cfg, body.get("endpoint", ""))
            if not ep or ep.get("type") != "remote":
                return self._send(400, {"error": "remote endpoint id required"})
            return self._send(200, check_remote(cfg, ep))

        if path == "/api/runtime":
            allowed = ("thinking", "context_shift", "cache_reuse", "reason_effort",
                       "reason_format",
                       "image_min_tokens", "image_max_tokens", "batch", "ubatch",
                       "extra_flags", "fit", "flash_attn", "kv_unified", "kv_unified_per_slot",
                       "spec_type", "spec_n_max", "spec_n_min", "spec_draft_kv",
                       "threads", "threads_batch", "numa", "load_mode")
            upd = {k: body[k] for k in allowed if k in body}
            save_config(lambda d: d.setdefault("runtime", {}).update(upd))
            applied = False
            if body.get("apply"):
                cfg2 = load_config()
                run = running_server(cfg2)
                if run and run.get("engine") == "llamacpp" and "kv" in run:
                    # Rebuild from the preset that is actually running so ITS overrides
                    # (spec_type/MTP, thinking, model, ...) survive the restart. A bare launch
                    # dropped them and silently fell back to the globals — and hardcoding
                    # qwen3-vl restarted on the wrong model entirely.
                    mp = match_preset(cfg2, run)
                    launch, label = (resolve_launch(cfg2, {"preset": mp}) if mp else (None, None))
                    if not launch:
                        mid = model_id_for_file(cfg2, run.get("model_file"))
                        launch = {"ctx": run.get("ctx", 98304), "kv": run.get("kv", "q4_0"),
                                  "parallel": run.get("parallel"), "ngl": run.get("ngl", 99),
                                  "model_dir": (cfg2.get("models", {}).get(mid or "") or {}).get("dir")}
                        label = "runtime-update"
                    ok, _ = start_switch(cfg2, launch, label or "runtime-update")
                    applied = ok
            return self._send(200, {"ok": True, "runtime": load_config().get("runtime"),
                                    "applied": applied,
                                    "hint": "" if applied else "saved — restart the local model (apply any preset) to take effect"})

        if path == "/api/engine":
            eng = body.get("engine")
            if eng not in ("llamacpp", "vllm", "ollama"):
                return self._send(400, {"error": "engine must be llamacpp, vllm, or ollama"})
            if not engine_installed(cfg, eng):
                spec = cfg.get("engines", {}).get(eng, {})
                return self._send(400, {"error": "not_installed", "engine": eng,
                                        "install": spec.get("install", ""),
                                        "hint": f"{eng} is not installed on this box"})
            ok, msg = start_engine(cfg, eng, body)
            return self._send(202 if ok else 409, {"ok": ok, "message": msg, "engine": eng})

        if path == "/api/power":
            w = body.get("watts")
            if w is None:
                return self._send(400, {"error": "watts required"})
            return self._send(200, set_power_limit(w))

        if path == "/api/system":
            op = body.get("op")
            if op == "cancel":
                return self._send(200, cancel_power())
            if op not in ("poweroff", "reboot"):
                return self._send(400, {"error": "op must be poweroff, reboot or cancel"})
            cap = power_capability(force=True)
            pw = body.get("password") or None
            if not cap.get("can") and not pw:
                return self._send(403, dict({"error": "needs_privilege"}, **cap))
            if not cap.get("can"):
                # verify the password NOW rather than at fire time, so a typo surfaces
                # while the user is still looking at the dialog
                chk = subprocess.run(["sudo", "-S", "-k", "-p", "", "true"],
                                     input=pw + "\n", capture_output=True, text=True, timeout=15)
                if chk.returncode != 0:
                    return self._send(403, {"error": "needs_privilege", "bad_password": True,
                                            "err": "sudo rejected that password"})
            delay = max(0, min(600, int(body.get("delay_s") or 0)))
            stop_first = cfg if body.get("stop_model", True) else None
            return self._send(202, schedule_power(op, delay, stop_first, pw))

        if path == "/api/engine-install":
            eng = body.get("engine")
            if eng not in ("llamacpp", "vllm", "ollama"):
                return self._send(400, {"error": "engine must be llamacpp, vllm, or ollama"})
            force = bool(body.get("force"))
            if engine_installed(cfg, eng) and not force:
                return self._send(200, {"ok": True, "already_installed": True})
            run = running_server(cfg)
            if run and run.get("engine") == eng:
                return self._send(409, {"error": f"{eng} is serving right now — stop it first"})
            logp, msg = start_install(cfg, eng)
            if not logp:
                return self._send(409, {"error": msg})
            return self._send(202, {"ok": True, "engine": eng, "log": logp, "message": msg})

        if path == "/api/engine-config":
            eng = body.get("engine")
            if eng not in ("llamacpp", "vllm", "ollama"):
                return self._send(400, {"error": "engine must be llamacpp, vllm, or ollama"})
            fields = {k: v for k, v in body.items() if k != "engine"}

            def mut(d):
                d.setdefault("engines", {}).setdefault(eng, {}).update(fields)
            save_config(mut)
            return self._send(200, {"ok": True, "engine": load_config()["engines"][eng]})

        if path == "/api/download":
            repo = body.get("repo") or body.get("url")
            if not repo:
                return self._send(400, {"error": "repo (org/name or HF URL) required"})
            inc = body.get("include")
            if isinstance(inc, str):
                inc = [p for p in inc.split() if p]
            jid, msg = start_download(cfg, repo, body.get("dest"), body.get("name"), include=inc)
            if not jid:
                return self._send(400, {"error": msg})
            return self._send(202, {"ok": True, "id": jid, "message": msg})

        if path == "/api/bench":
            if body.get("op") == "forget":
                lab = body.get("label")
                save_config(lambda d: d.get("benchmarks", {}).pop(lab, None))
                return self._send(200, {"ok": True})
            if body.get("mode") == "run":
                ok, msg = run_bench_settings(cfg, body)
            elif body.get("mode") == "server":
                ok, msg = start_server_bench(cfg, body)
            else:
                ok, msg = start_bench(cfg, body)
            return self._send(202 if ok else 409, {"ok": ok, "message": msg})

        if path == "/api/models":
            op = body.get("op", "add")
            if op == "rescan":
                # Re-read every model directory and write the detected facts back. This is
                # what makes "drop a projector into the folder" or "move the -mtp build to
                # its own directory" just work, instead of needing the config hand-edited.
                changes = apply_model_resync(add_new=body.get("add_new", True))
                return self._send(200, {"ok": True, "changes": changes,
                                        "models": scan_models(load_config())})
            mid = (body.get("id") or "").strip()
            if not mid:
                return self._send(400, {"error": "id is required"})
            if op == "delete":
                # purge=False (default) only unregisters; the files stay on disk.
                if body.get("purge"):
                    used = presets_using_model(cfg, mid)
                    if used and not body.get("force"):
                        return self._send(409, {"error": "model_in_use", "presets": used,
                                                "hint": "delete or repoint these presets first, "
                                                        "or resend with force:true"})
                    ok, msg, freed = purge_model_dir(cfg, mid)
                    if not ok:
                        return self._send(400, {"error": msg})
                    save_config(lambda d: d.get("models", {}).pop(mid, None))
                    return self._send(200, {"ok": True, "purged": True,
                                            "freed_gb": freed, "message": msg})
                save_config(lambda d: d.get("models", {}).pop(mid, None))
                return self._send(200, {"ok": True, "purged": False})
            if not body.get("dir"):
                return self._send(400, {"error": "dir is required"})
            entry = {"dir": body["dir"], "label": body.get("label", mid),
                     "vision": bool(body.get("vision", False)), "note": body.get("note", "")}
            save_config(lambda d: d.setdefault("models", {}).__setitem__(mid, entry))
            return self._send(200, {"ok": True, "models": load_config().get("models")})

        if path == "/api/preset":
            op = body.get("op", "upsert")
            if op == "reorder":
                order = [str(x) for x in (body.get("order") or [])]
                if not order:
                    return self._send(400, {"error": "order (list of preset ids) is required"})

                def reorder(d):
                    ps = d.get("presets", [])
                    by = {p["id"]: p for p in ps}
                    seen = set()
                    new_ps = []
                    for i in order:                      # the ids the panel sent, in their new order
                        if i in by and i not in seen:
                            new_ps.append(by[i])
                            seen.add(i)
                    for p in ps:                         # anything not listed keeps its relative place
                        if p["id"] not in seen:
                            new_ps.append(p)
                    d["presets"] = new_ps
                save_config(reorder)
                return self._send(200, {"ok": True,
                                        "order": [p["id"] for p in load_config()["presets"]]})
            pid = (body.get("id") or "").strip()
            if not pid:
                return self._send(400, {"error": "id is required"})
            if op == "delete":
                def del_mut(d):
                    d.update(
                        presets=[p for p in d.get("presets", []) if p["id"] != pid],
                        profiles={k: (v if v != pid else None) for k, v in d.get("profiles", {}).items()})
                    sync_auto_switch_map(d)
                save_config(del_mut)
                return self._send(200, {"ok": True,
                                        "auto_switch": load_config().get("auto_switch")})
            # upsert
            kv = body.get("kv")
            if kv and kv not in ("f16", "q8_0", "q4_0"):
                return self._send(400, {"error": "kv must be f16, q8_0, or q4_0"})
            if not re.fullmatch(r"[A-Za-z0-9][\w.\-]{0,63}", pid):
                return self._send(400, {"error": "id may use letters, digits, dot, dash and "
                                                 "underscore (max 64 chars)"})
            orig = (body.get("orig_id") or "").strip()
            renaming = bool(orig and orig != pid)
            if renaming:
                if not any(p["id"] == orig for p in cfg.get("presets", [])):
                    return self._send(404, {"error": f"preset '{orig}' no longer exists"})
                if any(p["id"] == pid for p in cfg.get("presets", [])):
                    return self._send(409, {"error": f"a preset named '{pid}' already exists"})
            want_model = (body.get("model") or "").strip()
            if want_model:
                known = set(cfg.get("models", {}).keys())
                try:
                    for m in scan_models(cfg).values():
                        known.add(m.get("config_id") or m.get("id"))
                except Exception:
                    pass
                if want_model not in known:
                    return self._send(400, {"error": f"unknown model '{want_model}' — "
                                                     "download or register it first"})
            look_for = orig if renaming else pid
            existing = next((p for p in cfg.get("presets", []) if p["id"] == look_for), None)
            if not existing:
                if body.get("ctx") in (None, "") or not body.get("kv"):
                    return self._send(400, {"error": "new preset needs at least ctx and kv"})
            int_keys = ("ctx", "cache_reuse", "image_min_tokens", "image_max_tokens",
                        "n_cpu_moe", "spec_n_max", "spec_n_min", "kv_unified_per_slot",
                        "fit_target", "fit_ctx", "threads", "threads_batch")
            # ngl is NOT an int key: it may be the string "auto" (let --fit decide)
            # keys that can be cleared back to "inherit" by sending "" / null
            # every per-preset override the editor can send must be listed here, or it is
            # silently dropped on save and the preset quietly falls back to the globals
            clearable = ("thinking", "reason_effort", "reason_format", "context_shift",
                         "cache_reuse",
                         "image_min_tokens", "image_max_tokens", "tag", "note", "parallel",
                         "spec_type", "spec_n_max", "spec_n_min", "spec_draft_kv", "mtp_head",
                         "n_cpu_moe", "cpu_moe", "lazy_mode", "vision",
                         "load_mode", "fit", "fit_target", "fit_ctx", "override_tensor",
                         "numa", "no_host", "threads", "threads_batch", "extra_flags",
                         "kv_unified", "kv_unified_per_slot")

            def mut(d):
                ps = d.setdefault("presets", [])
                base = dict(existing) if existing else {"id": pid}
                base["id"] = pid
                # the chosen model must win: setdefault() here silently ignored it,
                # so every preset ended up on the default model no matter what was picked
                if want_model:
                    base["model"] = want_model
                else:
                    base.setdefault("model", "qwen3-vl")
                if body.get("kv"):
                    base["kv"] = body["kv"]
                for k in ("ctx", "ngl"):
                    if body.get(k) not in (None, ""):
                        try:
                            base[k] = int(body[k])
                        except (TypeError, ValueError):
                            pass
                if body.get("parallel") not in (None, ""):
                    v = body["parallel"]
                    if str(v).lower() == "auto":
                        base["parallel"] = "auto"      # llama.cpp picks; shared KV pool
                    else:
                        try:
                            base["parallel"] = max(1, int(v))
                        except (TypeError, ValueError):
                            pass
                for k in clearable:
                    if k in body:
                        v = body[k]
                        if v in (None, ""):
                            base.pop(k, None)
                        elif k in int_keys:
                            try:
                                base[k] = int(v)
                            except (TypeError, ValueError):
                                pass
                        else:
                            base[k] = v
                if existing:
                    ps[ps.index(existing)] = base
                else:
                    ps.append(base)
                if renaming:
                    # follow the rename anywhere the old id was referenced
                    prof = d.setdefault("profiles", {})
                    for k, v in list(prof.items()):
                        if v == orig:
                            prof[k] = pid
            def mut2(d):
                mut(d)
                sync_auto_switch_map(d)
            save_config(mut2)
            saved = next((p for p in load_config()["presets"] if p["id"] == pid), None)
            return self._send(200, {"ok": True, "preset": saved,
                                    "auto_switch": load_config().get("auto_switch")})

        return self._send(404, {"error": "not found"})


def main():
    cfg = load_config()
    host = cfg["server"].get("host", "0.0.0.0")
    pport = int(cfg["server"].get("panel_port", 8080))
    rport = int(cfg["server"].get("router_port", 8001))

    adopt_downloads()   # re-attach to any download still running from a previous panel

    router = ThreadingHTTPServer((host, rport), RouterHandler)
    threading.Thread(target=router.serve_forever, daemon=True).start()
    print(f"router  (OpenAI-compatible) on http://{host}:{rport}/v1", flush=True)

    # Optional: bring the local model up on boot/start (guarded so a panel restart
    # never disturbs an already-running model). Only for the llama.cpp engine.
    if cfg["server"].get("autostart_local", False):
        def _autostart():
            time.sleep(6)
            c = load_config()
            if not running_server(c) and c.get("active_engine", "llamacpp") == "llamacpp":
                pid = c.get("profiles", {}).get("default") or (c["presets"][0]["id"] if c.get("presets") else None)
                if pid:
                    launch, label = resolve_launch(c, {"preset": pid})
                    if launch:
                        print(f"autostart: launching local preset '{label}'", flush=True)
                        start_switch(c, launch, label)
        threading.Thread(target=_autostart, daemon=True).start()

    panel = ThreadingHTTPServer((host, pport), PanelHandler)
    print(f"panel   (web UI)            on http://{host}:{pport}", flush=True)
    try:
        panel.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
