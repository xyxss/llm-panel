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
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, CONFIG_PATH)
    return data


# ---------------------------------------------------------------- models / version
MODELS_ROOT = os.path.expanduser("~/models")


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
            ggufs = [f for f in files if f.endswith(".gguf") and not f.lower().startswith("mmproj")]
            if not ggufs:
                continue
            mmproj = [f for f in files if f.lower().startswith("mmproj") and f.endswith(".gguf")]
            try:
                size = sum(os.path.getsize(os.path.join(d, f)) for f in ggufs)
            except OSError:
                size = 0
            found[name] = {"id": name, "dir": d, "gguf": sorted(ggufs)[0],
                           "vision": bool(mmproj), "size_gb": round(size / 1e9, 1),
                           "configured": False}
    for mid, m in cfg.get("models", {}).items():
        d = expand(m.get("dir", ""))
        base = os.path.basename(d.rstrip("/")) if d else mid
        if base in found:
            found[base].update(configured=True, config_id=mid, label=m.get("label", mid))
        else:
            found[mid] = {"id": mid, "dir": d, "vision": m.get("vision", False),
                          "configured": True, "config_id": mid, "label": m.get("label", mid),
                          "missing": not (d and os.path.isdir(d))}
    return list(found.values())


def panel_version():
    """Changes whenever panel.py or index.html changes — drives the UI refresh banner."""
    t = 0
    for f in (os.path.join(HERE, "panel.py"), os.path.join(HERE, "index.html")):
        try:
            t = max(t, int(os.path.getmtime(f)))
        except OSError:
            pass
    return str(t)


# ---------------------------------------------------------------- GPU / process
def gpu_stats():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=5).strip().splitlines()[0]
        u, t, fr, util, temp = [x.strip() for x in out.split(",")]
        return {"used_mib": int(u), "total_mib": int(t), "free_mib": int(fr),
                "util_pct": int(util), "temp_c": int(temp)}
    except Exception as e:
        return {"error": str(e)}


_ARG_RE = {
    "ctx": re.compile(r"(?:-c|--ctx-size)\s+(\d+)"),
    "parallel": re.compile(r"(?:--parallel|-np)\s+(\d+)"),
    "kv": re.compile(r"--cache-type-k\s+(\S+)"),
    "ngl": re.compile(r"(?:-ngl|--n-gpu-layers|--gpu-layers)\s+(\d+)"),
    "model": re.compile(r"(?:-m|--model)\s+(\S+)"),
    "port": re.compile(r"--port\s+(\d+)"),
    "alias": re.compile(r"--alias\s+(\S+)"),
    "thinking": re.compile(r"(?:--reasoning|-rea)\s+(on|off|auto)"),
    "reason_effort": re.compile(r"--reasoning-effort\s+(\S+)"),
    "cache_reuse": re.compile(r"--cache-reuse\s+(\d+)"),
    "max_model_len": re.compile(r"--max-model-len\s+(\d+)"),
}


# ---------------------------------------------------------------- engines
def engine_installed(cfg, eng):
    """Is a given engine actually available on this box?"""
    if eng == "llamacpp":
        return os.path.exists(expand(cfg["server"]["llama_bin"]))
    if eng == "vllm":
        return os.path.isdir(os.path.expanduser("~/vllm-env")) or bool(shutil.which("vllm"))
    if eng == "ollama":
        return bool(shutil.which("ollama"))
    return False


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
    for k in ("ctx", "parallel", "ngl", "port", "cache_reuse", "max_model_len"):
        if k in info:
            info[k] = int(info[k])
    # context shift is a bare boolean flag
    if "--no-context-shift" in args:
        info["context_shift"] = "off"
    elif "--context-shift" in args:
        info["context_shift"] = "on"
    if "model" in info:
        info["model_file"] = os.path.basename(info.pop("model"))
    return info


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


def estimate_vram(cfg, ctx, kv):
    base, per1k = cfg["estimator"]["coeffs_gb"].get(kv, cfg["estimator"]["coeffs_gb"]["q4_0"])
    return round(base + per1k * ctx / 1000.0, 2)


def match_preset(cfg, run):
    if not run:
        return None
    for p in cfg["presets"]:
        if p["ctx"] == run.get("ctx") and p["kv"] == run.get("kv") \
           and str(p.get("parallel", 1)) == str(run.get("parallel", 1)):
            return p["id"]
    return None


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
    env.update(CTX=str(launch["ctx"]), KV_QUANT=launch["kv"],
               PARALLEL=str(launch["parallel"]), NGL=str(launch["ngl"]),
               PORT=str(cfg["server"].get("vlm_port", 8000)))
    if launch.get("model_dir"):
        env["MDIR"] = expand(launch["model_dir"])
    # Optional runtime knobs — only set when specified, so serve-vlm.sh
    # defaults (incl. its thinking-aware sampling profile) still apply.
    rt = cfg.get("runtime", {})
    for key, envname in (("thinking", "THINKING"), ("context_shift", "CONTEXT_SHIFT"),
                         ("cache_reuse", "CACHE_REUSE"), ("reason_effort", "REASON_EFFORT"),
                         ("image_min_tokens", "IMAGE_MIN_TOKENS"),
                         ("image_max_tokens", "IMAGE_MAX_TOKENS"),
                         ("ubatch", "UBATCH"), ("batch", "BATCH"),
                         ("extra_flags", "EXTRA_FLAGS"), ("fit", "FIT")):
        val = launch.get(key, rt.get(key))
        if val not in (None, ""):
            env[envname] = str(val)
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
    """Free the GPU by stopping whichever local engine is running (only ONE fits 16GB)."""
    for pat in (expand(cfg["server"]["llama_bin"]), "vllm serve", "ollama serve"):
        subprocess.run(["pkill", "-f", pat], check=False)
    _wait_gpu_free()


def _launch_vllm(cfg, body):
    serve = expand(cfg.get("engines", {}).get("vllm", {}).get("serve_script", "~/serve-vllm.sh"))
    logp = expand(cfg["server"].get("vllm_log", "~/vllm-server.log"))
    v = cfg.get("engines", {}).get("vllm", {})
    env = os.environ.copy()
    env["PORT"] = str(engine_port(cfg, "vllm"))
    for key, envname in (("model", "VLLM_MODEL"), ("max_model_len", "MAX_MODEL_LEN"),
                         ("gpu_util", "GPU_UTIL"), ("quantization", "QUANT"),
                         ("kv_cache_dtype", "KV_CACHE_DTYPE"), ("max_num_seqs", "MAX_NUM_SEQS"),
                         ("reasoning_parser", "REASONING_PARSER"), ("enable_prefix_caching", "PREFIX_CACHE")):
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
    logf = open(logp, "wb")
    subprocess.Popen(["ollama", "serve"], stdout=logf, stderr=subprocess.STDOUT,
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
                     "extra_flags", "fit")


def resolve_launch(cfg, body):
    if body.get("preset"):
        p = next((x for x in cfg["presets"] if x["id"] == body["preset"]), None)
        if not p:
            return None, f"unknown preset '{body['preset']}'"
        model = cfg["models"].get(p.get("model", ""), {})
        launch = {"ctx": int(p["ctx"]), "kv": p["kv"],
                  "parallel": int(body.get("parallel", p.get("parallel", 1))),
                  "ngl": int(p.get("ngl", 99)), "model_dir": model.get("dir")}
        # request body wins over preset; only include keys actually specified so the
        # global runtime fallback in _launch still applies for the rest.
        for k in _RT_OVERRIDE_KEYS:
            if body.get(k) is not None:
                launch[k] = body[k]
            elif p.get(k) is not None:
                launch[k] = p[k]
        return launch, p["id"]
    c = body.get("custom")
    if c:
        if c.get("kv", "q4_0") not in ("f16", "q8_0", "q4_0"):
            return None, "kv must be f16, q8_0, or q4_0"
        model = cfg["models"].get(c.get("model", ""), {})
        launch = {"ctx": int(c["ctx"]), "kv": c.get("kv", "q4_0"),
                  "parallel": int(c.get("parallel", 1)), "ngl": int(c.get("ngl", 99)),
                  "model_dir": model.get("dir")}
        for k in _RT_OVERRIDE_KEYS:
            if c.get(k) is not None:
                launch[k] = c[k]
        return launch, "custom"
    return None, "body needs 'preset' or 'custom'"


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
            base = f"http://127.0.0.1:{engine_port(cfg)}/v1"  # follows the active engine
            key = read_api_key(cfg)
            force_model = None  # let local alias pass through
        else:
            base, key, model = remote_target(cfg, ep)
            if not key:
                return self._err(401, f"no API key set for '{eid}' (fill secrets.json)")
            force_model = model

        # read request body
        n = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(n) if n else b""

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
                p["vram_gb"] = estimate_vram(cfg, p["ctx"], p["kv"])
            view["endpoints"] = [endpoint_public(cfg, e) for e in cfg.get("endpoints", [])]
            view["router_url"] = f"http://{self._lan_ip()}:{cfg['server'].get('router_port', 8001)}/v1"
            view["version"] = panel_version()
            return self._send(200, view)

        if path == "/api/models":
            return self._send(200, {"models": scan_models(cfg), "models_root": MODELS_ROOT})

        if path == "/api/status":
            run = running_server(cfg)
            if run and "kv" in run:
                run["vram_gb_est"] = estimate_vram(cfg, run.get("ctx", 0), run["kv"])
            eng_status = {e: {"installed": engine_installed(cfg, e), "port": engine_port(cfg, e)}
                          for e in ("llamacpp", "vllm", "ollama")}
            det_eng = run.get("engine") if run else None
            return self._send(200, {
                "gpu": gpu_stats(), "running": run, "health": local_health(cfg),
                "matched_preset": match_preset(cfg, run), "profiles": cfg.get("profiles", {}),
                "tps": last_tps(cfg), "switch": SWITCH_STATE,
                "active_endpoint": cfg.get("active_endpoint"),
                "active_engine": cfg.get("active_engine", "llamacpp"),
                "running_engine": det_eng, "engines_status": eng_status,
                "runtime": cfg.get("runtime", {}), "version": panel_version(),
                "vram_total_gb": cfg["gpu"]["vram_total_gb"], "server_time": time.time(),
            })

        if path == "/api/log":
            q = urllib.parse.parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
            n = min(int(q.get("n", ["40"])[0]), 400)
            return self._send(200, {"lines": tail(expand(cfg["server"]["server_log"]), n)})

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

        if path == "/api/switch":
            launch, label = resolve_launch(cfg, body)
            if not launch:
                return self._send(400, {"error": label})
            est = estimate_vram(cfg, launch["ctx"], launch["kv"])
            if est > cfg["estimator"]["cap_gb"] and not body.get("force"):
                return self._send(409, {"error": "over_vram_budget", "est_gb": est,
                                        "cap_gb": cfg["estimator"]["cap_gb"]})
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
                       "image_min_tokens", "image_max_tokens", "batch", "ubatch",
                       "extra_flags", "fit")
            upd = {k: body[k] for k in allowed if k in body}
            save_config(lambda d: d.setdefault("runtime", {}).update(upd))
            applied = False
            if body.get("apply"):
                run = running_server(cfg)
                if run and run.get("engine") == "llamacpp" and "kv" in run:
                    launch = {"ctx": run.get("ctx", 98304), "kv": run.get("kv", "q4_0"),
                              "parallel": run.get("parallel", 1), "ngl": run.get("ngl", 99),
                              "model_dir": cfg["models"].get("qwen3-vl", {}).get("dir")}
                    ok, _ = start_switch(load_config(), launch, "runtime-update")
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

        if path == "/api/engine-config":
            eng = body.get("engine")
            if eng not in ("llamacpp", "vllm", "ollama"):
                return self._send(400, {"error": "engine must be llamacpp, vllm, or ollama"})
            fields = {k: v for k, v in body.items() if k != "engine"}

            def mut(d):
                d.setdefault("engines", {}).setdefault(eng, {}).update(fields)
            save_config(mut)
            return self._send(200, {"ok": True, "engine": load_config()["engines"][eng]})

        if path == "/api/models":
            op = body.get("op", "add")
            mid = (body.get("id") or "").strip()
            if not mid:
                return self._send(400, {"error": "id is required"})
            if op == "delete":
                save_config(lambda d: d.get("models", {}).pop(mid, None))
                return self._send(200, {"ok": True})
            if not body.get("dir"):
                return self._send(400, {"error": "dir is required"})
            entry = {"dir": body["dir"], "label": body.get("label", mid),
                     "vision": bool(body.get("vision", False)), "note": body.get("note", "")}
            save_config(lambda d: d.setdefault("models", {}).__setitem__(mid, entry))
            return self._send(200, {"ok": True, "models": load_config().get("models")})

        if path == "/api/preset":
            op = body.get("op", "upsert")
            pid = (body.get("id") or "").strip()
            if not pid:
                return self._send(400, {"error": "id is required"})
            if op == "delete":
                save_config(lambda d: d.update(
                    presets=[p for p in d.get("presets", []) if p["id"] != pid],
                    profiles={k: (v if v != pid else None) for k, v in d.get("profiles", {}).items()}))
                return self._send(200, {"ok": True})
            # upsert
            kv = body.get("kv")
            if kv and kv not in ("f16", "q8_0", "q4_0"):
                return self._send(400, {"error": "kv must be f16, q8_0, or q4_0"})
            existing = next((p for p in cfg.get("presets", []) if p["id"] == pid), None)
            if not existing:
                if body.get("ctx") in (None, "") or not body.get("kv"):
                    return self._send(400, {"error": "new preset needs at least ctx and kv"})
            int_keys = ("ctx", "parallel", "ngl", "cache_reuse", "image_min_tokens", "image_max_tokens")
            # keys that can be cleared back to "inherit" by sending "" / null
            clearable = ("thinking", "reason_effort", "context_shift", "cache_reuse",
                         "image_min_tokens", "image_max_tokens", "tag", "note")

            def mut(d):
                ps = d.setdefault("presets", [])
                base = dict(existing) if existing else {"id": pid, "model": "qwen3-vl"}
                base["id"] = pid
                base.setdefault("model", body.get("model", "qwen3-vl"))
                if body.get("kv"):
                    base["kv"] = body["kv"]
                for k in ("ctx", "parallel", "ngl"):
                    if body.get(k) not in (None, ""):
                        try:
                            base[k] = int(body[k])
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
            save_config(mut)
            saved = next((p for p in load_config()["presets"] if p["id"] == pid), None)
            return self._send(200, {"ok": True, "preset": saved})

        return self._send(404, {"error": "not found"})


def main():
    cfg = load_config()
    host = cfg["server"].get("host", "0.0.0.0")
    pport = int(cfg["server"].get("panel_port", 8080))
    rport = int(cfg["server"].get("router_port", 8001))

    router = ThreadingHTTPServer((host, rport), RouterHandler)
    threading.Thread(target=router.serve_forever, daemon=True).start()
    print(f"router  (OpenAI-compatible) on http://{host}:{rport}/v1", flush=True)

    panel = ThreadingHTTPServer((host, pport), PanelHandler)
    print(f"panel   (web UI)            on http://{host}:{pport}", flush=True)
    try:
        panel.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
