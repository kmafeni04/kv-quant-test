"""llama-server lifecycle: command construction, health, stderr capture, VRAM sampling.

One server per arm. The command line is fully determined by (model, kv, ctx, knobs), which is
what keeps the three KV arms of a model comparable.
"""
import os
import signal
import socket
import subprocess
import threading
import time

from . import client, config


def kv_ok(kv):
    return kv in ("f16", "q8_0", "q4_0", "bf16", "q5_1", "q5_0")


def build_cmd(model_key, kv, ctx, knobs=None, port=None, extra_ctx_flags=None):
    m = config.MODELS[model_key]
    knobs = dict(m["knobs"] if knobs is None else knobs)
    cmd = [config.SERVER_BIN, "-m", m["path"], "--alias", f"{model_key}-{kv}",
           "--port", str(port or config.PORT),
           "-c", str(ctx), "-ctk", kv, "-ctv", kv]
    cmd += list(config.SERVER_COMMON)
    cmd += list(m.get("extra") or [])
    if knobs.get("ncmoe"):
        cmd += ["--n-cpu-moe", str(knobs["ncmoe"])]
    if knobs.get("ncffn"):
        cmd += ["--n-cpu-ffn", str(knobs["ncffn"])]
    cmd += list(extra_ctx_flags or [])
    return cmd


def port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def gpu_mib():
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return int(p.stdout.strip().splitlines()[0])
    except Exception:                                              # noqa: BLE001
        return -1


class VramSampler:
    """Peak GPU memory (MiB) while the server is up, sampled every second."""

    def __init__(self, period=1.0):
        self.period, self.peak, self.samples = period, 0, []
        self._stop = threading.Event()
        self._t = None

    def start(self):
        self.peak = 0
        self.samples = []
        self._stop.clear()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while not self._stop.is_set():
            v = gpu_mib()
            if v >= 0:
                self.peak = max(self.peak, v)
                self.samples.append(v)
            self._stop.wait(self.period)

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join(timeout=5)


class Server:
    def __init__(self, model_key, kv, ctx, knobs=None, port=None, log_path=None):
        self.model_key, self.kv, self.ctx = model_key, kv, ctx
        self.knobs = dict(config.MODELS[model_key]["knobs"] if knobs is None else knobs)
        self.port = port or config.PORT
        self.base = f"http://127.0.0.1:{self.port}"
        self.log_path = log_path or os.path.join("/tmp", f"kv-{model_key}-{kv}.log")
        self.proc = None
        self.vram = VramSampler()
        self.start_s, self.load_s = None, None

    def start(self, timeout=900):
        if not port_free(self.port):
            print(f"[server] port {self.port} still in use; refusing to start", flush=True)
            return False
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        cmd = build_cmd(self.model_key, self.kv, self.ctx, self.knobs, self.port)
        self.cmd = cmd
        t0 = time.time()
        self.log = open(self.log_path, "wb")
        self.log.write(("# " + " ".join(cmd) + "\n").encode())
        self.log.flush()
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        self.vram.start()
        ok, waited, payload = client.wait_ready(self.base, timeout=timeout, proc=self.proc)
        self.start_s = round(time.time() - t0, 1)
        if not ok:
            print(f"[server] {self.model_key} {self.kv} ctx={self.ctx} knobs={self.knobs} "
                  f"failed after {waited:.0f}s: {payload}", flush=True)
            self.stop()
            return False
        self.load_s = self.start_s
        return True

    def log_tail(self, n=4000):
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - n))
                return f.read().decode("utf-8", "replace")
        except OSError:
            return ""

    def oom_reason(self):
        """Classify the server log, so the fit probe can tell OOM from a slow load."""
        tail = self.log_tail(20000).lower()
        for marker in ("out of memory", "failed to allocate", "cuda error", "cuda malloc",
                       "ggml_backend_cuda_buffer_type_alloc_buffer", "erroroutofdevicememory"):
            if marker in tail:
                return "oom"
        if self.proc is not None and self.proc.poll() is not None:
            return "exited"
        return None

    def stop(self, hard_after=120.0):
        self.vram.stop()
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            t0 = time.time()
            while self.proc.poll() is None and time.time() - t0 < hard_after:
                time.sleep(0.5)
            if self.proc.poll() is None:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait(timeout=30)
        try:
            self.log.close()
        except Exception:                                          # noqa: BLE001
            pass
        t0 = time.time()
        while not port_free(self.port) and time.time() - t0 < 60:
            time.sleep(0.5)
        self.proc = None

    def __enter__(self):
        if not self.start():
            raise RuntimeError(f"server did not start: {self.model_key} {self.kv}")
        return self

    def __exit__(self, *exc):
        self.stop()
        return False
