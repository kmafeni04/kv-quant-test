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


def wait_vram_free(below=1024, timeout=240):
    """Block until the GPU is released, or report that it never was.

    A CUDA context outlives its process by a moment: `llama-perplexity` launched two seconds after a
    server exits allocates weights into a still-occupied device and dies on a 300 MiB cudaMalloc.
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        used = gpu_mib()
        if 0 <= used < below:
            return True
        time.sleep(2)
    holders = _gpu_holders()
    print(f"[server] WARNING: GPU still holds {gpu_mib()} MiB after {timeout}s"
          + (f" (holder pids: {holders})" if holders else ""), flush=True)
    return False


def _gpu_holders():
    """PIDs the driver reports as holding device memory, for the timeout message."""
    try:
        p = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return ", ".join(line.strip() for line in p.stdout.strip().splitlines() if line.strip())
    except Exception:                                              # noqa: BLE001
        return ""


def gpu_mib():
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return int(p.stdout.strip().splitlines()[0])
    except Exception:                                              # noqa: BLE001
        return -1


def _tail_until(path, needle, window=4 << 20):
    """Last occurrence of `needle` and the following 2 KiB, scanning the file backwards."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            end = f.tell()
            carry = b""
            while end > 0:
                start = max(0, end - window)
                f.seek(start)
                chunk = f.read(end - start) + carry
                idx = chunk.rfind(needle.encode())
                if idx >= 0:
                    return chunk[idx:idx + 2048].decode("utf-8", "replace")
                carry = chunk[:200]
                end = start
    except OSError:
        return None
    return None


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

    def wait_vram_free(self, below=1024, timeout=240):
        """A previous arm's CUDA context can outlive its process for a moment; do not race it."""
        return wait_vram_free(below=below, timeout=timeout)

    def start(self, timeout=900, attempts=2):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        for attempt in range(attempts):
            if not port_free(self.port):
                print(f"[server] port {self.port} still in use; refusing to start", flush=True)
                return False
            self.wait_vram_free()
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
            if ok:
                self.load_s = self.start_s
                return True
            reason = self.oom_reason() or str(payload)
            print(f"[server] {self.model_key} {self.kv} ctx={self.ctx} knobs={self.knobs} "
                  f"attempt {attempt + 1}/{attempts} failed after {waited:.0f}s "
                  f"(gpu {gpu_mib()} MiB used): {reason}", flush=True)
            self.stop()
            if attempt + 1 < attempts:
                # A wedged loader can hold tens of GiB long after its process was killed; the
                # retry is only worth making once the driver has actually handed the memory back.
                wait_vram_free(timeout=600)
        return False

    def log_tail(self, n=4000):
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - n))
                return f.read().decode("utf-8", "replace")
        except OSError:
            return ""

    def sampler_resolved(self):
        """Parse the last resolved sampler block from the `-v` trace, per arm.

        The verbose log is tens of MB per arm, so scan backwards in chunks instead of reading a
        fixed tail that may start after the last sampler line.
        """
        import re
        block = _tail_until(self.log_path, "sampler params:", window=4 << 20)
        if not block:
            return {"error": "no sampler params line in server log", "greedy": None}
        out = {"raw": block.strip()[:1200]}
        for key in ("repeat_penalty", "top_k", "top_p", "min_p", "temp"):
            m = re.search(rf"{key} = ([-+]?\d+(?:\.\d+)?)", block)
            if m:
                out["temperature" if key == "temp" else key] = float(m.group(1))
        out["greedy"] = out.get("temperature") == 0.0
        return out

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
