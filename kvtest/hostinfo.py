"""Capture the hardware and build a run actually used.

Everything is read from the machine at call time and every probe is allowed to fail into "unknown":
a report that names the GPU, the driver, the kernel, the CUDA toolkit, the core count, the RAM and
the exact llama.cpp build flags is reproducible; one that says "an RTX 3080" is not.
"""
import json
import os
import platform
import re
import subprocess

LLAMA_DIR = os.environ.get("LLAMA_DIR", "/home/kome/Projects/code/clones/llama.cpp")
CMAKE_FLAGS = ("GGML_CUDA", "GGML_CUDA_FA", "GGML_CUDA_FA_ALL_QUANTS", "GGML_CUDA_FA_QUANTS",
               "CMAKE_CUDA_ARCHITECTURES", "CMAKE_BUILD_TYPE", "GGML_NATIVE", "GGML_CUDA_NO_VMM")


def _run(cmd, timeout=10):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (p.stdout or "") + (p.stderr or "")
    except Exception:                                              # noqa: BLE001
        return ""


def _parse_gpu(out):
    """nvidia-smi CSV -> dict, or {} when the line cannot be parsed (toolkit without a driver)."""
    lines = [ln for ln in (out or "").splitlines() if ln.strip()]
    if not lines:
        return {}
    parts = [p.strip() for p in lines[0].split(",")]
    if len(parts) < 5 or not parts[0]:
        return {}
    return {"name": parts[0], "memory_total_mib": _int(parts[1]),
            "memory_used_mib": _int(parts[2]), "driver": parts[3],
            "compute_capability": parts[4]}


def gpu():
    """Never returns a bare {}: either it names the GPU, or it says why it could not."""
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version,"
                               "compute_cap", "--format=csv,noheader"])
    if not out.strip():
        return {"error": "nvidia-smi unavailable (no tool, or no reachable driver)"}
    rec = _parse_gpu(out)
    if not rec.get("name"):
        return {"error": "nvidia-smi produced no parsable GPU line", "raw": out.strip()[:200]}
    rec["usable_budget_note"] = ("the study leaves ~1 GiB for the display/driver; the fit probe "
                                 "records the peak actually reached per arm")
    return rec


def _int(s):
    m = re.match(r"\s*(\d+)", s or "")
    return int(m.group(1)) if m else None


def cpu():
    model, cores = None, os.cpu_count()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {"model": model, "logical_cores": cores, "threads": _nproc()}


def _nproc():
    out = _run(["nproc", "--all"])
    return _int(out) or os.cpu_count()


def ram_gib():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kib = int(line.split()[1])
                    return round(kib / 1024 / 1024, 1)
    except OSError:
        pass
    return None


def os_name():
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    return platform.platform()


def cuda():
    out = _run(["nvcc", "--version"])
    m = re.search(r"release (\d+\.\d+)", out)
    return m.group(1) if m else "unknown"


def llama_build():
    cache = os.path.join(LLAMA_DIR, "build", "CMakeCache.txt")
    flags = {}
    if os.path.exists(cache):
        with open(cache) as f:
            for line in f:
                for k in CMAKE_FLAGS:
                    if line.startswith(f"{k}:"):
                        flags[k] = line.split("=", 1)[1].strip()
    commit = _run(["git", "-C", LLAMA_DIR, "rev-parse", "HEAD"]).strip() or None
    dirty = bool(_run(["git", "-C", LLAMA_DIR, "status", "--porcelain"]).strip())
    version = _run([os.path.join(LLAMA_DIR, "build", "bin", "llama-server"), "--version"]).strip()
    return {"commit": commit, "tree_dirty": dirty,
            "version": version.splitlines()[1].strip() if len(version.splitlines()) > 1 else version,
            "flags": flags}


def disk_free_gib(path=None):
    try:
        st = os.statvfs(path or os.path.expanduser("~"))
        return round(st.f_bavail * st.f_frsize / (1 << 30), 1)
    except OSError:
        return None


def snapshot():
    """-> dict suitable for embedding in raw.json / hardware.json."""
    return {
        "captured_at_utc": _now(),
        "gpu": gpu(),
        "cpu": cpu(),
        "ram_gib": ram_gib(),
        "os": os_name(),
        "kernel": platform.release(),
        "cuda_toolkit": cuda(),
        "llama_cpp": llama_build(),
        "home_free_gib": disk_free_gib(),
        "python": platform.python_version(),
    }


def _now():
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def one_line(hw):
    """Compact string for a report header."""
    if not hw:
        return "unknown"
    g = hw.get("gpu") or {}
    ll = hw.get("llama_cpp") or {}
    bits = [g.get("name") or "GPU unknown"]
    if g.get("memory_total_mib"):
        bits[-1] += f" {g['memory_total_mib']} MiB"
    if g.get("compute_capability"):
        bits[-1] += f" sm_{str(g['compute_capability']).replace('.', '')}"
    if g.get("driver"):
        bits.append(f"driver {g['driver']}")
    if hw.get("cuda_toolkit"):
        bits.append(f"CUDA {hw['cuda_toolkit']}")
    if hw.get("cpu"):
        bits.append(f"{hw['cpu'].get('model') or 'CPU'} ({hw['cpu'].get('logical_cores')} cores)")
    if hw.get("ram_gib"):
        bits.append(f"{hw['ram_gib']} GiB RAM")
    if hw.get("kernel"):
        bits.append(f"kernel {hw['kernel']}")
    if ll.get("commit"):
        bits.append(f"llama.cpp {ll['commit'][:9]}")
    return " · ".join(bits)


if __name__ == "__main__":
    print(json.dumps(snapshot(), indent=2, sort_keys=True))
