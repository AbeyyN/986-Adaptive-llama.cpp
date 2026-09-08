#!/usr/bin/env python3
"""Runtime benchmark adapters for 986 Adaptive."""

from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from statistics import median
from typing import List, Optional, Tuple

from adaptive_state import BenchmarkCandidate, BenchmarkMetrics, MIB


def find_binary(name: str, explicit: Optional[str] = None) -> str:
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return str(path)
        resolved = shutil.which(explicit)
        if resolved:
            return resolved
        raise ValueError(f"binary not found: {explicit}")

    resolved = shutil.which(name)
    if resolved:
        return resolved

    executable = name + (".exe" if platform.system() == "Windows" else "")
    for candidate in (
        Path("build") / "bin" / executable,
        Path("bin") / executable,
        Path(executable),
    ):
        if candidate.is_file():
            return str(candidate)
    raise ValueError(f"{name} not found; pass its path explicitly")


def _rss_mib(pid: int) -> Optional[float]:
    system = platform.system()
    if system == "Linux":
        try:
            for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
        except (OSError, ValueError, IndexError):
            return None

    if system == "Darwin":
        try:
            output = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True, timeout=1.0)
            value = output.strip()
            return float(value) / 1024.0 if value else None
        except (OSError, ValueError, subprocess.SubprocessError):
            return None

    if system == "Windows":
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        try:
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            PROCESS_VM_READ = 0x0010
            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, False, pid
            )
            if not handle:
                return None
            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
            ctypes.windll.kernel32.CloseHandle(handle)
            return counters.WorkingSetSize / MIB if ok else None
        except (AttributeError, OSError):
            return None

    return None


def _nvidia_vram_mib(pid: int) -> Optional[float]:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=1.5,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    total = 0.0
    found = False
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            row_pid = int(parts[0])
            used = float(parts[1])
        except ValueError:
            continue
        if row_pid == pid:
            total += used
            found = True
    return total if found else None


class MemorySampler:
    def __init__(self, pid: int, interval: float = 0.05):
        self.pid = pid
        self.interval = interval
        self.peak_rss_mib: Optional[float] = None
        self.peak_vram_mib: Optional[float] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = _rss_mib(self.pid)
            vram = _nvidia_vram_mib(self.pid)
            if rss is not None:
                self.peak_rss_mib = max(self.peak_rss_mib or 0.0, rss)
            if vram is not None:
                self.peak_vram_mib = max(self.peak_vram_mib or 0.0, vram)
            self._stop.wait(self.interval)


def parse_llama_bench_json(payload: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        records = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid llama-bench JSON: {exc}") from exc
    if not isinstance(records, list):
        raise ValueError("llama-bench JSON must be a list")

    pp_values: List[float] = []
    tg_values: List[float] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        try:
            n_prompt = int(record.get("n_prompt", 0))
            n_gen = int(record.get("n_gen", 0))
            avg_ts = float(record["avg_ts"])
        except (KeyError, TypeError, ValueError):
            continue
        if n_prompt > 0 and n_gen == 0:
            pp_values.append(avg_ts)
        elif n_prompt == 0 and n_gen > 0:
            tg_values.append(avg_ts)

    pp = sum(pp_values) / len(pp_values) if pp_values else None
    tg = sum(tg_values) / len(tg_values) if tg_values else None
    return pp, tg


def run_llama_bench(
    binary: str,
    model: str,
    candidate: BenchmarkCandidate,
    repetitions: int = 3,
    pp_tokens: int = 512,
    tg_tokens: int = 128,
    timeout: float = 600.0,
) -> BenchmarkMetrics:
    command = [
        binary,
        "-m", model,
        "-o", "json",
        "-r", str(repetitions),
        "-p", str(pp_tokens),
        "-n", str(tg_tokens),
        *candidate.bench_args(),
    ]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    sampler = MemorySampler(proc.pid)
    sampler.start()
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        proc.communicate()
        raise RuntimeError(f"llama-bench timed out for candidate {candidate.id}") from exc
    finally:
        sampler.stop()

    if proc.returncode != 0:
        detail = stderr.strip().splitlines()[-1] if stderr.strip() else f"exit {proc.returncode}"
        raise RuntimeError(f"llama-bench failed for {candidate.id}: {detail}")

    pp, tg = parse_llama_bench_json(stdout)
    if pp is None or tg is None:
        raise RuntimeError(f"llama-bench returned incomplete PP/TG metrics for {candidate.id}")
    return BenchmarkMetrics(pp, tg, None, sampler.peak_rss_mib, sampler.peak_vram_mib)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_health(base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    health_url = base_url.rstrip("/") + "/health"
    last_error: Optional[Exception] = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=1.0) as response:
                if 200 <= response.status < 300:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(0.25)
    raise RuntimeError(f"llama-server did not become healthy: {last_error}")


def _server_model_id(base_url: str) -> str:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/v1/models", timeout=2.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
        data = payload.get("data", [])
        if data and isinstance(data[0], dict) and data[0].get("id"):
            return str(data[0]["id"])
    except (OSError, ValueError, KeyError, urllib.error.URLError):
        pass
    return "local"


def probe_ttft(base_url: str, prompt: str = "Reply with one short word.", timeout: float = 30.0) -> float:
    model_id = _server_model_id(base_url)
    body = json.dumps(
        {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8,
            "temperature": 0,
            "stream": True,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                event = json.loads(data)
                content = event["choices"][0]["delta"].get("content")
            except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                continue
            if content:
                return (time.perf_counter() - start) * 1000.0
    raise RuntimeError("stream ended before the first generated token")


def measure_candidate_ttft(
    server_binary: str,
    model: str,
    candidate: BenchmarkCandidate,
    repetitions: int = 3,
    startup_timeout: float = 120.0,
) -> float:
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    command = [
        server_binary,
        "-m", model,
        "--host", "127.0.0.1",
        "--port", str(port),
        *candidate.llama_args(),
    ]
    proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait_for_health(base_url, startup_timeout)
        samples = [probe_ttft(base_url) for _ in range(max(1, repetitions))]
        return float(median(samples))
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)
