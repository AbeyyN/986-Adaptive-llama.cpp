#!/usr/bin/env python3
"""986 Adaptive llama.cpp bootstrap AutoTune control plane.

This module is intentionally dependency-free and additive to upstream llama.cpp.
It detects host capabilities, applies a portable policy profile, and emits a
candidate llama-server argument set. The candidate is a bootstrap heuristic;
benchmark-backed tuning is the next stage and remains authoritative.

No hostnames, IP addresses, device models, or owner-specific values are
hard-coded here. User overrides always win.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

GIB = 1024 ** 3

PROFILE_MEMORY_FRACTION: Dict[str, float] = {
    "balanced": 0.80,
    "latency": 0.75,
    "throughput": 0.85,
    "memory": 0.65,
    "thermal-safe": 0.70,
    "cluster": 0.80,
}

PROFILE_THREAD_FRACTION: Dict[str, float] = {
    "balanced": 0.75,
    "latency": 0.75,
    "throughput": 1.00,
    "memory": 0.60,
    "thermal-safe": 0.50,
    "cluster": 0.70,
}


@dataclass(frozen=True)
class SystemProfile:
    os: str
    arch: str
    logical_cpus: int
    memory_gib: float
    backend_candidates: List[str]
    preferred_backend: str


@dataclass(frozen=True)
class TuneResult:
    profile: str
    backend: str
    threads: int
    ctx_size: int
    batch_size: int
    ubatch_size: int
    flash_attn: str
    cache_type_k: str
    cache_type_v: str
    gpu_layers: Optional[int]
    memory_budget_gib: float
    model_gib: Optional[float]
    notes: List[str]

    def llama_args(self) -> List[str]:
        args = [
            "--threads", str(self.threads),
            "--ctx-size", str(self.ctx_size),
            "--batch-size", str(self.batch_size),
            "--ubatch-size", str(self.ubatch_size),
            "--flash-attn", self.flash_attn,
            "--cache-type-k", self.cache_type_k,
            "--cache-type-v", self.cache_type_v,
        ]
        if self.gpu_layers is not None:
            args.extend(["--gpu-layers", str(self.gpu_layers)])
        return args


def _run_quiet(command: List[str], timeout: float = 2.0) -> bool:
    try:
        proc = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def total_memory_bytes() -> int:
    system = platform.system()

    if system == "Linux":
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass

    if system == "Darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, timeout=2.0)
            return int(out.strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            pass

    if system == "Windows":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        try:
            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        except (AttributeError, OSError):
            pass

    # Conservative fallback for restricted/containerized environments.
    return 8 * GIB


def detect_backends() -> List[str]:
    candidates: List[str] = []
    system = platform.system()
    machine = platform.machine().lower()

    if shutil.which("nvidia-smi") and _run_quiet(["nvidia-smi", "-L"]):
        candidates.append("cuda")

    if shutil.which("rocminfo") and _run_quiet(["rocminfo"]):
        candidates.append("hip")

    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        candidates.append("metal")

    if shutil.which("sycl-ls") and _run_quiet(["sycl-ls"]):
        candidates.append("sycl")

    if shutil.which("vulkaninfo") and _run_quiet(["vulkaninfo", "--summary"]):
        candidates.append("vulkan")

    candidates.append("cpu")
    return list(dict.fromkeys(candidates))


def preferred_backend(candidates: List[str]) -> str:
    # Preference only selects the first benchmark candidate. Later stages must
    # use measured PP/TG/TTFT data rather than assuming this order is fastest.
    order = ["cuda", "hip", "metal", "sycl", "vulkan", "cpu"]
    return next((item for item in order if item in candidates), "cpu")


def detect_system() -> SystemProfile:
    cpus = max(1, os.cpu_count() or 1)
    memory_gib = total_memory_bytes() / GIB
    candidates = detect_backends()
    return SystemProfile(
        os=platform.system() or "Unknown",
        arch=platform.machine() or "unknown",
        logical_cpus=cpus,
        memory_gib=round(memory_gib, 2),
        backend_candidates=candidates,
        preferred_backend=preferred_backend(candidates),
    )


def model_size_gib(path: Optional[str], explicit_gib: Optional[float]) -> Optional[float]:
    if explicit_gib is not None:
        if explicit_gib <= 0:
            raise ValueError("--model-gib must be greater than zero")
        return round(explicit_gib, 3)
    if path:
        model = Path(path)
        if not model.is_file():
            raise ValueError(f"model file not found: {model}")
        return round(model.stat().st_size / GIB, 3)
    return None


def _context_for_memory(available_after_model: float, requested: Optional[int]) -> int:
    if requested is not None:
        if requested < 256:
            raise ValueError("--context must be >= 256")
        return requested

    # Conservative bootstrap tiers. Precise KV sizing depends on architecture,
    # KV data types, layer count and embedding dimensions; benchmark stage will
    # replace these heuristics with model-aware measurements.
    if available_after_model >= 24:
        return 32768
    if available_after_model >= 12:
        return 16384
    if available_after_model >= 6:
        return 8192
    return 4096


def tune(
    system: SystemProfile,
    profile: str = "balanced",
    model_gib: Optional[float] = None,
    requested_context: Optional[int] = None,
    backend_override: Optional[str] = None,
    threads_override: Optional[int] = None,
) -> TuneResult:
    if profile not in PROFILE_MEMORY_FRACTION:
        raise ValueError(f"unknown profile: {profile}")

    backend = backend_override or system.preferred_backend
    if backend_override and backend_override not in system.backend_candidates:
        # Explicit user override still wins, but surface the mismatch.
        override_note = f"backend override '{backend_override}' was not auto-detected"
    else:
        override_note = ""

    memory_budget = max(1.0, system.memory_gib * PROFILE_MEMORY_FRACTION[profile])
    reserve_for_model = model_gib or 0.0
    available_after_model = max(0.5, memory_budget - reserve_for_model)

    if threads_override is not None:
        if threads_override < 1:
            raise ValueError("--threads must be >= 1")
        threads = threads_override
    else:
        threads = max(1, round(system.logical_cpus * PROFILE_THREAD_FRACTION[profile]))

    ctx = _context_for_memory(available_after_model, requested_context)

    if profile == "latency":
        batch, ubatch = 512, 256
    elif profile == "throughput":
        batch, ubatch = 2048, 512
    elif profile == "memory":
        batch, ubatch = 256, 128
    elif profile == "thermal-safe":
        batch, ubatch = 256, 128
    elif profile == "cluster":
        batch, ubatch = 1024, 256
    else:
        batch, ubatch = 1024, 256

    # Quantized KV is the portable bootstrap default for a memory-aware fork.
    # Benchmark/quality validation may promote individual hosts to f16/q8_0.
    if system.memory_gib >= 48 and profile not in {"memory", "thermal-safe"}:
        cache_k, cache_v = "q8_0", "q8_0"
    else:
        cache_k, cache_v = "q8_0", "q5_1"

    accelerator = backend != "cpu"
    gpu_layers = 999 if accelerator else None
    flash_attn = "auto"

    notes = [
        "bootstrap heuristic only; measured benchmark data must override heuristics",
        "user-supplied backend/context/thread overrides take precedence over AutoTune",
    ]
    if override_note:
        notes.append(override_note)
    if model_gib is None:
        notes.append("model size unknown; pass --model or --model-gib for safer memory planning")
    if model_gib is not None and model_gib >= memory_budget:
        notes.append("model size meets/exceeds memory budget; consider stronger quantization or RPC offload")
    if backend in {"sycl", "vulkan"}:
        notes.append("integrated/discrete GPU topology is not yet classified; benchmark gpu-layers before locking")

    return TuneResult(
        profile=profile,
        backend=backend,
        threads=threads,
        ctx_size=ctx,
        batch_size=batch,
        ubatch_size=ubatch,
        flash_attn=flash_attn,
        cache_type_k=cache_k,
        cache_type_v=cache_v,
        gpu_layers=gpu_layers,
        memory_budget_gib=round(memory_budget, 2),
        model_gib=model_gib,
        notes=notes,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="986-adaptive", description="986 Adaptive llama.cpp hardware profiler and bootstrap AutoTune")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("detect", help="print detected system capabilities as JSON")

    tune_parser = sub.add_parser("tune", help="emit a candidate llama-server configuration")
    tune_parser.add_argument("--profile", choices=sorted(PROFILE_MEMORY_FRACTION), default="balanced")
    tune_parser.add_argument("--model", help="path to a local GGUF file; used only for size budgeting")
    tune_parser.add_argument("--model-gib", type=float, help="model size in GiB when no local path is available")
    tune_parser.add_argument("--context", type=int, help="explicit context size; overrides AutoTune")
    tune_parser.add_argument("--backend", help="explicit backend override; user override wins")
    tune_parser.add_argument("--threads", type=int, help="explicit thread override; user override wins")
    tune_parser.add_argument("--format", choices=["json", "args"], default="json")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    system = detect_system()

    if args.command == "detect":
        print(json.dumps(asdict(system), indent=2, sort_keys=True))
        return 0

    try:
        size = model_size_gib(args.model, args.model_gib)
        result = tune(
            system,
            profile=args.profile,
            model_gib=size,
            requested_context=args.context,
            backend_override=args.backend,
            threads_override=args.threads,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.format == "args":
        print(" ".join(result.llama_args()))
    else:
        payload = {"system": asdict(system), "tuning": asdict(result), "llama_args": result.llama_args()}
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
