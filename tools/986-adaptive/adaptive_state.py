#!/usr/bin/env python3
"""State, fingerprints and candidate generation for 986 Adaptive."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from adaptive986 import SystemProfile, TuneResult

MIB = 1024 ** 2
SCHEMA_VERSION = 1

PROFILE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "balanced": {"pp": 0.34, "tg": 0.34, "ttft": 0.17, "memory": 0.15},
    "latency": {"pp": 0.20, "tg": 0.35, "ttft": 0.35, "memory": 0.10},
    "throughput": {"pp": 0.50, "tg": 0.38, "ttft": 0.04, "memory": 0.08},
    "memory": {"pp": 0.12, "tg": 0.18, "ttft": 0.05, "memory": 0.65},
    "thermal-safe": {"pp": 0.20, "tg": 0.25, "ttft": 0.10, "memory": 0.45},
    "cluster": {"pp": 0.42, "tg": 0.38, "ttft": 0.10, "memory": 0.10},
}


@dataclass(frozen=True)
class BenchmarkCandidate:
    id: str
    backend: str
    threads: int
    ctx_size: int
    batch_size: int
    ubatch_size: int
    gpu_layers: Optional[int]
    flash_attn: str
    cache_type_k: str
    cache_type_v: str

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

    def bench_args(self) -> List[str]:
        args = [
            "-t", str(self.threads),
            "-b", str(self.batch_size),
            "-ub", str(self.ubatch_size),
            "-fa", self.flash_attn,
            "-ctk", self.cache_type_k,
            "-ctv", self.cache_type_v,
        ]
        if self.gpu_layers is not None:
            args.extend(["-ngl", str(self.gpu_layers)])
        return args


@dataclass(frozen=True)
class BenchmarkMetrics:
    pp_ts: Optional[float]
    tg_ts: Optional[float]
    ttft_ms: Optional[float]
    peak_rss_mib: Optional[float]
    peak_vram_mib: Optional[float]

    @property
    def peak_memory_mib(self) -> Optional[float]:
        values = [v for v in (self.peak_rss_mib, self.peak_vram_mib) if v is not None]
        return sum(values) if values else None


@dataclass(frozen=True)
class CandidateResult:
    candidate: BenchmarkCandidate
    metrics: BenchmarkMetrics
    score: float


@dataclass(frozen=True)
class GuardResult:
    ok: bool
    regressions: List[str]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stable_hash(payload: Dict[str, object], prefix: str) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(raw).hexdigest()[:20]}"


def hardware_fingerprint(system: SystemProfile) -> str:
    material: Dict[str, object] = {
        "schema": 1,
        "os": system.os,
        "arch": system.arch,
        "logical_cpus": system.logical_cpus,
        "memory_mib": round(system.memory_gib * 1024),
        "backend_candidates": sorted(system.backend_candidates),
    }
    return _stable_hash(material, "hw")


def model_fingerprint(path: str, chunk_bytes: int = 2 * MIB) -> str:
    model = Path(path)
    if not model.is_file():
        raise ValueError(f"model file not found: {model}")

    size = model.stat().st_size
    digest = hashlib.sha256()
    digest.update(size.to_bytes(16, "big", signed=False))
    with model.open("rb") as handle:
        digest.update(handle.read(chunk_bytes))
        if size > chunk_bytes:
            handle.seek(max(0, size - chunk_bytes))
            digest.update(handle.read(chunk_bytes))
    return f"model-{digest.hexdigest()[:20]}"


def state_root() -> Path:
    override = os.environ.get("986_ADAPTIVE_HOME")
    if override:
        return Path(override).expanduser()
    if platform.system() == "Windows":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "986-adaptive"
    xdg = os.environ.get("XDG_STATE_HOME")
    if xdg:
        return Path(xdg) / "986-adaptive"
    return Path.home() / ".local" / "state" / "986-adaptive"


def profile_path(hardware_id: str, model_id: str, profile: str) -> Path:
    return state_root() / "profiles" / hardware_id / model_id / f"{profile}.json"


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_from_tune(result: TuneResult, candidate_id: str, **changes: object) -> BenchmarkCandidate:
    values: Dict[str, object] = {
        "id": candidate_id,
        "backend": result.backend,
        "threads": result.threads,
        "ctx_size": result.ctx_size,
        "batch_size": result.batch_size,
        "ubatch_size": result.ubatch_size,
        "gpu_layers": result.gpu_layers,
        "flash_attn": result.flash_attn,
        "cache_type_k": result.cache_type_k,
        "cache_type_v": result.cache_type_v,
    }
    values.update(changes)
    return BenchmarkCandidate(**values)  # type: ignore[arg-type]


def generate_candidates(system: SystemProfile, result: TuneResult, limit: int = 7) -> List[BenchmarkCandidate]:
    """Generate a small, deterministic search set around the bootstrap tune.

    The search is intentionally bounded. P0 optimizes common high-impact knobs
    without creating a combinatorial benchmark matrix on every new host.
    """

    if limit < 1:
        raise ValueError("candidate limit must be >= 1")

    candidates: List[BenchmarkCandidate] = [_candidate_from_tune(result, "base")]

    thread_values = [
        max(1, round(result.threads * 0.67)),
        min(system.logical_cpus, max(result.threads + 1, round(system.logical_cpus * 0.9))),
    ]
    for index, threads in enumerate(thread_values, start=1):
        if threads != result.threads:
            candidates.append(_candidate_from_tune(result, f"threads-{index}", threads=threads))

    smaller_batch = max(128, result.batch_size // 2)
    smaller_ubatch = min(max(64, result.ubatch_size // 2), smaller_batch)
    if (smaller_batch, smaller_ubatch) != (result.batch_size, result.ubatch_size):
        candidates.append(
            _candidate_from_tune(
                result,
                "batch-low",
                batch_size=smaller_batch,
                ubatch_size=smaller_ubatch,
            )
        )

    larger_batch = min(4096, result.batch_size * 2)
    larger_ubatch = min(larger_batch, min(1024, result.ubatch_size * 2))
    if (larger_batch, larger_ubatch) != (result.batch_size, result.ubatch_size):
        candidates.append(
            _candidate_from_tune(
                result,
                "batch-high",
                batch_size=larger_batch,
                ubatch_size=larger_ubatch,
            )
        )

    if result.gpu_layers is not None:
        candidates.append(_candidate_from_tune(result, "cpu-fallback", gpu_layers=0))

    if (result.cache_type_k, result.cache_type_v) != ("f16", "f16"):
        candidates.append(_candidate_from_tune(result, "kv-f16", cache_type_k="f16", cache_type_v="f16"))

    deduped: List[BenchmarkCandidate] = []
    signatures = set()
    for candidate in candidates:
        signature = tuple(candidate.llama_args())
        if signature not in signatures:
            signatures.add(signature)
            deduped.append(candidate)
    return deduped[:limit]
