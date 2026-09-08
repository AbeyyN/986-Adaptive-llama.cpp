#!/usr/bin/env python3
"""Measured scoring, persistence and regression policy for 986 Adaptive."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from adaptive986 import SystemProfile, detect_system, model_size_gib, tune
from adaptive_runtime import measure_candidate_ttft, run_llama_bench
from adaptive_state import (
    PROFILE_WEIGHTS,
    SCHEMA_VERSION,
    BenchmarkCandidate,
    BenchmarkMetrics,
    CandidateResult,
    GuardResult,
    generate_candidates,
    hardware_fingerprint,
    load_json,
    model_fingerprint,
    profile_path,
    save_json,
    utc_now,
)


def _normalize_positive(values: Sequence[Optional[float]]) -> List[Optional[float]]:
    available = [v for v in values if v is not None and v > 0]
    if not available:
        return [None] * len(values)
    best = max(available)
    return [(v / best if v is not None and v > 0 else None) for v in values]


def _normalize_lower(values: Sequence[Optional[float]]) -> List[Optional[float]]:
    available = [v for v in values if v is not None and v > 0]
    if not available:
        return [None] * len(values)
    best = min(available)
    return [(best / v if v is not None and v > 0 else None) for v in values]


def score_results(profile: str, raw: Sequence[Tuple[BenchmarkCandidate, BenchmarkMetrics]]) -> List[CandidateResult]:
    if profile not in PROFILE_WEIGHTS:
        raise ValueError(f"unknown profile: {profile}")
    if not raw:
        return []

    pp_norm = _normalize_positive([metrics.pp_ts for _, metrics in raw])
    tg_norm = _normalize_positive([metrics.tg_ts for _, metrics in raw])
    ttft_norm = _normalize_lower([metrics.ttft_ms for _, metrics in raw])
    mem_norm = _normalize_lower([metrics.peak_memory_mib for _, metrics in raw])
    weights = PROFILE_WEIGHTS[profile]

    scored: List[CandidateResult] = []
    for index, (candidate, metrics) in enumerate(raw):
        components = {
            "pp": pp_norm[index],
            "tg": tg_norm[index],
            "ttft": ttft_norm[index],
            "memory": mem_norm[index],
        }
        numerator = 0.0
        denominator = 0.0
        for name, normalized in components.items():
            if normalized is None:
                continue
            weight = weights[name]
            numerator += normalized * weight
            denominator += weight
        score = numerator / denominator if denominator else 0.0
        scored.append(CandidateResult(candidate, metrics, round(score, 6)))

    return sorted(scored, key=lambda item: item.score, reverse=True)


def _result_payload(
    system: SystemProfile,
    hardware_id: str,
    model_id: str,
    model: str,
    profile: str,
    results: Sequence[CandidateResult],
) -> Dict[str, object]:
    if not results:
        raise ValueError("cannot persist an empty benchmark result")
    winner = results[0]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "hardware_fingerprint": hardware_id,
        "model_fingerprint": model_id,
        "model_size_gib": model_size_gib(model, None),
        "profile": profile,
        "system": asdict(system),
        "winner_id": winner.candidate.id,
        "winner": {
            "candidate": asdict(winner.candidate),
            "metrics": asdict(winner.metrics),
            "score": winner.score,
        },
        "results": [
            {
                "candidate": asdict(item.candidate),
                "metrics": asdict(item.metrics),
                "score": item.score,
            }
            for item in results
        ],
    }


def benchmark(
    model: str,
    profile: str,
    llama_bench: str,
    llama_server: Optional[str] = None,
    repetitions: int = 3,
    ttft_repetitions: int = 3,
    candidate_limit: int = 7,
    save: bool = True,
) -> Tuple[Dict[str, object], Optional[Path]]:
    system = detect_system()
    size = model_size_gib(model, None)
    bootstrap = tune(system, profile=profile, model_gib=size)
    candidates = generate_candidates(system, bootstrap, limit=candidate_limit)

    raw: List[Tuple[BenchmarkCandidate, BenchmarkMetrics]] = []
    for candidate in candidates:
        metrics = run_llama_bench(llama_bench, model, candidate, repetitions=repetitions)
        if llama_server:
            ttft = measure_candidate_ttft(
                llama_server,
                model,
                candidate,
                repetitions=ttft_repetitions,
            )
            metrics = BenchmarkMetrics(
                metrics.pp_ts,
                metrics.tg_ts,
                ttft,
                metrics.peak_rss_mib,
                metrics.peak_vram_mib,
            )
        raw.append((candidate, metrics))

    scored = score_results(profile, raw)
    hardware_id = hardware_fingerprint(system)
    model_id = model_fingerprint(model)
    payload = _result_payload(system, hardware_id, model_id, model, profile, scored)

    path: Optional[Path] = None
    if save:
        path = profile_path(hardware_id, model_id, profile)
        save_json(path, payload)
    return payload, path


def persisted_profile_for(
    model: str, profile: str = "balanced", system: Optional[SystemProfile] = None
) -> Tuple[Dict[str, object], Path]:
    active_system = system or detect_system()
    path = profile_path(hardware_fingerprint(active_system), model_fingerprint(model), profile)
    if not path.is_file():
        raise ValueError(f"no persisted profile for this hardware/model/profile: {path}")
    return load_json(path), path


def winner_candidate(payload: Dict[str, object]) -> BenchmarkCandidate:
    winner = payload.get("winner")
    if not isinstance(winner, dict):
        raise ValueError("benchmark JSON has no winner")
    candidate = winner.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError("benchmark JSON has no winner candidate")
    try:
        return BenchmarkCandidate(**candidate)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"invalid persisted candidate: {exc}") from exc


def winner_metrics(payload: Dict[str, object]) -> Dict[str, Optional[float]]:
    winner = payload.get("winner")
    if not isinstance(winner, dict):
        raise ValueError("benchmark JSON has no winner")
    metrics = winner.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("benchmark JSON has no winner metrics")
    return {
        "pp_ts": _optional_float(metrics.get("pp_ts")),
        "tg_ts": _optional_float(metrics.get("tg_ts")),
        "ttft_ms": _optional_float(metrics.get("ttft_ms")),
        "peak_memory_mib": _peak_memory_from_dict(metrics),
    }


def _optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _peak_memory_from_dict(metrics: Dict[str, object]) -> Optional[float]:
    values = [_optional_float(metrics.get("peak_rss_mib")), _optional_float(metrics.get("peak_vram_mib"))]
    available = [value for value in values if value is not None]
    return sum(available) if available else None


def _pct_drop(previous: float, current: float) -> float:
    return ((previous - current) / previous) * 100.0 if previous > 0 else 0.0


def _pct_rise(previous: float, current: float) -> float:
    return ((current - previous) / previous) * 100.0 if previous > 0 else 0.0


def compare_performance(
    baseline: Dict[str, object],
    current: Dict[str, object],
    pp_drop: float = 7.0,
    tg_drop: float = 7.0,
    ttft_rise: float = 10.0,
    memory_rise: float = 10.0,
) -> GuardResult:
    old = winner_metrics(baseline)
    new = winner_metrics(current)
    regressions: List[str] = []

    checks = [
        ("PP", "pp_ts", "drop", pp_drop),
        ("TG", "tg_ts", "drop", tg_drop),
        ("TTFT", "ttft_ms", "rise", ttft_rise),
        ("memory", "peak_memory_mib", "rise", memory_rise),
    ]
    for label, key, direction, threshold in checks:
        before = old[key]
        after = new[key]
        if before is None or after is None:
            continue
        change = _pct_drop(before, after) if direction == "drop" else _pct_rise(before, after)
        if change > threshold:
            regressions.append(f"{label} regression {change:.2f}% exceeds {threshold:.2f}%")

    return GuardResult(ok=not regressions, regressions=regressions)
