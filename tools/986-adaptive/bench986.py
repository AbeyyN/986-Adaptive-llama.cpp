#!/usr/bin/env python3
"""Command-line interface for measured 986 Adaptive tuning."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Sequence

from adaptive986 import detect_system
from adaptive_runtime import find_binary, parse_llama_bench_json
from adaptive_score import (
    benchmark,
    compare_performance,
    persisted_profile_for,
    score_results,
    winner_candidate,
)
from adaptive_state import (
    PROFILE_WEIGHTS,
    BenchmarkCandidate,
    BenchmarkMetrics,
    generate_candidates,
    hardware_fingerprint,
    load_json,
    model_fingerprint,
    save_json,
)


def _cmd_fingerprint(args: argparse.Namespace) -> int:
    system = detect_system()
    payload: Dict[str, object] = {
        "hardware_fingerprint": hardware_fingerprint(system),
        "system": asdict(system),
    }
    if args.model:
        payload["model_fingerprint"] = model_fingerprint(args.model)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    try:
        bench_binary = find_binary("llama-bench", args.llama_bench)
        server_binary = find_binary("llama-server", args.llama_server) if args.with_ttft else None
        payload, path = benchmark(
            model=args.model,
            profile=args.profile,
            llama_bench=bench_binary,
            llama_server=server_binary,
            repetitions=args.repetitions,
            ttft_repetitions=args.ttft_repetitions,
            candidate_limit=args.candidate_limit,
            save=not args.no_save,
        )
    except (ValueError, RuntimeError, OSError, urllib.error.URLError) as exc:
        print(f"986-adaptive benchmark: {exc}", file=sys.stderr)
        return 2

    if args.output:
        save_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if path:
        print(f"saved: {path}", file=sys.stderr)
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    try:
        payload, path = persisted_profile_for(args.model, profile=args.profile)
        candidate = winner_candidate(payload)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"986-adaptive replay: {exc}", file=sys.stderr)
        return 2

    if args.format == "args":
        print(" ".join(candidate.llama_args()))
    else:
        print(json.dumps({"path": str(path), "winner": payload.get("winner")}, indent=2, sort_keys=True))
    return 0


def _cmd_guard(args: argparse.Namespace) -> int:
    try:
        baseline = load_json(Path(args.baseline))
        current = load_json(Path(args.current))
        result = compare_performance(
            baseline,
            current,
            pp_drop=args.pp_drop,
            tg_drop=args.tg_drop,
            ttft_rise=args.ttft_rise,
            memory_rise=args.memory_rise,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"986-adaptive guard: {exc}", file=sys.stderr)
        return 2

    payload = asdict(result)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench986",
        description="Measured benchmark and performance guard for 986 Adaptive llama.cpp",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fingerprint = sub.add_parser("fingerprint", help="print stable hardware/model fingerprints")
    fingerprint.add_argument("--model", help="optional local GGUF file")
    fingerprint.set_defaults(func=_cmd_fingerprint)

    bench = sub.add_parser("benchmark", help="benchmark candidates and persist the measured winner")
    bench.add_argument("--model", required=True, help="local GGUF model path")
    bench.add_argument("--profile", choices=sorted(PROFILE_WEIGHTS), default="balanced")
    bench.add_argument("--llama-bench", help="path to llama-bench; auto-detected when omitted")
    bench.add_argument("--llama-server", help="path to llama-server; required only with --with-ttft")
    bench.add_argument(
        "--with-ttft",
        action="store_true",
        help="launch llama-server per candidate and measure streaming TTFT",
    )
    bench.add_argument("--repetitions", type=int, default=3)
    bench.add_argument("--ttft-repetitions", type=int, default=3)
    bench.add_argument("--candidate-limit", type=int, default=7)
    bench.add_argument("--no-save", action="store_true", help="do not persist the winning profile")
    bench.add_argument("--output", help="also write the full benchmark JSON to this path")
    bench.set_defaults(func=_cmd_benchmark)

    replay = sub.add_parser("replay", help="load the persisted winner for the current hardware/model")
    replay.add_argument("--model", required=True, help="local GGUF model path")
    replay.add_argument("--profile", choices=sorted(PROFILE_WEIGHTS), default="balanced")
    replay.add_argument("--format", choices=["json", "args"], default="args")
    replay.set_defaults(func=_cmd_replay)

    guard = sub.add_parser("guard", help="fail when a benchmark regresses beyond configured thresholds")
    guard.add_argument("--baseline", required=True)
    guard.add_argument("--current", required=True)
    guard.add_argument("--pp-drop", type=float, default=7.0)
    guard.add_argument("--tg-drop", type=float, default=7.0)
    guard.add_argument("--ttft-rise", type=float, default=10.0)
    guard.add_argument("--memory-rise", type=float, default=10.0)
    guard.set_defaults(func=_cmd_guard)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "repetitions", 1) < 1 or getattr(args, "ttft_repetitions", 1) < 1:
        parser.error("repetitions must be >= 1")
    if getattr(args, "candidate_limit", 1) < 1:
        parser.error("candidate limit must be >= 1")
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
