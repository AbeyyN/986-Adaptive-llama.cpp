# 986 Adaptive llama.cpp

Self-tuning local LLM inference built on top of [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp).

[![986 Adaptive CI](https://github.com/AbeyyN/986-Adaptive-llama.cpp/actions/workflows/986-adaptive-ci.yml/badge.svg)](https://github.com/AbeyyN/986-Adaptive-llama.cpp/actions/workflows/986-adaptive-ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Upstream](https://img.shields.io/badge/upstream-ggml--org%2Fllama.cpp-555.svg)](https://github.com/ggml-org/llama.cpp)

986 Adaptive is an experimental, upstream-oriented fork of `llama.cpp`. The inference engine remains upstream `llama.cpp`; the 986 layer adds a portable control plane that detects the host, benchmarks candidate configurations, selects a measured winner, persists that result, and detects later performance regressions.

The project is not tied to a specific machine, hostname, network, GPU brand, or homelab. Hardware-specific decisions are derived from runtime capability and benchmark data.

## Status

**P0 — adaptive foundation: implemented; real-hardware validation is ongoing.**

Current P0 capabilities:

| Area | Implementation |
| --- | --- |
| Host detection | OS, architecture, logical CPU count, system memory, available backend candidates |
| Bootstrap tuning | `balanced`, `latency`, `throughput`, `memory`, `thermal-safe`, `cluster` |
| Hardware identity | Anonymous stable hardware fingerprint; no hostname or IP in the key |
| Model identity | Fast content fingerprint using model size plus first/last file regions |
| PP benchmark | Measured with upstream `llama-bench` |
| TG benchmark | Measured with upstream `llama-bench` |
| TTFT | End-to-end streaming probe against a per-candidate `llama-server` instance |
| Memory | Peak process RSS on Linux/macOS/Windows; NVIDIA process VRAM when available |
| Selection | Profile-weighted normalized score from measured results |
| Persistence | Winning result stored by hardware + model fingerprint and optimization profile |
| Performance guard | Threshold-based PP/TG/TTFT/memory regression check with CI-friendly exit codes |

The bootstrap tuner remains intentionally conservative. A heuristic configuration is only the starting point; measured benchmark data is authoritative whenever it exists.

## Architecture

```text
                 +----------------------+
                 |   Host / model input |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Capability detection |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Bootstrap candidates |
                 +----------+-----------+
                            |
              +-------------+-------------+
              |                           |
              v                           v
     +------------------+        +------------------+
     |   llama-bench    |        |   llama-server   |
     | PP / TG / memory |        | streaming TTFT   |
     +---------+--------+        +---------+--------+
               |                           |
               +-------------+-------------+
                             |
                             v
                  +-----------------------+
                  | Profile-aware scoring |
                  +-----------+-----------+
                              |
                              v
                  +-----------------------+
                  | Persist winner/profile|
                  +-----------+-----------+
                              |
                              v
                  +-----------------------+
                  | Performance guardian  |
                  +-----------------------+
```

Design rule:

```text
Detect -> Benchmark -> Decide -> Persist -> Revalidate
```

Not:

```text
Device name -> hard-coded settings
```

## Quick start

Build the fork using the normal upstream build process. The adaptive layer expects `llama-bench`; `llama-server` is only required when TTFT measurement is enabled.

```sh
cmake -B build
cmake --build build --config Release -j
```

Upstream build options, backends, platform notes, Docker images and deployment instructions remain documented in [`docs/build.md`](docs/build.md) and the [upstream project](https://github.com/ggml-org/llama.cpp).

### 1. Inspect the host

```sh
python tools/986-adaptive/adaptive986.py detect
```

### 2. Generate a bootstrap configuration

```sh
python tools/986-adaptive/adaptive986.py tune \
  --profile balanced \
  --model /path/to/model.gguf \
  --format args
```

Explicit values still take precedence:

```sh
python tools/986-adaptive/adaptive986.py tune \
  --profile balanced \
  --model /path/to/model.gguf \
  --context 16384 \
  --threads 8
```

### 3. Benchmark and select a measured winner

```sh
python tools/986-adaptive/bench986.py benchmark \
  --model /path/to/model.gguf \
  --profile balanced \
  --llama-bench build/bin/llama-bench
```

Add real streaming TTFT measurement:

```sh
python tools/986-adaptive/bench986.py benchmark \
  --model /path/to/model.gguf \
  --profile latency \
  --llama-bench build/bin/llama-bench \
  --llama-server build/bin/llama-server \
  --with-ttft
```

The harness generates a bounded search set around the bootstrap tune rather than brute-forcing every possible flag combination. Candidate selection is based on measured PP/TG, TTFT when enabled, and peak memory pressure.

## Persisted profiles

Each measured winner is stored under a hardware/model fingerprint key and kept separately by optimization profile.

Default locations:

- Linux/macOS: `~/.local/state/986-adaptive/`
- Windows: `%LOCALAPPDATA%\986-adaptive\`
- `XDG_STATE_HOME` is respected where present
- `986_ADAPTIVE_HOME` overrides the state root on any platform

Example layout:

```text
986-adaptive/
└── profiles/
    └── hw-<fingerprint>/
        └── model-<fingerprint>/
            ├── balanced.json
            ├── latency.json
            └── memory.json
```

The hardware fingerprint deliberately excludes hostname, IP address and account identity.

Load the stored winner later and emit the exact `llama-server` arguments:

```sh
python tools/986-adaptive/bench986.py replay \
  --model /path/to/model.gguf \
  --profile balanced \
  --format args
```

## Performance guardian

A benchmark result can be compared with a known-good baseline:

```sh
python tools/986-adaptive/bench986.py guard \
  --baseline baseline.json \
  --current current.json
```

Default regression thresholds:

| Metric | Default failure threshold |
| --- | ---: |
| PP throughput | > 7% drop |
| TG throughput | > 7% drop |
| TTFT | > 10% increase |
| Peak memory | > 10% increase |

Exit code `0` means the result is within thresholds. Exit code `1` means a regression was detected. This makes the guardian suitable for scheduled validation or CI jobs without requiring a separate service.

## Profiles

Profiles change the optimization objective, not the detected hardware.

| Profile | Primary objective |
| --- | --- |
| `balanced` | General interactive use without excessive memory pressure |
| `latency` | Lower response latency and TTFT |
| `throughput` | Higher sustained prompt/token throughput |
| `memory` | Lower memory footprint, accepting lower peak speed |
| `thermal-safe` | Conservative CPU/memory pressure for constrained hosts |
| `cluster` | Baseline policy for later RPC/network-aware placement |

When a metric is unavailable, its weight is removed and the remaining measured metrics are renormalized. Missing TTFT or GPU telemetry therefore does not fabricate a score.

## Benchmark methodology

`llama-bench` is used for prompt processing and token generation because it is maintained upstream and exposes machine-readable JSON. The 986 layer does not reinterpret its PP/TG timing as TTFT. TTFT is measured separately from an OpenAI-compatible streaming request against `llama-server`.

`llama-bench` does not include tokenization and sampling time in its benchmark numbers; this distinction is preserved in 986 reports. See [`docs/986-adaptive/BENCHMARKING.md`](docs/986-adaptive/BENCHMARKING.md) for the measurement model, scoring rules and reproducibility notes.

## Upstream compatibility

The fork is deliberately additive.

- Upstream inference code remains the source of truth for model execution and backend kernels.
- 986-specific policy code lives under `tools/986-adaptive/` and `docs/986-adaptive/` wherever practical.
- Device-specific magic values are not accepted in generic policy code.
- Deep scheduler/kernel changes are deferred until a reproducible benchmark demonstrates a bottleneck that cannot be solved cleanly outside the core.
- Upstream fixes and backend support should remain straightforward to merge.

This structure is intended to keep the delta reviewable and reduce rebase cost as `llama.cpp` evolves.

## Project layout

```text
tools/986-adaptive/
├── adaptive986.py        # host detection + bootstrap policy
├── adaptive_state.py     # fingerprints, state + candidate generation
├── adaptive_runtime.py   # llama-bench, TTFT + memory adapters
├── adaptive_score.py     # scoring, persistence + performance guard
├── bench986.py           # command-line entry point
└── test_adaptive986.py   # cross-platform unit tests

docs/986-adaptive/
├── ARCHITECTURE.md
├── BENCHMARKING.md
└── ROADMAP.md
```

## Validation

The dedicated `986 Adaptive CI` workflow runs on:

- Ubuntu
- Windows
- macOS

CI covers policy logic, fingerprint behavior, llama-bench JSON parsing, bounded candidate generation, profile-aware scoring and regression-guard behavior. Large-model performance validation is intentionally not performed on hosted CI runners; reproducible hardware benchmark reports are the next validation layer.

## Roadmap

P0 establishes measured single-host tuning and regression control. Planned work then moves to adaptive runtime policy, RPC/cluster placement, and only later to deeper engine changes where benchmark evidence justifies them.

See [`docs/986-adaptive/ROADMAP.md`](docs/986-adaptive/ROADMAP.md).

## Relationship to llama.cpp

986 Adaptive is based on [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp), which provides the inference engine, model support, quantization stack, server, benchmark tooling, and hardware backends used by this fork.

For upstream documentation, supported backends and normal `llama.cpp` usage, refer to:

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [`docs/build.md`](docs/build.md)
- [`tools/llama-bench/README.md`](tools/llama-bench/README.md)
- [`tools/server/README.md`](tools/server/README.md)

## License

MIT. This fork retains the upstream license and attribution.
