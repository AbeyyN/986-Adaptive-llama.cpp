# Benchmarking and selection

This document defines how 986 Adaptive measures candidate configurations and decides whether a result is acceptable.

## Measurement sources

### Prompt processing and text generation

PP and TG are taken directly from upstream `llama-bench` JSON output.

The harness runs a candidate with both a prompt-processing workload and a token-generation workload, then reads `avg_ts` from the corresponding records:

- PP: `n_prompt > 0`, `n_gen == 0`
- TG: `n_prompt == 0`, `n_gen > 0`

The default workload is 512 prompt tokens and 128 generated tokens with three repetitions. These values are configurable in code and can be expanded later into workload classes.

`llama-bench` explicitly excludes tokenization and sampling time. 986 reports PP/TG as engine benchmark throughput and does not present those numbers as end-to-end application latency.

### Time to first token

TTFT is measured independently because `llama-bench` does not expose end-to-end first-token latency.

When `--with-ttft` is enabled, the harness:

1. launches `llama-server` for one candidate on a temporary local port;
2. waits for `/health` to become ready;
3. sends an OpenAI-compatible streaming chat-completion request;
4. records elapsed wall time from request start until the first non-empty generated content chunk;
5. repeats the probe and uses the median;
6. terminates the server before testing the next candidate.

This measurement includes request handling, prompt processing, queueing inside the local server, sampling, and delivery of the first streamed token. It is therefore intentionally different from raw `llama-bench` throughput.

### Memory

The benchmark process is sampled while `llama-bench` is running.

Current adapters:

- Linux: `/proc/<pid>/status` RSS
- macOS: `ps` RSS
- Windows: `GetProcessMemoryInfo`
- NVIDIA: per-process `nvidia-smi` used memory when available

Peak RSS and peak NVIDIA VRAM are stored separately. A combined peak-memory value is used only as a relative pressure signal during candidate scoring. Unsupported GPU telemetry remains `null`; it is not estimated.

## Candidate search

P0 deliberately avoids a full combinatorial search. The bootstrap tune defines the center of a bounded candidate set. The harness then tests a small number of high-impact variations such as:

- lower and higher thread counts;
- smaller and larger batch/ubatch pairs;
- CPU fallback when accelerator offload is active;
- an unquantized KV-cache candidate when the bootstrap selected quantized KV.

Duplicate argument sets are removed and the candidate count is capped.

The intent is to make first-run tuning useful without turning model startup into an uncontrolled benchmark campaign. Later revisions can add model-specific search strategies if measured data shows that they are worthwhile.

## Scoring

Each candidate receives normalized scores for the metrics that were actually measured.

Higher is better:

- PP tokens/s
- TG tokens/s

Lower is better:

- TTFT
- peak memory

For a higher-is-better metric, the best candidate receives `1.0`; other candidates are divided by the best value. For a lower-is-better metric, the lowest value receives `1.0`; other candidates receive `best / candidate`.

The active profile supplies metric weights:

| Profile | PP | TG | TTFT | Memory |
| --- | ---: | ---: | ---: | ---: |
| balanced | 0.34 | 0.34 | 0.17 | 0.15 |
| latency | 0.20 | 0.35 | 0.35 | 0.10 |
| throughput | 0.50 | 0.38 | 0.04 | 0.08 |
| memory | 0.12 | 0.18 | 0.05 | 0.65 |
| thermal-safe | 0.20 | 0.25 | 0.10 | 0.45 |
| cluster | 0.42 | 0.38 | 0.10 | 0.10 |

If a metric is unavailable, its weight is removed and the remaining weights are renormalized for that candidate set. No synthetic TTFT or VRAM number is inserted.

The highest final score becomes the measured winner for that hardware/model/profile run.

## Fingerprints and persistence

### Hardware fingerprint

The hardware key is a SHA-256-derived identifier built from:

- operating system;
- architecture;
- logical CPU count;
- total system memory;
- detected backend candidate set.

Hostname, IP address, username and account identity are excluded.

### Model fingerprint

Hashing a multi-gigabyte GGUF on every startup is unnecessary for P0. The model key uses:

- file size;
- the first model region;
- the last model region.

This gives a stable content-derived identity at much lower startup cost than a full-file hash. It is not intended to be a cryptographic supply-chain verification mechanism.

### State

Profiles are written atomically beneath the platform state directory. `986_ADAPTIVE_HOME` can be used to relocate the entire state tree.

A persisted result includes the system profile, fingerprints, candidate arguments, metrics, scores, and selected winner. This provides enough information to reproduce and audit the decision without storing user-specific machine names.

## Performance guardian

The guardian compares the winner metrics of a current benchmark with a known-good baseline.

Default failure thresholds:

- PP drop greater than 7%
- TG drop greater than 7%
- TTFT increase greater than 10%
- peak-memory increase greater than 10%

Unavailable metrics are skipped. A detected regression returns exit code `1`; malformed input or an execution error returns `2`.

Thresholds are policy, not universal constants. They are CLI-configurable and should eventually be set per workload class once the project has enough benchmark history.

## Reproducibility notes

A performance result is only meaningful when the surrounding conditions are recorded. For serious comparisons, keep the following stable where possible:

- exact model/quantization;
- `llama.cpp` commit/build;
- backend build options;
- driver/runtime version;
- power mode;
- thermal state;
- background system load;
- NUMA policy;
- benchmark repetitions;
- prompt/generation lengths.

P0 persists the configuration and measurements needed by the adaptive selector. A later benchmark-report schema can add full environment capture for public result comparison.
