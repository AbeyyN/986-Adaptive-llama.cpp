# 986 Adaptive llama.cpp — Architecture

## Purpose

986 Adaptive llama.cpp is an upstream-compatible fork of `ggml-org/llama.cpp` that adds a self-tuning runtime/control plane around the inference engine.

The project is **not tied to one owner, one homelab, one CPU, one GPU, or one network**. Hardware-specific behavior must be discovered at runtime or supplied through explicit configuration.

## Portability contract

The following are prohibited in the generic engine/control plane:

- hard-coded private IP addresses, hostnames or DNS names;
- hard-coded user names or home-lab node names;
- assumptions about a specific CPU/GPU model;
- owner-specific filesystem paths;
- fixed thermal limits presented as universally safe;
- silently overriding explicit user configuration.

The following rules are mandatory:

1. **Detect first.** Runtime discovery produces a capability profile.
2. **Benchmark second.** Measured PP/TG/TTFT/memory data supersedes heuristics.
3. **User override wins.** Explicit flags/configuration always take precedence over AutoTune.
4. **Fail safe.** If confidence is low, prefer conservative settings and expose the reason.
5. **Upstream first.** Prefer additive modules and narrow integration points over invasive rewrites.
6. **Reproducible decisions.** AutoTune results must be serializable and explainable.

## Layer model

```text
Applications / OpenAI-compatible clients
                |
        llama-server / llama cli
                |
        986 Adaptive policy layer
        |       |       |       |
    profiler  tuner  governor  router
        |       |       |       |
        +-------+-------+-------+
                |
        upstream llama.cpp
                |
       ggml backend scheduler
                |
 CPU / CUDA / HIP / Metal / SYCL / Vulkan / RPC
```

### Layer 0 — Upstream core

Keep upstream llama.cpp and ggml as close to upstream as practical. Kernel-level changes require benchmark evidence and regression tests.

### Layer 1 — Capability profiler

Collect portable facts such as:

- OS and architecture;
- logical/physical CPU topology where available;
- RAM and available memory;
- accelerator/backend candidates;
- VRAM/unified-memory topology where available;
- runtime/backend versions;
- thermal/power telemetry where supported;
- RPC node capabilities and link measurements for cluster mode.

### Layer 2 — Bootstrap tuner

Generate a safe initial configuration before benchmark history exists. Current implementation lives in:

`tools/986-adaptive/adaptive986.py`

Profiles:

- `balanced`
- `latency`
- `throughput`
- `memory`
- `thermal-safe`
- `cluster`

Bootstrap output is a candidate only. It must not be described as the optimal configuration until measured.

### Layer 3 — Benchmark tuner

Benchmark candidate configurations using reproducible workloads and record:

- prompt processing throughput (PP);
- token generation throughput (TG);
- time to first token (TTFT);
- peak RAM/VRAM;
- power/temperature when available;
- stability/error status.

The benchmark tuner chooses settings from measurements rather than static device tables.

### Layer 4 — Runtime governors

Planned governors:

- adaptive KV-cache policy;
- thermal/power governor;
- model residency/eviction governor;
- request QoS scheduler;
- speculative decoding governor;
- cluster/RPC placement governor.

A governor may modify only parameters declared runtime-safe. Anything requiring model reload must be scheduled as a controlled transition.

## Configuration precedence

Highest precedence first:

```text
CLI explicit override
    > environment/config explicit override
    > saved benchmark profile
    > detected capability policy
    > conservative defaults
```

AutoTune must never silently defeat an explicit user choice.

## Hardware profiles

Hardware profiles are data, not code branches. A profile should contain facts and measurements, for example:

```json
{
  "schema": 1,
  "host_fingerprint": "...",
  "backend": "vulkan",
  "model_fingerprint": "...",
  "measurements": {
    "pp_tokens_s": 0.0,
    "tg_tokens_s": 0.0,
    "ttft_ms": 0.0,
    "peak_ram_gib": 0.0
  }
}
```

No public profile should contain private hostnames, IP addresses, credentials or identifying local paths.

## Upstream sync strategy

- Track `ggml-org/llama.cpp` as upstream.
- Keep 986-specific files under clearly named paths where practical.
- Avoid renaming upstream symbols solely for branding.
- Prefer wrappers/hooks over broad rewrites.
- Every core patch must document the upstream touch points.
- Maintain a benchmark gate for performance-sensitive patches.

## Success criteria

The fork is successful when a new user can run one high-level command, receive a reasonable configuration without knowing dozens of llama.cpp flags, and later improve that configuration automatically through measured benchmark data — while retaining full manual control.
