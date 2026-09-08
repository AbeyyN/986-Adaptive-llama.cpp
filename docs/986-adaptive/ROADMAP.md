# 986 Adaptive llama.cpp — Roadmap

## P0 — Measured adaptive foundation

Status: **IMPLEMENTED / VALIDATION**

- [x] Portable host capability profiler
- [x] Generic policy profiles
- [x] Bootstrap AutoTune argument generation
- [x] User-override precedence
- [x] Cross-platform unit/smoke CI
- [x] Portability and upstream-sync architecture contract
- [x] Persisted hardware/model fingerprint schema
- [x] `llama-bench` benchmark harness integration
- [x] Prompt-processing (PP) measurement
- [x] Text-generation (TG) measurement
- [x] Streaming `llama-server` TTFT probe
- [x] Peak RAM sampling on Linux/macOS/Windows
- [x] NVIDIA per-process VRAM sampling when available
- [x] Profile-aware measured candidate scoring
- [x] Automatic winner selection and profile persistence
- [x] Regression/performance guardian

Exit criterion: the implementation can detect a new machine, generate a bounded candidate set, benchmark the candidates, persist the measured winner, reproduce the selected arguments, and reject a later regression. The remaining P0 work is real-hardware validation across representative CPU, integrated-GPU and discrete-GPU systems rather than additional foundation code.

## P1 — Adaptive single-node runtime

- [ ] Reuse persisted winner automatically at launch
- [ ] Staleness policy for driver/build/model changes
- [ ] Adaptive KV-cache governor
- [ ] Speculative-decoding acceptance-rate governor
- [ ] Model residency and eviction manager
- [ ] Latency / throughput / balanced QoS runtime policies
- [ ] Thermal and power telemetry adapters
- [ ] Safe runtime transition rules
- [ ] Benchmark-history compaction and retention policy

Exit criterion: a long-running `llama-server` can adapt to workload and resource pressure without owner-specific configuration while retaining deterministic fallback behavior.

## P2 — Adaptive cluster / RPC

- [ ] Node capability advertisement
- [ ] Network latency/bandwidth benchmark
- [ ] Compute + memory + network placement score
- [ ] Failure-aware node health states
- [ ] Re-placement/reload strategy after node loss
- [ ] Cluster benchmark history
- [ ] Heterogeneous-node scoring and exclusion rules

Exit criterion: cluster mode selects placement from measured hardware and network characteristics rather than static node names or memory size alone.

## P3 — Deep engine optimization

Only after benchmark evidence identifies a bottleneck that cannot be addressed cleanly by existing upstream interfaces:

- [ ] Scheduler/core hooks upstream cannot provide cleanly
- [ ] Active-slot shared-prefix KV experiments
- [ ] Per-op/backend scheduling experiments
- [ ] Backend-specific kernel optimization where justified

Exit criterion: every deep patch has a reproducible performance gain, correctness coverage, and a documented upstream rebase strategy.

## Non-goals

- Branding-only changes to upstream internals
- Device-specific hacks in generic code
- Claiming heuristic settings are universally optimal
- Inventing missing benchmark metrics
- Replacing mature upstream functionality merely to own a duplicate implementation
- Breaking OpenAI-compatible server behavior without a documented compatibility reason
