# 986 Adaptive llama.cpp — Roadmap

## P0 — Foundation

Status: **ACTIVE**

- [x] Portable host capability profiler
- [x] Generic policy profiles
- [x] Bootstrap AutoTune argument generation
- [x] User-override precedence
- [x] Cross-platform unit/smoke CI
- [x] Portability and upstream-sync architecture contract
- [ ] Persisted hardware/model fingerprint schema
- [ ] Benchmark harness integration with `llama-bench`
- [ ] Score PP/TG/TTFT/RAM/VRAM candidates
- [ ] Auto-select measured winner
- [ ] Regression/performance guardian

Exit criterion: AutoTune can detect a new machine, benchmark candidate configs, save the winning profile, and reproduce it.

## P1 — Adaptive single-node runtime

- [ ] Adaptive KV-cache governor
- [ ] Speculative decoding acceptance-rate governor
- [ ] Model residency and eviction manager
- [ ] Latency / throughput / balanced QoS policies
- [ ] Thermal and power telemetry adapters
- [ ] Safe runtime transition rules

Exit criterion: long-running `llama-server` can adapt to workload and resource pressure without owner-specific configuration.

## P2 — Adaptive cluster / RPC

- [ ] Node capability advertisement
- [ ] Network latency/bandwidth benchmark
- [ ] Compute + memory + network placement score
- [ ] Failure-aware node health states
- [ ] Re-placement/reload strategy after node loss
- [ ] Cluster benchmark history

Exit criterion: cluster mode selects placement from measured hardware and network characteristics rather than static node names or memory size alone.

## P3 — Deep engine optimization

Only after benchmark evidence identifies a real bottleneck:

- [ ] Scheduler/core hooks upstream cannot provide cleanly
- [ ] Active-slot shared-prefix KV experiments
- [ ] Per-op/backend scheduling experiments
- [ ] Backend-specific kernel optimization where justified

Exit criterion: each deep patch has reproducible performance gain, correctness tests, and an upstream rebase plan.

## Non-goals

- Branding-only changes to upstream internals
- Device-specific hacks in generic code
- Claiming heuristic settings are universally optimal
- Replacing mature upstream functionality merely to own a duplicate implementation
- Breaking OpenAI-compatible server behavior without a documented compatibility reason
