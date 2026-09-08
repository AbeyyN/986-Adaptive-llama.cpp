# 986 Adaptive llama.cpp

An upstream-compatible experimental fork of `ggml-org/llama.cpp` focused on **self-tuning local LLM inference**.

The project adds a portable adaptive control plane around llama.cpp. It is intended for any user and any supported host; it is not tied to a specific homelab or machine.

## Current status

**P0 / Foundation**

Implemented now:

- dependency-free hardware capability detection;
- generic tuning profiles: balanced, latency, throughput, memory, thermal-safe, cluster;
- bootstrap generation of llama.cpp runtime arguments;
- explicit user-override precedence;
- Linux / Windows / macOS CI tests;
- architecture contract designed to keep upstream rebases manageable.

The current tuner is deliberately labelled **bootstrap heuristic**. The next P0 milestone connects it to `llama-bench` so measured PP/TG/TTFT/memory results replace heuristics.

## Try it

Detect the current host:

```bash
python tools/986-adaptive/adaptive986.py detect
```

Generate a balanced candidate configuration when the GGUF model is local:

```bash
python tools/986-adaptive/adaptive986.py tune \
  --profile balanced \
  --model /path/to/model.gguf \
  --format args
```

Or supply only the expected model size:

```bash
python tools/986-adaptive/adaptive986.py tune \
  --profile memory \
  --model-gib 8.5
```

Explicit settings override AutoTune:

```bash
python tools/986-adaptive/adaptive986.py tune \
  --profile balanced \
  --context 16384 \
  --threads 8 \
  --backend vulkan
```

## Design principle

```text
Detect -> Benchmark -> Decide -> Observe -> Adapt
```

Never:

```text
Device name -> hard-coded magic values
```

## Documentation

- `docs/986-adaptive/ARCHITECTURE.md`
- `docs/986-adaptive/ROADMAP.md`

## Upstream

Core inference remains based on `ggml-org/llama.cpp`. 986-specific changes should stay additive whenever possible so upstream fixes and new hardware/model support remain easy to merge.

## License

This fork retains the upstream MIT license and upstream attribution.
