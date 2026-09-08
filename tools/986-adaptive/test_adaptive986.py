#!/usr/bin/env python3
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest

MODULE_DIR = pathlib.Path(__file__).parent

adaptive_spec = importlib.util.spec_from_file_location("adaptive986", MODULE_DIR / "adaptive986.py")
assert adaptive_spec and adaptive_spec.loader
adaptive986 = importlib.util.module_from_spec(adaptive_spec)
sys.modules[adaptive_spec.name] = adaptive986
adaptive_spec.loader.exec_module(adaptive986)

bench_spec = importlib.util.spec_from_file_location("bench986", MODULE_DIR / "bench986.py")
assert bench_spec and bench_spec.loader
bench986 = importlib.util.module_from_spec(bench_spec)
sys.modules[bench_spec.name] = bench986
bench_spec.loader.exec_module(bench986)


class AdaptiveTuneTests(unittest.TestCase):
    def system(self, memory=16.0, cpus=8, backends=None):
        backends = backends or ["cpu"]
        return adaptive986.SystemProfile(
            os="TestOS",
            arch="x86_64",
            logical_cpus=cpus,
            memory_gib=memory,
            backend_candidates=backends,
            preferred_backend=adaptive986.preferred_backend(backends),
        )

    def test_balanced_cpu_is_generic(self):
        result = adaptive986.tune(self.system(), profile="balanced", model_gib=6.0)
        self.assertEqual(result.backend, "cpu")
        self.assertIsNone(result.gpu_layers)
        self.assertEqual(result.threads, 6)
        self.assertIn("--ctx-size", result.llama_args())

    def test_accelerator_candidate_enables_offload(self):
        result = adaptive986.tune(self.system(backends=["cuda", "cpu"]), model_gib=4.0)
        self.assertEqual(result.backend, "cuda")
        self.assertEqual(result.gpu_layers, 999)

    def test_user_overrides_win(self):
        result = adaptive986.tune(
            self.system(),
            backend_override="custom-backend",
            threads_override=3,
            requested_context=12288,
        )
        self.assertEqual(result.backend, "custom-backend")
        self.assertEqual(result.threads, 3)
        self.assertEqual(result.ctx_size, 12288)

    def test_profiles_change_resource_policy(self):
        system = self.system(memory=32.0, cpus=16)
        thermal = adaptive986.tune(system, profile="thermal-safe")
        throughput = adaptive986.tune(system, profile="throughput")
        self.assertLess(thermal.threads, throughput.threads)
        self.assertLess(thermal.batch_size, throughput.batch_size)
        self.assertLess(thermal.memory_budget_gib, throughput.memory_budget_gib)

    def test_invalid_context_rejected(self):
        with self.assertRaises(ValueError):
            adaptive986.tune(self.system(), requested_context=128)


class BenchmarkTests(unittest.TestCase):
    def system(self, memory=16.0, cpus=8, backends=None):
        backends = backends or ["cpu"]
        return adaptive986.SystemProfile(
            os="TestOS",
            arch="x86_64",
            logical_cpus=cpus,
            memory_gib=memory,
            backend_candidates=backends,
            preferred_backend=adaptive986.preferred_backend(backends),
        )

    def candidate(self, name="a", threads=6, batch=1024):
        return bench986.BenchmarkCandidate(
            id=name,
            backend="cpu",
            threads=threads,
            ctx_size=8192,
            batch_size=batch,
            ubatch_size=min(256, batch),
            gpu_layers=None,
            flash_attn="auto",
            cache_type_k="q8_0",
            cache_type_v="q5_1",
        )

    def test_hardware_fingerprint_is_stable_and_anonymous(self):
        system = self.system(memory=32.0, cpus=16, backends=["vulkan", "cpu"])
        first = bench986.hardware_fingerprint(system)
        second = bench986.hardware_fingerprint(system)
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("hw-"))
        self.assertNotIn("TestOS", first)

    def test_model_fingerprint_changes_with_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "model.gguf"
            path.write_bytes(b"GGUF" + b"a" * 4096)
            first = bench986.model_fingerprint(str(path), chunk_bytes=128)
            second = bench986.model_fingerprint(str(path), chunk_bytes=128)
            self.assertEqual(first, second)
            path.write_bytes(b"GGUF" + b"b" * 4096)
            self.assertNotEqual(first, bench986.model_fingerprint(str(path), chunk_bytes=128))

    def test_llama_bench_json_extracts_pp_and_tg(self):
        payload = json.dumps(
            [
                {"n_prompt": 512, "n_gen": 0, "avg_ts": 1000.0},
                {"n_prompt": 0, "n_gen": 128, "avg_ts": 50.0},
            ]
        )
        pp, tg = bench986.parse_llama_bench_json(payload)
        self.assertEqual(pp, 1000.0)
        self.assertEqual(tg, 50.0)

    def test_candidate_generation_is_bounded_and_variable(self):
        system = self.system(memory=32.0, cpus=16)
        bootstrap = adaptive986.tune(system, profile="balanced", model_gib=5.0)
        candidates = bench986.generate_candidates(system, bootstrap, limit=5)
        self.assertGreaterEqual(len(candidates), 2)
        self.assertLessEqual(len(candidates), 5)
        self.assertEqual(candidates[0].id, "base")
        self.assertTrue(any(item.threads != bootstrap.threads for item in candidates[1:]))

    def test_score_selects_measured_winner(self):
        fast = self.candidate("fast")
        slow = self.candidate("slow", threads=4, batch=512)
        raw = [
            (fast, bench986.BenchmarkMetrics(1200.0, 60.0, 100.0, 3000.0, None)),
            (slow, bench986.BenchmarkMetrics(800.0, 40.0, 150.0, 3200.0, None)),
        ]
        scored = bench986.score_results("balanced", raw)
        self.assertEqual(scored[0].candidate.id, "fast")
        self.assertGreater(scored[0].score, scored[1].score)

    def test_memory_profile_can_prefer_lower_memory_candidate(self):
        fast = self.candidate("fast")
        lean = self.candidate("lean", threads=4, batch=512)
        raw = [
            (fast, bench986.BenchmarkMetrics(1100.0, 55.0, None, 7000.0, None)),
            (lean, bench986.BenchmarkMetrics(900.0, 48.0, None, 3000.0, None)),
        ]
        scored = bench986.score_results("memory", raw)
        self.assertEqual(scored[0].candidate.id, "lean")

    def test_winner_candidate_replays_llama_args(self):
        candidate = self.candidate("winner")
        payload = {"winner": {"candidate": dict(candidate.__dict__)}}
        restored = bench986.winner_candidate(payload)
        self.assertEqual(restored.id, "winner")
        self.assertIn("--ctx-size", restored.llama_args())

    def test_guard_detects_regression(self):
        def payload(pp, tg, ttft, rss):
            return {
                "winner": {
                    "metrics": {
                        "pp_ts": pp,
                        "tg_ts": tg,
                        "ttft_ms": ttft,
                        "peak_rss_mib": rss,
                        "peak_vram_mib": None,
                    }
                }
            }

        baseline = payload(1000.0, 50.0, 100.0, 3000.0)
        current = payload(850.0, 49.0, 130.0, 3100.0)
        result = bench986.compare_performance(baseline, current)
        self.assertFalse(result.ok)
        self.assertTrue(any("PP" in item for item in result.regressions))
        self.assertTrue(any("TTFT" in item for item in result.regressions))


if __name__ == "__main__":
    unittest.main()
