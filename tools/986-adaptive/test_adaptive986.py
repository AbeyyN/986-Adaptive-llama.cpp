#!/usr/bin/env python3
import importlib.util
import pathlib
import sys
import unittest

MODULE_PATH = pathlib.Path(__file__).with_name("adaptive986.py")
spec = importlib.util.spec_from_file_location("adaptive986", MODULE_PATH)
assert spec and spec.loader
adaptive986 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = adaptive986
spec.loader.exec_module(adaptive986)


class TuneTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
