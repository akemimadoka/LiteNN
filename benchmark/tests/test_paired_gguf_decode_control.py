import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_paired_gguf_decode_control import (  # noqa: E402
    assess_window_host_stability,
    assess_window_process_affinity,
    affinity_domain_metrics,
    build_llama_in_process_command,
    build_litenn_command,
    cpu_set,
    load_in_process_decode_report,
    load_litenn_generated_token_ids,
    load_tokenizer_token_ids,
    normalize_completion_text,
    paired_power_policy_stable,
    parse_forced_replay_metrics,
    process_power_policy_stable,
    token_ids_identity,
    wait_for_host_admission,
)


class FixedTokenReplayTest(unittest.TestCase):
    def test_parses_shared_process_cpu_set(self) -> None:
        self.assertEqual(cpu_set("0-3,6,8-9"), [0, 1, 2, 3, 6, 8, 9])
        with self.assertRaisesRegex(Exception, "CPU set"):
            cpu_set("3-1")

    def test_validates_observed_window_process_affinity(self) -> None:
        report = {
            "windows": [
                {
                    "phase": "measured",
                    "telemetry": {"allowedCPUIntersection": [0, 1, 2, 3]},
                },
                {
                    "phase": "measured",
                    "telemetry": {"allowedCPUIntersection": [0, 1, 2, 3]},
                },
            ]
        }
        self.assertTrue(assess_window_process_affinity(report, [0, 1, 2, 3])["passed"])
        self.assertFalse(assess_window_process_affinity(report, [0, 1])["passed"])

    def test_waits_for_consecutive_quiet_host_samples(self) -> None:
        class Monitor:
            def __init__(self) -> None:
                self.values = iter([100.0, 55.0, 22.0, 21.0])
                self.closed = False

            def sample(self) -> dict[str, object]:
                return {
                    "monotonic_ns": 1,
                    "host_utility_percent_mean": next(self.values),
                    "weighted_actual_mhz": 5000.0,
                }

            def close(self) -> None:
                self.closed = True

        monitor = Monitor()
        with patch("run_paired_gguf_decode_control.create_frequency_monitor", return_value=monitor), patch(
            "run_paired_gguf_decode_control.time.sleep"
        ):
            result = wait_for_host_admission(25.0, 2, 1, 1.0, 0.01)
        self.assertTrue(result["passed"])
        self.assertEqual(result["accepted_streak"], 2)
        self.assertEqual(result["sample_count"], 4)
        self.assertTrue(monitor.closed)

    def test_host_admission_uses_requested_affinity_domain(self) -> None:
        class Monitor:
            def __init__(self) -> None:
                self.closed = False

            def sample(self) -> dict[str, object]:
                return {
                    "monotonic_ns": 1,
                    "host_utility_percent_mean": 90.0,
                    "processor_observations": [
                        {
                            "group": 0,
                            "processor": processor,
                            "actual_mhz": 5000.0,
                            "utility_percent": 10.0 if processor < 2 else 100.0,
                        }
                        for processor in range(4)
                    ],
                }

            def close(self) -> None:
                self.closed = True

        monitor = Monitor()
        with patch("run_paired_gguf_decode_control.create_frequency_monitor", return_value=monitor), patch(
            "run_paired_gguf_decode_control.time.sleep"
        ):
            result = wait_for_host_admission(25.0, 2, 0, 1.0, 0.01, [0, 1])
        self.assertTrue(result["passed"])
        self.assertEqual(result["activity_source"], "affinity_domain_utility_mean")

    def test_computes_affinity_domain_metrics(self) -> None:
        sample = {
            "processor_observations": [
                {"group": 0, "processor": 0, "actual_mhz": 4000.0, "utility_percent": 25.0},
                {"group": 0, "processor": 1, "actual_mhz": 5000.0, "utility_percent": 75.0},
                {"group": 0, "processor": 2, "actual_mhz": 3000.0, "utility_percent": 100.0},
            ]
        }
        self.assertEqual(affinity_domain_metrics(sample, [0, 1]), (50.0, 4750.0))
        self.assertIsNone(affinity_domain_metrics(sample, [0, 3]))

    def test_rejects_decode_window_host_activity_excursion(self) -> None:
        def window(index: int, utility: float, frequency: float) -> dict[str, object]:
            return {
                "phase": "measured",
                "index": index,
                "telemetry": {
                    "hostUtilityPercentMean": {"median": utility},
                    "hostLoad1m": {"median": None},
                    "weightedActualMHz": {"median": frequency},
                },
            }

        report = {"windows": [window(0, 20.0, 5000.0), window(1, 21.0, 4980.0), window(2, 60.0, 4900.0)]}
        stability = assess_window_host_stability(report, 2.0, 0.95)
        self.assertTrue(stability["available"])
        self.assertFalse(stability["activity_passed"])
        self.assertTrue(stability["frequency_passed"])
        self.assertFalse(stability["passed"])

    def test_window_stability_prefers_affinity_domain_metrics(self) -> None:
        def window(index: int, domain_utility: float) -> dict[str, object]:
            return {
                "phase": "measured",
                "index": index,
                "telemetry": {
                    "affinityDomainUtilityPercentMean": {"median": domain_utility},
                    "affinityDomainWeightedActualMHz": {"median": 5000.0},
                    "hostUtilityPercentMean": {"median": 100.0 if index == 2 else 10.0},
                    "hostLoad1m": {"median": None},
                    "weightedActualMHz": {"median": 3000.0 if index == 2 else 5000.0},
                },
            }

        stability = assess_window_host_stability(
            {"windows": [window(0, 20.0), window(1, 21.0), window(2, 22.0)]}, 2.0, 0.95
        )
        self.assertEqual(stability["activity_metric"], "affinityDomainUtilityPercentMean")
        self.assertTrue(stability["passed"])

    def test_loads_valid_tokenizer_manifest(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "tokens.json"
            path.write_text(
                json.dumps({"schema": "litenn.llamacpp_tokens.v1", "tokenIds": [4, 8, 15, 16, 23, 42]}),
                encoding="utf-8",
            )
            self.assertEqual(load_tokenizer_token_ids(path), [4, 8, 15, 16, 23, 42])

    def test_rejects_invalid_tokenizer_manifest(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "tokens.json"
            path.write_text(
                json.dumps({"schema": "litenn.llamacpp_tokens.v1", "tokenIds": [1, -2]}), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "invalid token ids"):
                load_tokenizer_token_ids(path)

    def test_parses_natural_sampler_divergence(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "decode.txt"
            path.write_text(
                "[1,2,3]\n[]\ngenerated_tokens=2 forced_replay=true "
                "forced_token_mismatch_count=1 first_forced_token_mismatch_index=1\n",
                encoding="utf-8",
            )
            self.assertEqual(
                parse_forced_replay_metrics(path),
                {"enabled": True, "natural_mismatch_count": 1, "first_natural_mismatch_index": 1},
            )

    def test_loads_generated_suffix_after_prompt(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "decode.txt"
            path.write_text("[10,11,20,21,22]\n[]\nmetrics\n", encoding="utf-8")
            self.assertEqual(load_litenn_generated_token_ids(path, 2), [20, 21, 22])

    def test_token_identity_is_order_sensitive(self) -> None:
        self.assertNotEqual(token_ids_identity([1, 2, 3])["sha256"], token_ids_identity([3, 2, 1])["sha256"])
        self.assertEqual(token_ids_identity([1, 2, 3])["count"], 3)

    def test_normalizes_windows_completion_output_to_model_bytes(self) -> None:
        self.assertEqual(normalize_completion_text("hello\r\nworld\r\n\r\n"), "hello\nworld")

    def test_rejects_in_process_power_policy_transition(self) -> None:
        stable = {
            "power_policy_before": {"source": "powercfg", "value": "high-performance"},
            "power_policy_after": {"source": "powercfg", "value": "high-performance"},
        }
        changed = {
            **stable,
            "power_policy_after": {"source": "powercfg", "value": "balanced"},
        }
        self.assertTrue(process_power_policy_stable(stable))
        self.assertFalse(process_power_policy_stable(changed))
        self.assertFalse(process_power_policy_stable({}))

    def test_rejects_cross_runtime_power_policy_mismatch(self) -> None:
        high_performance = {
            "power_policy_before": {"source": "powercfg", "value": "high-performance"},
            "power_policy_after": {"source": "powercfg", "value": "high-performance"},
        }
        balanced = {
            "power_policy_before": {"source": "powercfg", "value": "balanced"},
            "power_policy_after": {"source": "powercfg", "value": "balanced"},
        }
        self.assertTrue(paired_power_policy_stable(high_performance, high_performance))
        self.assertFalse(paired_power_policy_stable(high_performance, balanced))

    def test_litenn_command_preserves_explicit_cache_capacity(self) -> None:
        args = SimpleNamespace(
            python="python311",
            prompt="hello",
            predict=128,
            llvm_opt_level=0,
            litenn_threads=8,
            litenn_worker_wait="adaptive",
            litenn_max_cache_length=256,
            litenn_affinity="default",
            shared_weights_cache_dir=Path("shared-weights"),
        )
        command = build_litenn_command(
            args,
            Path("model.gguf"),
            Path("litenn_gguf_convert"),
            Path("litenn_llamacpp_adapter"),
            Path("workdir"),
            Path("cache"),
        )

        index = command.index("--max-cache-length")
        self.assertEqual(command[index + 1], "256")
        shared_index = command.index("--shared-weights-cache-dir")
        self.assertEqual(command[shared_index + 1], "shared-weights")

    def test_builds_unmeasured_cache_preparation_command(self) -> None:
        args = SimpleNamespace(
            python="python311",
            prompt="hello",
            predict=16,
            llvm_opt_level=0,
            litenn_threads=2,
            litenn_worker_wait="adaptive",
            litenn_max_cache_length=256,
            litenn_affinity="compact",
            shared_weights_cache_dir=None,
            in_process_warmup_windows=1,
            in_process_windows=3,
        )
        command = build_litenn_command(
            args,
            Path("model.gguf"),
            Path("litenn_gguf_convert"),
            Path("litenn_llamacpp_adapter"),
            Path("workdir"),
            Path("cache"),
            require_aot_cache_hit=False,
            compile_only=True,
        )
        self.assertIn("--compile-only", command)
        self.assertNotIn("--require-aot-cache-hit", command)
        self.assertNotIn("--benchmark-windows", command)

    def test_validates_in_process_window_report_from_raw_windows(self) -> None:
        windows = [
            {
                "phase": "warmup",
                "index": 0,
                "windowStartMonotonicNs": 1_000_000_000,
                "decodeStartMonotonicNs": 2_000_000_000,
                "decodeEndMonotonicNs": 2_008_000_000,
                "windowEndMonotonicNs": 2_008_000_000,
                "stateResetMs": 0.01,
                "prefillMs": 10.0,
                "decodeWallMs": 8.0,
                "moduleRunMs": 7.9,
                "decodeTokens": 4,
                "msPerToken": 2.0,
                "tokensPerSecond": 500.0,
            },
            {
                "phase": "measured",
                "index": 0,
                "windowStartMonotonicNs": 3_000_000_000,
                "decodeStartMonotonicNs": 4_000_000_000,
                "decodeEndMonotonicNs": 4_004_000_000,
                "windowEndMonotonicNs": 4_004_000_000,
                "stateResetMs": 0.01,
                "prefillMs": 2.0,
                "decodeWallMs": 4.0,
                "moduleRunMs": 3.9,
                "decodeTokens": 4,
                "msPerToken": 1.0,
                "tokensPerSecond": 1000.0,
            },
            {
                "phase": "measured",
                "index": 1,
                "windowStartMonotonicNs": 5_000_000_000,
                "decodeStartMonotonicNs": 6_000_000_000,
                "decodeEndMonotonicNs": 6_005_000_000,
                "windowEndMonotonicNs": 6_005_000_000,
                "stateResetMs": 0.01,
                "prefillMs": 2.0,
                "decodeWallMs": 5.0,
                "moduleRunMs": 4.9,
                "decodeTokens": 4,
                "msPerToken": 1.25,
                "tokensPerSecond": 800.0,
            },
        ]
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "windows.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "litenn.in_process_decode_windows.v2",
                        "producer": "LiteNN",
                        "warmupWindows": 1,
                        "measuredWindows": 2,
                        "promptTokens": 3,
                        "decodeTokensPerWindow": 4,
                        "windows": windows,
                        "summary": {
                            "tokensPerSecondMedian": 900.0,
                            "tokensPerSecondCVPercent": 15.713484026367723,
                        },
                    }
                ),
                encoding="utf-8",
            )
            report = load_in_process_decode_report(
                path,
                producer="LiteNN",
                warmup_windows=1,
                measured_windows=2,
                prompt_tokens=3,
                decode_tokens=4,
                frequency_samples=[
                    {
                        "monotonic_ns": 4_001_000_000,
                        "weighted_actual_mhz": 5100.0,
                        "host_utility_percent_mean": 25.0,
                        "active_logical_cpus": ["0,0", "0,1"],
                    },
                    {
                        "monotonic_ns": 6_001_000_000,
                        "weighted_actual_mhz": 5000.0,
                        "host_utility_percent_mean": 30.0,
                        "active_logical_cpus": ["0,1", "0,2"],
                    },
                ],
                resource_samples=[
                    {
                        "monotonic_ns": 4_000_500_000,
                        "cpu_user_ms": 10.0,
                        "cpu_system_ms": 2.0,
                        "rss_bytes": 100,
                        "private_bytes": 80,
                        "allowed_cpu_ids": [0, 1],
                    },
                    {
                        "monotonic_ns": 4_003_500_000,
                        "cpu_user_ms": 15.0,
                        "cpu_system_ms": 3.0,
                        "rss_bytes": 120,
                        "private_bytes": 90,
                        "allowed_cpu_ids": [0, 1],
                    },
                    {
                        "monotonic_ns": 6_000_500_000,
                        "cpu_user_ms": 20.0,
                        "cpu_system_ms": 4.0,
                        "rss_bytes": 130,
                        "private_bytes": 95,
                        "allowed_cpu_ids": [1, 2],
                    },
                    {
                        "monotonic_ns": 6_004_500_000,
                        "cpu_user_ms": 26.0,
                        "cpu_system_ms": 5.0,
                        "rss_bytes": 140,
                        "private_bytes": 100,
                        "allowed_cpu_ids": [1, 2],
                    },
                ],
            )
            self.assertEqual(report["validated_statistics"]["tokens_per_second"]["median"], 900.0)
            self.assertEqual(report["validated_statistics"]["temporal_drift"]["direction"], "decreasing")
            self.assertAlmostEqual(
                report["validated_statistics"]["temporal_drift"]["first_to_last_percent"], -20.0
            )
            self.assertTrue(report["telemetry"]["allMeasuredWindowsCovered"])
            self.assertEqual(report["windows"][1]["telemetry"]["rssBytes"]["maximum"], 120.0)
            self.assertAlmostEqual(report["windows"][1]["telemetry"]["processCPUTimeDeltaMs"], 6.0)

            document = json.loads(path.read_text(encoding="utf-8"))
            document["summary"]["tokensPerSecondMedian"] = 901.0
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "median"):
                load_in_process_decode_report(
                    path,
                    producer="LiteNN",
                    warmup_windows=1,
                    measured_windows=2,
                    prompt_tokens=3,
                    decode_tokens=4,
                )

    def test_builds_matched_in_process_commands(self) -> None:
        args = SimpleNamespace(
            python="python311",
            prompt="hello",
            predict=5,
            context_size=256,
            llvm_opt_level=0,
            litenn_threads=8,
            litenn_worker_wait="adaptive",
            litenn_max_cache_length=256,
            litenn_affinity="default",
            llama_threads=2,
            in_process_warmup_windows=1,
            in_process_windows=3,
        )
        prompt = [10, 11]
        generated = [20, 21, 22, 23, 24]
        litenn = build_litenn_command(
            args,
            Path("model.gguf"),
            Path("litenn_gguf_convert"),
            Path("litenn_llamacpp_adapter"),
            Path("workdir"),
            Path("cache"),
            generated,
            prompt,
            Path("litenn-windows.json"),
        )
        llama = build_llama_in_process_command(
            args,
            Path("model.gguf"),
            Path("litenn_llamacpp_adapter"),
            prompt,
            generated,
            Path("llama-windows.json"),
        )
        self.assertIn("10,11", litenn)
        self.assertNotIn("--llamacpp-tokenizer-tool", litenn)
        self.assertEqual(litenn[litenn.index("--benchmark-windows") + 1], "3")
        self.assertEqual(llama[1], "benchmark-fixed-decode")
        self.assertEqual(llama[3:5], ["10,11", "20,21,22,23,24"])
        self.assertEqual(llama[5:7], ["1", "3"])


if __name__ == "__main__":
    unittest.main()
