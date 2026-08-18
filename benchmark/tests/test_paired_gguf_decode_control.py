import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_paired_gguf_decode_control import (  # noqa: E402
    build_llama_in_process_command,
    build_litenn_command,
    load_in_process_decode_report,
    load_litenn_generated_token_ids,
    load_tokenizer_token_ids,
    normalize_completion_text,
    paired_power_policy_stable,
    parse_forced_replay_metrics,
    process_power_policy_stable,
    token_ids_identity,
)


class FixedTokenReplayTest(unittest.TestCase):
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

    def test_validates_in_process_window_report_from_raw_windows(self) -> None:
        windows = [
            {
                "phase": "warmup",
                "index": 0,
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
                        "schema": "litenn.in_process_decode_windows.v1",
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
            )
            self.assertEqual(report["validated_statistics"]["tokens_per_second"]["median"], 900.0)

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
