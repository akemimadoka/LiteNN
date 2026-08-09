import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark"))

from benchmark.run_llama_cpp_stage_control import (
    AGGREGATE_STAGES,
    normalize_pair,
    parse_output,
    position_bins,
    summarize_pairs,
    token_ids,
)


class LlamaCppStageControlTest(unittest.TestCase):
    def test_parses_exact_replay_token_ids(self) -> None:
        self.assertEqual(token_ids("151644,872,198"), [151644, 872, 198])
        self.assertEqual(position_bins("1-16,17-48,49-80"), [(1, 16), (17, 48), (49, 80)])

    def test_parses_complete_aggregate_profile(self) -> None:
        output = """\
mode=aggregate threads=8 warmup=9 steps=15 mean_decode_ms=155.500000 tokens_per_second=6.430868
stage=attention ms_per_token=34.250000 calls_per_token=49.000 percent_of_decode=22.026
stage=ffn.gate_up ms_per_token=67.250000 calls_per_token=48.000 percent_of_decode=43.248
stage=ffn.activation ms_per_token=0.750000 calls_per_token=48.000 percent_of_decode=0.482
stage=ffn.down ms_per_token=40.500000 calls_per_token=48.000 percent_of_decode=26.045
stage=logits ms_per_token=10.800000 calls_per_token=1.000 percent_of_decode=6.945
"""
        metrics = parse_output(output, "aggregate")

        self.assertEqual(set(metrics["stages"]), AGGREGATE_STAGES)
        self.assertEqual(metrics["stages"]["logits"]["calls_per_token"], 1.0)

    def test_normalizes_stages_and_reports_profile_coverage(self) -> None:
        baseline = {
            "metrics": {"mean_decode_ms": 150.0},
            "process": {"frequency": {"available": False}},
        }
        profile = {
            "metrics": {
                "mean_decode_ms": 153.0,
                "stages": {
                    "attention": {"ms_per_token": 30.0, "calls_per_token": 49.0},
                    "ffn.gate_up": {"ms_per_token": 65.0, "calls_per_token": 48.0},
                    "ffn.activation": {"ms_per_token": 1.0, "calls_per_token": 48.0},
                    "ffn.down": {"ms_per_token": 39.0, "calls_per_token": 48.0},
                    "logits": {"ms_per_token": 12.0, "calls_per_token": 1.0},
                },
            },
            "process": {"frequency": {"available": False}},
        }

        pair = normalize_pair(baseline, profile)

        self.assertAlmostEqual(pair["profile_overhead_percent"], 2.0)
        self.assertAlmostEqual(pair["stage_coverage_percent"], 100.0 * 147.0 / 153.0)
        summary = summarize_pairs("clang", "aggregate", [pair, pair, pair])
        self.assertAlmostEqual(summary["stage_coverage_percent"]["median"], 100.0 * 147.0 / 153.0)

    def test_parses_and_normalizes_position_binned_aggregate_steps(self) -> None:
        baseline = parse_output(
            """\
mode=baseline threads=2 warmup=0 steps=2 mean_decode_ms=15.000000 tokens_per_second=66.666667
decode_step=1 decode_ms=10.000000
decode_step=2 decode_ms=20.000000
""",
            "baseline",
        )
        profile = parse_output(
            """\
mode=aggregate threads=2 warmup=0 steps=2 mean_decode_ms=16.500000 tokens_per_second=60.606061
decode_step=1 decode_ms=11.000000
decode_step=2 decode_ms=22.000000
stage=attention ms_per_token=5.500000 calls_per_token=49.000 percent_of_decode=33.333
stage=ffn.gate_up ms_per_token=4.400000 calls_per_token=48.000 percent_of_decode=26.667
stage=ffn.activation ms_per_token=0.550000 calls_per_token=48.000 percent_of_decode=3.333
stage=ffn.down ms_per_token=3.300000 calls_per_token=48.000 percent_of_decode=20.000
stage=logits ms_per_token=2.200000 calls_per_token=1.000 percent_of_decode=13.333
stage_step=1 stage=attention stage_ms=5.000000 calls=49
stage_step=1 stage=ffn.gate_up stage_ms=4.000000 calls=48
stage_step=1 stage=ffn.activation stage_ms=0.500000 calls=48
stage_step=1 stage=ffn.down stage_ms=3.000000 calls=48
stage_step=1 stage=logits stage_ms=2.000000 calls=1
stage_step=2 stage=attention stage_ms=6.000000 calls=49
stage_step=2 stage=ffn.gate_up stage_ms=4.800000 calls=48
stage_step=2 stage=ffn.activation stage_ms=0.600000 calls=48
stage_step=2 stage=ffn.down stage_ms=3.600000 calls=48
stage_step=2 stage=logits stage_ms=2.400000 calls=1
""",
            "aggregate",
        )
        pair = normalize_pair(
            {"metrics": baseline, "process": {"frequency": {"available": False}}},
            {"metrics": profile, "process": {"frequency": {"available": False}}},
            [(1, 1), (2, 2)],
        )

        self.assertEqual(len(pair["position_bins"]), 2)
        self.assertAlmostEqual(
            pair["position_bins"][0]["normalized_stages"]["attention"]["ms_per_token"],
            5.0 * 10.0 / 11.0,
        )
        summary = summarize_pairs("clang", "aggregate", [pair, pair, pair])
        self.assertEqual(summary["position_bins"][1]["start"], 2)
        self.assertAlmostEqual(summary["position_bins"][1]["baseline_ms_per_token"]["median"], 20.0)

    def test_counter_patch_applies_to_pinned_llama_cpp(self) -> None:
        source = ROOT / "third_party" / "llama.cpp"
        patch = ROOT / "benchmark" / "llama_cpp_stage_profile" / "llama_cpp_cpu_stage_counters.patch"
        result = subprocess.run(
            ["git", "apply", "--check", str(patch)],
            cwd=source,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
