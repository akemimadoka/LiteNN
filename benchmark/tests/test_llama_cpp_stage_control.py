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
    summarize_pairs,
    token_ids,
)


class LlamaCppStageControlTest(unittest.TestCase):
    def test_parses_exact_replay_token_ids(self) -> None:
        self.assertEqual(token_ids("151644,872,198"), [151644, 872, 198])

    def test_parses_complete_aggregate_profile(self) -> None:
        output = """\
mode=aggregate threads=8 warmup=9 steps=15 mean_decode_ms=155.500000 tokens_per_second=6.430868
stage=attention ms_per_token=34.250000 calls_per_token=49.000 percent_of_decode=22.026
stage=ffn.gate_up ms_per_token=67.250000 calls_per_token=48.000 percent_of_decode=43.248
stage=ffn.down ms_per_token=41.250000 calls_per_token=48.000 percent_of_decode=26.527
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
                    "ffn.down": {"ms_per_token": 40.0, "calls_per_token": 48.0},
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
