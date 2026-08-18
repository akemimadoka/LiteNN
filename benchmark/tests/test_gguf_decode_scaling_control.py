import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_gguf_decode_scaling_control import positive_thread_counts, summarize_scaling_reports  # noqa: E402


def report(threads: int, llama_tps: float, litenn_tps: float, llama_cpu_ms: float, litenn_cpu_ms: float):
    def runtime_window(cpu_ms: float) -> dict[str, object]:
        return {
            "phase": "measured",
            "decodeWallMs": 1000.0,
            "decodeTokens": 10,
            "telemetry": {"processCPUTimeDeltaMs": cpu_ms},
        }

    return {
        "configuration": {
            "litenn_threads": threads,
            "llama_threads": threads,
            "process_cpu_set": [0, 1, 2, 3],
        },
        "summary": {
            "llama_cpp": {"median": llama_tps, "coefficient_of_variation_percent": 1.0},
            "litenn": {"median": litenn_tps, "coefficient_of_variation_percent": 2.0},
        },
        "pairs": [
            {
                "llama_cpp": {"in_process": {"windows": [runtime_window(llama_cpu_ms)]}},
                "litenn": {"in_process": {"windows": [runtime_window(litenn_cpu_ms)]}},
            }
        ],
        "gate": {"accepted": True},
    }


class GGUFDecodeScalingControlTest(unittest.TestCase):
    def test_parses_sorted_unique_thread_counts(self) -> None:
        self.assertEqual(positive_thread_counts("8,1,4,2"), [1, 2, 4, 8])
        with self.assertRaisesRegex(Exception, "unique"):
            positive_thread_counts("1,1")

    def test_summarizes_wall_and_cpu_scaling(self) -> None:
        summary = summarize_scaling_reports(
            [
                (1, report(1, 2.0, 1.8, 1200.0, 1400.0)),
                (2, report(2, 3.0, 3.2, 1800.0, 2000.0)),
            ]
        )
        rows = {(row["runtime"], row["threads"]): row for row in summary["runtime_rows"]}
        self.assertAlmostEqual(rows[("llama_cpp", 2)]["speedup_vs_baseline"], 1.5)
        self.assertAlmostEqual(rows[("llama_cpp", 2)]["parallel_efficiency_percent"], 75.0)
        self.assertAlmostEqual(rows[("litenn", 1)]["process_cpu_ms_per_token_median"], 140.0)
        paired = summary["paired_rows"][1]
        self.assertAlmostEqual(paired["litenn_vs_llama_wall_throughput_percent"], 100.0 / 15.0)
        self.assertAlmostEqual(paired["litenn_vs_llama_process_cpu_time_ratio"], 2000.0 / 1800.0)


if __name__ == "__main__":
    unittest.main()
