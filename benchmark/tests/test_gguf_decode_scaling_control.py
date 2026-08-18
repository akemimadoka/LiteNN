import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_gguf_decode_scaling_control import (  # noqa: E402
    load_completed_child_report,
    option_value,
    positive_thread_counts,
    resume_identity,
    resume_input_facts,
    summarize_scaling_reports,
)


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

    def test_resume_identity_covers_all_invocation_inputs_without_exposing_them(self) -> None:
        model_argument = "private/model.gguf"
        identity = resume_identity([1, 2, 4, 8], "python311", ["--model", model_argument])
        self.assertEqual(len(identity), 64)
        self.assertNotIn(model_argument, identity)
        self.assertNotEqual(identity, resume_identity([1, 2, 4], "python311", ["--model", model_argument]))
        self.assertNotEqual(identity, resume_identity([1, 2, 4, 8], "python311", ["--model", "other.gguf"]))

    def test_fingerprints_resume_input_metadata_without_retaining_paths(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            model = Path(directory) / "private-model.gguf"
            model.write_bytes(b"gguf")
            arguments = ["--model", str(model), "--prompt=hello"]
            self.assertEqual(option_value(arguments, "--model"), str(model))
            self.assertEqual(option_value(arguments, "--prompt"), "hello")
            facts = resume_input_facts(arguments)
            self.assertEqual(facts["--model"]["size_bytes"], 4)
            self.assertNotIn(str(model), json.dumps(facts))
            before = resume_identity([1], "python311", arguments, facts)
            model.write_bytes(b"changed")
            after = resume_identity([1], "python311", arguments, resume_input_facts(arguments))
            self.assertNotEqual(before, after)

    def test_reuses_only_complete_matching_thread_reports(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "child.json"
            document = report(4, 5.0, 4.5, 100.0, 110.0)
            document["status"] = "complete"
            path.write_text(json.dumps(document), encoding="utf-8")
            self.assertIsNotNone(load_completed_child_report(path, 4))
            self.assertIsNone(load_completed_child_report(path, 8))

            document["status"] = "running"
            path.write_text(json.dumps(document), encoding="utf-8")
            self.assertIsNone(load_completed_child_report(path, 4))


if __name__ == "__main__":
    unittest.main()
