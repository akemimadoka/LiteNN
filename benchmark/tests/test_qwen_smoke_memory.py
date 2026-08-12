import json
import sys
import tempfile
import unittest
from pathlib import Path

from example.gguf.qwen_smoke import build_parser, run_step, write_profile_artifacts


class QwenSmokeMemoryTest(unittest.TestCase):
    def test_parses_selected_layer_checkpoints(self) -> None:
        args = build_parser().parse_args(
            [
                "--model",
                "fixture.gguf",
                "--token-ids",
                "1,2",
                "--stateful",
                "--paged-reference-decode",
                "--layer-checkpoint-dir",
                "checkpoints",
                "--layer-checkpoint-generated-indices",
                "23,7",
                "--sub-layer-checkpoint-blocks",
                "46,44,46",
            ]
        )
        self.assertEqual(args.layer_checkpoint_dir, Path("checkpoints"))
        self.assertEqual(args.layer_checkpoint_generated_indices, (23, 7))
        self.assertEqual(args.sub_layer_checkpoint_blocks, (46, 44, 46))

    def test_records_stage_labeled_memory_and_trace_counters(self) -> None:
        script = (
            "import sys,time; "
            "print('[LiteNN gguf] allocate fixture...', file=sys.stderr, flush=True); "
            "payload=bytearray(8*1024*1024); time.sleep(0.08); "
            "print('[LiteNN gguf] allocate fixture: ok 80.0 ms', file=sys.stderr, flush=True); "
            "time.sleep(0.03)"
        )
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            step = run_step(
                "memory_probe",
                [sys.executable, "-c", script],
                root,
                memory_sample_interval_ms=10,
            )
            self.assertEqual(step["returncode"], 0)
            memory_path = Path(step["memory"])
            document = json.loads(memory_path.read_text(encoding="utf-8"))
            self.assertGreaterEqual(document["sample_count"], 2)
            self.assertIsNotNone(document["peaks"]["rss_bytes"])
            self.assertTrue(any(sample["stage"] == "allocate fixture" for sample in document["samples"]))

            trace_path, waterfall_path = write_profile_artifacts(root, [step])
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertTrue(
                any(event.get("cat") == "litenn.process.memory" and event.get("ph") == "C" for event in trace["traceEvents"])
            )
            self.assertIn("peak rss_bytes", waterfall_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
