import json
import tempfile
import unittest
from pathlib import Path

from benchmark.profile_bundle import (
    GGUFDecodeAnalysis,
    GGUFDecodeStep,
    GGUFNodeEvent,
    classify_native_residual_node,
    write_gguf_decode_analysis,
)


class NativeResidualLedgerTest(unittest.TestCase):
    def test_classifies_high_level_residual_categories(self) -> None:
        self.assertEqual(classify_native_residual_node("CallNode"), "call_control")
        self.assertEqual(classify_native_residual_node("GroupedQuantizedMatMulNode"), "projection_wrapper")
        self.assertEqual(classify_native_residual_node("NormalizationNode"), "normalization")
        self.assertEqual(classify_native_residual_node("BinaryOpNode"), "elementwise")
        self.assertEqual(classify_native_residual_node("RoPENode"), "attention_position_state")
        self.assertEqual(classify_native_residual_node("ReshapeNode"), "view_data_movement")

    def test_closes_module_non_helper_with_unemitted_self_and_markers(self) -> None:
        analysis = GGUFDecodeAnalysis(
            steps=[
                GGUFDecodeStep(
                    step=1,
                    phase="generation",
                    step_ms=20.0,
                    generated_tokens=1,
                    tokens_per_second=50.0,
                    module_run_ms=18.0,
                    helper_profile_enabled=True,
                    helper_total_ms=10.0,
                    module_non_helper_ms=8.0,
                    node_self_total_ms=5.0,
                    node_instrumentation_ms=2.0,
                    module_unattributed_ms=1.0,
                )
            ],
            helpers=[],
            nodes=[
                GGUFNodeEvent(1, 0, 0, "CallNode", 1, 1, 3.0, 2.5, 0.0),
                GGUFNodeEvent(1, 0, 1, "NormalizationNode", 2, 1, 1.5, 1.0, 0.0),
            ],
            node_summaries=[],
        )
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            write_gguf_decode_analysis(Path(directory), analysis)
            summary = json.loads((Path(directory) / "gguf_decode_summary.json").read_text(encoding="utf-8"))

        generation = next(item for item in summary["native_residual_ledger"] if item["phase"] == "generation")
        categories = {item["category"]: item for item in generation["categories"]}
        self.assertEqual(categories["call_control"]["total_ms"], 2.5)
        self.assertEqual(categories["normalization"]["total_ms"], 1.0)
        self.assertEqual(categories["unemitted_node_self"]["total_ms"], 1.5)
        self.assertEqual(categories["node_instrumentation"]["total_ms"], 2.0)
        self.assertEqual(categories["module_unattributed"]["total_ms"], 1.0)
        self.assertAlmostEqual(generation["accounted_ms"], 8.0)
        self.assertAlmostEqual(generation["closure_error_ms"], 0.0)
        self.assertAlmostEqual(generation["instrumentation_percent_of_module"], 100.0 / 9.0)
        self.assertFalse(generation["gate"]["instrumentation_within_3_percent"])
        self.assertFalse(generation["gate"]["accepted"])


if __name__ == "__main__":
    unittest.main()
