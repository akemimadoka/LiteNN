import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark"))

from benchmark.run_litenn_position_stage_control import (
    load_token_ids_file,
    normalize_pair,
    parse_run_logs,
    qwen_stage_shape_passes,
    stage_for_helper,
    stage_variance_passes,
    summarize_pairs,
)
from benchmark.profile_bundle import GGUFHelperEvent


def helper(operator: str, role: str, name: str = "helper") -> GGUFHelperEvent:
    return GGUFHelperEvent(1, name, "", operator, role, 1, 1.0, 1.0)


class LiteNNPositionStageControlTest(unittest.TestCase):
    def test_loads_token_ids_from_supported_json_shapes(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            plain = root / "plain.json"
            plain.write_text("[1, 2, 3]\n", encoding="utf-8")
            tokenizer = root / "tokenizer.json"
            tokenizer.write_text(json.dumps({"tokenIds": [4, 5]}), encoding="utf-8")
            generated = root / "generated.json"
            generated.write_text(json.dumps({"generatedTokenIds": [6, 7]}), encoding="utf-8")
            self.assertEqual(load_token_ids_file(plain), [1, 2, 3])
            self.assertEqual(load_token_ids_file(tokenizer), [4, 5])
            self.assertEqual(load_token_ids_file(generated), [6, 7])

    def test_maps_qwen_helpers_without_hiding_fused_swiglu_down(self) -> None:
        self.assertEqual(stage_for_helper(helper("projection", "qkv_grouped")), "attention.qkv")
        self.assertEqual(stage_for_helper(helper("attention", "active_prefix")), "attention.core")
        self.assertEqual(stage_for_helper(helper("activation", "swiglu")), "ffn.activation")
        self.assertEqual(stage_for_helper(helper("projection", "ffn_down")), "ffn.down")
        self.assertEqual(
            stage_for_helper(
                helper(
                    "projection",
                    "ffn_down",
                    "litenn_cpu_swiglu_ggml_block_matmul_field_interleaved_v4_q8k_f32",
                )
            ),
            "ffn.swiglu_down_fused",
        )
        self.assertEqual(
            stage_for_helper(
                helper(
                    "projection",
                    "ffn_down",
                    "litenn_cpu_swiglu_bounded_ggml_block_matmul_field_interleaved_v4_q8k_f32",
                )
            ),
            "ffn.swiglu_down_fused",
        )

    def test_parses_generation_steps_and_helper_stages(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            stdout = root / "stdout.txt"
            stderr = root / "stderr.txt"
            stdout.write_text(
                "stream stats step=1 phase=prompt_replay step_ms=2 module_run_ms=1 "
                "helper_profile_enabled=true helper_total_ms=0.5 module_non_helper_ms=0.5 "
                "generated_tokens=0 generated_tokens_per_second=0 sampling_ms=0\n"
                "stream stats step=2 phase=generation step_ms=12 module_run_ms=10 "
                "helper_profile_enabled=true helper_total_ms=8 module_non_helper_ms=2 "
                "generated_tokens=1 generated_tokens_per_second=83 sampling_ms=0.1\n",
                encoding="utf-8",
            )
            stderr.write_text(
                '[LiteNN gguf] decode step 2 helper litenn_cpu_swiglu_f32 '
                'detail="gate=1x4 up=1x4 out=1x4" calls=48 total_ms=8 avg_ms=0.166667\n',
                encoding="utf-8",
            )
            parsed = parse_run_logs(stdout, stderr, 1, True)
            self.assertEqual(len(parsed["steps"]), 1)
            stages = parsed["steps"][0]["stages"]
            self.assertEqual(stages["ffn.activation"]["calls"], 48)
            self.assertEqual(stages["module.residual"]["ms"], 2.0)

    def test_normalizes_bins_and_summarizes_growth(self) -> None:
        stage_shape = {
            "embedding": {"ms": 1.0, "calls": 1},
            "attention.qkv": {"ms": 1.0, "calls": 48},
            "attention.rope": {"ms": 1.0, "calls": 2304},
            "attention.kv_append": {"ms": 1.0, "calls": 96},
            "attention.core": {"ms": 1.0, "calls": 48},
            "attention.output": {"ms": 1.0, "calls": 48},
            "ffn.gate_up": {"ms": 1.0, "calls": 48},
            "ffn.swiglu_down_fused": {"ms": 1.0, "calls": 48},
            "logits": {"ms": 1.0, "calls": 1},
            "module.residual": {"ms": 1.0, "calls": 0},
        }

        def run(profile: bool) -> dict[str, object]:
            return {
                "metrics": {
                    "steps": [
                        {
                            "step_ms": value + 2.0,
                            "module_ms": value,
                            "stages": (
                                {
                                    name: {**stage, "ms": float(stage["ms"]) * value / 10.0}
                                    for name, stage in stage_shape.items()
                                }
                                if profile
                                else {}
                            ),
                        }
                        for value in (10.0, 12.0)
                    ]
                }
            }

        pair = normalize_pair(run(False), run(True), [(1, 1), (2, 2)])
        summary = summarize_pairs([pair, pair, pair])
        self.assertAlmostEqual(pair["stage_coverage_percent"], 100.0)
        self.assertAlmostEqual(summary["first_to_last_growth"]["profile_module_ms"]["median"], 2.0)
        self.assertTrue(qwen_stage_shape_passes(summary))

    def test_absolute_variance_gate_only_exempts_tiny_stages(self) -> None:
        tiny = {"mean": 0.2, "coefficient_of_variation_percent": 20.0, "standard_deviation": 0.02}
        noisy = {"mean": 0.2, "coefficient_of_variation_percent": 20.0, "standard_deviation": 1.0}
        large = {"mean": 2.0, "coefficient_of_variation_percent": 20.0, "standard_deviation": 0.02}
        self.assertTrue(stage_variance_passes(tiny, 15.0, 0.05))
        self.assertFalse(stage_variance_passes(noisy, 15.0, 0.05))
        self.assertFalse(stage_variance_passes(large, 15.0, 0.05))


if __name__ == "__main__":
    unittest.main()
