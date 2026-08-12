import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compare_sub_layer_checkpoints import aggregate_reports, markdown_report  # noqa: E402


def metric(target: float, median: float, maximum: float, z_score: float, above: bool) -> dict[str, object]:
    return {
        "target": target,
        "control_median": median,
        "control_maximum": maximum,
        "ratio_to_control_median": target / median,
        "modified_z_score": z_score,
        "above_control_maximum": above,
    }


class SubLayerCheckpointComparisonTest(unittest.TestCase):
    def test_aggregates_boundary_order_and_joint_outliers(self) -> None:
        reports = {
            "attention_norm": {
                "target_outlier_analysis": {
                    "control_generated_indices": [1, 2, 3],
                    "rows": [
                        {
                            "layer": 46,
                            "normalized_rms_error": metric(0.3, 0.2, 0.25, 3.0, True),
                            "cosine_distance": metric(0.1, 0.05, 0.08, 4.0, True),
                        }
                    ],
                }
            },
            "ffn_down": {
                "target_outlier_analysis": {
                    "control_generated_indices": [1, 2, 3],
                    "rows": [
                        {
                            "layer": 46,
                            "normalized_rms_error": metric(0.5, 0.2, 0.3, 8.0, True),
                            "cosine_distance": metric(0.2, 0.06, 0.1, 7.0, True),
                        }
                    ],
                }
            },
        }
        report = aggregate_reports(reports, 4)
        self.assertEqual(report["first_joint_outlier_boundary_by_layer"], {"46": "attention_norm"})
        self.assertEqual(
            report["ranked_coordinates_by_joint_modified_z"][0], {"boundary": "ffn_down", "layer": 46}
        )
        self.assertIn("ffn_down", markdown_report(report))


if __name__ == "__main__":
    unittest.main()
