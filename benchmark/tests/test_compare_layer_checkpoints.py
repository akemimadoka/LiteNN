import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compare_layer_checkpoints import compare_manifests, markdown_report  # noqa: E402


def write_fixture(root: Path, values: list[list[float]]) -> Path:
    payload = root / "generated-000023.bin"
    offsets: list[int] = []
    with payload.open("wb") as stream:
        for layer in values:
            offsets.append(stream.tell())
            stream.write(struct.pack(f"<{len(layer)}f", *layer))
    manifest = root / "manifest.tsv"
    lines = [
        "# litenn-layer-checkpoints-v1",
        "generated_index\tabsolute_step\tposition\tinput_token_id\tfile\tlayer\tname\tdtype\tshape\t"
        "byte_offset\tbyte_size\tminimum\tmaximum\tmean\trms\tnon_finite\tchecksum_fnv1a64",
    ]
    for layer, row in enumerate(values):
        lines.append(
            f"23\t32\t31\t7\t{payload.name}\t{layer}\tlayer_hidden_{layer}\tFloat32\t1x{len(row)}\t"
            f"{offsets[layer]}\t{len(row) * 4}\t0\t0\t0\t0\t0\t0"
        )
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def write_neighborhood_fixture(root: Path, values: dict[int, list[list[float]]]) -> Path:
    lines = [
        "# litenn-layer-checkpoints-v1",
        "generated_index\tabsolute_step\tposition\tinput_token_id\tfile\tlayer\tname\tdtype\tshape\t"
        "byte_offset\tbyte_size\tminimum\tmaximum\tmean\trms\tnon_finite\tchecksum_fnv1a64",
    ]
    for generated_index, layers in sorted(values.items()):
        payload = root / f"generated-{generated_index:06d}.bin"
        with payload.open("wb") as stream:
            for layer, row in enumerate(layers):
                offset = stream.tell()
                stream.write(struct.pack(f"<{len(row)}f", *row))
                lines.append(
                    f"{generated_index}\t{generated_index + 9}\t{generated_index + 8}\t7\t{payload.name}\t"
                    f"{layer}\tlayer_hidden_{layer}\tFloat32\t1x{len(row)}\t{offset}\t{len(row) * 4}\t"
                    "0\t0\t0\t0\t0\t0"
                )
    manifest = root / "manifest.tsv"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


class LayerCheckpointComparisonTest(unittest.TestCase):
    def test_reports_first_failing_layer_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            reference = root / "reference"
            candidate = root / "candidate"
            reference.mkdir()
            candidate.mkdir()
            reference_manifest = write_fixture(reference, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            candidate_manifest = write_fixture(candidate, [[1.0, 2.0], [3.0, 4.25], [5.0, 6.5]])

            report = compare_manifests(reference_manifest, candidate_manifest, 1.0e-4, 1.0e-4)
            self.assertFalse(report["passed"])
            self.assertEqual(report["failing_rows"], 2)
            self.assertEqual(report["first_failing_layer_by_generated_index"], {"23": 1})
            self.assertAlmostEqual(report["rows"][1]["max_absolute_error"], 0.25)
            self.assertAlmostEqual(report["rows"][1]["reference_rms"], math.sqrt(12.5))
            self.assertAlmostEqual(report["rows"][1]["normalized_rms_error"], 0.25 / math.sqrt(25.0))
            self.assertIn("| 23 | 1 |", markdown_report(report))

    def test_accepts_identical_bundles(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            reference = root / "reference"
            candidate = root / "candidate"
            reference.mkdir()
            candidate.mkdir()
            values = [[-1.0, 0.0, 1.0], [2.0, 4.0, 8.0]]
            report = compare_manifests(
                write_fixture(reference, values), write_fixture(candidate, values), 0.0, 0.0, {23}
            )
            self.assertTrue(report["passed"])
            self.assertEqual(report["first_failing_layer_by_generated_index"], {"23": None})
            self.assertTrue(all(row["cosine_similarity"] == 1.0 for row in report["rows"]))

    def test_rejects_truncated_payload(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            reference = root / "reference"
            candidate = root / "candidate"
            reference.mkdir()
            candidate.mkdir()
            reference_manifest = write_fixture(reference, [[1.0, 2.0]])
            candidate_manifest = write_fixture(candidate, [[1.0, 2.0]])
            (candidate / "generated-000023.bin").write_bytes(struct.pack("<f", 1.0))
            with self.assertRaisesRegex(RuntimeError, "truncated"):
                compare_manifests(reference_manifest, candidate_manifest, 0.0, 0.0)

    def test_ranks_target_against_generated_index_neighborhood(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            reference = root / "reference"
            candidate = root / "candidate"
            reference.mkdir()
            candidate.mkdir()
            reference_values = {index: [[1.0, 1.0], [2.0, 2.0]] for index in (20, 21, 22, 23)}
            candidate_values = {
                20: [[1.10, 1.10], [2.20, 2.20]],
                21: [[1.11, 1.11], [2.30, 2.30]],
                22: [[1.09, 1.09], [2.10, 2.10]],
                23: [[1.30, 1.30], [2.20, 2.20]],
            }
            report = compare_manifests(
                write_neighborhood_fixture(reference, reference_values),
                write_neighborhood_fixture(candidate, candidate_values),
                0.0,
                0.0,
                target_generated_index=23,
            )
            analysis = report["target_outlier_analysis"]
            self.assertEqual(analysis["control_generated_indices"], [20, 21, 22])
            self.assertEqual(analysis["ranked_layers_by_normalized_rms_modified_z"][0], 0)
            self.assertTrue(analysis["rows"][0]["normalized_rms_error"]["above_control_maximum"])
            self.assertFalse(analysis["rows"][1]["normalized_rms_error"]["above_control_maximum"])
            self.assertIn("Target Index 23 Neighborhood", markdown_report(report))


if __name__ == "__main__":
    unittest.main()
