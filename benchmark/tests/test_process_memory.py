import os
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.process_memory import parse_linux_status, sample_current_process, summarize_samples


class ProcessMemoryTest(unittest.TestCase):
    def test_parses_linux_resident_breakdown(self) -> None:
        metrics = parse_linux_status(
            """\
VmSize:     8192 kB
VmHWM:      4096 kB
VmRSS:      3072 kB
RssAnon:    2048 kB
RssFile:     768 kB
RssShmem:    256 kB
Threads:       7
Cpus_allowed_list: 0-3,8
"""
        )
        self.assertEqual(metrics["virtual_bytes"], 8192 * 1024)
        self.assertEqual(metrics["peak_rss_bytes"], 4096 * 1024)
        self.assertEqual(metrics["rss_bytes"], 3072 * 1024)
        self.assertEqual(metrics["anonymous_resident_bytes"], 2048 * 1024)
        self.assertEqual(metrics["mapped_resident_bytes"], 1024 * 1024)
        self.assertEqual(metrics["thread_count"], 7)
        self.assertEqual(metrics["allowed_cpu_ids"], [0, 1, 2, 3, 8])

    def test_summarizes_each_peak_with_its_stage(self) -> None:
        summary = summarize_samples(
            [
                {"elapsed_ms": 10.0, "stage": "import", "rss_bytes": 100, "private_bytes": 80},
                {"elapsed_ms": 20.0, "stage": "prepack", "rss_bytes": 160, "private_bytes": 140},
                {"elapsed_ms": 30.0, "stage": "write", "rss_bytes": 150, "private_bytes": 145},
            ]
        )
        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["peaks"]["rss_bytes"]["bytes"], 160)
        self.assertEqual(summary["peaks"]["rss_bytes"]["stage"], "prepack")
        self.assertEqual(summary["peaks"]["private_bytes"]["bytes"], 145)
        self.assertEqual(summary["peaks"]["private_bytes"]["stage"], "write")
        self.assertIsNone(summary["peaks"]["mapped_resident_bytes"])

    def test_reads_current_process_without_optional_dependencies(self) -> None:
        self.assertGreater(os.getpid(), 0)
        metrics = sample_current_process()
        self.assertIsInstance(metrics["rss_bytes"], int)
        self.assertGreater(metrics["rss_bytes"], 0)
        self.assertIsInstance(metrics["cpu_user_ms"], float)
        self.assertIsInstance(metrics["cpu_system_ms"], float)
        self.assertGreater(metrics["allowed_cpu_count"], 0)


if __name__ == "__main__":
    unittest.main()
