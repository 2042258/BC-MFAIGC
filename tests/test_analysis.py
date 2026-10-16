"""Collector denominators/schema and explicit workspace ownership."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis import collect_runs, collect_comparison_runs
from analysis.build_reports import detection_table, mean_std


class AnalysisTests(unittest.TestCase):
    def test_shared_collectors_keep_vote_and_worker_round_denominators_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            run = project / "logs/blogs/20260101_000000_fixture"
            run.mkdir(parents=True)
            votes = [
                {
                    "round": 1,
                    "worker_id": "bad",
                    "worker_malicious": True,
                    "score": 2.0,
                    "accepted_for_aggregation": False,
                },
                {
                    "round": 1,
                    "worker_id": "bad",
                    "worker_malicious": True,
                    "score": 1.0,
                    "accepted_for_aggregation": True,
                },
                {
                    "round": 1,
                    "worker_id": "good",
                    "worker_malicious": False,
                    "score": 0.0,
                    "accepted_for_aggregation": True,
                },
            ]
            decisions = [
                {
                    "round": 1,
                    "worker_id": "bad",
                    "worker_malicious": True,
                    "filter_accepted": False,
                }
            ]
            (run / "validation_records.jsonl").write_text(
                "\n".join(map(json.dumps, votes))
            )
            (run / "aggregation_decisions.jsonl").write_text(
                "\n".join(map(json.dumps, decisions))
            )
            (run / "timing_records.jsonl").write_text(
                json.dumps({"round": 1, "phase": "round_summary", "defense_on": 1})
            )
            modern = collect_runs.summarize(
                run, "fixture", "test", project_root=project
            )
            classic = collect_comparison_runs.summarize(
                run, "fixture", "test", project_root=project
            )
            self.assertEqual(modern["malicious_vote_total"], 2)
            self.assertEqual(modern["malicious_worker_round_total"], 1)
            self.assertEqual(modern["malicious_worker_round_filter_rejected"], 1)
            self.assertEqual(
                modern["benign_worker_round_total"], 0
            )  # Unknown final decision is not counted.
            self.assertEqual(modern["log_dir"], "logs/blogs/20260101_000000_fixture")
            for key, value in classic.items():
                self.assertEqual(value, modern[key])
            self.assertNotIn("protocol_phase_counts", classic)
            self.assertIn("protocol_phase_counts", modern)

    def test_manifest_supplies_formal_run_tags_to_both_collector_clis(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            run_id = "core-clean-fedavg-91000"
            run = project / "logs/blogs" / ("20260101_000000_" + run_id)
            run.mkdir(parents=True)
            (run / "timing_records.jsonl").write_text(
                json.dumps({"round": 1, "phase": "round_summary"})
            )
            manifest = project / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "common": [],
                        "runs": [{"id": run_id, "tag": "formal-fedavg", "args": []}],
                    }
                )
            )
            for module in (collect_runs, collect_comparison_runs):
                output = project / (module.__name__ + ".jsonl")
                with patch(
                    "sys.argv",
                    ["collector", "--manifest", str(manifest), "--output", str(output)],
                ):
                    module.main(project_root=project)
                rows = [
                    json.loads(line)
                    for line in output.read_text().splitlines()
                    if line.strip()
                ]
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["run_id"], run_id)
                self.assertEqual(rows[0]["tag"], "formal-fedavg")

    def test_detection_statistics_remain_population_std_and_known_rounds(self):
        self.assertEqual(mean_std([1.0, 3.0]), {"n": 2, "mean": 2.0, "std": 1.0})
        table = detection_table(
            [
                {
                    "tag": "x",
                    "malicious_worker_round_filter_rejected": 1,
                    "malicious_worker_round_total": 2,
                    "benign_worker_round_filter_rejected": 0,
                    "benign_worker_round_total": 4,
                }
            ]
        )["x"]
        self.assertEqual(table["pooled_tpr"], 0.5)
        self.assertEqual(table["pooled_fpr"], 0.0)


if __name__ == "__main__":
    unittest.main()
