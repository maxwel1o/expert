import csv
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReleaseInventoryTests(unittest.TestCase):
    def test_exactly_five_role_profiles_exist(self):
        actual = {p.parent.name for p in (ROOT / "roles").glob("*/SOUL.md")}
        self.assertEqual(actual, {"leader", "deployer", "tester", "profiler", "analyst"})

    def test_manifest_has_expected_role_counts(self):
        with (ROOT / "manifests" / "skills.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        counts = {}
        for row in rows:
            counts[row["role"]] = counts.get(row["role"], 0) + 1
        self.assertEqual(
            counts,
            {"leader": 1, "deployer": 127, "tester": 19, "profiler": 5, "analyst": 49},
        )


if __name__ == "__main__":
    unittest.main()
