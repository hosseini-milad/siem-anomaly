import tempfile
import unittest
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from data.audit import audit_dataset

class DatasetAuditTest(unittest.TestCase):
    def test_counts_and_clean_violation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for kind in ("normal", "attacks"):
                folder = root / "date=2026-06-15" / f"type={kind}"
                folder.mkdir(parents=True)
                pq.write_table(pa.table({"value": [1, 2]}), folder / f"nfcapd.202606151200_{kind}.parquet")
            report = audit_dataset(root)
            self.assertEqual(report["totals"]["normal"]["rows"], 2)
            self.assertEqual(report["schema_variants"], 1)
            self.assertTrue(any(i["issue"] == "attack_rows_in_clean_period" for i in report["issues"]))
            self.assertTrue(any(i["issue"] == "empty_or_missing_partition" for i in report["issues"]))

    def test_missing_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                audit_dataset(Path(tmp) / "missing")
