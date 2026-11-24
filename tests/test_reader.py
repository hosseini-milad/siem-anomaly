import unittest
from datetime import date, datetime
import pyarrow as pa
from data.reader import canonicalize

class ReaderTest(unittest.TestCase):
    def test_logical_date_and_original(self):
        stamp = datetime(2026, 7, 29, 12, 34, 56)
        batch = pa.record_batch([pa.array([stamp, None], type=pa.timestamp("us")), pa.array([80.0, 443.0])], names=["flow_start", "dst_port"])
        result = canonicalize(batch, date(2026, 6, 15), "nfcapd.202607291235_normal.parquet")
        self.assertEqual(result.column(result.schema.get_field_index("flow_start")).to_pylist(), [datetime(2026, 6, 15, 12, 34, 56), datetime(2026, 6, 15, 12, 35)])
        self.assertEqual(result.column(result.schema.get_field_index("original_flow_start")).to_pylist(), [stamp, None])
        self.assertEqual(result.column(result.schema.get_field_index("dst_port")).type, pa.int64())

    def test_fractional_and_nonfinite_integers(self):
        from data.reader import safe_integer_cast
        values = pa.array([3.1, 80.0, None, float("nan"), float("inf"), -1.5, 2**63 * 1.0])
        self.assertEqual(safe_integer_cast(values).to_pylist(), [None, 80, None, None, None, None, None])
        batch = pa.record_batch([pa.array([None]*7, type=pa.timestamp("us")), values], names=["flow_start", "dst_port"])
        diagnostics = {}
        result = canonicalize(batch, date(2026, 6, 15), "nfcapd.202607291235_normal.parquet", diagnostics)
        self.assertEqual(diagnostics, {"dst_port": 5})
