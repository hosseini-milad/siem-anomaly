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
