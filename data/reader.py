"""Read bounded batches with canonical types and logical partition dates."""
from pathlib import Path
from datetime import date, datetime
import re
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

FLOATS = set("duration_s iat_mean iat_std payload_entropy packets_per_sec bytes_per_packet bytes_per_sec iat_cv log_bytes log_packets log_duration threat_intel_score".split())
INTS = set("src_port dst_port packets bytes tos flag_count flag_syn flag_ack flag_fin flag_rst flag_psh flag_urg is_syn_only is_encrypted_port proto_tcp proto_udp proto_icmp label_binary label_multiclass".split())
STRINGS = set("mitre_technique mitre_tactic".split())

def canonicalize(batch, day, filename):
    """Preserve original timestamp; replace date only, retaining time of day."""
    cols = {}
    for name in batch.schema.names:
        arr = batch.column(batch.schema.get_field_index(name))
        target = pa.float64() if name in FLOATS else pa.int64() if name in INTS else pa.string() if name in STRINGS else arr.type
        cols[name] = pc.cast(arr, target, safe=True)
    original = pc.cast(cols["flow_start"], pa.timestamp("us"), safe=True)
    match = re.search(r"nfcapd\.(\d{12})_", filename)
    fallback = datetime.strptime(match[1], "%Y%m%d%H%M") if match else None
    values = [datetime.combine(day, (v or fallback).time()) if (v or fallback) is not None else None for v in original.to_pylist()]
    cols["original_flow_start"] = original
    cols["flow_start"] = pa.array(values, type=pa.timestamp("us"))
    cols["date"] = pa.array([day] * batch.num_rows, type=pa.date32())
    return pa.record_batch([cols[n] for n in sorted(cols)], names=sorted(cols))

def iter_batches(root: Path, batch_size=8192):
    """Yield partition date, kind, source path and canonical record batch."""
    if not root.is_dir():
        raise FileNotFoundError(root)
    for path in sorted(root.glob("date=*/type=*/*.parquet")):
        day = date.fromisoformat(path.parent.parent.name[5:])
        kind = path.parent.name[5:]
        if kind not in ("normal", "attacks"):
            continue
        with pq.ParquetFile(path) as pf:
            for batch in pf.iter_batches(batch_size=batch_size):
                yield day, kind, path, canonicalize(batch, day, path.name)
