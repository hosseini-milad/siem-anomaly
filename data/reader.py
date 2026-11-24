"""Read bounded batches with canonical types and logical partition dates."""
from pathlib import Path
from datetime import date, datetime
import re
from config.settings import BATCH_SIZE
import numpy as np
from collections import Counter
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

FLOATS = set("duration_s iat_mean iat_std payload_entropy packets_per_sec bytes_per_packet bytes_per_sec iat_cv log_bytes log_packets log_duration threat_intel_score".split())
INTS = set("src_port dst_port packets bytes tos flag_count flag_syn flag_ack flag_fin flag_rst flag_psh flag_urg is_syn_only is_encrypted_port proto_tcp proto_udp proto_icmp label_binary label_multiclass".split())
STRINGS = set("mitre_technique mitre_tactic".split())

def safe_integer_cast(arr):
    """Null invalid float integers without truncating; preserve original nulls."""
    if pa.types.is_floating(arr.type):
        arr = pc.cast(arr, pa.float64())
        valid = pc.and_kleene(pc.is_finite(arr), pc.equal(arr, pc.floor(arr)))
        valid = pc.and_kleene(valid, pc.and_(pc.greater_equal(arr, -9223372036854775808.0), pc.less(arr, 9223372036854775808.0)))
        arr = pc.if_else(pc.fill_null(valid, False), arr, pa.scalar(None, type=arr.type))
    return pc.cast(arr, pa.int64(), safe=True)


def canonicalize(batch, day, filename, diagnostics=None):
    """Preserve original timestamp; replace date only, retaining time of day."""
    cols = {}
    for name in batch.schema.names:
        arr = batch.column(batch.schema.get_field_index(name))
        target = pa.float64() if name in FLOATS else pa.int64() if name in INTS else pa.string() if name in STRINGS else arr.type
        try:
            converted = safe_integer_cast(arr) if name in INTS else pc.cast(arr, target, safe=True)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
            raise ValueError(f"{filename}: column {name}: {exc}") from exc
        added = converted.null_count - arr.null_count
        if added and diagnostics is not None:
            diagnostics[name] = diagnostics.get(name, 0) + added
        cols[name] = converted
    original = pc.cast(cols["flow_start"], pa.timestamp("us"), safe=True)
    match = re.search(r"nfcapd\.(\d{12})_", filename)
    fallback = datetime.strptime(match[1], "%Y%m%d%H%M") if match else None
    valid = pc.is_valid(original).to_numpy(zero_copy_only=False)
    # Use integer Arrow buffers to preserve microsecond precision.
    safe = pc.fill_null(pc.cast(original, pa.int64()), 0).to_numpy(zero_copy_only=False)
    midnight = int(np.datetime64(day.isoformat(), 'us').astype(np.int64))
    values = midnight + np.mod(safe, 86400000000)
    if fallback is not None:
        offset = (fallback.hour * 3600 + fallback.minute * 60 + fallback.second) * 1000000
        values = np.where(valid, values, midnight + offset)
    values = pa.array(values, mask=(~valid if fallback is None else None), type=pa.int64()).cast(pa.timestamp('us'))
    cols["original_flow_start"] = original
    cols["flow_start"] = values
    cols["date"] = pa.repeat(pa.scalar(day, type=pa.date32()), batch.num_rows)
    return pa.record_batch([cols[n] for n in sorted(cols)], names=sorted(cols))

def iter_batches(root: Path, batch_size=BATCH_SIZE, diagnostics=None):
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
                counts = {}
                result = canonicalize(batch, day, path.name, counts)
                if diagnostics is not None and counts:
                    diagnostics.setdefault(str(path.relative_to(root)), Counter()).update(counts)
                yield day, kind, path, result
