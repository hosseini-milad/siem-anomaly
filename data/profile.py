"""Bounded-memory validation; output aggregate counts only."""
import argparse
from collections import Counter
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.compute as pc
from data.reader import iter_batches

BINARY = set("flag_syn flag_ack flag_fin flag_rst flag_psh flag_urg is_syn_only is_encrypted_port proto_tcp proto_udp proto_icmp".split())

def count_true(mask):
    return pc.sum(pc.cast(pc.fill_null(mask, False), pa.int64())).as_py() or 0

def profile_dataset(root, batch_size=8192):
    rows = 0
    nulls, invalid, labels = Counter(), Counter(), Counter()
    cast_issues = {}
    for day, kind, path, batch in iter_batches(root, batch_size, cast_issues):
        rows += batch.num_rows
        for name, arr in zip(batch.schema.names, batch.columns):
            nulls[name] += arr.null_count
            if pa.types.is_floating(arr.type):
                invalid[name + ":nonfinite"] += count_true(pc.invert(pc.is_finite(arr)))
            if name in ("src_port", "dst_port"):
                invalid[name + ":range"] += count_true(pc.or_(pc.less(arr, 0), pc.greater(arr, 65535)))
            if name in BINARY:
                invalid[name + ":binary"] += count_true(pc.and_(pc.is_valid(arr), pc.invert(pc.is_in(arr, value_set=pa.array([0, 1], type=arr.type)))))
            if name == "label_binary":
                invalid["label_binary:partition_mismatch"] += count_true(pc.not_equal(arr, 0 if kind == "normal" else 1))
            if name == "type":
                invalid["type:partition_mismatch"] += count_true(pc.not_equal(arr, kind))
        labels[kind] += batch.num_rows
    for counts in cast_issues.values():
        for name, count in counts.items():
            invalid[name + ":invalid_integer_cast"] += count
    return {"cast_issues_by_file": cast_issues, "rows": rows, "partition_rows": dict(labels), "nulls": dict(nulls), "invalid": dict(invalid)}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("dataset/parquet"))
    p.add_argument("--output", type=Path, default=Path("artifacts/dataset_profile.json"))
    p.add_argument("--batch-size", type=int, default=8192)
    a = p.parse_args()
    report = profile_dataset(a.root, a.batch_size)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Profile saved: {a.output}; rows={report['rows']}")
    return 1 if any(report["invalid"].values()) or not report["rows"] else 0

if __name__ == "__main__":
    raise SystemExit(main())
