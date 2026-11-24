"""Metadata-only audit of the local historical Parquet dataset."""
import argparse
from collections import Counter
from datetime import date, timedelta, datetime
import json
from pathlib import Path
import re

import pyarrow.parquet as pq

START = date(2026, 6, 15)
END = date(2026, 7, 10)
CLEAN_END = date(2026, 6, 20)
NAME = re.compile(r"nfcapd\.(\d{12})_(normal|attacks)\.parquet")


def audit_dataset(root: Path) -> dict:
    """Read footers only; never load event rows or publish field values."""
    if not root.is_dir():
        raise FileNotFoundError(root)
    days, schemas, issues = {}, Counter(), []
    for offset in range((END - START).days + 1):
        day = START + timedelta(days=offset)
        key = day.isoformat()
        days[key] = {"expected_clean": day <= CLEAN_END,
                     "normal": {"files": 0, "rows": 0},
                     "attacks": {"files": 0, "rows": 0}}
        for kind in ("normal", "attacks"):
            folder = root / f"date={key}" / f"type={kind}"
            paths = sorted(folder.glob("*.parquet"))
            if not paths and (kind == "normal" or day > CLEAN_END):
                issues.append({"path": str(folder.relative_to(root)), "issue": "empty_or_missing_partition"})
            for path in paths:
                rel = str(path.relative_to(root))
                match = NAME.fullmatch(path.name)
                try:
                    if not match or match[2] != kind or datetime.strptime(match[1], "%Y%m%d%H%M").date() != day:
                        raise ValueError("filename mismatch")
                except ValueError:
                    issues.append({"path": rel, "issue": "filename_partition_mismatch"})
                try:
                    with pq.ParquetFile(path) as pf:
                        schema = tuple((f.name, str(f.type), f.nullable) for f in pf.schema_arrow)
                        schemas[schema] += 1
                        rows = pf.metadata.num_rows
                    days[key][kind]["files"] += 1
                    days[key][kind]["rows"] += rows
                    if day <= CLEAN_END and kind == "attacks" and rows:
                        issues.append({"path": rel, "issue": "attack_rows_in_clean_period"})
                except Exception as exc:
                    issues.append({"path": rel, "issue": "unreadable_footer", "error_type": type(exc).__name__})
    for path in sorted(root.rglob("*.parquet")):
        parts = path.relative_to(root).parts
        if len(parts) != 3 or not parts[0].startswith("date=") or parts[0][5:] not in days or parts[1] not in ("type=normal", "type=attacks"):
            issues.append({"path": str(path.relative_to(root)), "issue": "unexpected_partition"})
    return {"days": days, "schemas": [{"fields": [{"name": n, "type": t, "nullable": v} for n,t,v in s], "files": c} for s,c in schemas.items()],
            "issues": issues, "schema_variants": len(schemas),
            "totals": {kind: {metric: sum(d[kind][metric] for d in days.values()) for metric in ("files", "rows")} for kind in ("normal", "attacks")},
            "limitations": ["Footer audit does not verify event timestamps, actual null counts, row labels or data-page integrity.", "Partition labels are supplied labels, not independently validated."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("dataset/parquet"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/dataset_audit.json"))
    args = parser.parse_args()
    report = audit_dataset(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Audit saved: {args.output}; issues={len(report['issues'])}; schemas={report['schema_variants']}")
    return 1 if report["issues"] or report["schema_variants"] != 1 else 0


if __name__ == "__main__":
    raise SystemExit(main())
