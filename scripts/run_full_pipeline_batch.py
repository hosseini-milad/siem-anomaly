"""Run the complete raw-day -> chunk SCRAC -> daily evaluation -> BRR -> monthly pipeline.

Only days on/after 2026-06-21 are processed. Earlier normal-only days remain
available to the already-fitted preprocessing transform, but are never treated
as attack-analysis days. The transform is reused, never refitted here.
"""
from __future__ import annotations
import argparse
import json
import shutil
from datetime import date
from pathlib import Path

CUTOFF = date(2026, 6, 21)

def discover_days(root: Path) -> list[str]:
    root = Path(root)
    days = []
    for folder in root.glob("date=*"):
        raw = folder.name.removeprefix("date=")
        try:
            parsed = date.fromisoformat(raw)
        except ValueError:
            continue
        if parsed < CUTOFF:
            continue
        if any((folder / f"type={kind}").is_dir() and
               any((folder / f"type={kind}").glob("*.parquet"))
               for kind in ("normal", "attacks")):
            days.append(parsed.isoformat())
    return sorted(set(days))

def select_days(root: Path, requested: list[str] | None) -> list[str]:
    available = discover_days(root)
    if requested is None:
        if not available:
            raise ValueError(f"No Parquet days found on/after {CUTOFF} under {root}")
        return available
    selected = sorted(set(requested))
    invalid = []
    for value in selected:
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            invalid.append(value)
            continue
        if parsed < CUTOFF or value not in available:
            invalid.append(value)
    if invalid:
        raise ValueError(f"Requested days are invalid, before cutoff, or missing Parquet data: {invalid}. Available: {available}")
    if not selected:
        raise ValueError("No days selected")
    return selected

def run(root, transform, output, requested_days=None, capacity=200,
        representative_ratio=.10, graph_k=12, degree_cap=20,
        edge_threshold=.85, anchor_threshold=.85, lambda_size=.20,
        acceptance_threshold=.75, temporal_threshold=.85,
        rare_max=10, redundancy_threshold=.999, overwrite=False):
    from scripts.run_chunked_scrac_day import run_day
    from evaluation.hierarchical_evaluation import evaluate
    from refinement.run import run as run_brr
    from scripts.run_period_batch import build_monthly

    root, transform, output = Path(root), Path(transform), Path(output)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    if not transform.is_file():
        raise FileNotFoundError(f"Fitted transform not found: {transform}; this pipeline deliberately does not refit it.")
    days = select_days(root, requested_days)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output}. Choose a new --output or pass --overwrite to replace only this dedicated run folder.")
        shutil.rmtree(output)
    structural_root = output / "structural"
    evaluation_root = output / "evaluation"
    refinement_root = output / "behavioral_refinement"
    for folder in (structural_root, evaluation_root, refinement_root):
        folder.mkdir(parents=True, exist_ok=True)

    summaries = []
    for day in days:
        print(f"\n=== STRUCTURAL DAY {day} ===", flush=True)
        structural = structural_root / day
        report = run_day(root, day, transform, structural, capacity=capacity,
            representative_ratio=representative_ratio, graph_k=graph_k,
            degree_cap=degree_cap, edge_threshold=edge_threshold,
            anchor_threshold=anchor_threshold, lambda_size=lambda_size,
            acceptance_threshold=acceptance_threshold,
            temporal_threshold=temporal_threshold)
        print(f"\n=== EXACT EVALUATION {day} ===", flush=True)
        evaluated = evaluate(root, day, structural, transform, evaluation_root / day)
        print(f"\n=== BEHAVIORAL REFINEMENT {day} ===", flush=True)
        refined = run_brr(evaluation_root / day, refinement_root / day,
            threshold=temporal_threshold, rare_max=rare_max,
            redundancy_threshold=redundancy_threshold)
        summaries.append({
            "day": day,
            "raw_rows": report["raw_rows"],
            "chunks": report["chunk_count"],
            "chunk_components": report["chunk_scrac_output_components"],
            "daily_supercomponents": report["daily_supercomponents"],
            "evaluation": str(evaluation_root / day / "evaluation.json"),
            "behavioral_refinement": str(refinement_root / day / "report.json"),
            "valid_behavioral_coverage": refined["valid_coverage"],
            "uncovered_rows": refined["uncovered_rows"],
        })

    print("\n=== MONTHLY SUPERCOMPONENTS ===", flush=True)
    daily_eval_folders = {day: evaluation_root / day for day in days}
    monthly = build_monthly(daily_eval_folders, output / "period_batch",
        threshold=temporal_threshold, capacity=capacity,
        lambda_size=lambda_size, acceptance_threshold=acceptance_threshold)

    result = {
        "status": "completed",
        "pipeline": "raw_parquet_to_chunk_scrac_daily_hierarchy_evaluation_behavioral_refinement_monthly_supercomponents",
        "cutoff_inclusive": CUTOFF.isoformat(),
        "normal_only_days_before_cutoff_excluded": True,
        "transform_reused_not_refitted": str(transform),
        "days_processed": days,
        "daily_results": summaries,
        "monthly_results": monthly,
        "capacity": capacity,
        "output": str(output.resolve()),
        "notes": [
            "Every selected day is read in full; chunk size remains the established 10,000 rows.",
            "Preprocessing transform is reused. Earlier normal-only days are not included as analysis days.",
            "Labels are not used for representative selection, graph construction, SCRAC or monthly grouping; they are used only in post-hoc evaluation/reporting.",
            "Monthly supercomponents use the available processed days only; absent calendar dates are not synthesized.",
            "Yearly aggregation is not run because this command creates monthly outputs and the available period may not contain 12 complete months."
        ]
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "full_pipeline_report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="daily_results"}, indent=2))
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="dataset/parquet")
    p.add_argument("--transform", default="artifacts/preprocessing/transform.npz")
    p.add_argument("--output", default="artifacts/full_pipeline_run")
    p.add_argument("--days", default=None,
        help="Comma-separated ISO dates. Omit to process every available day >= 2026-06-21.")
    p.add_argument("--capacity", type=int, choices=(200,300,500), default=200)
    p.add_argument("--representative-ratio", type=float, default=.10)
    p.add_argument("--graph-k", type=int, default=12)
    p.add_argument("--degree-cap", type=int, default=20)
    p.add_argument("--edge-threshold", type=float, default=.85)
    p.add_argument("--anchor-threshold", type=float, default=.85)
    p.add_argument("--lambda-size", type=float, default=.20)
    p.add_argument("--acceptance-threshold", type=float, default=.75)
    p.add_argument("--temporal-threshold", type=float, default=.85)
    p.add_argument("--rare-max", type=int, default=10)
    p.add_argument("--redundancy-threshold", type=float, default=.999)
    p.add_argument("--overwrite", action="store_true",
        help="Delete and recreate only the dedicated --output folder.")
    a = p.parse_args()
    days = None if not a.days else [v.strip() for v in a.days.split(",") if v.strip()]
    run(a.root, a.transform, a.output, days, a.capacity,
        a.representative_ratio, a.graph_k, a.degree_cap, a.edge_threshold,
        a.anchor_threshold, a.lambda_size, a.acceptance_threshold,
        a.temporal_threshold, a.rare_max, a.redundancy_threshold, a.overwrite)

if __name__ == "__main__":
    main()
