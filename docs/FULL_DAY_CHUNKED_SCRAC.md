# Full-day chunk-local SCRAC experiment

Run the entire day using the established label-blind representative selection and
SCRAC model independently on each chronological 10,000-row chunk. The script
then builds day/month/year super-components from the per-chunk SCRAC groups.

```powershell
python -m scripts.run_chunked_scrac_day --day 2026-06-21 --capacity 200 --output artifacts/temporal/2026-06-21-cap200
```

Important properties:
- Full-day input; no 50k sampling.
- Representative selection is the existing hybrid `select_representatives`.
- Each chunk gets its own indexed graph, connected-component labels, SC-FAGC
  optimization and SCRAC assignment.
- Topology eligibility is local to each chunk graph.
- SCRAC hard capacity is applied to representative-node membership.
- Chunk SCRAC groups are converted to mass-weighted descriptors and passed to
  temporal Complete-Link aggregation with a 200-child limit.
- No attack/normal source labels are consumed by clustering or aggregation.

Outputs:
- `chunks/<chunk-id>/nodes.npz` and `scrac.npz`
- `chunks.json`, `atoms.npz`, `hierarchy.json`, `report.json`

The run aborts if any chunk's SC-FAGC optimization is not validated, if any
assignment is incomplete, or if the hard group capacity is violated. It does
not silently fall back to a different clustering algorithm.

Capacity 300 or 500 is permitted for controlled follow-up experiments, but 200
is the baseline and must be evaluated first. The daily hierarchy's capacity is
measured in immediate child components; raw-row mass is conserved separately.
