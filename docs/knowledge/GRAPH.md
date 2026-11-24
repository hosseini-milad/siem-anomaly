# Knowledge graph — planned architecture
Stage 1 audit code exists; execution and tests pending. The JSON graph is the canonical index; update both views at every stage. Paths are planned, not verified existing directories.
```mermaid
flowchart TD
 R[Historical Parquet] --> D[data.audit.audit_dataset]
 M[data.audit.main] --> D
 D --> O[Local audit JSON]
 D --> P[preprocessing]
 P --> G[graph]
 G --> C[compression]
 C --> H[hierarchy]
 H --> L[learning]
 L --> E[evaluation]
 E --> S[storage]
 X[external DB] --> I[inference]
 S --> I
 I --> A[api]
```
For every implemented function add its qualified name, repository path, inputs, outputs, calls, dependencies, stage, tests and status to graph.json. Add cross-module edges with explicit relation types. Do not infer call relationships without checking code. This view documents dependencies, not generated runtime images.

Stage 1 symbols: `data.audit.audit_dataset(root)` reads Parquet footers; `data.audit.main()` invokes it and writes local JSON. Tests: `tests/test_dataset_audit.py`. See [dataset instructions](../DATASET.md).

## Stage 1 canonical reader
```mermaid
flowchart TD
 P[Parquet] --> R[data.reader.iter_batches]
 R --> C[data.reader.canonicalize]
 C --> V[data.profile.profile_dataset]
 V --> O[Local profile JSON]
```
Original flow timestamp retained; logical date follows folder. Tests and profile execution pending.

Stage 1 fix: `data.reader.canonicalize` calls `data.reader.safe_integer_cast`; invalid float-to-integer values become null and `iter_batches` forwards per-file counts to `data.profile.profile_dataset`. Regression test added; local retry pending.

## Stage 1 profile review
Profile received: 48,759,366 rows (29,411,750 normal; 19,347,616 attacks). 19,051 invalid dst_port casts, all normal, across 5,340 files; converted to null. No other checked invalid values; 7,319,029 missing label_binary; non-null labels and type match partitions. Missing country 16,840,679; behavioral_flags 41,473,304; threat_intel_score 13,887,652; each MITRE field 29,411,750. Profile execution confirmed, tests not confirmed. Proposed next: feature allowlist, exclude label-derived fields, unknown port handling without class-dependent deletion, source-overlap check and train-only preprocessing. Stage 1 review pending approval.

Stage 2 initial implementation: centralized 10000-row batch target; NumPy vectorized feature extraction, port categories, robust scaling and regularized ZCA; bounded uniform priority sample of 200000 rows from reconstructed clean reference only. Fit is provisional, not an independent temporal evaluation. Source overlap and final split pending. Reader retains per-file boundaries for provenance; preprocessing coalescing across files pending. Output transform NPZ and validation report only; no full transformed dataset persisted. Tests and local run pending.

```mermaid
flowchart TD
 R[data.reader.iter_batches] --> F[preprocessing.features.extract_features]
 F --> S[preprocessing.transform.sample_training]
 S --> T[preprocessing.transform.fit_transform]
 F --> A[Transform.apply]
 T --> A
 A --> O[Validation report]
```
