# Roadmap
Status: stage 1 metadata audit implemented; tests and local dataset audit pending.
| Stage | Deliverable | Status |
|---|---|---|
| 0 | Architecture and knowledge graph | documented |
| 1 | Confirm historical source, schema, timestamps, labels, volumes, missingness and split | in progress; source layout supplied; metadata audit implemented, validation pending |
| 2 | Fit preprocessing on training data; persist transforms | planned |
| 3 | Sparse mixed-similarity graph | planned |
| 4 | Connected components including isolates | planned |
| 5 | Representatives, mapping and traceable compression | planned |
| 6 | Daily/period hierarchy | planned |
| 7 | Normal behavior model | planned |
| 8 | Validation, baselines, sensitivity and profiling | planned |
| 9 | Versioned model artifact and reload test | planned |
| 10 | External database inference and REST API | planned |
Each stage requires review, acceptance criteria, tests, measured outputs and graph updates before advancing. Do not treat proposed parameters as verified results.

Metadata audit received. User confirmed reconstructed clean days and approved logical folder date. Canonical reader and bounded profile implemented; local tests and profile pending. Do not advance until reviewed.

Stage 1 fix: `data.reader.canonicalize` calls `data.reader.safe_integer_cast`; invalid float-to-integer values become null and `iter_batches` forwards per-file counts to `data.profile.profile_dataset`. Regression test added; local retry pending.

## Stage 1 profile review
Profile received: 48,759,366 rows (29,411,750 normal; 19,347,616 attacks). 19,051 invalid dst_port casts, all normal, across 5,340 files; converted to null. No other checked invalid values; 7,319,029 missing label_binary; non-null labels and type match partitions. Missing country 16,840,679; behavioral_flags 41,473,304; threat_intel_score 13,887,652; each MITRE field 29,411,750. Profile execution confirmed, tests not confirmed. Proposed next: feature allowlist, exclude label-derived fields, unknown port handling without class-dependent deletion, source-overlap check and train-only preprocessing. Stage 1 review pending approval.

Stage 2 initial implementation: centralized 10000-row batch target; NumPy vectorized feature extraction, port categories, robust scaling and regularized ZCA; bounded uniform priority sample of 200000 rows from reconstructed clean reference only. Fit is provisional, not an independent temporal evaluation. Source overlap and final split pending. Reader retains per-file boundaries for provenance; preprocessing coalescing across files pending. Output transform NPZ and validation report only; no full transformed dataset persisted. Tests and local run pending.

Stage 2 execution report reviewed: all rows processed, batch target 10000, 200000 fit rows, 25 features. NPZ inspected: finite float64 arrays, positive scales, symmetric positive definite whitening, condition number approximately 871444. Four features have zero median and scaled mean with unit stored scale; variance requires verification. Stage 2 not scientifically accepted yet; feature distribution, covariance diagnostics, source overlap, final split and cross-file batching pending. Unit test results not supplied.
