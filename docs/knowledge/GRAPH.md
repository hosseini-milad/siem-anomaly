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
