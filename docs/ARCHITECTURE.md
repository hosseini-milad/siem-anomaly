# Architecture
Proposed modular monolith. Python core independent of API and database. FastAPI proposed for REST, MongoDB for project metadata/results with versioned model artifacts, NumPy/SciPy/Pandas/scikit-learn; NetworkX only if needed. Dependency choices remain provisional.
## Planned modules
| Path | Responsibility |
|---|---|
| config/ | validated settings |
| data/ | historical and external database adapters |
| preprocessing/ | schema, encoding, train-only fit and transform |
| graph/ | sparse similarity graph and connected components |
| compression/ | representative selection, mapping, traceable shortcuts |
| hierarchy/ | daily and period aggregation |
| learning/ | normal behavior model |
| evaluation/ | metrics and validation |
| storage/ | artifact persistence/versioning |
| inference/ | scoring new events using frozen model |
| api/ | REST orchestration |
| tests/ | unit and integration tests |
| artifacts/ | local generated outputs, not committed |
Training: data -> preprocessing -> graph -> compression -> hierarchy -> learning -> evaluation -> storage. Inference: external adapter -> frozen preprocessing -> model scoring -> API. Evaluation gates release. No Django, GUI or visual reporting. Avoid API/DB coupling in scientific core.
Upstream conceptual mappings: ingestion to data/preprocessing/graph/compression; pooling to compression; hierarchy to hierarchy; spectral_learning to learning. Audit actual code before reuse. Stage 1 data and tests directories created; remaining Python modules are planned.

## Pre-graph reduction architecture revision
Architecture revision requested by user: before initial similarity graph, process logical 10000-row chunks with fixed preprocessing and local unsupervised KNN-based grouping, retaining approximately 10% actual representative records (1000 per full chunk). This is distinct from later structure-aware graph compression. KNN grouping algorithm not yet specified or implemented; KNN alone is not clustering. Preserve rare/isolated behaviors via label-free novelty/coverage allocation, avoid density-only selection, retain source-to-representative mapping, multiplicity, distances and provenance; flag uncovered records rather than silently discarding them. Labels only for evaluation. Compare rare-pattern recall, coverage error, downstream PR-AUC/recall, cross-chunk neighbor recall, runtime/RAM against unreduced and random baselines. Fixed similarity threshold strictly >0.85; current ZCA provisional, mixed formula parity pending. Full chunk pair count 49,995,000 versus 499,500 for 1000 representatives; 99% reduction in graph-stage comparisons only, not end-to-end KNN cost. Use bounded block or indexed candidate search, no full dataset dense matrix. Cross-chunk representative edges must remain possible. Two successive 10% reductions would retain 1% overall; second-stage budget must be separately validated. Protect mapping and raw security data locally; do not publish raw records or identifiers. Pilot before acceptance; exact grouping, rarity budget, mapping format and cross-file coalescing pending.
