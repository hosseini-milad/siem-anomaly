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
