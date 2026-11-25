# SIEM Anomaly
Lightweight modular Python implementation of the scientific architecture of spectral_project. Training first, inference and REST API later. No Django, UI, or image generation.
Start with [AGENTS.md](AGENTS.md), [architecture](docs/ARCHITECTURE.md), [roadmap](docs/ROADMAP.md), [model design](docs/MODEL_DESIGN.md), [rules](docs/PROJECT_RULES.md), [knowledge graph](docs/knowledge/GRAPH.md) and [machine-readable graph](docs/knowledge/graph.json).
Status: architecture proposal approved for documentation; stage 1 metadata audit implemented; tests and real dataset audit pending. See [dataset instructions](docs/DATASET.md). Do not publish sensitive data or private upstream source.

Stage 4.1 component-level SC-FAGC is implemented on branch feature/component-sc-fagc. See [SC-FAGC design and runbook](docs/SC_FAGC.md). Synthetic tests: python -m pytest tests/test_sc_fagc.py -q. Real-data scientific validation remains pending.
