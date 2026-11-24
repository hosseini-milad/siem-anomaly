# Project rules
1. Model first, API later; no executable code before approval.
2. Preserve scientific architecture; simplify software, not methodology without approval.
3. No Django, graphical UI or image generation.
4. One stage at a time; independent testable modules; configuration centralized.
5. Use sparse representations and profile runtime and memory. Cache only with input/config/version validation.
6. Prevent temporal leakage; preserve provenance and reproducibility.
7. Do not publish secrets, sensitive records, private source or model artifacts.
8. Update roadmap, machine-readable and Mermaid knowledge graphs with verified symbols each stage. Planned entities are not implemented entities.
9. External database access read-only by default; schema and connection details require confirmation.

Stage 2 initial implementation: centralized 10000-row batch target; NumPy vectorized feature extraction, port categories, robust scaling and regularized ZCA; bounded uniform priority sample of 200000 rows from reconstructed clean reference only. Fit is provisional, not an independent temporal evaluation. Source overlap and final split pending. Reader retains per-file boundaries for provenance; preprocessing coalescing across files pending. Output transform NPZ and validation report only; no full transformed dataset persisted. Tests and local run pending.
10. Vectorize record operations; central default batch size 10000, configurable. Graph operations may require separate block limits. Fit transforms on training only, never independently per batch.
