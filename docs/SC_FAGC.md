# Component-level SC-FAGC

## Purpose and boundary
This stage consumes connected components produced by graph.components. Each
connected component is one structural representative of the sparse graph.
SC-FAGC groups these representatives into higher-level behavioral groups; it
does not replace SCRAC, the sparse graph, or the connected-component audit.
No attack/normal labels enter feature construction, anchor selection,
optimization, or assignment.

## Method
For component feature vectors, construct the bipartite component-to-anchor
matrix using the paper's parameter-free assignment:
B_ij = (d_i,k+1 - d_ij) / (k*d_i,k+1 - sum_h d_i,h), for k nearest anchors,
with d_ij = 2*(1-S_ij) and zero weights elsewhere. Rows sum to one.
Anchor affinity is A = B.T @ B. SC-FAGC maximizes
Tr(U.T @ A @ U) + epsilon*H(U), with row-simplex constraints and bilateral
anchor-cluster mass bounds. IRW/MM linearizes the cohesion term using 2*A@U;
each subproblem is solved with double-bounded entropic optimal transport and
generalized Sinkhorn scaling.

Transport row marginals are weighted by each component's represented raw-row
count. This is a project-specific extension so large components contribute
proportionally to cluster mass. The final assignment retains the SCRAC
Complete-Link gate: a component may join a group only when its minimum mixed
similarity to every current member is strictly greater than 0.85. Candidate
groups are ranked by 0.4*S_anchor + 0.3*S_mean + 0.3*S_min - P(new_raw_mass),
where P(n)=0.02*ln(1+n/50)+0.10*(max(0,n-100)/100)^2. Groups above 100
represented raw rows become progressively less attractive, rather than being
forbidden by a hard capacity. Components that cannot join an existing group
are retained as standalone groups; no component is discarded.

## Anchor and descriptor adaptation
The paper uses BKHK for anchor generation and N2HI for initialization. This
implementation currently uses deterministic farthest-first component medoids
and leading-eigenvector initialization. These are explicit substitutions, not
claims of exact reproduction of BKHK/N2HI. Component descriptors are means of
continuous representative-node features and per-feature modes of discrete
representative-node features. The component audit's raw_rows array supplies
component mass.

## Run
First produce the component audit from an existing graph-pilot directory:
python -m graph.components --input artifacts/graph/pilot --output artifacts/graph/components

Then run SC-FAGC:
python -m graph.component_sc_fagc --nodes artifacts/graph/pilot/nodes.npz --components artifacts/graph/components/components.npz --output artifacts/graph/sc_fagc/report.json --capacity 100

Outputs: report.json contains parameters, objective history, convergence
diagnostics, assignments, cluster masses and oversized indices. report.npz
contains component assignments, soft probabilities, component masses and
anchor indices. Keep these artifacts local if they expose sensitive structure.

Optional controls: --anchors, --clusters, --anchor-k, --epsilon, --max-iter,
and --sinkhorn-iter.

## Tests
Install test dependencies with: python -m pip install -r requirements.txt pytest
Run: python -m pytest tests/test_sc_fagc.py -q

## Validation status and limitations
Synthetic tests cover bipartite row normalization, weighted DB-OT bounds,
the approved SCRAC size penalty, Complete-Link thresholding, total component
mapping, descriptor aggregation, and end-to-end execution. They do not establish detection quality on real data.
Real-data evaluation must compare SC-FAGC with SCRAC-only and no-size-penalty
baselines, reporting component-size distribution, oversized fraction, graph
coverage, objective history, runtime, peak memory, and downstream anomaly
metrics on agreed held-out days. Weighted DB-OT and final capacity assignment
are project adaptations; the paper's anchor-level bounds alone do not imply a
hard capacity guarantee for final components.


## Adaptive anchor expansion (2026-11-25)

The initial 5% anchor budget is a starting seed, not a cap. Each component is compared against current anchors using the configured mixed-similarity threshold (default 0.85). If at least one existing anchor has similarity strictly greater than the threshold, the component is assigned to its nearest qualifying anchors (up to anchor-k). Otherwise the component becomes a new anchor and can represent later components. The bipartite matrix is sparse; each row sums to one and anchor rows self-assign. Reports distinguish initial coverage, components requiring new anchors, number of newly created anchors, final anchor count, and each creation reason. Threshold is an experimental hyperparameter and must be evaluated separately from SCRAC Complete-Link.


## Evaluation run for adaptive anchors

The initial anchor budget defaults to 5% but is not a cap. Dynamic expansion adds a component as a new anchor only when its best similarity to all current anchors is not strictly above `--anchor-threshold` (default 0.85). Existing qualifying anchors always take precedence. `--anchor-threshold` is distinct from the final SCRAC Complete-Link threshold and should be tuned on held-out data.

Run the model (single-line command):
`python -m graph.component_sc_fagc --nodes artifacts/graph/pilot/nodes.npz --components artifacts/graph/components/components.npz --output artifacts/graph/sc_fagc/report_v4.json --capacity 100 --anchor-threshold 0.85 --max-iter 500`

Run independent assignment audit (single-line command):
`python -m scripts.evaluate_sc_fagc --nodes artifacts/graph/pilot/nodes.npz --components artifacts/graph/components/components.npz --report artifacts/graph/sc_fagc/report_v4.json --output artifacts/graph/sc_fagc/evaluation_v4.json --threshold 0.85`

The audit checks every pair within each final group for the Complete-Link threshold, reports component-count and raw-row-mass size distributions, oversized groups, singleton groups, and anchor expansion coverage. Runtime in this audit is not model runtime. Compare at least threshold 0.80, 0.85, and 0.90 on the same pilot before selecting a production value. The new adaptive anchor tests are synthetic; real-data quality and runtime are not yet validated.


## SCRAC cubic hard-capacity design (2026-11-25)

This section supersedes the earlier SCRAC raw-row-mass size-penalty description for final behavioral groups. SC-FAGC/DB-OT continues to use represented raw-row mass for transport marginals; that quantity is not the final SCRAC group capacity.

Final SCRAC groups count input component representatives. The hard capacity defaults to 200 representatives, independently of each representative's raw-row mass. A candidate may join a group only when its mixed similarity to every current member is strictly greater than the Complete-Link threshold (default 0.85), the proposed member count does not exceed capacity, and its score meets the configurable acceptance threshold (default 0.75).

The proposed score is:
`Score(i,C)=0.4*S_anchor+0.3*S_mean+0.3*S_min-P(n+1)`
where `P(k)=lambda_size*(k/capacity)^3` and lambda_size defaults to 0.20. The hard capacity is independent of the penalty and is always enforced. The score penalty uses proposed member count, never raw-row mass. Raw-row mass remains a separate reporting and SC-FAGC transport quantity.

CLI controls: `--capacity 200 --lambda-size 0.20 --acceptance-threshold 0.75`. The CLI capacity controls the final SCRAC hard member-count cap; it must not be interpreted as a cap on raw flows or represented rows.

Model run:
`python -m graph.component_sc_fagc --nodes artifacts/graph/pilot/nodes.npz --components artifacts/graph/components/components.npz --output artifacts/graph/sc_fagc/report_cubic200.json --capacity 200 --lambda-size 0.20 --acceptance-threshold 0.75 --anchor-threshold 0.85 --max-iter 500`

Independent audit:
`python -m scripts.evaluate_sc_fagc --nodes artifacts/graph/pilot/nodes.npz --components artifacts/graph/components/components.npz --report artifacts/graph/sc_fagc/report_cubic200.json --output artifacts/graph/sc_fagc/evaluation_cubic200.json --threshold 0.85 --capacity 200`

The audit reports group member-count distribution, raw-row-mass distribution separately, hard-capacity violations, Complete-Link violations, and assignment coverage. Compare capacity 100/150/200 and lambda 0/0.10/0.20/0.30, using identical input and deterministic settings. Also run multiple input orders to quantify greedy-order sensitivity. Synthetic tests do not establish real-data quality; inspect runtime and peak memory on the pilot before considering production use.


## Post-hoc attack/normal mixture audit

Class labels are strictly evaluation-only. The graph pilot, representative selection, graph construction, SC-FAGC anchors, and SCRAC assignments must not consume labels. The audit reconstructs raw-row class membership only after the final assignment exists, using the saved `mapping_*.npz` source partition paths, graph-node-to-initial-component mapping in `components.npz`, and final `assignment` in the SCRAC report.

Run from repository root:

`python -m scripts.audit_scrac_label_mix --mapping-dir artifacts/graph/pilot --components artifacts/graph/components/components.npz --assignment-report artifacts/graph/sc_fagc/report_cubic200.json --output artifacts/graph/sc_fagc/label_mix_cubic200.json`

The audit independently counts raw attack and normal rows per final group, reports attack/normal percentages, mixed/pure groups, macro group impurity, and raw-row-weighted impurity. It cross-checks reconstructed raw-row totals against `components.raw_rows` and attack counts against `components.attack_rows`; mismatches fail closed. If all mapping paths belong to `type=attacks`, the report explicitly flags an attack-only input rather than claiming clustering purity.

This audit requires the original pilot `mapping_*.npz` files. `nodes.npz` and `components.npz` alone are insufficient to reconstruct source-partition labels. The audit is post-hoc and must never be imported by the clustering pipeline.


## Label-blind 50k temporal pilot (2026-11-25)

The earlier pilot used the first five chronological 10k chunks after stable sorting. Since source paths were sorted with attacks before normal, equal-timestamp rows and the first-window cutoff could yield an attack-only pilot even when the full day contains both classes. This is a sampling defect, not evidence that the day lacks normal traffic.

The revised default samples exactly 50,000 rows from the selected day using five equal-duration time bins (10,000 rows per bin), uniform random selection without replacement within each bin, and a fixed seed. Sampling uses timestamps and row positions only. It does not read partition labels, enforce a 61.4/38.6 class ratio, or stratify by attack/normal. The natural class ratio is expected to be approximated by a representative sample, then measured only in the separate post-hoc label-mix audit.

The pilot no longer calculates label-derived diagnostics during graph construction. Mapping files retain source paths solely for the independent post-hoc audit. The 10,000-record processing batch remains unchanged. If any temporal bin has fewer than its quota, the sampler fails clearly rather than silently changing the design.

Run the 50k pilot:
`python -m graph.pilot --day 2026-06-21 --sample-rows 50000 --temporal-bins 5 --seed 42 --backend exact --output artifacts/graph/pilot_50k`

Then rebuild connected components:
`python -m graph.components --input artifacts/graph/pilot_50k --output artifacts/graph/components_50k`

Then run SC-FAGC/SCRAC with the approved settings:
`python -m graph.component_sc_fagc --nodes artifacts/graph/pilot_50k/nodes.npz --components artifacts/graph/components_50k/components.npz --output artifacts/graph/sc_fagc/report_50k_cubic200.json --capacity 200 --lambda-size 0.20 --acceptance-threshold 0.75 --anchor-threshold 0.85 --max-iter 500`

Then run post-hoc label-mix audit:
`python -m scripts.audit_scrac_label_mix --mapping-dir artifacts/graph/pilot_50k --components artifacts/graph/components_50k/components.npz --assignment-report artifacts/graph/sc_fagc/report_50k_cubic200.json --output artifacts/graph/sc_fagc/label_mix_50k_cubic200.json`

The full-day run is a separate later experiment using `--full-day --backend indexed`; it is not part of the 50k pilot. The 61.4/38.6 ratio is a population reference from the complete day, not a sampling constraint. Never use attack/normal labels to select pilot rows or build the graph.
