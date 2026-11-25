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
proportionally to cluster mass. Final assignment keeps components indivisible
and uses a default maximum group mass of 100 raw rows. Components whose own
mass exceeds 100 are reported as oversized standalone components.

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
capacity assignment, oversized-component handling, descriptor aggregation,
and end-to-end execution. They do not establish detection quality on real data.
Real-data evaluation must compare SC-FAGC with SCRAC-only and no-size-penalty
baselines, reporting component-size distribution, oversized fraction, graph
coverage, objective history, runtime, peak memory, and downstream anomaly
metrics on agreed held-out days. Weighted DB-OT and final capacity assignment
are project adaptations; the paper's anchor-level bounds alone do not imply a
hard capacity guarantee for final components.
