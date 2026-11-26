"""Audit the observed representative graph, not a raw-flow graph."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def contract(a, labels, n):
    s=csr_matrix((np.ones(len(labels)),(np.arange(len(labels)),labels)),shape=(len(labels),n))
    return (s.T@a@s).tocsr()

def audit(a, labels, n):
    q=contract(a,labels,n)
    _,cc=connected_components(a,directed=False)
    splits=sum(len(np.unique(cc[labels==j]))>1 for j in np.unique(labels))
    co=a.tocoo()
    external=labels[co.row]!=labels[co.col]
    return q,{"nodes":a.shape[0],"edges":a.nnz//2,"weight":float(a.sum()),
              "contracted_weight":float(q.sum()),"internal_weight":float(q.diagonal().sum()),
              "external_weight":float(co.data[external].sum()),
              "cross_original_component_groups":int(splits),
              "isolated_nodes":int(np.count_nonzero(a.getnnz(axis=1)==0))}
