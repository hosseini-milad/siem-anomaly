import numpy as np

def audit(structural, behavioral, score, valid, n, rare_max=10):
    original=np.bincount(structural,minlength=n)
    mapped=np.bincount(behavioral,minlength=n)
    covered=np.bincount(behavioral[valid],minlength=n)
    rare=original<=rare_max
    rows=rare[structural]
    return {"rare_max_structural_rows":rare_max,"rare_representatives":int(rare.sum()),
            "rare_rows":int(rows.sum()),"rare_rows_covered":int((rows&valid).sum()),
            "rare_representatives_without_valid_behavioral_rows":int((rare&(covered==0)).sum()),
            "uncovered_rows":int((~valid).sum())},original,mapped,covered,rare
