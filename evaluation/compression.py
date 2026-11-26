from pathlib import Path
import numpy as np

def summarize_compression(raw, reps, atoms, daily, source_bytes, artifact_bytes):
    return {"counts":[raw,reps,atoms,daily],"count_ratio":raw/daily,
            "source_bytes":source_bytes,"artifact_bytes_including_mappings":artifact_bytes,
            "storage_ratio":source_bytes/artifact_bytes if artifact_bytes else None}

def distribution(values):
    a=np.asarray(values)
    return {"min":float(a.min()),"max":float(a.max()),"mean":float(a.mean()),
            "p50":float(np.median(a)),"p95":float(np.quantile(a,.95))}
