import numpy as np
import pytest

from graph.pilot import temporal_sample_indices


def test_temporal_sampler_is_reproducible_and_exact_size():
    times = np.arange(1000, dtype=np.int64)
    a = temporal_sample_indices(times, 100, 5, 42)
    b = temporal_sample_indices(times, 100, 5, 42)
    assert len(a) == 100
    assert np.array_equal(a, b)
    assert np.all(np.diff(a) > 0)


def test_temporal_sampler_uses_equal_duration_bins():
    times = np.arange(1000, dtype=np.int64)
    chosen = temporal_sample_indices(times, 100, 5, 42)
    edges = np.linspace(times.min(), times.max() + 1, 6, dtype=np.int64)
    counts = [int(((times[chosen] >= edges[i]) & (times[chosen] < edges[i + 1])).sum()) for i in range(5)]
    assert counts == [20, 20, 20, 20, 20]


def test_temporal_sampler_rejects_sparse_temporal_bin():
    times = np.concatenate([np.arange(0, 10), np.arange(100, 110)]).astype(np.int64)
    with pytest.raises(ValueError, match="temporal bin"):
        temporal_sample_indices(times, 10, 5, 7)


def test_temporal_sampler_requires_divisible_quota():
    with pytest.raises(ValueError, match="divisible"):
        temporal_sample_indices(np.arange(100), 11, 5, 1)
