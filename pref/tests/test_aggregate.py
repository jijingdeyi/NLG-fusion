"""Unit tests for y / R aggregation (score.md §3.1.4–3.1.5)."""

from __future__ import annotations

import numpy as np

from pref.aggregate import aggregate_pair, dirichlet_var, empirical_pref


def test_empirical_example():
    # 7A, 0T, 3B → (0.7, 0, 0.3)
    counts = np.array([7.0, 0.0, 3.0])
    p = empirical_pref(counts)
    assert np.allclose(p, [0.7, 0.0, 0.3])


def test_pool_annotations_default():
    # Paper method: pool all clicks → 4A+4B → (0.5,0,0.5)
    y, R, meta = aggregate_pair(
        {
            'u0': ['A', 'A', 'A', 'B'],
            'u1': ['A', 'B', 'B', 'B'],
        },
    )
    assert meta['pool_annotations'] is True
    assert np.isclose(y.sum(), 1.0)
    assert np.allclose(y, [0.5, 0.0, 0.5], atol=1e-6)
    assert 0.0 <= R <= 1.0


def test_pool_counts_matter():
    # 100A + 1B → mostly A when pooled
    y, _, meta = aggregate_pair(
        {'pro': ['A'] * 100, 'novice': ['B']},
    )
    assert meta['pool_annotations'] is True
    assert y[0] > 0.95


def test_weighted_annotator_mode():
    y, _, meta = aggregate_pair(
        {'pro': ['A'] * 100, 'novice': ['B']},
        expertise={'pro': 'professional', 'novice': 'professional'},
        pool_annotations=False,
    )
    assert np.isclose(meta['lambda']['pro'], 0.5)
    assert np.isclose(meta['lambda']['novice'], 0.5)
    assert np.allclose(y, [0.5, 0.0, 0.5])


def test_nonprofessional_eta():
    y, _, meta = aggregate_pair(
        {'pro': ['A'], 'nov': ['B']},
        expertise={'pro': 'professional', 'nov': 'non-professional'},
        eta=0.5,
        pool_annotations=False,
    )
    assert np.isclose(meta['lambda']['pro'], 1 / 1.5)
    assert np.isclose(meta['lambda']['nov'], 0.5 / 1.5)
    assert y[0] > y[2]


def test_reliability_increases_with_counts():
    y_small, R_small, _ = aggregate_pair({'u': ['A', 'B']})
    y_big, R_big, _ = aggregate_pair({'u': ['A'] * 50 + ['B'] * 50})
    assert np.allclose(y_small, y_big, atol=1e-6)
    assert R_big > R_small


def test_prior_total_var_one_sixth():
    v = dirichlet_var(np.zeros(3))
    assert np.isclose(v.sum(), 1.0 / 6.0)


if __name__ == '__main__':
    test_empirical_example()
    test_pool_annotations_default()
    test_pool_counts_matter()
    test_weighted_annotator_mode()
    test_nonprofessional_eta()
    test_reliability_increases_with_counts()
    test_prior_total_var_one_sixth()
    print('all aggregate tests passed')
