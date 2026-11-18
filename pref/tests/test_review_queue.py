"""The single re-annotation queue keeps its priority and its cap."""
from pref.qc import order_review_batch, votes_disagree


def test_review_batch_fills_higher_tiers_first():
    chosen = order_review_batch(
        ['m1', 'm2'],
        ['c1', 'm1'],
        ['s1', 's2'],
        ['o1', 'o2', 'o3'],
        limit=4,
    )
    assert chosen == [('m1', 'model'), ('m2', 'model'), ('c1', 'cycle'), ('s1', 'split')]


def test_split_votes_need_two_different_choices():
    assert votes_disagree((0.5, 0.0, 0.5), 2)
    assert votes_disagree((0.5, 0.5, 0.0), 2)
    assert not votes_disagree((1.0, 0.0, 0.0), 2)
    assert not votes_disagree((1.0, 0.0, 0.0), 1)
