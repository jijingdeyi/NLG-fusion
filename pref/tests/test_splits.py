"""Near-duplicate grouping (score.md §3.3.3)."""

import numpy as np

from pref.splits import cluster_by_distance, repair_splits_by_group


def test_close_scenes_share_a_group():
    ids = ['FMB/a', 'FMB/b', 'FMB/c']
    dataset_of = {i: 'FMB' for i in ids}
    vectors = {
        'FMB/a': np.zeros(4),
        'FMB/b': np.array([0.01, 0.0, 0.0, 0.0]),
        'FMB/c': np.ones(4),
    }
    groups = cluster_by_distance(ids, dataset_of, vectors, threshold=0.05)
    assert groups['FMB/a'] == groups['FMB/b']
    assert groups['FMB/c'] != groups['FMB/a']


def test_repair_pulls_test_neighbor_into_train():
    scenes = [
        {'scene_id': 'FMB/a', 'split': 'train'},
        {'scene_id': 'FMB/b', 'split': 'train'},
        {'scene_id': 'FMB/c', 'split': 'test'},
    ]
    group_of = {s['scene_id']: 'FMB::FMB/a' for s in scenes}
    splits = repair_splits_by_group(scenes, group_of)
    assert splits == {'FMB/a': 'train', 'FMB/b': 'train', 'FMB/c': 'train'}


if __name__ == '__main__':
    test_close_scenes_share_a_group()
    test_repair_pulls_test_neighbor_into_train()
    print('ok')
