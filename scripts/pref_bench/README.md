# Preference pipeline (`scripts/pref_bench`)

Public entrypoints (see root [`README.md`](../../README.md) for full usage):

| Script | Role |
|--------|------|
| `build.py` | Materialize scenes / sample pairs |
| `annotate.py` | Browser UI: new labels, re-annotation, and read-only review |
| `aggregate_preferences.py` | Soft preference aggregation |
| `browse_labels.py` | Read-only browser for labeled pairs |
| `train_scorer_stage1.py` | Train Stage-1 ResNet-18 scorer |
| `config.example.yaml` | Config template |

```bash
cp scripts/pref_bench/config.example.yaml /path/to/pref_bench/config.yaml
# edit paths, then:
python scripts/pref_bench/build.py materialize --config ...
python scripts/pref_bench/build.py sample_pairs --config ...
python scripts/pref_bench/annotate.py --config ... --port 8766
python scripts/pref_bench/aggregate_preferences.py --config ...
python scripts/pref_bench/browse_labels.py --config ... --port 8767
CUDA_VISIBLE_DEVICES=0 python scripts/pref_bench/train_scorer_stage1.py --config ... --gpu 0
```

Metric for end users: `pip install ./hpfm` then `from hpfm import hpfm`.
