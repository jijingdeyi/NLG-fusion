# Released checkpoints

| File | Description |
|------|-------------|
| `scorer_stage1_best.pt` | Stage-1 ResNet-18 scalar scorer (paper §5) |
| `scorer_stage1_report.json` | Training / selection metadata |

## Use as a metric (recommended)

```python
from pref.metric import preference_score, preference_prefer

# same spirit as EN / MI / Qabf: one call → one number
s = preference_score('ir.png', 'vi.png', 'fused.png')

# or numpy HxW arrays in [0,255] / [0,1]
# s = preference_score(ir, vi, fused)

# pairwise: P(A ≻ B) ≈ sigmoid(s_A - s_B)
out = preference_prefer('ir.png', 'vi.png', 'a.png', 'b.png')
# out['s_a'], out['s_b'], out['delta'], out['p_a']
```

Weights load once from `checkpoints/scorer_stage1_best.pt` (override with
`ckpt=`). For many images, keep a handle:

```python
from pref.metric import PreferenceScorer
scorer = PreferenceScorer(device='cuda:0')  # or 'cpu'
s = scorer(ir, vi, fused)
```

## Paper eval scripts

```bash
python scripts/pref_bench/eval_stage1.py \
  --config /path/to/pref_bench/config.yaml \
  --ckpt checkpoints/scorer_stage1_best.pt --gpu 0
```

Complexity benchmark:

```bash
python scripts/pref_bench/bench_complexity.py --ckpt checkpoints/scorer_stage1_best.pt
```
