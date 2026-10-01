# HPFM — Human-Preference Fusion Metric

Infrared–visible fusion quality score aligned with human pairwise preference
(paper \(S_\theta\), Stage-1 ResNet-18).

## Install

```bash
pip install hpfm
# or from this repo:
pip install ./hpfm
```

Weights (~45 MB) ship inside the package; no extra download step.

## Usage

```python
from hpfm import hpfm

score = hpfm(ir, vis, fused)   # paths or HxW arrays; higher is better
```

Pairwise:

```python
from hpfm import hpfm_prefer

out = hpfm_prefer(ir, vis, fused_a, fused_b)
# out["s_a"], out["s_b"], out["delta"], out["p_a"]
```

CLI:

```bash
hpfm ir.png vi.png fused.png
```

## Dependencies

`torch`, `torchvision`, `numpy`, and `opencv-python` or `Pillow`.

## Sync weights (maintainers)

```bash
python hpfm/scripts/sync_weights.py --ckpt checkpoints/scorer_stage1_best.pt
```
