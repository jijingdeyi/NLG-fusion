# Human-preference IVIF score (HPFM)

Learn and use a **source-conditioned** fusion quality score
\(S_\theta(I_{\mathrm{ir}}, I_{\mathrm{vis}}, F)\) from human pairwise preferences
for infrared–visible image fusion (IVIF).

## What is released

| Path | Role |
|------|------|
| [`hpfm/`](hpfm/) | Pip-installable metric (weights included) |
| [`pref/`](pref/) | Library used by build / annotate / train |
| [`scripts/pref_bench/`](scripts/pref_bench/) | Build pairs → annotate → aggregate → train |
| [`checkpoints/`](checkpoints/) | Released Stage-1 weights (same as in `hpfm`) |
| [`docs/DATA.md`](docs/DATA.md) | Preference-database layout |

## Setup

```bash
git clone <this-repo>
cd NLG-fusion
pip install -r requirements-pref.txt
# install a CUDA build of PyTorch from https://pytorch.org if needed

# install the metric package (recommended)
pip install ./hpfm
```

## 1) Use the metric (most users)

```python
from hpfm import hpfm

# paths or HxW arrays (uint8 / float in [0,255] or [0,1])
score = hpfm(ir, vis, fused)   # higher = better
```

Pairwise preference between two fusions of the same IR/VIS pair:

```python
from hpfm import hpfm_prefer

out = hpfm_prefer(ir, vis, fused_a, fused_b)
# out["s_a"], out["s_b"], out["delta"], out["p_a"] ≈ P(A ≻ B)
```

CLI:

```bash
hpfm ir.png vi.png fused.png
# or: python -m hpfm ir.png vi.png fused.png
```

Weights are bundled inside the package (`hpfm/src/hpfm/weights.pt`). After
retraining, refresh them with:

```bash
python hpfm/scripts/sync_weights.py --ckpt checkpoints/scorer_stage1_best.pt
```

---

## 2) Build a preference database

Copy the example config and edit paths (`out_root`, `sota_root`, dataset roots):

```bash
cp scripts/pref_bench/config.example.yaml /path/to/pref_bench/config.yaml
```

Materialize scenes/candidates and sample pairwise tasks:

```bash
python scripts/pref_bench/build.py materialize --config /path/to/pref_bench/config.yaml
python scripts/pref_bench/build.py sample_pairs --config /path/to/pref_bench/config.yaml
```

See [`docs/DATA.md`](docs/DATA.md) for the expected JSONL layout under `out_root`.

---

## 3) Annotate (browser UI)

```bash
python scripts/pref_bench/annotate.py \
  --config /path/to/pref_bench/config.yaml \
  --annotator ann0 \
  --port 8766
```

Open `http://127.0.0.1:8766`. Labels append to JSONL under `out_root`
(resume-safe). Use `--allow-reannotate` for multi-round labels on the same pair.

---

## 4) Aggregate soft preferences

```bash
python scripts/pref_bench/aggregate_preferences.py \
  --config /path/to/pref_bench/config.yaml
```

Writes aggregated supervision (soft \(y\), reliability \(R\)) consumed by training.

---

## 5) Train Stage-1 scorer

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/pref_bench/train_scorer_stage1.py \
  --config /path/to/pref_bench/config.yaml \
  --gpu 0
```

Best checkpoint (by validation) is written under `{out_root}/fit/`
(e.g. `scorer_stage1_best.pt`). Copy or sync into `checkpoints/` / `hpfm` as needed.

---

## Public scripts (only these under `scripts/`)

| Script | Purpose |
|--------|---------|
| `build.py` | Build scenes / candidates / sample pairs |
| `annotate.py` | Browser pairwise annotation UI |
| `aggregate_preferences.py` | Aggregate labels → soft targets |
| `train_scorer_stage1.py` | Train ResNet-18 preference scorer |
| `config.example.yaml` | Config template |

Other research/eval utilities are kept out of the public tree (local-only).

## License

MIT — see [`LICENSE`](LICENSE).

## Citation

Citation details will be added upon publication. See `CITATION.cff`.
