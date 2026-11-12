# Human-preference IVIF score (HPFM)

Learn and use a **source-conditioned** fusion quality score
\(S_\theta(I_{\mathrm{ir}}, I_{\mathrm{vis}}, F)\) from human pairwise preferences
for infrared–visible image fusion (IVIF).

## What is released

| Path | Role |
|------|------|
| [`hpfm/`](hpfm/) | Pip-installable metric (weights included) |
| [`pref/`](pref/) | Library for build / annotate / aggregate / train |
| [`scripts/pref_bench/`](scripts/pref_bench/) | CLI: build → annotate → aggregate → browse → train |
| [`checkpoints/`](checkpoints/) | Released Stage-1 weights (same as in `hpfm`) |

## Setup

```bash
git clone <this-repo>
cd NLG-fusion
pip install -r requirements-pref.txt
# install a CUDA build of PyTorch from https://pytorch.org if needed
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

```bash
hpfm ir.png vi.png fused.png
# or: python -m hpfm ir.png vi.png fused.png
```

Weights live in `hpfm/src/hpfm/weights.pt`. After retraining:

```bash
python hpfm/scripts/sync_weights.py --ckpt checkpoints/scorer_stage1_best.pt
```

---

## 2) Build a preference database

Copy the example config and edit `out_root`, `sota_root`, and dataset roots:

```bash
cp scripts/pref_bench/config.example.yaml /path/to/pref_bench/config.yaml
python scripts/pref_bench/build.py materialize --config /path/to/pref_bench/config.yaml
python scripts/pref_bench/build.py sample_pairs --config /path/to/pref_bench/config.yaml
```

You need aligned IR–VI datasets (e.g. FMB / M3FD / MSRS) and fusion outputs under
`{sota_root}/{Dataset}/{method}/` with the same filenames as IR/VI.

### Layout under `out_root`

```text
pref_bench/
  config.yaml
  scenes.jsonl
  candidates.jsonl
  pairs.jsonl
  annotators.jsonl
  annotations.jsonl
  labels_ann0.jsonl      # append-only UI labels
  supervision.jsonl      # aggregated (y_A, y_T, y_B, R)
  scenes/.../ir|vi|fusions/
  fit/scorer_stage1_best.pt
```

---

## 3) Annotate (browser UI)

```bash
python scripts/pref_bench/annotate.py \
  --config /path/to/pref_bench/config.yaml \
  --annotator ann0 \
  --port 8766
```

Open `http://127.0.0.1:8766`. Choose a database, then a task. Back or Escape returns to that menu.

- **New annotation** writes a first label on pairs this user has not judged. The same screen can add scenes from a folder with `ir/` and `vi/`, or add one method from a results folder. Infrared and visible images are enough: reference fusions are built automatically, and missing method projects are skipped. It appends comparisons and keeps existing labels. Do not re-run `sample_scenes` or `sample_pairs` on a database that already has labels; those commands rewrite the lists.
- **Re-annotate: routine check** reviews preference-cycle edges first. If there is no cycle, it takes a small set of the least reliable pairs, or a random sample when reliability does not vary.
- **Re-annotate: model disagreements** reviews decisive labels where the trained scorer gives the other candidate probability above 0.6.
- **View labeled results** shows the aggregated preference, with A and B as the stored candidates. RoadScene is filtered as seen methods (E1) and unseen methods (E2). Dataset and method filters narrow the list. **Re-annotate this pair** appends one more vote on the open pair. Earlier votes stay and are counted together.

Earlier labels are kept. Labels append to JSONL under the chosen database (resume-safe). A `browse_packs` entry whose directory contains `pairs.jsonl` is an additional database, not only a view.

---

## 4) Aggregate soft preferences

```bash
python scripts/pref_bench/aggregate_preferences.py \
  --config /path/to/pref_bench/config.yaml
```

Writes `supervision.jsonl` (soft \(y\), reliability \(R\)) for training.

---

## 5) Review labeled pairs

```bash
python scripts/pref_bench/browse_labels.py \
  --config /path/to/pref_bench/config.yaml \
  --port 8767
```

Open `http://127.0.0.1:8767`. The page is read-only and shows IR, VI, and the two
stored candidates. A and B follow `supervision.jsonl`, not the left/right order
used while annotating. Arrow keys move between pairs. Another aggregated file
can be added with `--extra name=/path/to/supervision.jsonl`, or under
`browse_packs` in the config. The same server is `build.py browse`.

---

## 6) Train Stage-1 scorer

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/pref_bench/train_scorer_stage1.py \
  --config /path/to/pref_bench/config.yaml \
  --gpu 0
```

Best checkpoint (by validation) → `{out_root}/fit/scorer_stage1_best.pt`.
Copy/sync into `checkpoints/` and `hpfm` as needed.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `build.py` | Build scenes / candidates / sample pairs |
| `annotate.py` | Browser pairwise annotation UI |
| `aggregate_preferences.py` | Aggregate labels → soft targets |
| `browse_labels.py` | Read-only browser for labeled pairs |
| `train_scorer_stage1.py` | Train ResNet-18 preference scorer |
| `config.example.yaml` | Config template |

```bash
PYTHONPATH=. python -m pytest pref/tests -q
```

## License

MIT — see [`LICENSE`](LICENSE).

## Citation

Citation details will be added upon publication. See `CITATION.cff`.
