# Human-preference IVIF score (HPFM)

Score an infrared–visible fusion from human pairwise preferences:

`score(IR, visible, fusion)` → one number. Higher is better. The number ranks fusions of the **same** infrared and visible pair. It is not a score you compare across different scenes.

The repository has two pieces:

- [`hpfm/`](hpfm/) is the released scorer you can install and call. Its weights are the published grayscale model: infrared, visible, and fusion are resized together to 224 and scored by one ResNet-18.
- [`pref/`](pref/) and [`scripts/pref_bench/`](scripts/pref_bench/) build a preference database, collect labels in a browser, and train the current baseline. That baseline keeps the vote-share target and adds two input changes: visible and fusion are read as Y, Cb, Cr, and the network sees both a 224 view of the whole image and a complete cover of 224 tiles. Overlapping tiles split the shared area by how many tiles cover each pixel. The released `hpfm` weights are not this baseline; training writes a separate checkpoint.

## Setup

```bash
git clone <this-repo>
cd NLG-fusion
pip install -r requirements-pref.txt
# install a CUDA build of PyTorch from https://pytorch.org if needed
pip install ./hpfm
```

## Score a fusion

```python
from hpfm import hpfm, hpfm_prefer

score = hpfm(ir, vis, fused)   # path or HxW array; higher = better

out = hpfm_prefer(ir, vis, fused_a, fused_b)
# out["s_a"], out["s_b"], out["delta"], out["p_a"]
```

```bash
hpfm ir.png vi.png fused.png
```

Weights are `hpfm/src/hpfm/weights.pt`. To point the package at another grayscale checkpoint in the same format:

```bash
python hpfm/scripts/sync_weights.py --ckpt checkpoints/scorer_stage1_best.pt
```

## Build a preference database

```bash
cp scripts/pref_bench/config.example.yaml /path/to/pref_bench/config.yaml
# edit out_root, sota_root, and the dataset roots
python scripts/pref_bench/build.py make_gallery --config /path/to/pref_bench/config.yaml
# open the gallery, click scenes, and download selection_<dataset>.json
python scripts/pref_bench/build.py apply_selection \
  --config /path/to/pref_bench/config.yaml \
  --selection /path/to/selection_FMB.json
python scripts/pref_bench/build.py materialize --config /path/to/pref_bench/config.yaml
python scripts/pref_bench/build.py sample_pairs --config /path/to/pref_bench/config.yaml
```

`make_gallery` writes thumbnails. Click a scene to keep it, then download `selection_<dataset>.json`. Repeat `--selection` once per dataset. `materialize` copies infrared, visible, and fusion images into the database. `sample_pairs` writes the comparisons. Fusion results are read from `{sota_root}/{Dataset}/{method}/` and must use the same filenames as the infrared and visible images.

To skip the gallery and draw scenes at random:

```bash
python scripts/pref_bench/build.py all_random --config /path/to/pref_bench/config.yaml
```

Do not run `sample_scenes`, `apply_selection`, or `sample_pairs` again on a database that already has labels. Those commands rewrite the scene or pair list.

```text
pref_bench/
  config.yaml
  scenes.jsonl
  pairs.jsonl
  annotations.jsonl
  labels_ann0.jsonl
  supervision.jsonl
  scenes/.../ir|vi|fusions/
```

A directory listed under `browse_packs` that contains `pairs.jsonl` is another database. The annotator can label it and view it. A path that is a file is treated as that file’s parent directory.

## Annotate

```bash
python scripts/pref_bench/annotate.py \
  --config /path/to/pref_bench/config.yaml \
  --annotator ann0 \
  --port 8766
```

Open the printed address. Pick the database, then one of the three tasks. On a labeling screen the keys are `1` A, `2` or Space tie, `3` B, and Escape back. In the viewer, Left and Right move one pair.

- **New annotation** stores the first vote for pairs this person has not labeled. The same screen can add scenes from a folder that contains `ir/` and `vi/`, or add one fusion method from a results folder. Adding images asks you to confirm before it writes. Existing votes stay.
- **Re-annotate** asks for one more vote, at most 40 pairs. The order is model disagreements, then preference cycles, then pairs whose votes disagree, then other labeled pairs. An unfinished batch resumes, and votes already submitted in that batch are kept. A later batch can include a pair again when the model still disagrees, the pair is still on a cycle, or the votes still disagree. Other pairs are asked once.
- **View labeled results** shows the pooled preference. A and B are the stored candidates. Filters narrow the list by dataset and method. **Re-annotate this pair** appends one vote on the pair that is open.

Votes are appended. Closing the browser stops the server. Votes that were submitted are already on disk. When new votes exist, stopping the server rebuilds `supervision.jsonl`.

## Aggregate

```bash
python scripts/pref_bench/aggregate_preferences.py \
  --config /path/to/pref_bench/config.yaml
```

This writes `supervision.jsonl`. Each pair keeps the vote share `y_A`, `y_T`, `y_B`. The training target is `q = y_A + 0.5 y_T`. Reliability `R` is stored for checks and is not a training weight.

## Train the baseline

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/pref_bench/train_scorer_stage1.py \
  --config /path/to/pref_bench/config.yaml \
  --gpu 0
```

The run reads native images. Infrared stays gray. Visible and fusion are Y, Cb, Cr, with Y taken from the grayscale image. The network scores a 224 view of the whole image and every 224 tile; edge tiles overlap so the cover has no blank border. The loss is the vote share above. The best checkpoint is `{out_root}/fit/scorer_baseline_best.pt`. It does not replace `checkpoints/scorer_stage1_best.pt` or the weights inside `hpfm`.

The annotator’s model-disagreement queue still uses the released grayscale checkpoint. It reads `scorer_ckpt` from the config when that key is set, and otherwise `checkpoints/scorer_stage1_best.pt`.

## Tests

```bash
PYTHONPATH=. python -m pytest pref/tests -q
```

## License

MIT — see [`LICENSE`](LICENSE).

## Citation

Citation details will be added upon publication. See `CITATION.cff`.
