# Data layout

Preference data are **not** required to live inside git. Portable release packs
(relative paths) can be uploaded to cloud storage; see [`PACKS.md`](PACKS.md).

## In-domain preference DB (`pref_bench`)

Used for train / val / test in §5.1–5.5.

```
pref_bench/
  config.yaml
  scenes.jsonl
  candidates.jsonl
  pairs.jsonl
  annotators.jsonl
  annotations.jsonl
  labels_ann0.jsonl          # append-only UI labels
  supervision.jsonl          # aggregated (y_A, y_T, y_B, R)
  features.jsonl             # optional classical metrics cache
  scenes/{FMB,M3FD,MSRS}/... # ir/, vi/, fusions/
  fit/scorer_stage1_best.pt  # trained checkpoint
```

Scene-held-out split (paper numbers): **136 / 35 / 29** scenes,
**1152 / 292 / 239** pairs (FMB + M3FD + MSRS).

Public methods used as candidates (aliases `method_a`…`method_d` in the main
pool): TarDAL, CDDFuse, SwinFusion, DDFM, plus max / \(m_{\mathrm{hf}}\) priors and
exposure degradations for QC.

## RoadScene companion pack (`pref_bench_roadscene`)

Separate folder for §5.6 zero-shot protocols. **Not** mixed into the in-domain
train/val/test tables.

```
pref_bench_roadscene/
  config.yaml
  pairs.jsonl
  pairs_e1.jsonl             # seen methods × RoadScene
  pairs_e2.jsonl             # unseen methods × RoadScene
  annotations.jsonl
  labels_ann0.jsonl
  supervision.jsonl
  scenes/RoadScene/...
```

Paper setting: **30 scenes per protocol**, 3 pairs/scene → 90 pairs each
(85 decisive after dropping ties).

| Protocol | Methods |
|----------|---------|
| E1 | TarDAL, CDDFuse, SwinFusion, DDFM |
| E2 | EMMA, TextIF, MetaFusion, DCEvo |

## External dependencies

| Asset | Used for |
|-------|----------|
| FMB / M3FD / MSRS (aligned IR-VI) | Main preference scenes |
| RoadScene test | Zero-shot scenes |
| Third-party fusion outputs under `sota_root/{Dataset}/{method}/` | Candidate fusions |

Placeholders in `scripts/pref_bench/config*.example.yaml` use `/path/to/...`.
