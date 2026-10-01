# Cloud release packs

Portable preference databases (relative paths) are exported by:

```bash
python scripts/pref_bench/export_release_packs.py \
  --main /data/ykx/pref_bench \
  --roadscene /data/ykx/pref_bench_roadscene \
  --out-main /data/ykx/pref_bench_release \
  --out-roadscene /data/ykx/pref_bench_roadscene_release
```

## Current export (this machine)

| Pack | Path | Size (approx.) |
|------|------|----------------:|
| In-domain | `/data/ykx/pref_bench_release` | ~1.3 GB |
| RoadScene E1/E2 | `/data/ykx/pref_bench_roadscene_release` | ~59 MB |

Suggested cloud upload names:

```text
pref_bench_v1.zip
pref_bench_roadscene_v1.zip
```

```bash
cd /data/ykx
zip -r -0 pref_bench_v1.zip pref_bench_release
zip -r -0 pref_bench_roadscene_v1.zip pref_bench_roadscene_release
```

After users unzip, set `out_root` in `config.yaml` to the pack directory.
Code resolves relative `ir` / `vi` / `path_*` via `pref.io.resolve_pack_paths`.

Paper checkpoint is also mirrored under the repo:

```text
checkpoints/scorer_stage1_best.pt
```
