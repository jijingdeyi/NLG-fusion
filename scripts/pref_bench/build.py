#!/usr/bin/env python3
"""Preference-bench builder.

Recommended flow (click-to-select sources, 4 methods + 2 exposures):
  1) make_gallery   — build IR|VI thumbnails + HTML; open in browser, click until n_scenes
  2) apply_selection — write scenes.jsonl from gallery/selection.json
  3) materialize     — copy IR/VI (real files), copy 4 SOTA fusions, synthesize over/under-exp
  4) sample_pairs    — sparse pairwise JSONL for annotation UI
  5) annotate        — serve labeling UI (or: python scripts/pref_bench/annotate.py)

Also supported: sample_scenes (random, no clicking).

Over/under-exposure: scale luminance (Y in YCrCb for color; gray for mono), clip to 0–255.
SOTA / ckpt fusions are kept as color BGR when the source is color.
"""

from __future__ import annotations

import html
import json
import random
import shutil
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pref.ivif_io import list_paired_samples
from pref.fusion_prior import build_fusion_prior
import torch


def load_config(path: Path) -> dict:
    with path.open(encoding='utf-8') as f:
        return yaml.safe_load(f)


def _read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _write_gray(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def _write_img(path: Path, img: np.ndarray) -> None:
    """Write gray (H,W) or BGR color (H,W,3) as-is."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def _copy_file(src: Path, dst: Path) -> None:
    """Always materialize a real file (no symlinks)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.copy2(src, dst)


def _assign_splits(scenes: list[dict], cfg: dict) -> None:
    """§3.3.3: near-duplicate source images share one split, per dataset."""
    from pref.splits import DEFAULT_NEAR_DUP_MAE, apply_group_split, resolve_source_paths

    split_cfg = cfg['split']
    ratios = (
        float(split_cfg['train']),
        float(split_cfg['val']),
        float(split_cfg.get('test', 1.0 - float(split_cfg['train']) - float(split_cfg['val']))),
    )
    if scenes and all(resolve_source_paths(sc) for sc in scenes):
        apply_group_split(
            scenes,
            threshold=float(cfg.get('near_dup_mae', DEFAULT_NEAR_DUP_MAE)),
            ratios=ratios,
            seed=int(cfg.get('seed', 443)),
            repair=False,
        )
        return
    seed = int(cfg.get('seed', 443))
    rng = random.Random(seed)
    order = scenes.copy()
    rng.shuffle(order)
    n = len(order)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    for i, sc in enumerate(order):
        if i < n_train:
            sc['split'] = 'train'
        elif i < n_train + n_val:
            sc['split'] = 'val'
        else:
            sc['split'] = 'test'


def _write_scenes(out: Path, scenes: list[dict]) -> None:
    path = out / 'scenes.jsonl'
    with path.open('w', encoding='utf-8') as f:
        for sc in scenes:
            f.write(json.dumps(sc, ensure_ascii=False) + '\n')
    print(f'[scenes] wrote {len(scenes)} → {path}')


def _load_scenes(out: Path) -> list[dict]:
    path = out / 'scenes.jsonl'
    rows = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _degrade(img: np.ndarray, mode: str, gain: float = 1.4) -> np.ndarray:
    """Synthetic exposure: multiply (over) or divide (under), then clip.

    Color BGR: adjust Y in YCrCb so chroma stays; gray: scale directly.
    """
    if img.ndim == 3 and img.shape[2] == 3:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y = ycrcb[:, :, 0].astype(np.float32)
        if mode == 'overexp':
            y = np.clip(y * gain, 0, 255)
        elif mode == 'underexp':
            y = np.clip(y / max(gain, 1e-3), 0, 255)
        else:
            raise ValueError(f'unknown degrade mode {mode}')
        ycrcb[:, :, 0] = y.astype(np.uint8)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    x = img.astype(np.float32)
    if mode == 'overexp':
        x = np.clip(x * gain, 0, 255)
    elif mode == 'underexp':
        x = np.clip(x / max(gain, 1e-3), 0, 255)
    else:
        raise ValueError(f'unknown degrade mode {mode}')
    return x.astype(np.uint8)


def _prior_max(ir: np.ndarray, vi: np.ndarray) -> np.ndarray:
    return np.maximum(ir, vi)


def _prior_mhf(ir: np.ndarray, vi: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Luminance prior: hf_alpha=1 → m_hf = LP(max)+grad-selected HF."""
    t_ir = torch.from_numpy(ir.astype(np.float32) / 255.0)[None, None]
    t_vi = torch.from_numpy(vi.astype(np.float32) / 255.0)[None, None]
    out = build_fusion_prior(t_ir, t_vi, hf_alpha=float(alpha))
    return (out[0, 0].numpy() * 255.0).clip(0, 255).astype(np.uint8)


def _colorize_y_with_vi(y: np.ndarray, vi_bgr: np.ndarray) -> np.ndarray:
    """Put luminance Y onto VI chroma (CrCb) for color display."""
    if vi_bgr.ndim != 3 or vi_bgr.shape[2] != 3:
        return y
    if y.shape[:2] != vi_bgr.shape[:2]:
        y = cv2.resize(y, (vi_bgr.shape[1], vi_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    ycrcb = cv2.cvtColor(vi_bgr, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = y
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def _sota_dir(cfg: dict, ds_name: str, spec: dict) -> Path:
    """Resolve fusion folder for a SOTA slot.

    Priority:
      1) datasets.<ds>.sota.<slot_name>
      2) {sota_root}/{ds}/{method}   if spec has ``method`` and cfg has ``sota_root``
      3) spec['dir']                 (legacy single-dataset path; often LLVIP-only — avoid)
    """
    ds = cfg['datasets'].get(ds_name, {})
    overrides = ds.get('sota') or {}
    if spec['name'] in overrides:
        return Path(overrides[spec['name']])
    method = spec.get('method')
    sota_root = cfg.get('sota_root')
    if method and sota_root:
        return Path(sota_root) / ds_name / str(method)
    if 'dir' in spec:
        return Path(spec['dir'])
    raise KeyError(
        f'no SOTA path for slot {spec["name"]} on {ds_name}; '
        f'set sota_root+method or datasets.{ds_name}.sota.{spec["name"]}'
    )


def _find_fusion_file(folder: Path, name: str) -> Path | None:
    src = folder / name
    if src.is_file():
        return src
    alt = folder / (Path(name).stem + '.png')
    if alt.is_file():
        return alt
    for suf in ('.jpg', '.bmp', '.JPG', '.PNG'):
        alt = folder / (Path(name).stem + suf)
        if alt.is_file():
            return alt
    return None


def cmd_sample_scenes(cfg: dict) -> None:
    """Random sample (no UI)."""
    out = Path(cfg['out_root'])
    out.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get('seed', 443))
    scenes = []
    for ds_name, ds in cfg['datasets'].items():
        root = Path(ds['root'])
        n = int(ds['n_scenes'])
        names = list_paired_samples(root)
        rng = random.Random(seed + hash(ds_name) % 10_000)
        if n > len(names):
            raise ValueError(f'{ds_name}: need {n}, only {len(names)} in {root}')
        for name in rng.sample(names, n):
            scenes.append({
                'scene_id': f'{ds_name}/{Path(name).stem}',
                'dataset': ds_name,
                'root': str(root),
                'filename': name,
            })
    _assign_splits(scenes, cfg)
    _write_scenes(out, scenes)


def cmd_make_gallery(cfg: dict, dataset: str | None = None) -> None:
    """Build clickable IR|VI gallery HTML per dataset."""
    out = Path(cfg['out_root'])
    seed = int(cfg.get('seed', 443))
    thumb_h = 96
    datasets = cfg['datasets']
    if dataset:
        datasets = {dataset: datasets[dataset]}

    for ds_name, ds in datasets.items():
        root = Path(ds['root'])
        n_need = int(ds['n_scenes'])
        pool_n = int(ds.get('gallery_pool', 0))
        names = list_paired_samples(root)
        rng = random.Random(seed + hash(ds_name) % 10_000)
        if pool_n > 0 and pool_n < len(names):
            names = rng.sample(names, pool_n)
        else:
            names = list(names)
            rng.shuffle(names)

        gdir = out / 'gallery' / ds_name
        tdir = gdir / 'thumbs'
        tdir.mkdir(parents=True, exist_ok=True)
        items = []
        for name in names:
            ir = cv2.imread(str(root / 'ir' / name), cv2.IMREAD_GRAYSCALE)
            vi = cv2.imread(str(root / 'vi' / name), cv2.IMREAD_COLOR)
            if ir is None or vi is None:
                continue
            # IR as BGR gray, side-by-side with VI
            ir_bgr = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
            h = thumb_h
            def _resize(im):
                scale = h / im.shape[0]
                return cv2.resize(im, (max(1, int(im.shape[1] * scale)), h))
            strip = np.hstack([_resize(ir_bgr), _resize(vi)])
            stem = Path(name).stem
            thumb_rel = f'thumbs/{stem}.jpg'
            cv2.imwrite(str(tdir / f'{stem}.jpg'), strip, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            items.append({'filename': name, 'stem': stem, 'thumb': thumb_rel})

        catalog = {
            'dataset': ds_name,
            'root': str(root),
            'n_scenes': n_need,
            'items': items,
        }
        (gdir / 'catalog.json').write_text(
            json.dumps(catalog, ensure_ascii=False), encoding='utf-8',
        )

        # Self-contained HTML: click cards, download selection.json
        cards = []
        for it in items:
            cards.append(
                f'<div class="card" data-filename="{html.escape(it["filename"])}" '
                f'data-stem="{html.escape(it["stem"])}" onclick="toggle(this)">'
                f'<img src="{html.escape(it["thumb"])}" alt="">'
                f'<div class="cap">{html.escape(it["stem"])}</div></div>'
            )
        page = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<title>Select scenes — {html.escape(ds_name)}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 16px; background: #111; color: #eee; }}
.bar {{ position: sticky; top: 0; background: #1a1a1a; padding: 12px; z-index: 2;
        display: flex; gap: 12px; align-items: center; border-bottom: 1px solid #333; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 10px; margin-top: 12px; }}
.card {{ border: 3px solid #333; border-radius: 8px; cursor: pointer; overflow: hidden; background: #000; }}
.card.sel {{ border-color: #3d8bfd; box-shadow: 0 0 0 2px #3d8bfd55; }}
.card img {{ width: 100%; display: block; }}
.cap {{ font-size: 12px; padding: 4px 8px; color: #aaa; }}
button {{ padding: 8px 14px; cursor: pointer; }}
#count {{ font-weight: 700; color: #3d8bfd; }}
</style>
</head>
<body>
<div class="bar">
  <strong>{html.escape(ds_name)}</strong>
  <span>已选 <span id="count">0</span> / 目标 {n_need}</span>
  <button onclick="downloadSel()">下载 selection.json</button>
  <button onclick="clearSel()">清空</button>
  <span style="opacity:.7;font-size:13px">左 IR · 右 VIS · 点击选/取消 · 选满后下载，再 apply_selection</span>
</div>
<div class="grid">
{''.join(cards)}
</div>
<script>
const NEED = {n_need};
const DS = {json.dumps(ds_name)};
const ROOT = {json.dumps(str(root))};
const selected = new Set(JSON.parse(localStorage.getItem('pref_'+DS) || '[]'));
function refresh() {{
  document.querySelectorAll('.card').forEach(c => {{
    c.classList.toggle('sel', selected.has(c.dataset.filename));
  }});
  document.getElementById('count').textContent = selected.size;
  localStorage.setItem('pref_'+DS, JSON.stringify([...selected]));
}}
function toggle(el) {{
  const f = el.dataset.filename;
  if (selected.has(f)) selected.delete(f);
  else {{
    if (selected.size >= NEED) {{ alert('已达目标 '+NEED+'，先取消一张再选'); return; }}
    selected.add(f);
  }}
  refresh();
}}
function clearSel() {{ selected.clear(); refresh(); }}
function downloadSel() {{
  const filenames = [...selected];
  if (filenames.length === 0) {{ alert('还没选'); return; }}
  const payload = {{
    dataset: DS,
    root: ROOT,
    n_scenes: NEED,
    filenames,
  }};
  const blob = new Blob([JSON.stringify(payload, null, 2)], {{type: 'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'selection_' + DS + '.json';
  a.click();
}}
refresh();
</script>
</body>
</html>
"""
        index = gdir / 'index.html'
        index.write_text(page, encoding='utf-8')
        print(f'[gallery] {ds_name}: {len(items)} previews → {index}')
        print(f'         On a remote server, either:')
        print(f'           A) python scripts/pref_bench/build.py serve_gallery --config ... --port 8765')
        print(f'              then SSH tunnel: ssh -L 8765:127.0.0.1:8765 user@server')
        print(f'              open http://127.0.0.1:8765/{ds_name}/ on your laptop')
        print(f'           B) python scripts/pref_bench/build.py pick_cli --config ... --dataset {ds_name}')
        print(f'              (terminal y/n; open thumb paths in Cursor to preview)')


def cmd_serve_gallery(cfg: dict, port: int = 8765) -> None:
    """HTTP-serve gallery/ so a laptop browser can open via SSH -L."""
    import http.server
    import socketserver

    out = Path(cfg['out_root']) / 'gallery'
    if not out.is_dir():
        raise SystemExit(f'no gallery at {out}; run make_gallery first')

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(out), **kwargs)

    # Allow reuse
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(('0.0.0.0', port), Handler) as httpd:
        print(f'[serve] gallery at http://127.0.0.1:{port}/')
        for ds in cfg['datasets']:
            print(f'         http://127.0.0.1:{port}/{ds}/')
        print(f'[serve] From your laptop: ssh -L {port}:127.0.0.1:{port} <user>@<server>')
        print(f'        Or in Cursor: Ports → Forward {port}')
        print('[serve] Ctrl+C to stop')
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print('\n[serve] stopped')


def cmd_pick_cli(cfg: dict, dataset: str) -> None:
    """Terminal picker: y/n/q/u. Writes selection_<DS>.json under out_root/gallery/."""
    if dataset not in cfg['datasets']:
        raise SystemExit(f'unknown dataset {dataset}; choose from {list(cfg["datasets"])}')
    out = Path(cfg['out_root'])
    gdir = out / 'gallery' / dataset
    catalog_path = gdir / 'catalog.json'
    if not catalog_path.is_file():
        raise SystemExit(f'run make_gallery for {dataset} first ({catalog_path} missing)')

    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    n_need = int(catalog['n_scenes'])
    root = catalog['root']
    items = catalog['items']
    selected: list[str] = []
    skipped = 0

    print(f'[pick_cli] {dataset}: need {n_need} / pool {len(items)}')
    print('  commands: y=选中  n=跳过  u=撤销上一个  q=保存并退出  s=显示已选数量')
    print('  预览: 在 Cursor 里打开下面打印的 thumb 路径（远程文件可直接看图）\n')

    i = 0
    while i < len(items) and len(selected) < n_need:
        it = items[i]
        thumb = gdir / it['thumb']
        print(f'---- [{len(selected)}/{n_need}] {i+1}/{len(items)}  {it["stem"]}')
        print(f'  thumb: {thumb}')
        print(f'  ir:    {Path(root) / "ir" / it["filename"]}')
        print(f'  vi:    {Path(root) / "vi" / it["filename"]}')
        try:
            ans = input('  [y/n/u/q/s]? ').strip().lower()
        except EOFError:
            ans = 'q'
        if ans in ('y', 'yes', ''):
            # empty enter = skip? better require explicit y
            if ans == '':
                print('  (空回车忽略，请输入 y 或 n)')
                continue
            selected.append(it['filename'])
            i += 1
        elif ans in ('n', 'no'):
            skipped += 1
            i += 1
        elif ans == 'u':
            if selected:
                removed = selected.pop()
                print(f'  undone {removed}')
                # step back to that item if possible
                for j in range(i - 1, -1, -1):
                    if items[j]['filename'] == removed:
                        i = j
                        break
            else:
                print('  nothing to undo')
        elif ans == 's':
            print(f'  selected={len(selected)} skipped_so_far≈{skipped}')
        elif ans == 'q':
            break
        else:
            print('  use y / n / u / q / s')

    payload = {
        'dataset': dataset,
        'root': root,
        'n_scenes': n_need,
        'filenames': selected,
    }
    out_json = gdir / f'selection_{dataset}.json'
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[pick_cli] saved {len(selected)}/{n_need} → {out_json}')
    if len(selected) < n_need:
        print(f'[pick_cli] WARNING: fewer than target; re-run pick_cli or lower n_scenes')
    else:
        print(f'[pick_cli] next: apply_selection --selection {out_json}')


def cmd_apply_selection(cfg: dict, selection_paths: list[Path]) -> None:
    """Merge one or more selection_*.json into scenes.jsonl."""
    out = Path(cfg['out_root'])
    out.mkdir(parents=True, exist_ok=True)
    existing = []
    scenes_path = out / 'scenes.jsonl'
    if scenes_path.is_file():
        existing = _load_scenes(out)

    by_ds = {s['dataset']: [] for s in existing}
    for s in existing:
        by_ds.setdefault(s['dataset'], []).append(s)

    for sp in selection_paths:
        data = json.loads(sp.read_text(encoding='utf-8'))
        ds_name = data['dataset']
        root = data.get('root') or cfg['datasets'][ds_name]['root']
        n_need = int(data.get('n_scenes') or cfg['datasets'][ds_name]['n_scenes'])
        filenames = data['filenames']
        if len(filenames) < n_need:
            print(f'[warn] {ds_name}: selected {len(filenames)} < target {n_need} (still accepting)')
        if len(filenames) > n_need:
            filenames = filenames[:n_need]
            print(f'[warn] {ds_name}: truncated to {n_need}')
        rows = []
        for name in filenames:
            rows.append({
                'scene_id': f'{ds_name}/{Path(name).stem}',
                'dataset': ds_name,
                'root': str(root),
                'filename': name,
            })
        by_ds[ds_name] = rows
        print(f'[apply] {ds_name}: {len(rows)} scenes from {sp}')

    # keep datasets not updated
    for ds_name in cfg['datasets']:
        by_ds.setdefault(ds_name, [])

    scenes = []
    for ds_name in cfg['datasets']:
        scenes.extend(by_ds.get(ds_name, []))
    if not scenes:
        raise SystemExit('no scenes after apply_selection')
    _assign_splits(scenes, cfg)
    _write_scenes(out, scenes)


def _build_one_candidate(
    spec: dict,
    cfg: dict,
    sc: dict,
    name: str,
    ir: np.ndarray,
    vi: np.ndarray,
    vi_bgr: np.ndarray | None,
    built: dict[str, np.ndarray],
    scene_dir: Path,
) -> None:
    cname = spec['name']
    ctype = spec['type']
    dst = scene_dir / 'fusions' / f'{cname}.png'

    if ctype == 'prior':
        mode = spec.get('mode', 'max')
        if mode == 'max':
            img = built['prior_max']
        elif mode in ('m_hf', 'mhf', 'hf', 'inject'):
            img = _prior_mhf(ir, vi, float(spec.get('alpha', 1.0)))
        else:
            raise ValueError(f'unknown prior mode {mode}')
        if vi_bgr is not None:
            img = _colorize_y_with_vi(img, vi_bgr)
        _write_img(dst, img)
        built[cname] = img
    elif ctype == 'sota':
        folder = _sota_dir(cfg, sc['dataset'], spec)
        src = _find_fusion_file(folder, name)
        if src is None:
            print(f'[warn] missing SOTA {folder}/{name} ({sc["scene_id"]})')
            return
        # Keep color as stored by SOTA (BGR); do not force grayscale.
        raw = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if raw is None:
            raw = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if raw is None:
            return
        _write_img(dst, raw)
        built[cname] = raw
    elif ctype == 'degrade':
        seed_name = spec.get('seed', 'prior_max')
        seed_img = built.get(seed_name)
        if seed_img is None:
            seed_img = built.get('prior_max')
        if seed_img is None:
            print(f'[warn] no seed for degrade {cname} @ {sc["scene_id"]}')
            return
        img = _degrade(seed_img, spec.get('mode', 'overexp'), float(spec.get('gain', 1.4)))
        _write_img(dst, img)
        built[cname] = img
    elif ctype == 'ckpt':
        pass  # optional batch below
    else:
        raise ValueError(f'unknown type {ctype}')


def cmd_materialize(cfg: dict, skip_ckpt: bool = True) -> None:
    out = Path(cfg['out_root'])
    scenes = _load_scenes(out)
    cand_specs = cfg['candidates']
    # Pass 1: non-degrade; Pass 2: degrade (needs seed fusion in `built`)
    pass1 = [s for s in cand_specs if s['type'] != 'degrade']
    pass2 = [s for s in cand_specs if s['type'] == 'degrade']

    for sc in scenes:
        scene_dir = out / 'scenes' / sc['scene_id']
        root = Path(sc['root'])
        name = sc['filename']
        ir_src = root / 'ir' / name
        vi_src = root / 'vi' / name
        _copy_file(ir_src, scene_dir / 'ir' / name)
        _copy_file(vi_src, scene_dir / 'vi' / name)
        ir = _read_gray(ir_src)
        vi = _read_gray(vi_src)
        vi_bgr = cv2.imread(str(vi_src), cv2.IMREAD_COLOR)
        built: dict[str, np.ndarray] = {'prior_max': _prior_max(ir, vi)}

        for spec in pass1:
            _build_one_candidate(spec, cfg, sc, name, ir, vi, vi_bgr, built, scene_dir)
        for spec in pass2:
            _build_one_candidate(spec, cfg, sc, name, ir, vi, vi_bgr, built, scene_dir)

        meta = {
            **sc,
            'candidates': [s['name'] for s in cand_specs],
            'scene_dir': str(scene_dir),
        }
        (scene_dir / 'meta.json').write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8',
        )

    if skip_ckpt:
        print('[materialize] done (ckpt slots skipped by default)')
        print(f'             → {out / "scenes"}')
        return

    for spec in cand_specs:
        if spec['type'] != 'ckpt':
            continue
        ckpt = Path(spec['path'])
        if not ckpt.is_file():
            print(f'[warn] missing ckpt {ckpt}')
            continue
        by_root: dict[str, list[dict]] = {}
        for sc in scenes:
            by_root.setdefault(sc['root'], []).append(sc)
        for root, group in by_root.items():
            tmp = out / '_tmp_pred' / spec['name'] / Path(root).name
            ir_dir, vi_dir = tmp / 'ir', tmp / 'vi'
            ir_dir.mkdir(parents=True, exist_ok=True)
            vi_dir.mkdir(parents=True, exist_ok=True)
            for sc in group:
                _copy_file(Path(root) / 'ir' / sc['filename'], ir_dir / sc['filename'])
                _copy_file(Path(root) / 'vi' / sc['filename'], vi_dir / sc['filename'])
            pred_out = tmp / 'pred'
            cmd = [
                sys.executable, str(Path(__file__).resolve().parents[1] / 'pred.py'),
                '--ckpt', str(ckpt), '--data', str(tmp), '--dst', str(pred_out),
            ]
            print('[materialize]', ' '.join(cmd))
            subprocess.run(cmd, check=False)
            for sc in group:
                src = _find_fusion_file(pred_out, sc['filename'])
                if src is None:
                    continue
                raw = cv2.imread(str(src), cv2.IMREAD_COLOR)
                if raw is None:
                    raw = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
                if raw is None:
                    continue
                dst = out / 'scenes' / sc['scene_id'] / 'fusions' / f'{spec["name"]}.png'
                _write_img(dst, raw)
    print(f'[materialize] done → {out / "scenes"}')


def cmd_sample_pairs(cfg: dict) -> None:
    out = Path(cfg['out_root'])
    scenes = _load_scenes(out)
    n_pairs = int(cfg.get('pairs_per_scene', 8))
    seed = int(cfg.get('seed', 443))
    cand_names = [c['name'] for c in cfg['candidates']]
    degrade_names = {c['name'] for c in cfg['candidates'] if c.get('type') == 'degrade'}
    rng = random.Random(seed + 17)

    tasks = []
    for sc in scenes:
        scene_dir = out / 'scenes' / sc['scene_id']
        available = [
            name for name in cand_names
            if (scene_dir / 'fusions' / f'{name}.png').is_file()
        ]
        if len(available) < 2:
            print(f'[warn] <2 fusions for {sc["scene_id"]}')
            continue
        # Skip degrade↔degrade: both are negative anchors; not informative to rank.
        all_pairs = [
            (a, b) for i, a in enumerate(available) for b in available[i + 1:]
            if not (a in degrade_names and b in degrade_names)
        ]
        rng.shuffle(all_pairs)
        for a, b in all_pairs[: min(n_pairs, len(all_pairs))]:
            left, right = (a, b) if rng.random() < 0.5 else (b, a)
            tasks.append({
                'task_id': f'{sc["scene_id"]}::{left}__{right}',
                'scene_id': sc['scene_id'],
                'split': sc['split'],
                'dataset': sc['dataset'],
                'ir': str(scene_dir / 'ir' / sc['filename']),
                'vi': str(scene_dir / 'vi' / sc['filename']),
                'left': left,
                'right': right,
                'left_path': str(scene_dir / 'fusions' / f'{left}.png'),
                'right_path': str(scene_dir / 'fusions' / f'{right}.png'),
                'is_gold': False,
            })

    # NOTE: over/underexposure candidates may still appear in ordinary pairs.
    # Do NOT treat "method vs deg_overexp" as gold/control with a forced winner —
    # overexposure does not necessarily mean worse fusion quality under our protocol.

    rng.shuffle(tasks)
    path = out / 'pairs.jsonl'
    with path.open('w', encoding='utf-8') as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + '\n')
    print(f'[sample_pairs] {len(tasks)} ordinary tasks → {path}')


def main() -> None:
    p = ArgumentParser(description=__doc__)
    p.add_argument(
        'stage',
        choices=[
            'make_gallery', 'serve_gallery', 'pick_cli', 'apply_selection',
            'sample_scenes', 'materialize', 'sample_pairs', 'annotate', 'all_random',
        ],
    )
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--dataset', type=str, default=None, help='dataset for make_gallery / pick_cli')
    p.add_argument(
        '--selection', type=Path, action='append', default=[],
        help='selection_*.json from gallery (repeatable)',
    )
    p.add_argument('--port', type=int, default=8765, help='serve_gallery / annotate port')
    p.add_argument('--annotator', type=str, default=None, help='annotator id for annotate stage')
    p.add_argument('--split', type=str, default=None, help='optional train|val|test filter for annotate')
    p.add_argument('--skip_ckpt', action='store_true', default=True)
    p.add_argument('--run_ckpt', action='store_true', help='also run ckpt pred slots')
    args = p.parse_args()
    cfg = load_config(args.config)
    skip_ckpt = not args.run_ckpt

    if args.stage == 'make_gallery':
        cmd_make_gallery(cfg, dataset=args.dataset)
    elif args.stage == 'serve_gallery':
        cmd_serve_gallery(cfg, port=args.port)
    elif args.stage == 'pick_cli':
        if not args.dataset:
            raise SystemExit('pick_cli requires --dataset FMB|M3FD|MSRS')
        cmd_pick_cli(cfg, args.dataset)
    elif args.stage == 'apply_selection':
        if not args.selection:
            raise SystemExit('pass one or more --selection selection_XX.json')
        cmd_apply_selection(cfg, args.selection)
    elif args.stage == 'sample_scenes':
        cmd_sample_scenes(cfg)
    elif args.stage == 'materialize':
        cmd_materialize(cfg, skip_ckpt=skip_ckpt)
    elif args.stage == 'sample_pairs':
        cmd_sample_pairs(cfg)
    elif args.stage == 'annotate':
        # Delegate to annotate.py (long-running server).
        cmd = [
            sys.executable, str(Path(__file__).resolve().parent / 'annotate.py'),
            '--config', str(args.config),
            '--port', str(args.port if args.port != 8765 else 8766),
        ]
        if args.annotator:
            cmd.extend(['--annotator', args.annotator])
        if args.split:
            cmd.extend(['--split', args.split])
        raise SystemExit(subprocess.call(cmd))
    elif args.stage == 'all_random':
        cmd_sample_scenes(cfg)
        cmd_materialize(cfg, skip_ckpt=skip_ckpt)
        cmd_sample_pairs(cfg)


if __name__ == '__main__':
    main()
