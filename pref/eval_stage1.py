"""Stage-1 verification: test metrics, traditional baselines, buckets, Δs plot.

Does not touch Stage-2. Freezes the best checkpoint and evaluates held-out splits
under the same soft-pairwise protocol used in training.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from pref.dataset import PreferencePairDataset
from pref.io import resolve_pack_paths
from pref.model.loss import soft_preference_target
from pref.model.resnet_scorer import ResNet18ScalarScorer


METRIC_KEYS = ('en', 'mi', 'qabf', 'scd', 'vif', 'ssim', 'sf')
BUCKETS = (
    ('d>0.4', 0.4, 1.01),
    ('0.3<d≤0.4', 0.3, 0.4),
    ('0.2<d≤0.3', 0.2, 0.3),
    ('d≤0.2', 0.0, 0.2),
)


def image_entropy(path: str | Path) -> float:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    p = hist / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman ρ without requiring scipy; p-value left as nan if scipy missing."""
    if len(x) < 3:
        return float('nan'), float('nan')
    rx = x.argsort().argsort().astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    rho = float((rx * ry).sum() / denom) if denom > 0 else float('nan')
    try:
        from scipy.stats import spearmanr
        _, p = spearmanr(x, y)
        return rho, float(p)
    except Exception:
        return rho, float('nan')


def load_stage1_scorer(ckpt_path: Path | str, device: str = 'cpu') -> tuple[ResNet18ScalarScorer, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ResNet18ScalarScorer(pretrained=False).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt


def load_feature_table(path: Path) -> dict[tuple[str, str], dict]:
    table: dict[tuple[str, str], dict] = {}
    if not path.is_file():
        return table
    with path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            table[(row['scene_id'], row['candidate'])] = row
    return table


@torch.no_grad()
def score_rows(
    model: ResNet18ScalarScorer,
    rows: Sequence[dict],
    *,
    size: int,
    device: str,
    batch_size: int = 16,
) -> list[dict]:
    """Return per-pair records with s_A, s_B, Δs, q, R, prediction correctness."""
    ds = PreferencePairDataset(rows, size=size, augment=False, exclude_gold=False)
    # Keep alignment with original rows that have paths.
    valid = ds.rows
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    out: list[dict] = []
    cursor = 0
    model.eval()
    for x_a, x_b, y, R in loader:
        x_a = x_a.to(device)
        x_b = x_b.to(device)
        s_a = model(x_a)
        s_b = model(x_b)
        q = soft_preference_target(y.to(device))
        q_hat = torch.sigmoid(s_a - s_b)  # tau=1
        bs = x_a.size(0)
        for i in range(bs):
            r = valid[cursor + i]
            qi = float(q[i])
            qhi = float(q_hat[i])
            sai = float(s_a[i])
            sbi = float(s_b[i])
            ds_ = sai - sbi
            decisive = abs(qi - 0.5) >= 0.1
            soft_ok = (qhi >= 0.5) == (qi >= 0.5)
            dec_ok = soft_ok if decisive else None
            # soft BCE for this pair
            eps = 1e-8
            qhc = min(max(qhi, eps), 1.0 - eps)
            bce = -(qi * np.log(qhc) + (1.0 - qi) * np.log(1.0 - qhc))
            out.append({
                'pair_id': r.get('pair_id'),
                'scene_id': r.get('scene_id'),
                'split': r.get('split'),
                'candidate_a': r.get('candidate_a'),
                'candidate_b': r.get('candidate_b'),
                'path_a': r.get('path_a'),
                'path_b': r.get('path_b'),
                'y_A': float(r['y_A']),
                'y_T': float(r['y_T']),
                'y_B': float(r['y_B']),
                'R': float(R[i]),
                'q': qi,
                'q_hat': qhi,
                's_A': sai,
                's_B': sbi,
                'delta_s': ds_,
                'bce': bce,
                'decisive': decisive,
                'soft_ok': bool(soft_ok),
                'dec_ok': (bool(dec_ok) if dec_ok is not None else None),
                'd': abs(qi - 0.5),
            })
        cursor += bs
    return out


def summarize_scorer(records: Sequence[dict]) -> dict:
    if not records:
        return {'n': 0, 'loss': float('nan'), 'acc_soft': float('nan'), 'acc_decisive': float('nan')}
    bce = np.asarray([r['bce'] for r in records], dtype=np.float64)
    loss = float(bce.mean())
    soft = float(np.mean([r['soft_ok'] for r in records]))
    dec = [r for r in records if r['decisive']]
    acc_dec = float(np.mean([r['dec_ok'] for r in dec])) if dec else float('nan')
    return {
        'n': len(records),
        'n_decisive': len(dec),
        'loss': loss,
        'acc_soft': soft,
        'acc_decisive': acc_dec,
    }


def bucket_report(records: Sequence[dict]) -> list[dict]:
    rows = []
    for name, lo, hi in BUCKETS:
        # (lo, hi]: except first bucket uses > lo via hi exclusive lower handled:
        # d>0.4 → lo=0.4, hi=1.01 with d > lo
        # others: lo < d <= hi for middle; d<=0.2 for last
        if name.startswith('d>'):
            subset = [r for r in records if r['d'] > lo]
        elif name.startswith('d≤') or name.startswith('d<='):
            subset = [r for r in records if r['d'] <= hi]
        else:
            subset = [r for r in records if (r['d'] > lo and r['d'] <= hi)]
        soft = float(np.mean([r['soft_ok'] for r in subset])) if subset else float('nan')
        dec = [r for r in subset if r['decisive']]
        acc_dec = float(np.mean([r['dec_ok'] for r in dec])) if dec else float('nan')
        rows.append({
            'bucket': name,
            'n': len(subset),
            'n_decisive': len(dec),
            'acc_soft': soft,
            'acc_decisive': acc_dec,
            'mean_d': float(np.mean([r['d'] for r in subset])) if subset else float('nan'),
        })
    return rows


def _metric_value(
    feats: dict[tuple[str, str], dict],
    scene_id: str,
    cand: str,
    key: str,
    path: str | None,
    en_cache: dict[str, float],
) -> float | None:
    if key == 'en':
        if not path:
            return None
        if path not in en_cache:
            en_cache[path] = image_entropy(path)
        return en_cache[path]
    row = feats.get((scene_id, cand))
    if row is None or key not in row:
        return None
    return float(row[key])


def evaluate_metric_baselines(
    records: Sequence[dict],
    feats: dict[tuple[str, str], dict],
    *,
    decisive_only: bool = True,
) -> dict[str, dict]:
    """Same-protocol pairwise: m_A > m_B ⇒ predict A (higher-is-better)."""
    en_cache: dict[str, float] = {}
    out: dict[str, dict] = {}
    use = [r for r in records if (r['decisive'] if decisive_only else True)]
    for key in METRIC_KEYS:
        ok = 0
        n = 0
        ties_m = 0
        for r in use:
            ma = _metric_value(feats, r['scene_id'], r['candidate_a'], key, r.get('path_a'), en_cache)
            mb = _metric_value(feats, r['scene_id'], r['candidate_b'], key, r.get('path_b'), en_cache)
            if ma is None or mb is None:
                continue
            if ma == mb:
                ties_m += 1
                # metric tie → count as wrong vs decisive human (conservative)
                n += 1
                continue
            pred_a = ma > mb
            true_a = r['q'] >= 0.5
            ok += int(pred_a == true_a)
            n += 1
        out[key] = {
            'n': n,
            'acc': (ok / n) if n else float('nan'),
            'metric_ties': ties_m,
        }
    # ours on same subset
    if decisive_only:
        ours = [r for r in records if r['decisive'] and r['dec_ok'] is not None]
        out['ours_S'] = {
            'n': len(ours),
            'acc': float(np.mean([r['dec_ok'] for r in ours])) if ours else float('nan'),
            'metric_ties': 0,
        }
    else:
        out['ours_S'] = {
            'n': len(records),
            'acc': float(np.mean([r['soft_ok'] for r in records])) if records else float('nan'),
            'metric_ties': 0,
        }
    return out


def plot_delta_s_vs_q(records: Sequence[dict], dest: Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    q = np.asarray([r['q'] for r in records], dtype=np.float64)
    ds = np.asarray([r['delta_s'] for r in records], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(ds, q, s=12, alpha=0.45, c='#1f4e79', edgecolors='none')
    ax.axhline(0.5, color='#999', lw=0.8, ls='--')
    ax.axvline(0.0, color='#999', lw=0.8, ls='--')
    ax.set_xlabel(r'$\Delta s = s_A - s_B$')
    ax.set_ylabel(r'$q_A = y_A + \frac{1}{2} y_T$')
    ax.set_title('Human soft preference vs learned score gap')
    # Spearman
    if len(q) >= 3:
        rho, p = _spearman(ds, q)
        p_txt = f'p={p:.1e}' if p == p else 'p=n/a'
        ax.text(
            0.02, 0.98, f'Spearman ρ={rho:.3f}\n({p_txt})',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='#ccc'),
        )
    fig.tight_layout()
    fig.savefig(dest, dpi=160)
    plt.close(fig)
    return dest


def edge_consistency_by_scene(records: Sequence[dict]) -> dict:
    """Fraction of decisive human edges consistent with score order, per scene + overall."""
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r['decisive']:
            by_scene[str(r['scene_id'])].append(r)
    per = []
    ok = 0
    n = 0
    for sid, rs in sorted(by_scene.items()):
        s_ok = sum(1 for r in rs if r['dec_ok'])
        s_n = len(rs)
        ok += s_ok
        n += s_n
        per.append({'scene_id': sid, 'n_decisive': s_n, 'edge_consistency': s_ok / s_n if s_n else float('nan')})
    return {
        'overall': (ok / n) if n else float('nan'),
        'n_decisive_edges': n,
        'n_scenes': len(by_scene),
        'per_scene': per,
    }


def run_stage1_eval(
    out_root: Path | str,
    ckpt_path: Path | str | None = None,
    device: str = 'cuda:0',
    batch_size: int = 16,
) -> dict:
    out = Path(out_root)
    fit = out / 'fit'
    ckpt_path = Path(ckpt_path) if ckpt_path else fit / 'scorer_stage1_best.pt'
    superv = resolve_pack_paths(
        [json.loads(l) for l in (out / 'supervision.jsonl').read_text().splitlines() if l.strip()],
        out,
    )
    model, ckpt = load_stage1_scorer(ckpt_path, device=device)
    size = int(ckpt.get('size', 224))
    feats = load_feature_table(out / 'features.jsonl')

    # In-domain Stage-1 metrics use protocol==main (or non-RoadScene).
    # RoadScene pairs also have split=test but belong to zero-shot protocols e1/e2.
    def _indomain(rows: list[dict]) -> list[dict]:
        out = []
        for r in rows:
            proto = r.get('protocol')
            if proto in ('e1', 'e2', 'g1', 'g3'):
                continue
            if r.get('dataset') == 'RoadScene':
                continue
            out.append(r)
        return out

    split_records: dict[str, list[dict]] = {}
    split_summary: dict[str, dict] = {}
    for split in ('train', 'val', 'test'):
        rows = _indomain([r for r in superv if r.get('split') == split])
        recs = score_rows(model, rows, size=size, device=device, batch_size=batch_size)
        split_records[split] = recs
        split_summary[split] = summarize_scorer(recs)

    test_recs = split_records['test']
    buckets = bucket_report(test_recs)
    baselines_dec = evaluate_metric_baselines(test_recs, feats, decisive_only=True)
    baselines_soft = evaluate_metric_baselines(test_recs, feats, decisive_only=False)
    plot_path = plot_delta_s_vs_q(test_recs, fit / 'stage1_delta_s_vs_q_test.png')
    # also val plot
    plot_delta_s_vs_q(split_records['val'], fit / 'stage1_delta_s_vs_q_val.png')
    edge = edge_consistency_by_scene(test_recs)

    # spearman on test
    spearman = None
    if len(test_recs) >= 3:
        rho, p = _spearman(
            np.asarray([r['delta_s'] for r in test_recs], dtype=np.float64),
            np.asarray([r['q'] for r in test_recs], dtype=np.float64),
        )
        spearman = {'rho': rho, 'p': p}
    report = {
        'ckpt': str(ckpt_path),
        'ckpt_epoch': ckpt.get('epoch'),
        'size': size,
        'splits': split_summary,
        'test_buckets': buckets,
        'baselines_decisive': baselines_dec,
        'baselines_soft': baselines_soft,
        'edge_consistency_test': {
            'overall': edge['overall'],
            'n_decisive_edges': edge['n_decisive_edges'],
            'n_scenes': edge['n_scenes'],
        },
        'spearman_delta_s_vs_q_test': spearman,
        'plots': {
            'test': str(plot_path),
            'val': str(fit / 'stage1_delta_s_vs_q_val.png'),
        },
    }
    # save per-pair test table for further analysis
    pairs_path = fit / 'stage1_test_pair_scores.jsonl'
    with pairs_path.open('w', encoding='utf-8') as f:
        for r in test_recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    report['test_pair_scores'] = str(pairs_path)

    dest = fit / 'stage1_eval_report.json'
    dest.write_text(json.dumps(report, indent=2), encoding='utf-8')
    report['report_path'] = str(dest)
    return report


def format_markdown_table(baselines: dict[str, dict]) -> str:
    order = list(METRIC_KEYS) + ['ours_S']
    lines = ['| Metric | n | Human pairwise accuracy |', '|---|---:|---:|']
    for k in order:
        if k not in baselines:
            continue
        row = baselines[k]
        name = {'ours_S': r'Ours $S_\theta$', 'qabf': r'$Q^{AB/F}$'}.get(k, k.upper() if k != 'ssim' else 'SSIM')
        if k == 'ours_S':
            name = 'Ours S_θ'
        elif k == 'qabf':
            name = 'Qabf'
        else:
            name = k.upper()
        acc = row['acc']
        acc_s = f'{acc:.3f}' if acc == acc else 'nan'
        lines.append(f'| {name} | {row["n"]} | {acc_s} |')
    return '\n'.join(lines)
