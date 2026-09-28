#!/usr/bin/env python3
"""Preference-fit diagnostics (do these BEFORE scaling the CNN).

Experiments:
  1) Memorize N train pairs (no WD / no dropout) — can train acc hit ~100%?
  2) Easy (method vs degrade / gold) vs hard (method↔method) accuracy
  3) Per-scene Bradley–Terry |Δq| bins vs accuracy

  CUDA_VISIBLE_DEVICES=3 python scripts/pref_bench/diagnose_preference.py \\
      --config /data/ykx/pref_bench/config.yaml --gpu 3

Writes: {out_root}/fit/diagnose_report.json
"""

from __future__ import annotations

import json
import os
import random
import sys
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

# Pin GPU before torch CUDA init (argv parsed lightly).
def _early_pin() -> str:
    gpu = '3'
    for i, a in enumerate(sys.argv):
        if a == '--gpu' and i + 1 < len(sys.argv):
            gpu = sys.argv[i + 1]
        elif a.startswith('--gpu='):
            gpu = a.split('=', 1)[1]
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    return os.environ['CUDA_VISIBLE_DEVICES']


_PINNED = _early_pin()

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import importlib.util

_fp = Path(__file__).resolve().parent / 'fit_preference.py'
_spec = importlib.util.spec_from_file_location('fit_preference', _fp)
_fp_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_fp_mod)

ScoreNet = _fp_mod.ScoreNet
_load_triple = _fp_mod._load_triple
load_config = _fp_mod.load_config
load_features = _fp_mod.load_features
load_jsonl = _fp_mod.load_jsonl
load_labels = _fp_mod.load_labels
pairwise_examples = _fp_mod.pairwise_examples
score_linear = _fp_mod.score_linear



# ---------------------------------------------------------------------------
# Pair typing
# ---------------------------------------------------------------------------

def pair_bucket(win: str, lose: str, is_gold: bool) -> str:
    names = {win, lose}
    n_deg = sum(n.startswith('deg_') for n in names)
    if n_deg >= 1:
        return 'easy_vs_degrade'
    if all(n.startswith('method_') for n in names):
        return 'hard_method_vs_method'
    if any(n.startswith('prior_') for n in names) and any(n.startswith('method_') for n in names):
        return 'medium_method_vs_prior'
    if all(n.startswith('prior_') for n in names):
        return 'hard_prior_vs_prior'
    return 'other'


def is_degrade(name: str) -> bool:
    return name.startswith('deg_')


@torch.no_grad()
def eval_quality_direction(net: nn.Module, rows: list[dict], device: str, size: int, bs: int = 16) -> dict:
    """For pairs involving degrade: does S(non-deg) > S(deg)? (independent of human label)."""
    usable = []
    for e in rows:
        a, b = e['win'], e['lose']
        # recover both candidate names and paths
        if is_degrade(a) ^ is_degrade(b):
            if is_degrade(a):
                good_path, bad_path = e['lose_path'], e['win_path']
            else:
                good_path, bad_path = e['win_path'], e['lose_path']
            usable.append((e['ir'], e['vi'], good_path, bad_path))
    if not usable:
        return {'n': 0, 'acc': float('nan')}
    correct = 0
    for i in range(0, len(usable), bs):
        chunk = usable[i:i + bs]
        goods = torch.stack([_load_triple(ir, vi, g, size) for ir, vi, g, _ in chunk]).to(device)
        bads = torch.stack([_load_triple(ir, vi, b, size) for ir, vi, _, b in chunk]).to(device)
        correct += int((net(goods) > net(bads)).sum().item())
    return {'n': len(usable), 'acc': correct / len(usable)}


# ---------------------------------------------------------------------------
# Exp 1: memorize
# ---------------------------------------------------------------------------

class PairDataset(Dataset):
    def __init__(self, examples: list[dict], size: int = 128):
        self.examples = examples
        self.size = size

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        e = self.examples[idx]
        win = _load_triple(e['ir'], e['vi'], e['win_path'], self.size)
        lose = _load_triple(e['ir'], e['vi'], e['lose_path'], self.size)
        return win, lose


@torch.no_grad()
def eval_acc(net: nn.Module, rows: list[dict], device: str, size: int, bs: int = 16) -> float:
    if not rows:
        return float('nan')
    net.eval()
    correct = n = 0
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        wins = torch.stack([_load_triple(e['ir'], e['vi'], e['win_path'], size) for e in chunk]).to(device)
        loses = torch.stack([_load_triple(e['ir'], e['vi'], e['lose_path'], size) for e in chunk]).to(device)
        sw, sl = net(wins), net(loses)
        correct += int((sw > sl).sum().item())
        n += len(chunk)
    return correct / n


def exp_memorize(
    examples: list[dict],
    device: str,
    n_pairs: int = 200,
    size: int = 128,
    epochs: int = 80,
    seed: int = 443,
) -> dict:
    train_pool = [e for e in examples if e['split'] == 'train' and not e['is_gold']]
    rng = random.Random(seed)
    subset = train_pool.copy()
    rng.shuffle(subset)
    subset = subset[: min(n_pairs, len(subset))]

    torch.manual_seed(seed)
    net = ScoreNet().to(device)
    # No weight decay — force memorization
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0)
    loader = DataLoader(PairDataset(subset, size), batch_size=16, shuffle=True, num_workers=0)

    history = []
    best = 0.0
    for ep in range(1, epochs + 1):
        net.train()
        total = 0.0
        for win, lose in loader:
            win, lose = win.to(device), lose.to(device)
            opt.zero_grad()
            loss = F.softplus(-(net(win) - net(lose))).mean()
            loss.backward()
            opt.step()
            total += loss.item() * win.size(0)
        acc = eval_acc(net, subset, device, size)
        best = max(best, acc)
        if ep % 10 == 0 or ep == 1 or acc >= 0.99:
            print(f'[memorize] ep {ep:3d} loss={total/len(subset):.4f} train_acc={acc:.3f}')
        history.append({'epoch': ep, 'acc': acc, 'loss': total / max(len(subset), 1)})
        if acc >= 0.995:
            break

    return {
        'n_pairs': len(subset),
        'size': size,
        'epochs_run': len(history),
        'final_train_acc': history[-1]['acc'],
        'best_train_acc': best,
        'history_tail': history[-5:],
        'verdict': (
            'PIPELINE_OK_LIKELY_DATA_CEILING'
            if best >= 0.95
            else 'PIPELINE_OR_FORMULATION_BUG'
        ),
    }


# ---------------------------------------------------------------------------
# Exp 2: easy / hard buckets on a trained checkpoint
# ---------------------------------------------------------------------------

def load_net_ckpt(path: Path, device: str) -> tuple[ScoreNet, int]:
    ckpt = torch.load(path, map_location='cpu')
    net = ScoreNet()
    state = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    net.load_state_dict(state)
    net.to(device).eval()
    size = int(ckpt.get('size', 128)) if isinstance(ckpt, dict) else 128
    return net, size


def exp_easy_hard(net: nn.Module, examples: list[dict], device: str, size: int, split: str = 'test') -> dict:
    rows = [e for e in examples if e['split'] == split]
    buckets: dict[str, list[dict]] = defaultdict(list)
    for e in rows:
        buckets[pair_bucket(e['win'], e['lose'], e['is_gold'])].append(e)

    out = {}
    for name, rs in sorted(buckets.items()):
        out[name] = {'n': len(rs), 'acc_match_human': eval_acc(net, rs, device, size)}
    gold = [e for e in rows if e['is_gold']]
    out['gold_match_human'] = {'n': len(gold), 'acc': eval_acc(net, gold, device, size)}
    out['degrade_quality_direction'] = eval_quality_direction(net, rows, device, size)
    out['gold_quality_direction'] = eval_quality_direction(net, gold, device, size)
    all_rows = [e for e in rows if not e['is_gold']]
    out['overall_non_gold'] = {'n': len(all_rows), 'acc_match_human': eval_acc(net, all_rows, device, size)}
    return out


# ---------------------------------------------------------------------------
# Exp 3: BT |Δq| bins
# ---------------------------------------------------------------------------

def fit_bt_scene(cands: list[str], wins: list[tuple[str, str]], steps: int = 400) -> dict[str, float]:
    """Fit BT strengths q for candidates in one scene from (winner, loser) pairs."""
    idx = {c: i for i, c in enumerate(cands)}
    if len(cands) < 2 or not wins:
        return {c: 0.0 for c in cands}
    q = torch.zeros(len(cands), requires_grad=True)
    opt = torch.optim.LBFGS([q], lr=0.5, max_iter=50)

    wi = torch.tensor([idx[a] for a, b in wins], dtype=torch.long)
    lj = torch.tensor([idx[b] for a, b in wins], dtype=torch.long)

    def closure():
        opt.zero_grad()
        qq = q - q.mean()
        loss = F.softplus(-(qq[wi] - qq[lj])).mean()
        loss.backward()
        return loss.detach()

    for _ in range(8):
        opt.step(closure)
    with torch.no_grad():
        qq = (q - q.mean()).cpu().numpy()
    return {c: float(qq[i]) for c, i in idx.items()}


def exp_deltaq_bins(examples: list[dict], net: nn.Module | None, device: str, size: int,
                    linear_model: dict | None, feat: dict | None) -> dict:
    # group decisive labels by scene
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for e in examples:
        if e['choice_side'] is None:
            continue
        by_scene[e['scene_id']].append(e)

    # build BT from all non-tie labels (all splits) per scene
    # need original left/right choice — rebuild from win/lose
    records = []  # (split, dq, correct_net, correct_linear, bucket)
    for scene_id, rows in by_scene.items():
        # only use non-gold for BT strength? include all decisive for denser graph
        wins = [(e['win'], e['lose']) for e in rows]
        cands = sorted({x for w, l in wins for x in (w, l)})
        qmap = fit_bt_scene(cands, wins)
        for e in rows:
            if e['is_gold']:
                continue
            dq = abs(qmap[e['win']] - qmap[e['lose']])
            hit_net = None
            if net is not None:
                with torch.no_grad():
                    w = _load_triple(e['ir'], e['vi'], e['win_path'], size)[None].to(device)
                    l = _load_triple(e['ir'], e['vi'], e['lose_path'], size)[None].to(device)
                    hit_net = bool((net(w) > net(l)).item())
            hit_lin = None
            if linear_model is not None and feat is not None:
                a = feat.get((e['scene_id'], e['win']))
                b = feat.get((e['scene_id'], e['lose']))
                if a is not None and b is not None:
                    hit_lin = score_linear(linear_model, a) > score_linear(linear_model, b)
            records.append({
                'split': e['split'],
                'dq': dq,
                'hit_net': hit_net,
                'hit_lin': hit_lin,
            })

    edges = None  # use quantiles instead — BT scale is arbitrary per scene

    def bin_stats(split: str | None):
        rows = [r for r in records if split is None or r['split'] == split]
        if not rows:
            return {}
        dqs = np.array([r['dq'] for r in rows], dtype=np.float64)
        # normalize within split by rank percentile so bins are populated
        order = dqs.argsort()
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.linspace(0, 1, len(rows), endpoint=False)
        # also report raw-magnitude quartile bins after z-score within scene already mean0;
        # use percentile bins of |Δq|
        qs = [0.0, 0.25, 0.5, 0.75, 1.0]
        labels = ['Q1(hardest)', 'Q2', 'Q3', 'Q4(easiest)']
        out = {}
        for i, name in enumerate(labels):
            lo, hi = qs[i], qs[i + 1]
            if i < 3:
                sub_idx = [j for j, rnk in enumerate(ranks) if lo <= rnk < hi]
            else:
                sub_idx = [j for j, rnk in enumerate(ranks) if lo <= rnk <= hi]
            sub = [rows[j] for j in sub_idx]

            def acc(key):
                vals = [r[key] for r in sub if r[key] is not None]
                return (sum(vals) / len(vals)) if vals else float('nan')

            out[name] = {
                'n': len(sub),
                'dq_min': float(min((r['dq'] for r in sub), default=float('nan'))),
                'dq_max': float(max((r['dq'] for r in sub), default=float('nan'))),
                'mean_dq': float(np.mean([r['dq'] for r in sub])) if sub else float('nan'),
                'net_acc': acc('hit_net'),
                'linear_acc': acc('hit_lin'),
            }
        return out

    return {
        'all': bin_stats(None),
        'test': bin_stats('test'),
        'val': bin_stats('val'),
        'n_records': len(records),
    }


def annotate_choice_side(examples: list[dict]) -> list[dict]:
    """Attach dummy field used by BT grouping."""
    for e in examples:
        e['choice_side'] = 'win'
    return examples


def main() -> None:
    p = ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--annotator', default='ann0')
    p.add_argument('--gpu', default='3')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--size', type=int, default=128)
    p.add_argument('--memorize_n', type=int, default=200)
    p.add_argument('--memorize_epochs', type=int, default=80)
    p.add_argument('--skip_memorize', action='store_true')
    p.add_argument('--ckpt', type=Path, default=None,
                   help='optional ScoreNet ckpt; default net_s256.pt or net.pt')
    args = p.parse_args()

    cfg = load_config(args.config)
    out = Path(cfg['out_root'])
    fit_dir = out / 'fit'
    fit_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise SystemExit('CUDA not available')
    print(f'[diagnose] CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")} pin={_PINNED} device={device}')

    pairs = {r['task_id']: r for r in load_jsonl(out / 'pairs.jsonl')}
    labels = load_labels(out, args.annotator)
    examples = annotate_choice_side(pairwise_examples(labels, pairs))
    print(f'[diagnose] decisive pairs={len(examples)}')

    report: dict = {
        'note': (
            'Architecture already is shared scorer S(IR,VI,F) + pairwise softplus(-(s_w-s_l)); '
            'not a 4-image A/B classifier. BT is for analysis, not training targets.'
        ),
    }

    # ----- Exp 1 -----
    if not args.skip_memorize:
        print('\n=== Exp1: memorize subset ===')
        report['exp1_memorize'] = exp_memorize(
            examples, device, n_pairs=args.memorize_n, size=args.size,
            epochs=args.memorize_epochs,
        )
        print('[exp1]', report['exp1_memorize']['verdict'],
              'best=', report['exp1_memorize']['best_train_acc'])
    else:
        report['exp1_memorize'] = {'skipped': True}

    # ----- load ckpt for exp2/3 -----
    ckpt = args.ckpt
    if ckpt is None:
        for cand in (fit_dir / 'net_s256.pt', fit_dir / 'net.pt', fit_dir / 'net_s128.pt'):
            if cand.is_file():
                ckpt = cand
                break
    net = None
    size = args.size
    if ckpt is not None and ckpt.is_file():
        print(f'[diagnose] load ckpt {ckpt}')
        net, size = load_net_ckpt(ckpt, device)
        report['ckpt'] = str(ckpt)
        report['ckpt_size'] = size
    else:
        print('[diagnose] WARN: no ckpt; exp2/3 net metrics skipped')

    linear_model = None
    lw = fit_dir / 'linear_weights.json'
    if lw.is_file():
        linear_model = json.loads(lw.read_text(encoding='utf-8'))
    feat = load_features(out / 'features.jsonl') if (out / 'features.jsonl').is_file() else {}

    # ----- Exp 2 -----
    print('\n=== Exp2: easy vs hard ===')
    if net is not None:
        report['exp2_easy_hard'] = {
            'test': exp_easy_hard(net, examples, device, size, 'test'),
            'val': exp_easy_hard(net, examples, device, size, 'val'),
        }
        for k, v in report['exp2_easy_hard']['test'].items():
            if 'n' in v:
                acc = v.get('acc', v.get('acc_match_human', v.get('acc')))
                print(f"  test/{k}: n={v['n']} acc={acc:.3f}" if v['n'] else f"  test/{k}: empty")
    else:
        report['exp2_easy_hard'] = {'skipped': True}

    # ----- Exp 3 -----
    print('\n=== Exp3: BT |Δq| quantile bins ===')
    report['exp3_deltaq'] = exp_deltaq_bins(examples, net, device, size, linear_model, feat)
    for name, st in report['exp3_deltaq']['test'].items():
        print(f"  test {name}: n={st['n']} net={st['net_acc']:.3f} lin={st['linear_acc']:.3f}")

    # ----- Exp 4 placeholder -----
    report['exp4_self_consistency'] = {
        'status': 'NEEDS_HUMAN',
        'instruction': (
            'Re-annotate 200–500 pairs after several days with shuffled L/R; '
            'compute agreement. If self-consistency ≈70–80%, overall 70% model acc is near human ceiling.'
        ),
    }

    # keep previous memorize if skipped
    if args.skip_memorize:
        prev = fit_dir / 'diagnose_report.json'
        if prev.is_file():
            try:
                old = json.loads(prev.read_text(encoding='utf-8'))
                if 'exp1_memorize' in old and not old['exp1_memorize'].get('skipped'):
                    report['exp1_memorize'] = old['exp1_memorize']
            except Exception:
                pass

    # ----- Summary matrix -----
    e1 = report.get('exp1_memorize', {})
    e2 = report.get('exp2_easy_hard', {}).get('test', {})
    summary = {
        'memorize_200_best_acc': e1.get('best_train_acc'),
        'memorize_verdict': e1.get('verdict'),
        'gold_quality_direction_test': (e2.get('gold_quality_direction') or {}).get('acc'),
        'degrade_quality_direction_test': (e2.get('degrade_quality_direction') or {}).get('acc'),
        'hard_method_vs_method_test_acc_vs_human': (e2.get('hard_method_vs_method') or {}).get('acc_match_human'),
        'deltaq_test_bins': report['exp3_deltaq']['test'],
    }
    report['summary'] = summary

    path = fit_dir / 'diagnose_report.json'
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\n[diagnose] wrote {path}')
    print('[summary]', json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
