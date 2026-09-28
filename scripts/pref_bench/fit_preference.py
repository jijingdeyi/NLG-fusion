#!/usr/bin/env python3
"""Fit preference score: (A) linear metrics  S=w^T m̃  (B) small CNN score net.

  # after features.jsonl exists:
  CUDA_VISIBLE_DEVICES=3 python scripts/pref_bench/fit_preference.py \\
      --config /data/ykx/pref_bench/config.yaml --track both

Default GPU is physical **3** only (set before importing torch).
"""

from __future__ import annotations

import os
import sys

# Pin GPU before importing torch. Default: physical card 3 only.
def _early_pin_gpu() -> str:
    gpu = '3'
    for i, a in enumerate(sys.argv):
        if a == '--gpu' and i + 1 < len(sys.argv):
            gpu = sys.argv[i + 1]
            break
        if a.startswith('--gpu='):
            gpu = a.split('=', 1)[1]
            break
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    return os.environ['CUDA_VISIBLE_DEVICES']


_PINNED_GPU = _early_pin_gpu()

import json
import random
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ldt.fusion_metrics import ALL_METRIC_KEYS

METRIC_KEYS = tuple(ALL_METRIC_KEYS)


def load_config(path: Path) -> dict:
    with path.open(encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    with path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_labels(out: Path, annotator: str | None) -> list[dict]:
    files = sorted(out.glob('labels_*.jsonl')) if annotator is None else [out / f'labels_{annotator}.jsonl']
    rows = []
    for fp in files:
        rows.extend(load_jsonl(fp))
    # keep last label per task_id
    by = {}
    for r in rows:
        by[r['task_id']] = r
    return list(by.values())


def load_features(path: Path) -> dict[tuple[str, str], dict]:
    feat = {}
    for r in load_jsonl(path):
        feat[(r['scene_id'], r['candidate'])] = r
    return feat


def pairwise_examples(labels: list[dict], pairs: dict[str, dict]) -> list[dict]:
    """Non-tie preference pairs with winner/loser candidate names."""
    ex = []
    for lab in labels:
        if lab['choice'] not in ('left', 'right'):
            continue
        p = pairs.get(lab['task_id'])
        if p is None:
            continue
        if lab['choice'] == 'left':
            win, lose = p['left'], p['right']
        else:
            win, lose = p['right'], p['left']
        ex.append({
            'task_id': lab['task_id'],
            'scene_id': p['scene_id'],
            'split': p['split'],
            'dataset': p['dataset'],
            'win': win,
            'lose': lose,
            'ir': p['ir'],
            'vi': p['vi'],
            'win_path': p['left_path'] if win == p['left'] else p['right_path'],
            'lose_path': p['right_path'] if win == p['left'] else p['left_path'],
            'is_gold': bool(p.get('is_gold', False)),
        })
    return ex


# ---------------------------------------------------------------------------
# Track A: linear metrics
# ---------------------------------------------------------------------------

def fit_linear(
    examples: list[dict],
    feat: dict[tuple[str, str], dict],
    keys: tuple[str, ...] = METRIC_KEYS,
    seed: int = 443,
    steps: int = 2000,
    lr: float = 0.05,
    l1: float = 1e-3,
) -> dict:
    train = [e for e in examples if e['split'] == 'train' and not e['is_gold']]
    # train-set standardization stats over all train fusions appearing in pairs
    vecs = []
    for e in train:
        for c in (e['win'], e['lose']):
            row = feat.get((e['scene_id'], c))
            if row is None:
                continue
            vecs.append([float(row[k]) for k in keys])
    if not vecs:
        raise SystemExit('no train features; run compute_features.py first')
    X = np.asarray(vecs, dtype=np.float64)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)

    def mvec(scene: str, cand: str) -> np.ndarray | None:
        row = feat.get((scene, cand))
        if row is None:
            return None
        return (np.array([float(row[k]) for k in keys], dtype=np.float64) - mu) / sigma

    diffs, keep = [], []
    for e in train:
        a, b = mvec(e['scene_id'], e['win']), mvec(e['scene_id'], e['lose'])
        if a is None or b is None:
            continue
        diffs.append(a - b)  # want S(win)>S(lose) => w·(a-b) > 0
        keep.append(e)
    D = torch.tensor(np.asarray(diffs), dtype=torch.float32)
    # w >= 0 via softplus
    raw = torch.zeros(len(keys), requires_grad=True)
    opt = torch.optim.Adam([raw], lr=lr)
    rng = random.Random(seed)
    for step in range(steps):
        opt.zero_grad()
        w = F.softplus(raw)
        s = D @ w  # (N,)
        loss = F.softplus(-s).mean() + l1 * w.abs().sum()  # logistic pairwise
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            with torch.no_grad():
                acc = (s > 0).float().mean().item()
            print(f'[linear] step {step+1} loss={loss.item():.4f} train_acc={acc:.3f}')

    with torch.no_grad():
        w = F.softplus(raw).cpu().numpy()
    return {
        'keys': list(keys),
        'mu': mu.tolist(),
        'sigma': sigma.tolist(),
        'w': w.tolist(),
        'w_normalized': (w / (w.sum() + 1e-12)).tolist(),
    }


def score_linear(model: dict, feat_row: dict) -> float:
    keys = model['keys']
    mu = np.asarray(model['mu'])
    sigma = np.asarray(model['sigma'])
    w = np.asarray(model['w'])
    x = (np.array([float(feat_row[k]) for k in keys]) - mu) / sigma
    return float(w @ x)


def eval_linear(model: dict, examples: list[dict], feat: dict, split: str) -> dict:
    rows = [e for e in examples if e['split'] == split and not e['is_gold']]
    correct = 0
    n = 0
    by_scene = defaultdict(list)
    for e in rows:
        a = feat.get((e['scene_id'], e['win']))
        b = feat.get((e['scene_id'], e['lose']))
        if a is None or b is None:
            continue
        sa, sb = score_linear(model, a), score_linear(model, b)
        hit = sa > sb
        correct += int(hit)
        n += 1
        by_scene[e['scene_id']].append((e['win'], e['lose'], sa, sb))
    # Top-1: among candidates seen in this scene's pairs, does max S match any consistent winner? skip if sparse
    # Simpler Top-1 proxy: for scenes with >=2 candidates scored via features file
    return {
        'split': split,
        'n': n,
        'pairwise_acc': correct / n if n else float('nan'),
    }


# ---------------------------------------------------------------------------
# Track B: small CNN
# ---------------------------------------------------------------------------

class ScoreNet(nn.Module):
    """s(IR, VI, F) → scalar. Tiny CNN for preference fitting."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(-1)


def _load_triple(ir_p: str, vi_p: str, f_p: str, size: int = 128) -> torch.Tensor:
    ir = cv2.imread(ir_p, cv2.IMREAD_GRAYSCALE)
    vi = cv2.imread(vi_p, cv2.IMREAD_GRAYSCALE)
    f = cv2.imread(f_p, cv2.IMREAD_GRAYSCALE)
    if ir is None or vi is None or f is None:
        raise FileNotFoundError((ir_p, vi_p, f_p))
    ir = cv2.resize(ir, (size, size), interpolation=cv2.INTER_AREA)
    vi = cv2.resize(vi, (size, size), interpolation=cv2.INTER_AREA)
    f = cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)
    x = np.stack([ir, vi, f], axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(x)


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


def fit_net(
    examples: list[dict],
    device: str,
    seed: int = 443,
    epochs: int = 8,
    batch_size: int = 32,
    lr: float = 1e-3,
    size: int = 128,
) -> tuple[ScoreNet, dict]:
    train = [e for e in examples if e['split'] == 'train' and not e['is_gold']]
    val = [e for e in examples if e['split'] == 'val' and not e['is_gold']]
    torch.manual_seed(seed)
    random.seed(seed)
    net = ScoreNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(PairDataset(train, size), batch_size=batch_size, shuffle=True, num_workers=0)
    best = {'pairwise_acc': -1.0}
    best_state = None

    def eval_split(rows: list[dict]) -> float:
        if not rows:
            return float('nan')
        net.eval()
        correct = n = 0
        with torch.no_grad():
            for i in range(0, len(rows), batch_size):
                chunk = rows[i:i + batch_size]
                wins = torch.stack([_load_triple(e['ir'], e['vi'], e['win_path'], size) for e in chunk]).to(device)
                loses = torch.stack([_load_triple(e['ir'], e['vi'], e['lose_path'], size) for e in chunk]).to(device)
                sw, sl = net(wins), net(loses)
                correct += int((sw > sl).sum().item())
                n += len(chunk)
        return correct / n if n else float('nan')

    for ep in range(1, epochs + 1):
        net.train()
        total = 0.0
        for win, lose in loader:
            win, lose = win.to(device), lose.to(device)
            opt.zero_grad()
            sw, sl = net(win), net(lose)
            loss = F.softplus(-(sw - sl)).mean()
            loss.backward()
            opt.step()
            total += loss.item() * win.size(0)
        tr_acc = eval_split(train[: min(256, len(train))])
        va_acc = eval_split(val)
        print(f'[net] epoch {ep} loss={total/max(len(train),1):.4f} train@256={tr_acc:.3f} val={va_acc:.3f}')
        if va_acc == va_acc and va_acc > best['pairwise_acc']:
            best = {'pairwise_acc': va_acc, 'epoch': ep}
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    if best_state is not None:
        net.load_state_dict(best_state)
    return net, best


@torch.no_grad()
def eval_net(net: ScoreNet, examples: list[dict], split: str, device: str, size: int = 128, batch_size: int = 32) -> dict:
    rows = [e for e in examples if e['split'] == split and not e['is_gold']]
    net.eval()
    correct = n = 0
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        wins = torch.stack([_load_triple(e['ir'], e['vi'], e['win_path'], size) for e in chunk]).to(device)
        loses = torch.stack([_load_triple(e['ir'], e['vi'], e['lose_path'], size) for e in chunk]).to(device)
        sw, sl = net(wins), net(loses)
        correct += int((sw > sl).sum().item())
        n += len(chunk)
    return {'split': split, 'n': n, 'pairwise_acc': correct / n if n else float('nan')}


def main() -> None:
    p = ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--annotator', type=str, default='ann0')
    p.add_argument('--track', choices=['linear', 'net', 'both'], default='both')
    p.add_argument('--gpu', type=str, default='3',
                   help='physical GPU id; default 3. Applied via CUDA_VISIBLE_DEVICES before torch.')
    p.add_argument('--device', type=str, default='cuda:0',
                   help='torch device after GPU pin (default cuda:0 = the pinned card)')
    p.add_argument('--net_epochs', type=int, default=8)
    p.add_argument('--linear_steps', type=int, default=2000)
    p.add_argument('--size', type=int, default=128, help='IR/VI/F resize for ScoreNet (try 256)')
    p.add_argument('--batch_size', type=int, default=32)
    args = p.parse_args()

    cfg = load_config(args.config)
    out = Path(cfg['out_root'])
    fit_dir = out / 'fit'
    fit_dir.mkdir(parents=True, exist_ok=True)

    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise SystemExit('CUDA requested but not available')
        print(
            f'[fit] CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")} '
            f'(early-pin={_PINNED_GPU}) → {torch.cuda.get_device_name(0)} | torch device={args.device}'
        )


    pairs = {r['task_id']: r for r in load_jsonl(out / 'pairs.jsonl')}
    labels = load_labels(out, args.annotator)
    examples = pairwise_examples(labels, pairs)
    print(f'[fit] labels={len(labels)} pairwise_non_tie={len(examples)} device={args.device}')
    for sp in ('train', 'val', 'test'):
        print(f'  {sp}: {sum(1 for e in examples if e["split"]==sp and not e["is_gold"])}')

    compare = {}

    if args.track in ('linear', 'both'):
        feat_path = out / 'features.jsonl'
        feat = load_features(feat_path)
        print(f'[fit] features={len(feat)} from {feat_path}')
        if len(feat) < 100:
            raise SystemExit('features.jsonl too small; run compute_features.py first')
        model = fit_linear(examples, feat, steps=args.linear_steps)
        (fit_dir / 'linear_weights.json').write_text(json.dumps(model, indent=2), encoding='utf-8')
        report = {sp: eval_linear(model, examples, feat, sp) for sp in ('train', 'val', 'test')}
        # single-metric baselines on val
        baselines = {}
        for k in METRIC_KEYS:
            correct = n = 0
            for e in examples:
                if e['split'] != 'val' or e['is_gold']:
                    continue
                a, b = feat.get((e['scene_id'], e['win'])), feat.get((e['scene_id'], e['lose']))
                if a is None or b is None:
                    continue
                correct += int(float(a[k]) > float(b[k]))
                n += 1
            baselines[k] = correct / n if n else float('nan')
        report['val_single_metric'] = baselines
        report['w'] = dict(zip(model['keys'], model['w_normalized']))
        (fit_dir / 'linear_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        compare['linear'] = report
        print('[linear] val_acc', report['val']['pairwise_acc'], 'test_acc', report['test']['pairwise_acc'])
        print('[linear] w_norm', report['w'])
        print('[linear] val single-metric', baselines)

    if args.track in ('net', 'both'):
        bs = args.batch_size
        if args.size >= 256 and bs > 16:
            bs = 16  # keep GPU3 memory modest
        print(f'[net] size={args.size} batch_size={bs} epochs={args.net_epochs}')
        net, best = fit_net(
            examples, device=args.device, epochs=args.net_epochs,
            batch_size=bs, size=args.size,
        )
        tag = f's{args.size}'
        ckpt = {
            'state_dict': net.state_dict(),
            'best': best,
            'size': args.size,
            'batch_size': bs,
        }
        torch.save(ckpt, fit_dir / f'net_{tag}.pt')
        if args.size == 128:
            torch.save(ckpt, fit_dir / 'net.pt')
        report = {
            sp: eval_net(net, examples, sp, args.device, size=args.size, batch_size=bs)
            for sp in ('train', 'val', 'test')
        }
        report['best_val_during_train'] = best
        report['size'] = args.size
        (fit_dir / f'net_{tag}_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        if args.size == 128:
            (fit_dir / 'net_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        compare[f'net_{tag}'] = report
        print(f'[net {tag}] val_acc', report['val']['pairwise_acc'], 'test_acc', report['test']['pairwise_acc'])

    # merge with existing compare.json if present
    prev_path = fit_dir / 'compare.json'
    if prev_path.is_file():
        try:
            prev = json.loads(prev_path.read_text(encoding='utf-8'))
            prev.update(compare)
            compare = prev
        except Exception:
            pass
    prev_path.write_text(json.dumps(compare, indent=2), encoding='utf-8')
    print(f'[fit] wrote → {fit_dir}')


if __name__ == '__main__':
    main()
