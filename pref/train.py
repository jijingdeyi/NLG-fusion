"""Train SharedScorer with R-weighted preference CE (score.md §4.4)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pref.dataset import PreferencePairDataset
from pref.io import export_supervision_jsonl, load_jsonl, load_yaml
from pref.model import SharedScorer, scorer_pair_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def eval_loader(model: SharedScorer, loader: DataLoader, device: str, delta: float, sigma0: float) -> dict:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for x_a, x_b, y, R in loader:
        x_a = x_a.to(device)
        x_b = x_b.to(device)
        y = y.to(device)
        R = R.to(device)
        loss, stats = scorer_pair_loss(
            model, x_a, x_b, y, R, delta=delta, sigma0=sigma0, swap_prob=0.0,
        )
        bs = x_a.size(0)
        total_loss += float(loss) * bs
        total_acc += stats['acc_mode'] * bs
        n += bs
    return {
        'n': n,
        'loss': total_loss / n if n else float('nan'),
        'acc_mode': total_acc / n if n else float('nan'),
    }


def train_scorer(
    out_root: Path | str,
    cfg_path: Path | str | None = None,
    device: str = 'cuda:0',
    epochs: int = 12,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    size: int = 128,
    delta: float = 0.25,
    sigma0: float = 0.1,
    eta: float = 0.5,
    seed: int = 443,
    num_workers: int = 2,
) -> dict:
    out = Path(out_root)
    cfg = load_yaml(cfg_path) if cfg_path else {}
    set_seed(int(cfg.get('seed', seed)))

    superv_path = export_supervision_jsonl(out, eta=float(cfg.get('eta', eta)), cfg=cfg)
    rows = load_jsonl(superv_path)
    train_rows = [r for r in rows if r.get('split') == 'train']
    val_rows = [r for r in rows if r.get('split') == 'val']

    train_ds = PreferencePairDataset(train_rows, size=size, augment=True)
    val_ds = PreferencePairDataset(val_rows, size=size, augment=False)
    if len(train_ds) == 0:
        raise SystemExit(f'no train pairs with supervision under {out}')

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    model = SharedScorer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    fit_dir = out / 'fit'
    fit_dir.mkdir(parents=True, exist_ok=True)
    best = {'val_loss': float('inf'), 'epoch': -1}
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for x_a, x_b, y, R in train_loader:
            x_a, x_b, y, R = x_a.to(device), x_b.to(device), y.to(device), R.to(device)
            loss, _ = scorer_pair_loss(
                model, x_a, x_b, y, R, delta=delta, sigma0=sigma0, swap_prob=0.5,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss) * x_a.size(0)
            n += x_a.size(0)
        train_loss = running / max(n, 1)
        val_stats = eval_loader(model, val_loader, device, delta, sigma0) if len(val_ds) else {
            'n': 0, 'loss': float('nan'), 'acc_mode': float('nan'),
        }
        row = {'epoch': epoch, 'train_loss': train_loss, **{f'val_{k}': v for k, v in val_stats.items()}}
        history.append(row)
        print(
            f'[scorer] epoch {epoch}/{epochs} '
            f'train_loss={train_loss:.4f} val_loss={val_stats["loss"]:.4f} '
            f'val_acc={val_stats["acc_mode"]:.3f}'
        )
        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'delta': delta,
            'sigma0': sigma0,
            'size': size,
        }
        torch.save(ckpt, fit_dir / 'scorer_last.pt')
        if val_stats['n'] == 0 or val_stats['loss'] <= best['val_loss']:
            best = {'val_loss': val_stats['loss'], 'epoch': epoch, 'val_acc': val_stats['acc_mode']}
            torch.save(ckpt, fit_dir / 'scorer_best.pt')

    report = {
        'supervision': str(superv_path),
        'n_train': len(train_ds),
        'n_val': len(val_ds),
        'best': best,
        'history': history,
    }
    (fit_dir / 'scorer_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
