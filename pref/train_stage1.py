"""Stage-1 training: pairwise soft ranking → shared scalar scorer → s_F.

Does NOT train σ / A-Tie-B heads. Goal: show that (y, R) preferences alone
can produce a meaningful continuous score on unseen scenes.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pref.dataset import PreferencePairDataset, make_scene_loader
from pref.graph import connectivity_report
from pref.io import export_supervision_jsonl, load_jsonl, load_yaml
from pref.model.loss import scorer_rank_loss
from pref.model.resnet_scorer import ResNet18ScalarScorer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_optimizer(
    model: ResNet18ScalarScorer,
    *,
    lr_head: float,
    lr_backbone: float,
    weight_decay: float,
    backbone_trainable: bool,
) -> torch.optim.Optimizer:
    model.set_backbone_trainable(backbone_trainable)
    params = [{'params': list(model.head_parameters()), 'lr': lr_head}]
    if backbone_trainable:
        params.append({'params': list(model.backbone_parameters()), 'lr': lr_backbone})
    return torch.optim.AdamW(params, weight_decay=weight_decay)


@torch.no_grad()
def eval_rank(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    tau: float = 1.0,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_w = 0.0
    n_dec = 0
    n_dec_ok = 0
    n_soft_ok = 0
    n = 0
    for x_a, x_b, y, R in loader:
        x_a = x_a.to(device)
        x_b = x_b.to(device)
        y = y.to(device)
        R = R.to(device)
        loss, stats = scorer_rank_loss(
            model, x_a, x_b, y, R, tau=tau, swap_prob=0.0, use_R=False,
        )
        bs = x_a.size(0)
        # recompute decisive stats with R weighting for reporting
        s_a = model(x_a)
        s_b = model(x_b)
        if isinstance(s_a, tuple):
            s_a, s_b = s_a[0], s_b[0]
        q = y[:, 0] + 0.5 * y[:, 1]
        q_hat = torch.sigmoid((s_a - s_b) / float(tau))
        decisive = (q - 0.5).abs() >= 0.1
        soft_ok = ((q_hat >= 0.5) == (q >= 0.5)).float()
        n_soft_ok += float(soft_ok.sum())
        if decisive.any():
            n_dec += int(decisive.sum())
            n_dec_ok += int((((q_hat[decisive] >= 0.5) == (q[decisive] >= 0.5))).sum())
        wsum = float(R.sum().clamp_min(1e-8))
        total_loss += float(loss) * wsum
        total_w += wsum
        n += bs
    return {
        'n': n,
        'loss': total_loss / total_w if total_w else float('nan'),
        'acc_soft': n_soft_ok / n if n else float('nan'),
        'acc_decisive': (n_dec_ok / n_dec) if n_dec else float('nan'),
        'n_decisive': n_dec,
    }


# Candidates derived from a held-out method (same image content / style family).
# deg_* are brightness degradations of method_a — must leave with A.
METHOD_DERIVATIVES: dict[str, tuple[str, ...]] = {
    'method_a': ('deg_overexp', 'deg_underexp'),
}


def holdout_blocked_candidates(method: str) -> frozenset[str]:
    """Ids removed from train+val when leaving method out (incl. derivatives)."""
    return frozenset((method, *METHOD_DERIVATIVES.get(method, ())))


def pair_candidate_ids(row: dict) -> frozenset[str]:
    a = str(row.get('candidate_a') or row.get('left') or '')
    b = str(row.get('candidate_b') or row.get('right') or '')
    return frozenset({a, b})


def _pair_involves_method(row: dict, method: str) -> bool:
    """True iff the pair involves method itself (not derivatives). For test held-out."""
    return method in pair_candidate_ids(row)


def _pair_blocked_by_holdout(row: dict, method: str) -> bool:
    """True iff pair must be removed from train/val for a clean LOMO fold."""
    return bool(pair_candidate_ids(row) & holdout_blocked_candidates(method))


def train_scorer_stage1(
    out_root: Path | str,
    cfg_path: Path | str | None = None,
    device: str = 'cuda:0',
    epochs: int = 20,
    freeze_epochs: int = 4,
    batch_size: int = 16,
    lr_head: float = 1e-4,
    lr_backbone: float = 1e-5,
    weight_decay: float = 1e-4,
    size: int = 224,
    tau: float = 1.0,
    eta: float = 0.5,
    seed: int = 443,
    num_workers: int = 2,
    control_downweight: float = 0.35,
    pretrained: bool = True,
    input_mode: str = 'ir_vis_f',
    holdout_method: str | None = None,
    matched_train_size: int | None = None,
    run_tag: str | None = None,
    refresh_supervision: bool = False,
) -> dict:
    out = Path(out_root)
    cfg = load_yaml(cfg_path) if cfg_path else {}
    set_seed(int(cfg.get('seed', seed)))

    superv_path = out / 'supervision.jsonl'
    if refresh_supervision or not superv_path.is_file():
        superv_path = export_supervision_jsonl(out, eta=float(cfg.get('eta', eta)), cfg=cfg)
    rows = load_jsonl(superv_path)
    train_rows = [r for r in rows if r.get('split') == 'train']
    val_rows = [r for r in rows if r.get('split') == 'val']
    if holdout_method:
        blocked = holdout_blocked_candidates(holdout_method)
        before_tr, before_va = len(train_rows), len(val_rows)
        train_rows = [r for r in train_rows if not _pair_blocked_by_holdout(r, holdout_method)]
        # Checkpoint selection must not see M (or its derivatives) either.
        val_rows = [r for r in val_rows if not _pair_blocked_by_holdout(r, holdout_method)]
        print(
            f'[stage1] holdout_method={holdout_method} blocked={sorted(blocked)}: '
            f'train {before_tr} → {len(train_rows)}, '
            f'val {before_va} → {len(val_rows)}',
            flush=True,
        )
    elif matched_train_size is not None:
        # Size-matched control: random train subsample; M still allowed.
        n_keep = int(matched_train_size)
        if n_keep < len(train_rows):
            rng = random.Random(int(cfg.get('seed', seed)))
            train_rows = rng.sample(train_rows, n_keep)
            print(
                f'[stage1] matched_train_size={n_keep}: '
                f'random train subsample (val unchanged, methods kept)',
                flush=True,
            )
        else:
            print(
                f'[stage1] matched_train_size={n_keep} >= train {len(train_rows)}; no subsample',
                flush=True,
            )

    tag = run_tag or (
        f'lomo_{holdout_method}' if holdout_method else
        (
            f'matched_n{matched_train_size}' if matched_train_size is not None else
            (f'in_{input_mode}' if input_mode != 'ir_vis_f' else 'stage1')
        )
    )

    graph = {
        'train': connectivity_report(train_rows, split=None),
        'val': connectivity_report(val_rows, split=None),
        'all': connectivity_report(rows),
    }
    print(
        f'[stage1:{tag}] input_mode={input_mode} '
        f'train_scenes={graph["train"]["n_scenes"]} '
        f'disconnected={graph["train"]["n_disconnected"]}',
        flush=True,
    )

    train_ds = PreferencePairDataset(
        train_rows, size=size, augment=True, exclude_gold=True, exclude_control=False,
        input_mode=input_mode,
    )
    val_ds = PreferencePairDataset(
        val_rows, size=size, augment=False, exclude_gold=True, exclude_control=False,
        input_mode=input_mode,
    )
    if len(train_ds) == 0:
        raise SystemExit(f'no train pairs with supervision under {out}')

    train_loader = make_scene_loader(
        train_ds, batch_size, shuffle=True, num_workers=num_workers,
        control_downweight=control_downweight, seed=seed,
    )
    val_loader = make_scene_loader(
        val_ds, batch_size, shuffle=False, num_workers=num_workers,
    )

    model = ResNet18ScalarScorer(pretrained=pretrained, freeze_backbone=True).to(device)
    opt = _build_optimizer(
        model, lr_head=lr_head, lr_backbone=lr_backbone,
        weight_decay=weight_decay, backbone_trainable=False,
    )

    fit_dir = out / 'fit'
    fit_dir.mkdir(parents=True, exist_ok=True)
    best_path = fit_dir / f'scorer_{tag}_best.pt'
    last_path = fit_dir / f'scorer_{tag}_last.pt'
    report_path = fit_dir / f'scorer_{tag}_report.json'

    best = {'val_loss': float('inf'), 'epoch': -1}
    history = []
    backbone_on = False

    for epoch in range(1, epochs + 1):
        if (not backbone_on) and epoch > freeze_epochs:
            backbone_on = True
            opt = _build_optimizer(
                model, lr_head=lr_head, lr_backbone=lr_backbone,
                weight_decay=weight_decay, backbone_trainable=True,
            )
            print(f'[stage1:{tag}] unfreeze backbone at epoch {epoch}', flush=True)

        model.train()
        running = 0.0
        n = 0
        for x_a, x_b, y, R in train_loader:
            x_a, x_b, y, R = x_a.to(device), x_b.to(device), y.to(device), R.to(device)
            loss, _ = scorer_rank_loss(
                model, x_a, x_b, y, R, tau=tau, swap_prob=0.5, use_R=False,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.detach()) * x_a.size(0)
            n += x_a.size(0)
        train_loss = running / max(n, 1)
        val_stats = eval_rank(model, val_loader, device, tau=tau) if len(val_ds) else {
            'n': 0, 'loss': float('nan'), 'acc_soft': float('nan'),
            'acc_decisive': float('nan'), 'n_decisive': 0,
        }
        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'backbone': 'full' if backbone_on else 'frozen',
            **{f'val_{k}': v for k, v in val_stats.items()},
        }
        history.append(row)
        print(
            f'[stage1:{tag}] epoch {epoch}/{epochs} [{row["backbone"]}] '
            f'train_loss={train_loss:.4f} val_loss={val_stats["loss"]:.4f} '
            f'val_acc_soft={val_stats["acc_soft"]:.3f} '
            f'val_acc_dec={val_stats["acc_decisive"]:.3f}',
            flush=True,
        )
        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'stage': 1,
            'arch': 'resnet18_scalar',
            'tau': tau,
            'size': size,
            'pretrained': pretrained,
            'input_mode': input_mode,
            'holdout_method': holdout_method,
            'matched_train_size': matched_train_size,
            'run_tag': tag,
        }
        torch.save(ckpt, last_path)
        if val_stats['n'] == 0 or val_stats['loss'] <= best['val_loss']:
            best = {
                'val_loss': val_stats['loss'],
                'epoch': epoch,
                'val_acc_soft': val_stats['acc_soft'],
                'val_acc_decisive': val_stats['acc_decisive'],
            }
            torch.save(ckpt, best_path)

    report = {
        'supervision': str(superv_path),
        'run_tag': tag,
        'input_mode': input_mode,
        'holdout_method': holdout_method,
        'matched_train_size': matched_train_size,
        'n_train': len(train_ds),
        'n_val': len(val_ds),
        'best_ckpt': str(best_path),
        'best': best,
        'history': history,
        'hparams': {
            'epochs': epochs,
            'freeze_epochs': freeze_epochs,
            'batch_size': batch_size,
            'lr_head': lr_head,
            'lr_backbone': lr_backbone,
            'size': size,
            'tau': tau,
            'control_downweight': control_downweight,
            'input_mode': input_mode,
            'holdout_method': holdout_method,
            'matched_train_size': matched_train_size,
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
