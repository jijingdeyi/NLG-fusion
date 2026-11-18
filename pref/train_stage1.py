"""Train a scorer on vote-share preferences.

The default model is the region baseline: infrared is gray, visible and fusion
are Y, Cb, Cr, and the native-resolution feature map yields a region score
plus a whole-image score.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pref.dataset import BaselinePairDataset, baseline_collate, make_scene_loader
from pref.graph import connectivity_report
from pref.io import export_supervision_jsonl, load_jsonl, load_yaml, resolve_pack_paths
from pref.model.baseline import BaselineScorer
from pref.model.feature_scorer import FeatureScorer
from pref.model.region_scorer import RegionScorer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_optimizer(
    model: BaselineScorer,
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


def _freeze_batchnorm(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def _vote_share(y: torch.Tensor) -> torch.Tensor:
    return y[:, 0] + 0.5 * y[:, 1]


def _pair_loss(score_a: torch.Tensor, score_b: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(score_a - score_b, target)


def _score_fusion(model, ir, vi, fused, device) -> torch.Tensor:
    return model(
        ir.unsqueeze(0).to(device),
        vi.unsqueeze(0).to(device),
        fused.unsqueeze(0).to(device),
    ).reshape(())


@torch.no_grad()
def eval_rank(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    tau: float = 1.0,
) -> dict:
    del tau
    model.eval()
    total_loss = 0.0
    n_dec = 0
    n_dec_ok = 0.0
    n_soft_ok = 0.0
    n = 0
    for irs, vis, fused_as, fused_bs, y, _reliability in loader:
        y = y.to(device)
        q = _vote_share(y)
        scores_a, scores_b = [], []
        for ir, vi, fused_a, fused_b in zip(irs, vis, fused_as, fused_bs):
            scores_a.append(_score_fusion(model, ir, vi, fused_a, device))
            scores_b.append(_score_fusion(model, ir, vi, fused_b, device))
        score_a = torch.stack(scores_a)
        score_b = torch.stack(scores_b)
        total_loss += float(_pair_loss(score_a, score_b, q).detach()) * len(q)
        decisive = (q <= 0.4) | (q >= 0.6)
        credit = ((score_a > score_b) == (q > 0.5)).float()
        credit[score_a == score_b] = 0.5
        n_dec += int(decisive.sum())
        n_dec_ok += float(credit[decisive].sum())
        n_soft_ok += float(((score_a >= score_b) == (q >= 0.5)).float().sum())
        n += len(q)
    return {
        'n': n,
        'loss': total_loss / n if n else float('nan'),
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


def _make_scorer(model_name: str, *, pretrained: bool, size: int):
    if model_name == 'baseline':
        if size != 224:
            raise SystemExit('The global view and the tile size stay 224.')
        return BaselineScorer(pretrained=pretrained, global_size=size, tile_size=size), 'baseline_ycbcr_tiles'
    if model_name == 'region':
        return RegionScorer(pretrained=pretrained), 'region_ycbcr'
    if model_name == 'feature':
        return FeatureScorer(pretrained=pretrained), 'feature_ycbcr'
    raise SystemExit('model_name must be baseline, region, or feature')


def train_scorer_stage1(
    out_root: Path | str,
    cfg_path: Path | str | None = None,
    device: str = 'cuda:0',
    epochs: int = 100,
    freeze_epochs: int = 10,
    patience: int = 20,
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
    model_name: str = 'region',
) -> dict:
    out = Path(out_root)
    cfg = load_yaml(cfg_path) if cfg_path else {}
    set_seed(int(cfg.get('seed', seed)))

    superv_path = out / 'supervision.jsonl'
    if refresh_supervision or not superv_path.is_file():
        superv_path = export_supervision_jsonl(out, eta=float(cfg.get('eta', eta)), cfg=cfg)
    rows = resolve_pack_paths(load_jsonl(superv_path), out)
    train_rows = [
        r for r in rows
        if r.get('split') == 'train'
        and r.get('dataset') != 'RoadScene'
        and r.get('protocol', 'main') == 'main'
    ]
    val_rows = [
        r for r in rows
        if r.get('split') == 'val'
        and r.get('dataset') != 'RoadScene'
        and r.get('protocol', 'main') == 'main'
    ]
    if holdout_method:
        blocked = holdout_blocked_candidates(holdout_method)
        before_tr, before_va = len(train_rows), len(val_rows)
        train_rows = [r for r in train_rows if not _pair_blocked_by_holdout(r, holdout_method)]
        val_rows = [r for r in val_rows if not _pair_blocked_by_holdout(r, holdout_method)]
        print(
            f'[stage1] holdout_method={holdout_method} blocked={sorted(blocked)}: '
            f'train {before_tr} → {len(train_rows)}, '
            f'val {before_va} → {len(val_rows)}',
            flush=True,
        )
    elif matched_train_size is not None:
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
            (f'in_{input_mode}' if input_mode != 'ir_vis_f' else model_name)
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

    if input_mode != 'ir_vis_f':
        raise SystemExit(
            'The scorer reads infrared as gray and visible and fusion as Y, Cb, Cr.'
        )
    train_ds = BaselinePairDataset(
        train_rows, size=size, augment=True, exclude_gold=True, exclude_control=False,
    )
    val_ds = BaselinePairDataset(
        val_rows, size=size, augment=False, exclude_gold=True, exclude_control=False,
    )
    if len(train_ds) == 0:
        raise SystemExit(f'no train pairs with supervision under {out}')

    train_loader = make_scene_loader(
        train_ds, batch_size, shuffle=True, num_workers=num_workers,
        control_downweight=control_downweight, seed=seed, collate_fn=baseline_collate,
    )
    val_loader = make_scene_loader(
        val_ds, batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=baseline_collate,
    )

    model, arch = _make_scorer(model_name, pretrained=pretrained, size=size)
    model = model.to(device)
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
    stall = 0
    stopped_early = False

    for epoch in range(1, epochs + 1):
        if (not backbone_on) and epoch > freeze_epochs:
            backbone_on = True
            opt = _build_optimizer(
                model, lr_head=lr_head, lr_backbone=lr_backbone,
                weight_decay=weight_decay, backbone_trainable=True,
            )
            print(f'[stage1:{tag}] unfreeze backbone at epoch {epoch}', flush=True)

        model.train()
        _freeze_batchnorm(model)
        running = 0.0
        n = 0
        for irs, vis, fused_as, fused_bs, y, _reliability in train_loader:
            y = y.to(device)
            q = _vote_share(y)
            if random.random() < 0.5:
                fused_as, fused_bs, q = fused_bs, fused_as, 1.0 - q
            opt.zero_grad(set_to_none=True)
            parts = []
            for ir, vi, fused_a, fused_b, target in zip(irs, vis, fused_as, fused_bs, q):
                score_a = _score_fusion(model, ir, vi, fused_a, device)
                score_b = _score_fusion(model, ir, vi, fused_b, device)
                part = _pair_loss(score_a.reshape(1), score_b.reshape(1), target.reshape(1))
                (part / len(q)).backward()
                parts.append(part.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(torch.stack(parts).mean()) * len(q)
            n += len(q)
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
            'arch': arch,
            'tau': tau,
            'size': size,
            'pretrained': pretrained,
            'input_mode': input_mode,
            'holdout_method': holdout_method,
            'matched_train_size': matched_train_size,
            'run_tag': tag,
        }
        torch.save(ckpt, last_path)
        improved = val_stats['n'] == 0 or val_stats['loss'] < best['val_loss']
        if improved:
            best = {
                'val_loss': val_stats['loss'],
                'epoch': epoch,
                'val_acc_soft': val_stats['acc_soft'],
                'val_acc_decisive': val_stats['acc_decisive'],
            }
            torch.save(ckpt, best_path)
            stall = 0
        else:
            stall += 1
        report = _report(
            superv_path, tag, input_mode, holdout_method, matched_train_size,
            train_ds, val_ds, best_path, best, history, epochs, freeze_epochs,
            patience, batch_size, lr_head, lr_backbone, size, tau,
            control_downweight, arch, stopped_early,
        )
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        if epoch >= freeze_epochs and stall >= patience:
            stopped_early = True
            report['stopped_early'] = True
            report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(
                f'[stage1:{tag}] stop at epoch {epoch}: '
                f'val loss did not fall for {patience} epochs',
                flush=True,
            )
            break

    return report


def _report(
    superv_path, tag, input_mode, holdout_method, matched_train_size,
    train_ds, val_ds, best_path, best, history, epochs, freeze_epochs,
    patience, batch_size, lr_head, lr_backbone, size, tau,
    control_downweight, arch, stopped_early,
) -> dict:
    return {
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
            'patience': patience,
            'batch_size': batch_size,
            'lr_head': lr_head,
            'lr_backbone': lr_backbone,
            'size': size,
            'tau': tau,
            'target': 'vote_share',
            'model': arch,
            'stopped_early': stopped_early,
            'control_downweight': control_downweight,
            'input_mode': input_mode,
            'holdout_method': holdout_method,
            'matched_train_size': matched_train_size,
        },
        'stopped_early': stopped_early,
    }
