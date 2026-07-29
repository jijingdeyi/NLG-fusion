"""Fusion-only Lightning module (backbone train / val / predict)."""

from __future__ import annotations

import math
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import torch
from kornia.losses import ssim_loss
from torch import Tensor, nn
from torch.nn.functional import l1_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ldt.defaults import (
    DEFAULT_DIM,
    DEFAULT_F_WEIGHT,
    DEFAULT_LR,
    DEFAULT_LR_SCHED_PATIENCE,
    DEFAULT_PRIOR_HF_ALPHA,
)
from ldt.fusion_metrics import composite_validation_score, compute_fusion_metrics
from modules.fuser import Fuser


class GradLoss(nn.Module):
    def forward(self, vi: Tensor, ir: Tensor, fused: Tensor) -> Tensor:
        def _grad(x: Tensor) -> Tensor:
            dx = x[:, :, :, 1:] - x[:, :, :, :-1]
            dy = x[:, :, 1:, :] - x[:, :, :-1, :]
            dx = torch.nn.functional.pad(dx, (0, 1, 0, 0))
            dy = torch.nn.functional.pad(dy, (0, 0, 0, 1))
            return dx.abs() + dy.abs()

        return l1_loss(_grad(fused), torch.max(_grad(vi), _grad(ir)))


class LDTFusion(pl.LightningModule):
    """IR–VI fusion training on the lightweight backbone only."""

    def __init__(self, args: Namespace):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.dim = args.dim
        self.lr = args.lr
        self.f_weight = list(args.f_weight)
        self.lr_sched_patience = args.lr_sched_patience
        self.prior_hf_alpha = float(getattr(args, 'prior_hf_alpha', DEFAULT_PRIOR_HF_ALPHA))
        self.fuser = Fuser(dim=self.dim, use_bn=False, hf_alpha=self.prior_hf_alpha)
        self.grad_loss = GradLoss()

    def _fusion_terms(self, fused: Tensor, ir: Tensor, vi: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # Classical intensity target; residual base m blends max / max_hf via α.
        int_loss = l1_loss(fused, torch.max(ir, vi))
        tex_loss = self.grad_loss(vi, ir, fused)
        if self.f_weight[1] > 0:
            x_ssim = ssim_loss(fused, ir, window_size=11, reduction='none')
            y_ssim = ssim_loss(fused, vi, window_size=11, reduction='none')
            sim_loss = x_ssim * 0.5 + y_ssim * 0.5
        else:
            sim_loss = fused.new_zeros(())
        return int_loss, sim_loss, tex_loss

    def training_step(self, batch, batch_idx):
        x, y = batch['ir'], batch['vi']
        f = self.fuser(torch.cat([x, y], dim=1))
        int_l, sim_l, tex_l = self._fusion_terms(f, x, y)
        loss = (
            self.f_weight[0] * int_l
            + self.f_weight[1] * sim_l
            + self.f_weight[2] * tex_l
        ).mean()
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/int', int_l.mean(), on_step=False, on_epoch=True)
        self.log('train/tex', tex_l.mean(), on_step=False, on_epoch=True)
        return loss

    def on_before_optimizer_step(self, optimizer):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

    def validation_step(self, batch, batch_idx):
        x, y = batch['ir'], batch['vi']
        f = self.fuser(torch.cat([x, y], dim=1)).clamp(0, 1)
        if not torch.isfinite(f).all():
            f = torch.nan_to_num(f, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)

        totals = {'mi': 0.0, 'qabf': 0.0, 'scd': 0.0, 'vif': 0.0, 'ssim': 0.0}
        for i in range(x.shape[0]):
            ir_np = (x[i, 0].detach().cpu().numpy() * 255.0).astype('float32')
            vi_np = (y[i, 0].detach().cpu().numpy() * 255.0).astype('float32')
            f_np = (f[i, 0].detach().cpu().numpy() * 255.0).astype('float32')
            metrics = compute_fusion_metrics(ir_np, vi_np, f_np)
            for key in totals:
                totals[key] += metrics[key]

        n = x.shape[0]
        for key, value in totals.items():
            self.log(
                f'val/{key}', value / n,
                on_step=False, on_epoch=True, sync_dist=True, batch_size=n,
            )
        if batch_idx == 0:
            return {
                'ir': x.detach(),
                'vi': y.detach(),
                'fused': f.detach(),
                'name': batch.get('name'),
            }
        return None

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        required = ('val/mi', 'val/qabf', 'val/scd', 'val/vif', 'val/ssim')
        if not all(k in metrics for k in required):
            self.log('val/score', 0.0, prog_bar=True, on_epoch=True, sync_dist=True)
            return
        values = [float(metrics[k]) for k in required]
        if not all(math.isfinite(v) for v in values):
            self.log('val/score', 0.0, prog_bar=True, on_epoch=True, sync_dist=True)
            return
        score = composite_validation_score(*values)
        if not math.isfinite(score):
            score = 0.0
        self.log('val/score', score, prog_bar=True, on_epoch=True, sync_dist=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return batch['name'], self.f_forward(batch['ir'], batch['vi'])

    def f_forward(self, ir: Tensor, vi: Tensor, **_ignored) -> Tensor:
        return self.fuser(torch.cat([ir, vi], dim=1))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=self.lr_sched_patience,
        )
        return [optimizer], {
            'scheduler': scheduler,
            'monitor': 'val/score',
            'interval': 'epoch',
        }

    @staticmethod
    def add_model_specific_args(parent: ArgumentParser) -> ArgumentParser:
        parser = parent.add_argument_group('Fusion backbone')
        parser.add_argument('--dim', type=int, default=DEFAULT_DIM)
        parser.add_argument('--lr', type=float, default=DEFAULT_LR)
        parser.add_argument(
            '--f_weight', type=float, nargs=3, default=list(DEFAULT_F_WEIGHT),
            metavar=('INT', 'SSIM', 'TEX'),
        )
        parser.add_argument(
            '--prior_hf_alpha', type=float, default=DEFAULT_PRIOR_HF_ALPHA,
            help='Prior blend: m=(1-α)·max + α·max_hf',
        )
        parser.add_argument('--lr_sched_patience', type=int, default=DEFAULT_LR_SCHED_PATIENCE)
        return parent
