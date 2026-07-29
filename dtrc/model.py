"""Lightning module: frozen fusion backbone + trainable DTRC corrector (stage 2)."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW

from dtrc.corrector import DualModalityCorrector
from dtrc.losses import fusion_int_tex, rec_loss
from dtrc.vocab import IR_MVP_EXPERTS, VI_MVP_EXPERTS
from modules.fuser import load_backbone_from_checkpoint


class DTRCStage2(pl.LightningModule):
    """Teacher-forced experts: GT spatial gates, frozen F0, optimize L_src+L_ref+L_id."""

    def __init__(
        self,
        backbone_ckpt: str,
        lr: float = 1e-3,
        lambda_src: float = 1.0,
        lambda_ref: float = 1.0,
        lambda_id: float = 0.5,
        lambda_fus: float = 0.2,
        channels: int = 16,
        token_dim: int = 64,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.corrector = DualModalityCorrector(
            vi_experts=VI_MVP_EXPERTS,
            ir_experts=IR_MVP_EXPERTS,
            channels=channels,
            token_dim=token_dim,
        )
        self.backbone = load_backbone_from_checkpoint(backbone_ckpt, freeze=True)
        self.lr = lr
        self.lambda_src = lambda_src
        self.lambda_ref = lambda_ref
        self.lambda_id = lambda_id
        self.lambda_fus = lambda_fus

    def _fuse(self, ir: Tensor, vi: Tensor) -> Tensor:
        self.backbone.eval()
        return self.backbone(torch.cat([ir, vi], dim=1))

    @staticmethod
    def _enable_from_gates(gates: Tensor) -> Tensor:
        return (gates.flatten(2).amax(dim=-1) > 1e-4).to(dtype=gates.dtype)

    def training_step(self, batch, batch_idx):
        ir = batch['ir']
        vi = batch['vi']
        ir_deg = batch['ir_deg']
        vi_deg = batch['vi_deg']
        ir_gates = batch['ir_gates']
        vi_gates = batch['vi_gates']

        ir_en = self._enable_from_gates(ir_gates)
        vi_en = self._enable_from_gates(vi_gates)

        ir_hat, vi_hat, aux = self.corrector(
            ir_deg, vi_deg,
            ir_gates=ir_gates, vi_gates=vi_gates,
            ir_enable=ir_en, vi_enable=vi_en,
            neutral=False,
        )

        vi_mask = vi_gates.amax(dim=1, keepdim=True)
        ir_mask = ir_gates.amax(dim=1, keepdim=True)
        loss_src = rec_loss(vi_hat, vi, vi_mask) + rec_loss(ir_hat, ir, ir_mask)

        with torch.no_grad():
            u0 = self._fuse(ir, vi)
        ud = self._fuse(ir_hat, vi_hat)
        loss_ref = F.l1_loss(ud, u0.detach())
        loss_ref = loss_ref + 0.3 * F.l1_loss(
            ud[..., :, 1:] - ud[..., :, :-1],
            u0.detach()[..., :, 1:] - u0.detach()[..., :, :-1],
        )

        clean = (vi_mask.amax(dim=(1, 2, 3)) + ir_mask.amax(dim=(1, 2, 3))) < 1e-4
        if clean.any():
            loss_id = (
                (ir_hat[clean] - ir[clean]).abs().mean()
                + (vi_hat[clean] - vi[clean]).abs().mean()
            )
        else:
            loss_id = (
                ((1.0 - ir_mask) * (ir_hat - ir_deg).abs()).mean()
                + ((1.0 - vi_mask) * (vi_hat - vi_deg).abs()).mean()
            )

        loss_fus = fusion_int_tex(ud, ir, vi)
        loss = (
            self.lambda_src * loss_src
            + self.lambda_ref * loss_ref
            + self.lambda_id * loss_id
            + self.lambda_fus * loss_fus
        )

        self.log_dict(
            {
                'train/loss': loss,
                'train/src': loss_src,
                'train/ref': loss_ref,
                'train/gain_proxy': aux['vi']['residual'].abs().mean()
                + aux['ir']['residual'].abs().mean(),
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log('train/loss_step', loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        ir = batch['ir']
        vi = batch['vi']
        ir_deg = batch['ir_deg']
        vi_deg = batch['vi_deg']
        ir_gates = batch['ir_gates']
        vi_gates = batch['vi_gates']
        ir_en = self._enable_from_gates(ir_gates)
        vi_en = self._enable_from_gates(vi_gates)

        ir_m, vi_m, _ = self.corrector(
            ir_deg, vi_deg, ir_gates, vi_gates, ir_en, vi_en, neutral=False,
        )
        ir_n, vi_n, _ = self.corrector(
            ir_deg, vi_deg, ir_gates, vi_gates, ir_en, vi_en, neutral=True,
        )

        vi_mask = vi_gates.amax(dim=1, keepdim=True)
        ir_mask = ir_gates.amax(dim=1, keepdim=True)
        err_m = masked_mean_abs(vi_m, vi, vi_mask) + masked_mean_abs(ir_m, ir, ir_mask)
        err_n = masked_mean_abs(vi_n, vi, vi_mask) + masked_mean_abs(ir_n, ir, ir_mask)
        self.log('val/err_matched', err_m, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('val/err_neutral', err_n, on_epoch=True, sync_dist=True)
        self.log('val/gain', err_n - err_m, prog_bar=True, on_epoch=True, sync_dist=True)
        return err_m

    def configure_optimizers(self):
        params = [p for p in self.corrector.parameters() if p.requires_grad]
        return AdamW(params, lr=self.lr, weight_decay=1e-4)


def masked_mean_abs(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    w = mask.clamp(0, 1)
    if w.amax() < 1e-6:
        return (pred - target).abs().mean() * 0.0
    return ((pred - target).abs() * w).sum() / w.sum().clamp(min=1.0)
