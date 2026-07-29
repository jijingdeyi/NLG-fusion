#!/usr/bin/env python3
"""DTRC stage-2: teacher-forced experts + frozen fusion backbone.

Milestone: val/gain = err_neutral - err_matched should become clearly positive.
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dtrc.dataset import DTRCPairDataset
from dtrc.model import DTRCStage2


def main():
    p = ArgumentParser(description='Train DTRC stage-2 (GT-gated experts, frozen F0)')
    p.add_argument('--data', type=str, required=True)
    p.add_argument(
        '--backbone_ckpt', type=str,
        default='checkpoint/ldt_20260727_2110/ldt_20260727_2110-best-eepoch=007.ckpt',
    )
    p.add_argument('--ckpt', type=str, default='checkpoint')
    p.add_argument('--log_dir', type=str, default='logs')
    p.add_argument('--bs', type=int, default=8)
    p.add_argument('--val_bs', type=int, default=4)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--crop_size', type=int, default=320)
    p.add_argument('--val_samples', type=int, default=30)
    p.add_argument('--clean_prob', type=float, default=0.3)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--seed', type=int, default=443)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--no_cuda', action='store_true')
    args = p.parse_args()

    pl.seed_everything(args.seed)
    from datetime import datetime
    run_name = f'dtrc_stage2_{datetime.now().strftime("%Y%m%d_%H%M")}'
    run_dir = Path(args.ckpt) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model = DTRCStage2(backbone_ckpt=args.backbone_ckpt, lr=args.lr)

    train_loader = DataLoader(
        DTRCPairDataset(
            args.data, 'train', crop_size=args.crop_size,
            val_samples=args.val_samples, clean_prob=args.clean_prob,
        ),
        batch_size=args.bs, shuffle=True, num_workers=4,
        collate_fn=_collate, pin_memory=not args.no_cuda,
    )
    val_loader = DataLoader(
        DTRCPairDataset(
            args.data, 'val', crop_size=args.crop_size,
            val_samples=args.val_samples, clean_prob=0.2,
        ),
        batch_size=args.val_bs, shuffle=False, num_workers=4,
        collate_fn=_collate, pin_memory=not args.no_cuda,
    )

    logger = TensorBoardLogger(
        save_dir=args.log_dir, name='dtrc', version=run_name,
        default_hp_metric=False,
    )
    callbacks = [
        ModelCheckpoint(
            dirpath=str(run_dir),
            filename=f'{run_name}-best-e{{epoch:03d}}',
            monitor='val/gain',
            mode='max',
            save_top_k=1,
        ),
        EarlyStopping(monitor='val/gain', mode='max', patience=args.patience),
        LearningRateMonitor(logging_interval='epoch'),
    ]

    trainer = pl.Trainer(
        accelerator='cpu' if args.no_cuda else 'gpu',
        devices=1,
        logger=logger,
        callbacks=callbacks,
        max_epochs=args.epochs,
        log_every_n_steps=50,
    )
    print(f'[dtrc] run={run_name} backbone={args.backbone_ckpt}')
    trainer.fit(model, train_loader, val_loader)


def _collate(batch: list[dict]) -> dict:
    keys = batch[0].keys()
    out = {}
    for k in keys:
        items = [b[k] for b in batch]
        out[k] = items if isinstance(items[0], str) else torch.stack(items)
    return out


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    main()
