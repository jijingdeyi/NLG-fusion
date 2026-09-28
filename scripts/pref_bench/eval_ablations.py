#!/usr/bin/env python3
"""Evaluate source-input and leave-one-method-out Stage-1 ablations.

LOMO protocol (clean):
  - train+val never see held-out method M (nor method_a derivatives deg_*)
  - report on identical D_test^(M) for: default model, LOMO, matched-size, traditional metrics
"""

from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from pref.dataset import PreferencePairDataset
from pref.eval_stage1 import (
    evaluate_metric_baselines,
    load_feature_table,
    load_stage1_scorer,
    score_rows,
    summarize_scorer,
)
from pref.io import load_jsonl, load_yaml
from pref.train_stage1 import _pair_involves_method


def _score_mode(
    model,
    rows: list[dict],
    *,
    size: int,
    device: str,
    batch_size: int,
    input_mode: str,
) -> list[dict]:
    from torch.utils.data import DataLoader
    import math

    ds = PreferencePairDataset(
        rows, size=size, augment=False, exclude_gold=False, input_mode=input_mode,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    recs = []
    model.eval()
    cursor = 0
    with torch.no_grad():
        for x_a, x_b, y, R in loader:
            x_a, x_b = x_a.to(device), x_b.to(device)
            s_a, s_b = model(x_a), model(x_b)
            q = (y[:, 0] + 0.5 * y[:, 1]).to(device)
            q_hat = torch.sigmoid(s_a - s_b)
            for i in range(x_a.size(0)):
                r = ds.rows[cursor + i]
                qi, qhi = float(q[i]), float(q_hat[i])
                decisive = abs(qi - 0.5) >= 0.1
                soft_ok = (qhi >= 0.5) == (qi >= 0.5)
                eps = 1e-8
                qhc = min(max(qhi, eps), 1 - eps)
                bce = -(qi * math.log(qhc) + (1 - qi) * math.log(1 - qhc))
                recs.append({
                    'pair_id': r.get('pair_id'),
                    'scene_id': r.get('scene_id'),
                    'candidate_a': r.get('candidate_a'),
                    'candidate_b': r.get('candidate_b'),
                    'path_a': r.get('path_a'),
                    'path_b': r.get('path_b'),
                    'q': qi, 'R': float(R[i]), 'bce': bce,
                    'decisive': decisive, 'soft_ok': soft_ok,
                    'dec_ok': soft_ok if decisive else None,
                })
            cursor += x_a.size(0)
    return recs


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--skip-source', action='store_true')
    args = ap.parse_args()
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    cfg = load_yaml(args.config)
    out = Path(cfg['out_root'])
    fit = out / 'fit'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    rows = load_jsonl(out / 'supervision.jsonl')
    test_rows = [r for r in rows if r.get('split') == 'test']
    feats = load_feature_table(out / 'features.jsonl')

    report: dict = {'source_ablation': {}, 'lomo': {}}
    prev_path = fit / 'ablation_source_lomo_report.json'
    if args.skip_source and prev_path.is_file():
        try:
            prev = json.loads(prev_path.read_text(encoding='utf-8'))
            report['source_ablation'] = prev.get('source_ablation') or {}
        except (OSError, json.JSONDecodeError):
            pass

    # --- source ablation ---
    if not args.skip_source:
        modes = [
            ('ir_vis_f', fit / 'scorer_stage1_best.pt'),
            ('f_only', fit / 'scorer_in_f_only_best.pt'),
            ('ir_f', fit / 'scorer_in_ir_f_best.pt'),
            ('vis_f', fit / 'scorer_in_vis_f_best.pt'),
        ]
        for mode, ckpt in modes:
            if not ckpt.is_file():
                report['source_ablation'][mode] = {'error': f'missing {ckpt}'}
                continue
            model, meta = load_stage1_scorer(ckpt, device=device)
            size = int(meta.get('size', 224))
            recs = _score_mode(
                model, test_rows, size=size, device=device,
                batch_size=args.batch_size, input_mode=mode,
            )
            summ = summarize_scorer(recs)
            report['source_ablation'][mode] = {
                'ckpt': str(ckpt), 'epoch': meta.get('epoch'), **summ,
            }
            print(f'[source] {mode}: soft={summ["acc_soft"]:.3f} dec={summ["acc_decisive"]:.3f} n={summ["n"]}')

    # --- LOMO (clean protocol) ---
    default_ckpt = fit / 'scorer_stage1_best.pt'
    default_model, default_meta = load_stage1_scorer(default_ckpt, device=device)
    default_size = int(default_meta.get('size', 224))

    methods = ['method_a', 'method_b', 'method_c', 'method_d']
    for m in methods:
        held = [r for r in test_rows if _pair_involves_method(r, m)]
        other = [r for r in test_rows if not _pair_involves_method(r, m)]

        # traditional metrics on identical held-out decisive pairs
        # need scorer-style records with q/decisive — build lightly from supervision
        held_pseudo = []
        for r in held:
            q = float(r['y_A']) + 0.5 * float(r['y_T'])
            held_pseudo.append({
                'scene_id': r['scene_id'],
                'candidate_a': r['candidate_a'],
                'candidate_b': r['candidate_b'],
                'path_a': r.get('path_a'),
                'path_b': r.get('path_b'),
                'q': q,
                'decisive': abs(q - 0.5) >= 0.1,
                'dec_ok': None,
                'soft_ok': False,
            })
        trad = evaluate_metric_baselines(held_pseudo, feats, decisive_only=True)
        trad.pop('ours_S', None)
        best_trad_key = max(trad, key=lambda k: trad[k]['acc'] if trad[k]['n'] else -1)
        best_trad = {
            'metric': best_trad_key,
            'acc': trad[best_trad_key]['acc'],
            'n': trad[best_trad_key]['n'],
        }

        # default model on same held subset
        def_recs = score_rows(
            default_model, held, size=default_size, device=device, batch_size=args.batch_size,
        )
        def_summ = summarize_scorer(def_recs)

        out_m: dict = {
            'heldout_n': len(held),
            'heldout_n_decisive': def_summ['n_decisive'],
            'default_on_heldout': def_summ,
            'traditional_on_heldout': trad,
            'best_traditional_on_heldout': best_trad,
        }

        lomo_ckpt = fit / f'scorer_lomo_{m}_best.pt'
        if lomo_ckpt.is_file():
            model, meta = load_stage1_scorer(lomo_ckpt, device=device)
            size = int(meta.get('size', 224))
            for name, subset in [('heldout_pairs', held), ('other_pairs', other), ('all_test', test_rows)]:
                recs = score_rows(model, subset, size=size, device=device, batch_size=args.batch_size)
                summ = summarize_scorer(recs)
                out_m[name] = summ
                print(
                    f'[lomo holdout={m} {name}] '
                    f'soft={summ["acc_soft"]:.3f} dec={summ["acc_decisive"]:.3f} n={summ["n"]}'
                )
            out_m['ckpt'] = str(lomo_ckpt)
            out_m['epoch'] = meta.get('epoch')
            out_m['delta_vs_default_dec'] = (
                float(out_m['heldout_pairs']['acc_decisive']) - float(def_summ['acc_decisive'])
                if out_m['heldout_pairs']['n_decisive'] and def_summ['n_decisive']
                else float('nan')
            )
        else:
            out_m['heldout_pairs'] = {'error': f'missing {lomo_ckpt}'}

        matched_ckpt = fit / f'scorer_matched_{m}_best.pt'
        if matched_ckpt.is_file():
            model, meta = load_stage1_scorer(matched_ckpt, device=device)
            size = int(meta.get('size', 224))
            recs = score_rows(model, held, size=size, device=device, batch_size=args.batch_size)
            summ = summarize_scorer(recs)
            out_m['matched_size_on_heldout'] = {
                'ckpt': str(matched_ckpt), 'epoch': meta.get('epoch'), **summ,
            }
            print(
                f'[matched holdout={m} heldout] '
                f'soft={summ["acc_soft"]:.3f} dec={summ["acc_decisive"]:.3f}'
            )
        else:
            out_m['matched_size_on_heldout'] = {'error': f'missing {matched_ckpt}'}

        print(
            f'[lomo holdout={m}] default_dec={def_summ["acc_decisive"]:.3f} '
            f'best_trad={best_trad["metric"]}={best_trad["acc"]:.3f} '
            f'n_dec={def_summ["n_decisive"]}'
        )
        report['lomo'][m] = out_m

    dest = fit / 'ablation_source_lomo_report.json'
    try:
        dest.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('wrote', dest)
    except OSError:
        alt = Path('/home/ykx/NLG-fusion/logs/ablations/ablation_source_lomo_report.json')
        alt.parent.mkdir(parents=True, exist_ok=True)
        alt.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('wrote', alt)


if __name__ == '__main__':
    main()
