#!/usr/bin/env python3
"""Frozen HPFM vs LPIFM on VIFB scenes and methods outside both training sets.

Scenes are the five held out of LPIFM training. Methods are the six excluded
from LPIFM's label file. HPFM was not trained on VIFB. A/B follows the label
file. Official LPIFM resolution is 384; HPFM stays at 224.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/home/ykx/NLG-fusion")
LPIFM_ROOT = Path("/home/ykx/LPIFM")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LPIFM_ROOT))

from lpifm.inference import _ensure_rgb, decode_gap, load_frozen_model  # noqa: E402
from pref.dataset import load_triple  # noqa: E402
from pref.model.resnet_scorer import ResNet18ScalarScorer  # noqa: E402

VIFB = Path("/data/ykx/VIFB")
LABELS = LPIFM_ROOT / "Dataset/VIFB/VIFB_labels_unseen_images_unseen_methods_pairwise.json"
SEEN_SCENES = LPIFM_ROOT / "Dataset/VIFB/VIFB_labels_seen_images_seen_methods_pairwise.json"
TRAIN_LABELS = LPIFM_ROOT / "Dataset/VIFB/VIFB_labels_exclude_U2Fusion-Hybrid_MSD-MGFF-FPDE-GTF-NSCT_SR_allowself.json"
HPFM_CKPT = Path("/data/ykx/pref_bench/fit/scorer_stage1_best.pt")
LPIFM_CKPT = LPIFM_ROOT / "checkpoints/lpifm_vifb_baseline_v1.pt"
LPIFM_YAML = LPIFM_ROOT / "configs/release_inference.yaml"
OUT = Path("/home/ykx/NLG-fusion/results/lpifm_hpfm_comparison/vifb_unseen_both")
SEED = 443
N_BOOT = 10000


def scene_of(path: str) -> str:
    return Path(path).stem


def method_of(path: str) -> str:
    return Path(path).parent.name


def fused_path(method: str, scene: str) -> Path:
    return VIFB / "output/fused_images" / f"{scene}_{method}.jpg"


def check_holdout(rows: list[dict]) -> None:
    seen_rows = json.loads(SEEN_SCENES.read_text())
    train_rows = json.loads(TRAIN_LABELS.read_text())
    seen_scenes = {scene_of(r["a_path"]) for r in seen_rows}
    train_methods = {method_of(r["a_path"]) for r in train_rows} | {method_of(r["b_path"]) for r in train_rows}
    scenes = {scene_of(r["a_path"]) for r in rows}
    methods = {method_of(r["a_path"]) for r in rows} | {method_of(r["b_path"]) for r in rows}
    if scenes & seen_scenes:
        raise SystemExit(f"scenes overlap LPIFM training scenes: {sorted(scenes & seen_scenes)}")
    if methods & train_methods:
        raise SystemExit(f"methods overlap LPIFM training methods: {sorted(methods & train_methods)}")
    print("holdout scenes", sorted(scenes), flush=True)
    print("holdout methods", sorted(methods), flush=True)


def bootstrap(records: list[dict], hit_h: str, hit_l: str) -> tuple[float, float]:
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        if row["decisive"]:
            by_scene[row["scene_id"]].append(row)
    scenes = sorted(by_scene)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(scenes), size=(N_BOOT, len(scenes)))
    diffs = np.empty(N_BOOT, dtype=np.float64)
    for i, draw in enumerate(draws):
        h_ok = h_n = l_ok = l_n = 0
        for scene_i in draw:
            for row in by_scene[scenes[scene_i]]:
                h_n += 1
                l_n += 1
                h_ok += int(row[hit_h])
                l_ok += int(row[hit_l])
        diffs[i] = (h_ok / h_n) - (l_ok / l_n)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(low) * 100.0, float(high) * 100.0


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = json.loads(LABELS.read_text())
    check_holdout(rows)
    payload = torch.load(HPFM_CKPT, map_location="cpu", weights_only=False)
    h_model = ResNet18ScalarScorer(pretrained=False).to(device)
    h_model.load_state_dict(payload["model"])
    h_model.eval()
    bundle = load_frozen_model(LPIFM_CKPT, yaml_backup=LPIFM_YAML, device=device, t=0.3, t_cal=1.0)
    transform = bundle.transform

    def hpfm_score(ir: str, vi: str, fused: str) -> float:
        x = load_triple(ir, vi, fused, size=224).unsqueeze(0).to(device)
        with torch.no_grad():
            return float(h_model(x).item())

    def lpifm_gap(ir: str, vi: str, a: str, b: str) -> float:
        def t(path: str) -> torch.Tensor:
            return transform(_ensure_rgb(Image.open(path))).unsqueeze(0).to(device)
        with torch.no_grad():
            d = bundle.model(t(ir), t(vi), t(a), t(b))
        d_cal, _ = decode_gap(d, t=bundle.t, t_cal=bundle.t_cal)
        return float(d_cal.item())

    records = []
    h_cache: dict[str, float] = {}
    for i, row in enumerate(rows):
        scene = scene_of(row["a_path"])
        method_a, method_b = method_of(row["a_path"]), method_of(row["b_path"])
        ir = str(VIFB / "input/IR" / f"{scene}.jpg")
        vi = str(VIFB / "input/VI" / f"{scene}.jpg")
        path_a = str(fused_path(method_a, scene))
        path_b = str(fused_path(method_b, scene))
        for path in (ir, vi, path_a, path_b):
            if not Path(path).is_file():
                raise SystemExit(f"missing {path}")
        for path in (path_a, path_b):
            if path not in h_cache:
                h_cache[path] = hpfm_score(ir, vi, path)
        s_a, s_b = h_cache[path_a], h_cache[path_b]
        delta = s_a - s_b
        h_pred = "A" if delta >= 0 else "B"
        d_cal = lpifm_gap(ir, vi, path_a, path_b)
        if d_cal > 0.3:
            official = "A"
        elif d_cal < -0.3:
            official = "B"
        else:
            official = "Tie"
        if d_cal > 0:
            forced = "A"
        elif d_cal < 0:
            forced = "B"
        else:
            forced = "A"
        label = int(row["label"])
        decisive = label in (0, 1)
        human = {0: "A", 1: "B"}.get(label, "")
        if not decisive:
            outcome = ""
        elif official == "Tie":
            outcome = "tie"
        elif official == human:
            outcome = "correct"
        else:
            outcome = "reverse"
        records.append({
            "scene_id": scene,
            "candidate_a": method_a,
            "candidate_b": method_b,
            "label": label,
            "decisive": decisive,
            "human_side": human,
            "hpfm_delta": delta,
            "hpfm_pred": h_pred,
            "hpfm_zero": int(delta == 0.0),
            "lpifm_d_cal": d_cal,
            "lpifm_official": official,
            "lpifm_forced": forced,
            "lpifm_zero": int(d_cal == 0.0),
            "lpifm_a_outcome": outcome,
            "hit_A_hpfm": int(decisive and h_pred == human),
            "hit_A_lpifm": int(decisive and official == human),
            "hit_B_hpfm": int(decisive and h_pred == human),
            "hit_B_lpifm": int(decisive and forced == human),
            "lpifm_variant": bundle.variant,
        })
        print(f"{i + 1}/{len(rows)} {scene} {method_a} {method_b}", flush=True)

    decisive = [r for r in records if r["decisive"]]
    n = len(decisive)
    h_ok = sum(r["hit_A_hpfm"] for r in decisive)
    a_ok = sum(r["hit_A_lpifm"] for r in decisive)
    a_rev = sum(r["lpifm_a_outcome"] == "reverse" for r in decisive)
    a_tie = sum(r["lpifm_a_outcome"] == "tie" for r in decisive)
    b_ok = sum(r["hit_B_lpifm"] for r in decisive)
    ci_a = bootstrap(records, "hit_A_hpfm", "hit_A_lpifm")
    ci_b = bootstrap(records, "hit_B_hpfm", "hit_B_lpifm")
    summary = {
        "n_pairs": len(records),
        "n_ties": len(records) - n,
        "n_scenes": len({r["scene_id"] for r in decisive}),
        "n": n,
        "hpfm_correct": h_ok,
        "lpifm_A_correct": a_ok,
        "lpifm_A_reverse": a_rev,
        "lpifm_A_tie": a_tie,
        "lpifm_B_correct": b_ok,
        "only_hpfm_A": sum(r["hit_A_hpfm"] and not r["hit_A_lpifm"] for r in decisive),
        "only_lpifm_A": sum(r["hit_A_lpifm"] and not r["hit_A_hpfm"] for r in decisive),
        "only_hpfm_B": sum(r["hit_B_hpfm"] and not r["hit_B_lpifm"] for r in decisive),
        "only_lpifm_B": sum(r["hit_B_lpifm"] and not r["hit_B_hpfm"] for r in decisive),
        "diff_A_pp": 100.0 * ((h_ok / n) - (a_ok / n)),
        "diff_A_ci95_low_pp": ci_a[0],
        "diff_A_ci95_high_pp": ci_a[1],
        "diff_B_pp": 100.0 * ((h_ok / n) - (b_ok / n)),
        "diff_B_ci95_low_pp": ci_b[0],
        "diff_B_ci95_high_pp": ci_b[1],
        "hpfm_zero": sum(r["hpfm_zero"] for r in decisive),
        "lpifm_zero": sum(r["lpifm_zero"] for r in decisive),
        "variant": bundle.variant,
    }
    by_scene = []
    for scene in sorted({r["scene_id"] for r in decisive}):
        block = [r for r in decisive if r["scene_id"] == scene]
        by_scene.append({
            "scene_id": scene,
            "n": len(block),
            "hpfm": sum(r["hit_A_hpfm"] for r in block),
            "lpifm_A": sum(r["hit_A_lpifm"] for r in block),
            "lpifm_A_tie": sum(r["lpifm_a_outcome"] == "tie" for r in block),
            "lpifm_B": sum(r["hit_B_lpifm"] for r in block),
        })
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (OUT / "summary.json").write_text(json.dumps({"overall": summary, "by_scene": by_scene}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"overall": summary, "by_scene": by_scene}, indent=2), flush=True)


if __name__ == "__main__":
    main()
