#!/usr/bin/env python3
"""Frozen HPFM vs LPIFM on the expanded RoadScene v2 pack.

Reads aggregated supervision. Does not train, and does not change thresholds.
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
from torch.utils.data import DataLoader, Dataset

ROOT = Path("/home/ykx/NLG-fusion")
LPIFM_ROOT = Path("/home/ykx/LPIFM")
PACK = Path("/data/ykx/pref_bench_roadscene_v2")
OLD_PACK = Path("/data/ykx/pref_bench_roadscene")
OUT = Path("/home/ykx/NLG-fusion/results/lpifm_hpfm_comparison/roadscene_v2")
HPFM_CKPT = Path("/data/ykx/pref_bench/fit/scorer_stage1_best.pt")
LPIFM_CKPT = LPIFM_ROOT / "checkpoints/lpifm_vifb_baseline_v1.pt"
LPIFM_YAML = LPIFM_ROOT / "configs/release_inference.yaml"
SEED = 443
N_BOOT = 10000

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LPIFM_ROOT))

from lpifm.inference import _ensure_rgb, decode_gap, load_frozen_model  # noqa: E402
from pref.dataset import load_triple  # noqa: E402
from pref.model.resnet_scorer import ResNet18ScalarScorer  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def q_of(row: dict) -> float:
    return float(row["y_A"]) + 0.5 * float(row["y_T"])


def q_kind(row: dict, q: float) -> str:
    if q <= 0.4:
        return "decisive_B"
    if q >= 0.6:
        return "decisive_A"
    if float(row["y_T"]) == 1.0 and int(row["n_annotations"]) == 1:
        return "active_tie"
    if abs(float(row["y_A"]) - 0.5) < 1e-9 and abs(float(row["y_B"]) - 0.5) < 1e-9 and float(row["y_T"]) == 0.0:
        return "aggregated_half"
    return "other_half"


class PairPaths(Dataset):
    def __init__(self, rows: list[dict], transform) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        transform = self.transform
        return {
            "ir": transform(_ensure_rgb(Image.open(row["ir"]))),
            "vi": transform(_ensure_rgb(Image.open(row["vi"]))),
            "a": transform(_ensure_rgb(Image.open(row["path_a"]))),
            "b": transform(_ensure_rgb(Image.open(row["path_b"]))),
            "index": index,
        }


def score_hpfm(rows: list[dict], device: torch.device) -> dict[str, float]:
    payload = torch.load(HPFM_CKPT, map_location="cpu", weights_only=False)
    model = ResNet18ScalarScorer(pretrained=False).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    unique: dict[str, tuple[str, str, str]] = {}
    for row in rows:
        unique[row["path_a"]] = (row["ir"], row["vi"], row["path_a"])
        unique[row["path_b"]] = (row["ir"], row["vi"], row["path_b"])
    keys = list(unique)
    scores: dict[str, float] = {}
    with torch.no_grad():
        for start in range(0, len(keys), 16):
            batch = keys[start:start + 16]
            xs = torch.stack([
                load_triple(*unique[key], size=224) for key in batch
            ]).to(device)
            out = model(xs).detach().cpu().tolist()
            for key, value in zip(batch, out):
                scores[key] = float(value)
    return scores


def score_lpifm(rows: list[dict], device: torch.device) -> tuple[list[float], list[int], str]:
    bundle = load_frozen_model(
        LPIFM_CKPT,
        yaml_backup=LPIFM_YAML,
        device=device,
        t=0.3,
        t_cal=1.0,
        seed=SEED,
    )
    loader = DataLoader(PairPaths(rows, bundle.transform), batch_size=1, shuffle=False, num_workers=0)
    d_cals = [0.0] * len(rows)
    decisions = [0] * len(rows)
    bundle.model.eval()
    with torch.no_grad():
        for batch in loader:
            d = bundle.model(
                batch["ir"].to(device),
                batch["vi"].to(device),
                batch["a"].to(device),
                batch["b"].to(device),
            )
            d_cal, pred = decode_gap(d, t=bundle.t, t_cal=bundle.t_cal)
            idx = int(batch["index"][0])
            d_cals[idx] = float(d_cal[0].item())
            decisions[idx] = int(pred[0].item())
            if (idx + 1) % 40 == 0 or idx + 1 == len(rows):
                print(f"lpifm {idx + 1}/{len(rows)}", flush=True)
    return d_cals, decisions, bundle.variant


def side_from_code(code: int) -> str:
    return {0: "A", 1: "B", 2: "Tie"}[code]


def bootstrap(records: list[dict], hit_h: str, hit_l: str) -> tuple[float, float]:
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        if row["decisive"]:
            by_scene[row["scene_id"]].append(row)
    scenes = sorted(by_scene)
    if not scenes:
        return float("nan"), float("nan")
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


def summarize(name: str, subset: str, records: list[dict], paper: str) -> dict:
    decisive = [row for row in records if row["decisive"]]
    n = len(decisive)
    h_ok = sum(row["hit_A_hpfm"] for row in decisive)
    a_ok = sum(row["hit_A_lpifm"] for row in decisive)
    a_rev = sum(row["lpifm_a_outcome"] == "reverse" for row in decisive)
    a_tie = sum(row["lpifm_a_outcome"] == "tie" for row in decisive)
    directed = sum(row["lpifm_official"] != "Tie" for row in decisive)
    b_ok = sum(row["hit_B_lpifm"] for row in decisive)
    only_h_a = sum(row["hit_A_hpfm"] and not row["hit_A_lpifm"] for row in decisive)
    only_l_a = sum(row["hit_A_lpifm"] and not row["hit_A_hpfm"] for row in decisive)
    only_h_b = sum(row["hit_B_hpfm"] and not row["hit_B_lpifm"] for row in decisive)
    only_l_b = sum(row["hit_B_lpifm"] and not row["hit_B_hpfm"] for row in decisive)
    diff_a = 100.0 * ((h_ok / n) - (a_ok / n)) if n else float("nan")
    diff_b = 100.0 * ((h_ok / n) - (b_ok / n)) if n else float("nan")
    ci_a = bootstrap(records, "hit_A_hpfm", "hit_A_lpifm")
    ci_b = bootstrap(records, "hit_B_hpfm", "hit_B_lpifm")
    return {
        "protocol": name,
        "subset": subset,
        "n_scenes": len({row["scene_id"] for row in decisive}),
        "n": n,
        "hpfm_correct": h_ok,
        "lpifm_A_correct": a_ok,
        "lpifm_A_reverse": a_rev,
        "lpifm_A_tie": a_tie,
        "lpifm_A_directed": directed,
        "lpifm_B_correct": b_ok,
        "hpfm_zero": sum(row["hpfm_zero"] for row in decisive),
        "lpifm_zero": sum(row["lpifm_zero"] for row in decisive),
        "only_hpfm_A": only_h_a,
        "only_lpifm_A": only_l_a,
        "only_hpfm_B": only_h_b,
        "only_lpifm_B": only_l_b,
        "diff_A_pp": diff_a,
        "diff_A_ci95_low_pp": ci_a[0],
        "diff_A_ci95_high_pp": ci_a[1],
        "diff_B_pp": diff_b,
        "diff_B_ci95_low_pp": ci_b[0],
        "diff_B_ci95_high_pp": ci_b[1],
        "paper_hpfm": paper,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    rows = load_jsonl(PACK / "supervision.jsonl")
    old_ids = {row["pair_id"] for row in load_jsonl(OLD_PACK / "pairs.jsonl")}
    for row in rows:
        for key in ("ir", "vi", "path_a", "path_b"):
            if not Path(row[key]).is_file():
                raise SystemExit(f"missing {row[key]}")
    print("scoring hpfm", len(rows), flush=True)
    hpfm_scores = score_hpfm(rows, device)
    print("scoring lpifm", flush=True)
    d_cals, decisions, variant = score_lpifm(rows, device)

    records = []
    for row, d_cal, decision in zip(rows, d_cals, decisions):
        q = q_of(row)
        kind = q_kind(row, q)
        decisive = kind in {"decisive_A", "decisive_B"}
        human = "A" if q >= 0.5 else "B"
        s_a = hpfm_scores[row["path_a"]]
        s_b = hpfm_scores[row["path_b"]]
        delta = s_a - s_b
        h_pred = "A" if delta >= 0 else "B"
        official = side_from_code(decision)
        if d_cal > 0:
            forced = "A"
        elif d_cal < 0:
            forced = "B"
        else:
            forced = "A"
        if official == "Tie":
            outcome = "tie"
        elif official == human:
            outcome = "correct"
        else:
            outcome = "reverse"
        hit_h = int(decisive and h_pred == human)
        hit_a = int(decisive and official == human)
        hit_b_h = int(decisive and h_pred == human)
        hit_b_l = int(decisive and forced == human)
        records.append({
            "source_db": "pref_bench_roadscene_v2",
            "protocol": row["pair_id"].split("::", 1)[0],
            "dataset": row["dataset"],
            "scene_id": row["scene_id"],
            "pair_id": row["pair_id"],
            "candidate_a": row["candidate_a"],
            "candidate_b": row["candidate_b"],
            "ir": row["ir"],
            "vi": row["vi"],
            "path_a": row["path_a"],
            "path_b": row["path_b"],
            "y_A": row["y_A"],
            "y_T": row["y_T"],
            "y_B": row["y_B"],
            "n_annotations": row["n_annotations"],
            "q_A": q,
            "q_kind": kind,
            "decisive": decisive,
            "in_original_pack": row["pair_id"] in old_ids,
            "human_side": human if decisive else "",
            "hpfm_source": f"frozen:{HPFM_CKPT}",
            "hpfm_s_a": s_a,
            "hpfm_s_b": s_b,
            "hpfm_delta": delta,
            "hpfm_pred": h_pred,
            "hpfm_zero": int(delta == 0.0),
            "lpifm_d_cal": d_cal,
            "lpifm_official": official,
            "lpifm_forced": forced,
            "lpifm_zero": int(d_cal == 0.0),
            "lpifm_a_outcome": outcome if decisive else "",
            "hit_A_hpfm": hit_h,
            "hit_A_lpifm": hit_a,
            "hit_B_hpfm": hit_b_h,
            "hit_B_lpifm": hit_b_l,
            "lpifm_variant": variant,
        })

    OUT.mkdir(parents=True, exist_ok=True)
    fields = list(records[0])
    with (OUT / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    summaries = []
    for protocol in ("e1", "e2"):
        block = [row for row in records if row["protocol"] == protocol]
        paper = "0.812" if protocol == "e1" else "0.694"
        summaries.append(summarize(protocol, "v2_full", block, ""))
        original = [row for row in block if row["in_original_pack"]]
        summaries.append(summarize(protocol, "original_ids", original, paper))
    sum_fields = list(summaries[0])
    with (OUT / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sum_fields)
        writer.writeheader()
        writer.writerows(summaries)

    half_rows = []
    for protocol in ("e1", "e2"):
        block = [row for row in records if row["protocol"] == protocol and not row["decisive"]]
        for kind in ("active_tie", "aggregated_half", "other_half"):
            chosen = [row for row in block if row["q_kind"] == kind]
            if not chosen:
                continue
            half_rows.append({
                "protocol": protocol,
                "q_kind": kind,
                "n": len(chosen),
                "lpifm_official_tie": sum(row["lpifm_official"] == "Tie" for row in chosen),
            })
    with (OUT / "summary_q_half.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["protocol", "q_kind", "n", "lpifm_official_tie"])
        writer.writeheader()
        writer.writerows(half_rows)

    lines = ["# RoadScene v2 frozen HPFM vs LPIFM", ""]
    lines.append(f"Device: {device}. LPIFM state dict: {variant}.")
    lines.append(f"Pack: {PACK}. HPFM: {HPFM_CKPT}. LPIFM: {LPIFM_CKPT}.")
    lines.append("Decisive means q_A <= 0.4 or q_A >= 0.6. q_A=0.5 stays out of the denominator.")
    lines.append("Protocol A keeps LPIFM ties as misses. Protocol B forces the sign of d_cal; exact zero predicts A.")
    lines.append("original_ids uses pair ids from the previous 30-scene pack and the updated aggregated labels.")
    lines.append("paper_hpfm on that subset is the published reference, not a number to force.")
    lines.append("")
    for row in summaries:
        lines.append(
            f"{row['protocol']} {row['subset']}: scenes {row['n_scenes']} n {row['n']} "
            f"HPFM {row['hpfm_correct']}/{row['n']} "
            f"LPIFM-A {row['lpifm_A_correct']}/{row['n']} "
            f"(reverse {row['lpifm_A_reverse']} tie {row['lpifm_A_tie']}) "
            f"LPIFM-B {row['lpifm_B_correct']}/{row['n']} "
            f"diffA {row['diff_A_pp']:.2f} [{row['diff_A_ci95_low_pp']:.2f}, {row['diff_A_ci95_high_pp']:.2f}] "
            f"diffB {row['diff_B_pp']:.2f} [{row['diff_B_ci95_low_pp']:.2f}, {row['diff_B_ci95_high_pp']:.2f}]"
        )
    (OUT / "run_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
