#!/usr/bin/env python3
"""Classical IVIF metrics on the expanded RoadScene v2 labels.

Uses the same metric functions as the paper (pref.fusion_metrics at HEAD)
and the same entropy used for EN. Higher value predicts that side.
Equal metric values stay in the denominator and count as misses.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

PACK = Path("/data/ykx/pref_bench_roadscene_v2")
OLD_PACK = Path("/data/ykx/pref_bench_roadscene")
OLD_PRED = Path("/home/ykx/NLG-fusion/results/lpifm_hpfm_comparison/predictions.csv")
OUT = Path("/home/ykx/NLG-fusion/results/lpifm_hpfm_comparison/roadscene_v2")
PRED = OUT / "predictions.csv"
VENDOR = Path("/tmp/paper_metrics")
SEED = 443
N_BOOT = 10000
KEYS = ("en", "mi", "qabf", "scd", "vif", "ssim", "sf")


def ensure_vendor() -> None:
    if (VENDOR / "pref" / "fusion_metrics.py").is_file():
        return
    (VENDOR / "pref").mkdir(parents=True, exist_ok=True)
    (VENDOR / "pref" / "__init__.py").write_text("", encoding="utf-8")
    root = Path("/home/ykx/NLG-fusion")
    for name in ("qabf.py", "fusion_metrics.py"):
        text = subprocess.check_output(
            ["git", "show", f"HEAD:pref/{name}"], cwd=root, text=True,
        )
        (VENDOR / "pref" / name).write_text(text, encoding="utf-8")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def image_entropy(path: str) -> float:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    p = hist / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def score_one(job: tuple[str, str, str, str, str]) -> dict:
    scene_id, candidate, ir, vi, fused = job
    sys.path.insert(0, str(VENDOR))
    from pref.fusion_metrics import compute_fusion_metrics

    ir_img = cv2.imread(ir, cv2.IMREAD_GRAYSCALE)
    vi_img = cv2.imread(vi, cv2.IMREAD_GRAYSCALE)
    fused_img = cv2.imread(fused, cv2.IMREAD_GRAYSCALE)
    if ir_img is None or vi_img is None or fused_img is None:
        raise FileNotFoundError(fused)
    raw = compute_fusion_metrics(
        ir_img.astype(np.float32),
        vi_img.astype(np.float32),
        fused_img.astype(np.float32),
    )
    return {
        "scene_id": scene_id,
        "candidate": candidate,
        "fusion_path": fused,
        "en": image_entropy(fused),
        "mi": raw["mi"],
        "qabf": raw["qabf"],
        "scd": raw["scd"],
        "vif": raw["vif"],
        "ssim": raw["ssim"],
        "sf": raw["sf"],
    }


def q_of(row: dict) -> float:
    return float(row["y_A"]) + 0.5 * float(row["y_T"])


def decisive(q: float) -> bool:
    return q <= 0.4 or q >= 0.6


def bootstrap(records: list[dict], metric: str) -> tuple[float, float]:
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_scene[row["scene_id"]].append(row)
    scenes = sorted(by_scene)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(scenes), size=(N_BOOT, len(scenes)))
    diffs = np.empty(N_BOOT, dtype=np.float64)
    for i, draw in enumerate(draws):
        h_ok = h_n = m_ok = m_n = 0
        for scene_i in draw:
            for row in by_scene[scenes[scene_i]]:
                h_n += 1
                m_n += 1
                h_ok += int(row["hpfm"])
                m_ok += int(row[metric])
        diffs[i] = (h_ok / h_n) - (m_ok / m_n)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(low) * 100.0, float(high) * 100.0


def evaluate(name: str, subset: str, rows: list[dict], scores: dict[tuple[str, str], dict], hpfm: dict[str, int]) -> list[dict]:
    use = []
    for row in rows:
        q = q_of(row)
        if not decisive(q):
            continue
        a = scores[(row["scene_id"], row["candidate_a"])]
        b = scores[(row["scene_id"], row["candidate_b"])]
        rec = {
            "scene_id": row["scene_id"],
            "pair_id": row["pair_id"],
            "hpfm": hpfm[row["pair_id"]],
        }
        human_a = q >= 0.5
        for key in KEYS:
            ma, mb = a[key], b[key]
            if ma == mb:
                rec[key] = 0
                rec[key + "_tie"] = 1
            else:
                rec[key] = int((ma > mb) == human_a)
                rec[key + "_tie"] = 0
        use.append(rec)
    n = len(use)
    h_ok = sum(row["hpfm"] for row in use)
    out = []
    for key in KEYS:
        ok = sum(row[key] for row in use)
        ties = sum(row[key + "_tie"] for row in use)
        low, high = bootstrap(use, key)
        out.append({
            "protocol": name,
            "subset": subset,
            "metric": key,
            "n_scenes": len({row["scene_id"] for row in use}),
            "n": n,
            "correct": ok,
            "metric_ties": ties,
            "agreement": ok / n if n else float("nan"),
            "hpfm_correct": h_ok,
            "hpfm_agreement": h_ok / n if n else float("nan"),
            "diff_pp": 100.0 * ((h_ok / n) - (ok / n)) if n else float("nan"),
            "diff_ci95_low_pp": low,
            "diff_ci95_high_pp": high,
        })
    best = max(out, key=lambda row: row["agreement"])
    for row in out:
        row["best_on_subset"] = int(row["metric"] == best["metric"])
    return out


def main() -> None:
    ensure_vendor()
    rows = load_jsonl(PACK / "supervision.jsonl")
    jobs = {}
    for row in rows:
        jobs[(row["scene_id"], row["candidate_a"])] = (
            row["scene_id"], row["candidate_a"], row["ir"], row["vi"], row["path_a"],
        )
        jobs[(row["scene_id"], row["candidate_b"])] = (
            row["scene_id"], row["candidate_b"], row["ir"], row["vi"], row["path_b"],
        )
    print(f"unique fusions {len(jobs)}", flush=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with ProcessPoolExecutor(max_workers=8) as pool:
        scored = list(pool.map(score_one, jobs.values(), chunksize=4))
    scores = {(row["scene_id"], row["candidate"]): row for row in scored}
    with (OUT / "metric_cache.jsonl").open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row) + "\n")

    hpfm = {}
    with PRED.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["decisive"] == "True":
                hpfm[row["pair_id"]] = int(row["hit_A_hpfm"])

    old_rows = load_jsonl(OLD_PACK / "supervision.jsonl")
    old_ids = {row["pair_id"] for row in old_rows}
    paper_hpfm = {}
    with OLD_PRED.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["source_db"] == "pref_bench_roadscene" and row["decisive"] == "True":
                paper_hpfm[row["pair_id"]] = int(row["hit_A_hpfm"])
    summaries = []
    for protocol in ("e1", "e2"):
        block = [row for row in rows if row["pair_id"].startswith(protocol + "::")]
        summaries.extend(evaluate(protocol, "v2_full", block, scores, hpfm))
        original = [row for row in block if row["pair_id"] in old_ids]
        # HPFM hits for original ids are in the v2 predictions.
        summaries.extend(evaluate(protocol, "original_ids", original, scores, hpfm))
        old_block = [row for row in old_rows if row["pair_id"].startswith(protocol + "::")]
        summaries.extend(evaluate(protocol, "paper_labels", old_block, scores, paper_hpfm))

    fields = list(summaries[0])
    with (OUT / "metric_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    for row in summaries:
        if row["subset"] == "paper_labels":
            print(
                f"paper {row['protocol']} {row['metric']} {row['correct']}/{row['n']} "
                f"{row['agreement']:.3f} ties {row['metric_ties']}",
                flush=True,
            )
        elif row["subset"] == "v2_full":
            print(
                f"v2 {row['protocol']} {row['metric']} {row['correct']}/{row['n']} "
                f"{row['agreement']:.3f} diff {row['diff_pp']:.2f} "
                f"[{row['diff_ci95_low_pp']:.2f}, {row['diff_ci95_high_pp']:.2f}] "
                f"best {row['best_on_subset']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
