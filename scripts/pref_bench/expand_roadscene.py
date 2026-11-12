#!/usr/bin/env python3
"""Build the RoadScene transfer pack.

The finished pack is /data/ykx/pref_bench_roadscene: 40 scenes and 480 pairs.
Re-running would overwrite those labels, so main() refuses to continue.
"""

from __future__ import annotations

import json
import os
import shutil
from itertools import combinations
from pathlib import Path

import cv2

OLD = Path("/data/ykx/pref_bench_roadscene")
NEW = Path("/data/ykx/pref_bench_roadscene")
SRC = Path("/data/ykx/rs_train_add")
FUSION = Path("/data/ykx/sota/rs_train_add")
TRAIN_LIST = Path("/data/ykx/RoadScene/meta/train.txt")
TEST_IR = Path("/data/ykx/RoadScene/test/ir")
E1 = ("cddfuse", "ddfm", "swinfusion", "tardal")
E2 = ("dcevo", "emma", "metafusion", "textif")
METHODS = E1 + E2


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def rewrite(value: str) -> str:
    old = str(OLD)
    if isinstance(value, str) and value.startswith(old):
        return str(NEW) + value[len(old):]
    return value


def rewrite_row(row: dict) -> dict:
    return {key: rewrite(value) if isinstance(value, str) else value for key, value in row.items()}


def choose_scenes() -> list[dict]:
    train = {line.strip() for line in TRAIN_LIST.read_text(encoding="utf-8").splitlines() if line.strip()}
    test_names = {path.name for path in TEST_IR.iterdir() if path.is_file()}
    names = sorted(path.name for path in (SRC / "ir").iterdir() if path.is_file())
    if len(names) != 10:
        raise SystemExit(f"expected 10 source images, found {len(names)}")
    scenes = []
    for name in names:
        if name not in train:
            raise SystemExit(f"{name} is not in the RoadScene training list")
        if name in test_names:
            raise SystemExit(f"{name} is a RoadScene test image")
        ir = SRC / "ir" / name
        vi = SRC / "vi" / name
        if not ir.is_file() or not vi.is_file():
            raise SystemExit(f"missing IR/VI for {name}")
        for method in METHODS:
            fused = FUSION / method / name
            if not fused.is_file() or fused.stat().st_size == 0:
                raise SystemExit(f"missing fusion {fused}")
        stem = Path(name).stem
        scenes.append({
            "scene_id": f"RoadScene/{stem}",
            "dataset": "RoadScene",
            "filename": name,
            "stem": stem,
            "split": "test",
            "scene_group": f"RoadScene/{stem}",
            "ir": ir,
            "vi": vi,
        })
    return scenes


def link_tree(src: Path, dst: Path) -> None:
    """Put each file inside dst. Hardlinks stay under the new pack after resolve()."""
    if dst.exists() or dst.is_symlink():
        raise SystemExit(f"refusing to replace {dst}")
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)


def write_png(src: Path, dst: Path) -> None:
    image = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dst), image):
        raise SystemExit(f"cannot write {dst}")


def materialize(scene: dict) -> None:
    scene_dir = NEW / "scenes" / scene["scene_id"]
    ir_dst = scene_dir / "ir" / scene["filename"]
    vi_dst = scene_dir / "vi" / scene["filename"]
    ir_dst.parent.mkdir(parents=True, exist_ok=True)
    vi_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(scene["ir"], ir_dst)
    shutil.copy2(scene["vi"], vi_dst)
    for method in METHODS:
        write_png(FUSION / method / scene["filename"], scene_dir / "fusions" / f"{method}.png")


def scene_record(scene: dict) -> dict:
    scene_dir = NEW / "scenes" / scene["scene_id"]
    return {
        "scene_id": scene["scene_id"],
        "dataset": "RoadScene",
        "source_dataset": "RoadScene",
        "source_scene_id": scene["stem"],
        "filename": scene["filename"],
        "root": str(SRC),
        "scene_dir": str(scene_dir),
        "split": "test",
        "scene_group": scene["scene_group"],
        "ir": str(scene_dir / "ir" / scene["filename"]),
        "vi": str(scene_dir / "vi" / scene["filename"]),
        "protocol_pool": "roadscene_train",
        "methods": list(METHODS),
    }


def make_pair(protocol: str, scene: dict, left: str, right: str) -> dict:
    scene_id = scene["scene_id"]
    scene_dir = NEW / "scenes" / scene_id
    filename = scene["filename"]
    a, b = sorted((left, right))
    return {
        "task_id": f"{protocol}::{scene_id}::{left}__{right}",
        "pair_id": f"{protocol}::{scene_id}::{left}__{right}",
        "scene_id": scene_id,
        "scene_group": scene["scene_group"],
        "split": scene["split"],
        "dataset": "RoadScene",
        "protocol": protocol,
        "is_gold": False,
        "ir": str(scene_dir / "ir" / filename),
        "vi": str(scene_dir / "vi" / filename),
        "left": left,
        "right": right,
        "left_path": str(scene_dir / "fusions" / f"{left}.png"),
        "right_path": str(scene_dir / "fusions" / f"{right}.png"),
        "candidate_a": a,
        "candidate_b": b,
        "candidate_a_id": f"{scene_id}::{a}",
        "candidate_b_id": f"{scene_id}::{b}",
        "path_a": str(scene_dir / "fusions" / f"{a}.png"),
        "path_b": str(scene_dir / "fusions" / f"{b}.png"),
        "label_status": "pending",
    }


def main() -> None:
    raise SystemExit(
        "RoadScene is already built at /data/ykx/pref_bench_roadscene "
        "(40 scenes, 480 pairs)."
    )
    if NEW.exists():
        raise SystemExit(f"refusing to overwrite {NEW}")
    added = choose_scenes()
    old_scenes = load_jsonl(OLD / "scenes.jsonl")
    old_pairs = load_jsonl(OLD / "pairs.jsonl")
    old_ids = {row["scene_id"] for row in old_scenes}
    if any(scene["scene_id"] in old_ids for scene in added):
        raise SystemExit("a new scene id collides with the existing pack")

    NEW.mkdir(parents=True)
    for row in old_scenes:
        link_tree(OLD / "scenes" / row["scene_id"], NEW / "scenes" / row["scene_id"])
    for scene in added:
        materialize(scene)

    scenes = [rewrite_row(row) for row in old_scenes]
    scenes.extend(scene_record(scene) for scene in added)
    existing_keys = {
        (row["protocol"], row["scene_id"], frozenset((row["candidate_a"], row["candidate_b"])))
        for row in old_pairs
    }
    pairs = []
    for row in old_pairs:
        copied = rewrite_row(row)
        copied["label_status"] = "labeled"
        pairs.append(copied)
    pending = []
    pools = {"e1": E1, "e2": E2}
    for scene in scenes:
        for protocol, methods in pools.items():
            for left, right in combinations(methods, 2):
                key = (protocol, scene["scene_id"], frozenset((left, right)))
                if key in existing_keys:
                    continue
                pending.append(make_pair(protocol, scene, left, right))
    pairs.extend(pending)

    for row in pairs:
        for key in ("ir", "vi", "path_a", "path_b", "left_path", "right_path"):
            if not Path(row[key]).is_file():
                raise SystemExit(f"missing {row[key]}")

    write_jsonl(NEW / "scenes.jsonl", scenes)
    write_jsonl(NEW / "pairs.jsonl", pairs)
    write_jsonl(NEW / "pairs_e1.jsonl", [row for row in pairs if row["protocol"] == "e1"])
    write_jsonl(NEW / "pairs_e2.jsonl", [row for row in pairs if row["protocol"] == "e2"])
    write_jsonl(NEW / "pairs_pending.jsonl", pending)
    write_jsonl(NEW / "annotations.jsonl", [rewrite_row(row) for row in load_jsonl(OLD / "annotations.jsonl")])
    write_jsonl(NEW / "labels_ann0.jsonl", load_jsonl(OLD / "labels_ann0.jsonl"))
    write_jsonl(NEW / "annotators.jsonl", load_jsonl(OLD / "annotators.jsonl"))
    write_jsonl(NEW / "supervision.jsonl", [rewrite_row(row) for row in load_jsonl(OLD / "supervision.jsonl")])

    candidates = []
    for scene in scenes:
        for method in METHODS:
            candidates.append({
                "candidate_id": f"{scene['scene_id']}::{method}",
                "scene_id": scene["scene_id"],
                "method_id": method,
                "path": str(NEW / "scenes" / scene["scene_id"] / "fusions" / f"{method}.png"),
            })
    write_jsonl(NEW / "candidates.jsonl", candidates)
    (NEW / "config.yaml").write_text(
        "seed: 443\n"
        f"out_root: {NEW}\n"
        "sota_root: /data/ykx/sota\n"
        "pref_bench_root: /data/ykx/pref_bench\n"
        "annotator_ids:\n- ann0\n"
        "gen_eval:\n"
        "  seen_methods: [cddfuse, ddfm, swinfusion, tardal]\n"
        "  unseen_methods: [dcevo, emma, metafusion, textif]\n"
        "  n_scenes: 40\n"
        "  n_added_train_scenes: 10\n"
        "  pairs_per_scene: 6\n"
        "  n_e1: 240\n"
        "  n_e2: 240\n"
        "  n_labeled: 180\n"
        f"  n_pending: {len(pending)}\n",
        encoding="utf-8",
    )
    manifest = {
        "name": "pref_bench_roadscene",
        "based_on": str(OLD),
        "n_scenes": len(scenes),
        "n_original_roadscene_scenes": len(old_scenes),
        "added_scenes": [scene["scene_id"] for scene in added],
        "added_scene_source": "Images taken from the RoadScene training list. The whole pack is a test set and was not used in HPFM training.",
        "e1_seen_methods": list(E1),
        "e2_unseen_methods": list(E2),
        "n_pairs_e1": sum(row["protocol"] == "e1" for row in pairs),
        "n_pairs_e2": sum(row["protocol"] == "e2" for row in pairs),
        "n_labeled": 180,
        "n_pending": len(pending),
        "note": "Pending pairs have no human label. Seen and unseen methods are both present on every scene.",
    }
    (NEW / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "scenes": len(scenes),
        "pairs": len(pairs),
        "pending": len(pending),
        "added": manifest["added_scenes"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
