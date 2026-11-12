#!/usr/bin/env python3
"""Latency of one HPFM image and one LPIFM pair on the paper's hardware.

Same RoadScene scene as the complexity section: scene 1, TarDAL, plus
SwinFusion as LPIFM's second candidate. Timed region is the forward pass
with inputs already on the device. Warm-up 20, timed runs 100, GPU
synchronized only at the boundaries.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import torch
from PIL import Image

ROOT = Path("/home/ykx/NLG-fusion")
LPIFM_ROOT = Path("/home/ykx/LPIFM")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LPIFM_ROOT))

from lpifm.dataset import build_default_transform  # noqa: E402
from lpifm.inference import _ensure_rgb, load_frozen_model  # noqa: E402
from pref.dataset import load_triple  # noqa: E402
from pref.model.resnet_scorer import ResNet18ScalarScorer  # noqa: E402

SCENE = Path("/data/ykx/pref_bench_roadscene/scenes/RoadScene/1")
HPFM_CKPT = Path("/data/ykx/pref_bench/fit/scorer_stage1_best.pt")
LPIFM_CKPT = LPIFM_ROOT / "checkpoints/lpifm_vifb_baseline_v1.pt"
LPIFM_YAML = LPIFM_ROOT / "configs/release_inference.yaml"
OUT = Path("/home/ykx/NLG-fusion/results/lpifm_hpfm_comparison/latency.json")
WARM = 20
RUNS = 100
E1 = ("cddfuse", "ddfm", "swinfusion", "tardal")


def params_m(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def flops_g(model: torch.nn.Module, inputs: tuple) -> float:
    from thop import profile

    model = copy.deepcopy(model).cpu().eval()
    args = tuple(x.detach().cpu() for x in inputs)
    macs, _ = profile(model, inputs=args, verbose=False)
    return float(macs) / 1e9


def bench(fn, device: torch.device, warm: int = WARM, runs: int = RUNS) -> float:
    for _ in range(warm):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000.0


def main() -> None:
    gpu = torch.device("cuda:0")
    ir = str(SCENE / "ir/1.jpg")
    vi = str(SCENE / "vi/1.jpg")
    tardal = str(SCENE / "fusions/tardal.png")
    swin = str(SCENE / "fusions/swinfusion.png")

    h_cpu = load_triple(ir, vi, tardal, size=224).unsqueeze(0)
    payload = torch.load(HPFM_CKPT, map_location="cpu", weights_only=False)
    h_model = ResNet18ScalarScorer(pretrained=False)
    h_model.load_state_dict(payload["model"])
    h_model.eval()

    bundle = load_frozen_model(LPIFM_CKPT, yaml_backup=LPIFM_YAML, device=torch.device("cpu"), t=0.3, t_cal=1.0)
    # Match the paper's 224 comparison. Official LPIFM inference is 384;
    # the backbone is pooled, so the same weights run at 224.
    transform = build_default_transform(image_size=224)
    def tensor_of(path: str) -> torch.Tensor:
        image = transform(_ensure_rgb(Image.open(path))).unsqueeze(0)
        if image.shape[-2:] != (224, 224):
            raise RuntimeError(f"expected 224, got {tuple(image.shape)}")
        return image

    l_ir, l_vi, l_a, l_b = tensor_of(ir), tensor_of(vi), tensor_of(tardal), tensor_of(swin)
    l_model = bundle.model.eval()

    h_flops = flops_g(h_model, (h_cpu,))
    l_flops = flops_g(l_model, (l_ir, l_vi, l_a, l_b))
    result = {
        "gpu_name": torch.cuda.get_device_name(0),
        "cpu_threads": torch.get_num_threads(),
        "scene": "RoadScene/1",
        "hpfm_candidate": "tardal",
        "lpifm_pair": ["tardal", "swinfusion"],
        "warmup": WARM,
        "runs": RUNS,
        "hpfm": {
            "params_m": params_m(h_model),
            "flops_g": h_flops,
            "input": "1x3x224x224",
            "resolution": 224,
            "unit": "one fused image",
        },
        "lpifm": {
            "params_m": params_m(l_model),
            "flops_g": l_flops,
            "input": "four 3x224x224 tensors; context is 6-channel then projected",
            "resolution": 224,
            "variant": bundle.variant,
            "unit": "one pair",
        },
    }

    h_model.to(gpu)
    l_model.to(gpu)
    hx = h_cpu.to(gpu)
    tensors = [t.to(gpu) for t in (l_ir, l_vi, l_a, l_b)]
    with torch.no_grad():
        result["hpfm"]["gpu_ms"] = bench(lambda: h_model(hx), gpu)
        result["lpifm"]["gpu_ms"] = bench(lambda: l_model(*tensors), gpu)
        print("gpu done", result["hpfm"]["gpu_ms"], result["lpifm"]["gpu_ms"], flush=True)

        # Rank the four E1 fusions of this scene.
        h_inputs = [load_triple(ir, vi, str(SCENE / "fusions" / f"{name}.png"), size=224).unsqueeze(0).to(gpu) for name in E1]
        pairs = []
        for left, right in combinations(E1, 2):
            pairs.append((
                tensors[0], tensors[1],
                tensor_of(str(SCENE / "fusions" / f"{left}.png")).to(gpu),
                tensor_of(str(SCENE / "fusions" / f"{right}.png")).to(gpu),
            ))

        def rank_h():
            for x in h_inputs:
                h_model(x)

        def rank_l():
            for args in pairs:
                l_model(*args)

        result["hpfm"]["gpu_rank4_ms"] = bench(rank_h, gpu)
        result["lpifm"]["gpu_rank6pairs_ms"] = bench(rank_l, gpu)
        print("rank gpu done", result["hpfm"]["gpu_rank4_ms"], result["lpifm"]["gpu_rank6pairs_ms"], flush=True)

    h_model.to("cpu")
    l_model.to("cpu")
    with torch.no_grad():
        result["hpfm"]["cpu_ms"] = bench(lambda: h_model(h_cpu), torch.device("cpu"))
        print("hpfm cpu", result["hpfm"]["cpu_ms"], flush=True)
        result["lpifm"]["cpu_ms"] = bench(lambda: l_model(l_ir, l_vi, l_a, l_b), torch.device("cpu"))
        print("lpifm cpu", result["lpifm"]["cpu_ms"], flush=True)

    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
