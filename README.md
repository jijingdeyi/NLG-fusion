# NLG-fusion / DTRC

红外–可见光融合：**轻量 backbone** + 可选前置校正 **`dtrc/`**。

## 训练 backbone

```bash
python scripts/train.py --data /data/ykx/LLVIP/train
```

合成：\(u=\mathrm{clip}(m+s_r R)\)。  
- \(m=(1-\alpha)\max+\alpha\max_{\mathrm{hf}}\)，默认 \(\alpha=0.3\)（`--prior_hf_alpha`）  
- decoder：只吃 \([B_{\mathrm{ir}};B_{\mathrm{vi}}]\)，不吃 \(m\)  
- 损失：默认 `0.5 L_int(max) + 0.5 L_tex`（`--f_weight`）
- 选模：`val/score` 为竞争基准 max-min（LLVIP 对比方法 median→max 归一化，0.95·min+0.05·mean）

## 扫先验 α 与损失比（推荐先跑）

```bash
python scripts/sweep_prior_loss.py --data /data/ykx/LLVIP/train
```

用同一套 max-min `val/score` 选教师超参（α、int/tex）。

## 推理

```bash
python scripts/pred.py --ckpt checkpoint/<run>/<best>.ckpt --data ... --dst ...
```

## DTRC

```bash
python scripts/train_dtrc_stage2.py --data /data/ykx/LLVIP/train --backbone_ckpt <ckpt>
```

详见 `experiments.md` §11。
