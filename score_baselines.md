# 手工 Score / 指标分析归档（非主线）

> 本文档从 `score.md` 拆出，收录 **竞争归一化 + soft max-min 配方 A–G**、六集指标分析、教师扫参与先验网格等**历史/基线**材料。  
> **主线**（专家偏好 → 学习可部署 $S$）见 [`score.md`](score.md)。  
> 同场对比时，可将此处手工裁判作为基线；**不**再作为最终 $S$ 的拍板依据。

---

## 1. 手工配方与竞争归一化

历史 soft max-min 与配方 A–G **使用竞争归一化 $a_j$**（「过中位 / 追最优」语义）：

$$
a_j
=
\operatorname{clip}
\left(
\frac{m_j-\mathrm{floor}_j}{\mathrm{target}_j-\mathrm{floor}_j},
\,-1,\,1
\right),
\quad
\mathrm{floor}=\mathrm{median}_{\mathrm{methods}},\
\mathrm{target}=\max_{\mathrm{methods}}.
$$

$$
S_{\mathrm{raw}}
=
0.95\min_j a_j
+
0.05\operatorname{mean}_j a_j,
\qquad
S=\frac{S_{\mathrm{raw}}+1}{2}.
$$

| 方案 | 组成 | 角色 |
|------|------|------|
| A | MI, Qabf, SCD, VIF, SSIM | 社区五元组基线 |
| B/C | 去 VIF / 去 SSIM | 消融 |
| D | MI, Qabf, SF | 离线最稳的手工候选 |
| E/F/G/H | 加 SCD / 加 VIF / VIF换Qabf / 仅Qabf+SF | 对照 |

实现：`ldt/fusion_metrics.py`（`SCORE_RECIPES`、`COMPETITIVE_RANGES`、`score_from_metrics`）。

在偏好考试中，学到的 $S^*$ 宜与这些基线及单指标 **同场报告**。

---

## 2. 教师先验 $m(\alpha)$

用于强度先验，也可生成偏好集中的朴素候选与单调性链。

**（A）`blend`（历史默认，§4.1 已跑）**

$$
m(\alpha)=(1-\alpha)\,m_{\mathrm{int}}+\alpha\,m_{\mathrm{hf}},
\quad
m_{\mathrm{int}}=\max(\mathrm{IR},\mathrm{VI}),
\quad
m_{\mathrm{hf}}=\mathrm{clip}(\mathrm{LP}(m_{\mathrm{int}})+\mathrm{HF}).
$$

HF：5×5 box 后按梯度从 IR/VI 选边。$\alpha=1$ 是完整「低频+选边高频」图，不是纯高频残差。

**（B）`inject`（本意：扫高频注入量）**

$$
m(\alpha)=\mathrm{clip}\big(\mathrm{LP}(m_{\mathrm{int}})+\alpha\,\mathrm{HF}\big).
$$

$\alpha=0$ 为 $\mathrm{LP}(\max)$（比 max 更糊）；$\alpha=1$ 与 blend 的 α=1 相同。  
代码：`build_fusion_prior(..., mode='blend'|'inject')`。

---

## 3. 六集公开表

`/data/ykx/metric/*.xlsx`，用方法 **`mean` 行**（约 20–21 方法）。

| 数据集 | 文件 | 方法数 |
|--------|------|-------:|
| FMB | `FMB.xlsx` | 21 |
| LLVIP | `LLVIP.xlsx` | 21 |
| M3FD | `M3FD.xlsx` | 21 |
| MSRS | `MSRS.xlsx` | 21 |
| OpIVF | `OpIVF.xlsx` | 20 |
| RoadScene | `RoadScene.xlsx` | 21 |

### 3.1 相关 / 带宽 / 跨集（特征池启发）

| 现象 | 结论 |
|------|------|
| SF–AG $r\approx0.98$ | 二选一（默认倾向 SF） |
| MSE–PSNR 极高相关 | **禁止**进选模特征 |
| VIF–Qabf $r\approx0.82$ | 稳定假互补；稀疏学习宜只留一个 |
| SCD 与 Qabf/MI 符号跨集翻转 | 不宜当稳定轴 |
| SF–Qabf $r\approx0.18$ | 低相关，宜同时保留进池 |

| 指标 | median→max 带宽 | 含义 |
|------|-----------------|------|
| SSIM | 0.027–0.064 极窄 | 易主导手工 max-min |
| SF/AG | 很大 | 分辨力强 |
| MI | 中大 | 可用；偏 max |

跨集排序一致性大致：SF > AG > Qabf > MI > VIF ≫ SCD > SSIM。

| 指标 | 对特征池的建议 |
|------|----------------|
| Qabf, SF, MI | 优先进池（MI 宜封顶） |
| VIF | 与 Qabf 冗余 |
| SSIM, SCD | 可进池；预期常低权 |
| AG | 与 SF 二选一 |
| EN / CC / MS_SSIM | 不优先 |
| MSE/PSNR | 禁止 |

---

## 4. 教师超参扫参（2026-07-29，手工配方 A）

> 在 **A + soft max-min** 下的 α / int-tex 行为及 SSIM 过敏。主线应用学到的 $S^*$ 重打分后再定教师默认。

| 项 | 取值 |
|----|------|
| 脚本 | `scripts/sweep_prior_loss.py` |
| 数据 | LLVIP，`val_samples=30`，crop 320 |
| 结果 | `logs/sweeps/prior_loss_20260729_1411` |
| Phase1 | blend 先验当融合图；α 网格 |
| Phase2 | α=0.3 短训 int/tex |

当时 ranges（LLVIP median→max）：MI 2.00→3.07；Qabf 0.54→0.68；SCD 1.42→1.71；VIF 0.74→0.96；SSIM 0.65→0.68（**带宽 0.027**）。

**Phase1**

| α | score | worst | MI | Qabf | SCD | VIF | SSIM |
|--:|------:|-------|---:|-----:|----:|----:|-----:|
| 0.0 | 0.324 | ssim | 4.958 | 0.580 | 1.420 | 0.794 | 0.638 |
| 0.2 | 0.476 | ssim | 3.816 | 0.568 | 1.448 | 0.786 | 0.646 |
| **0.3** | **0.501** | ssim | 3.612 | 0.570 | 1.458 | 0.778 | 0.648 |
| 0.5 | 0.485 | ssim | 3.362 | 0.589 | 1.472 | 0.763 | 0.647 |
| 0.7 | 0.406 | ssim | 3.180 | 0.615 | 1.480 | 0.747 | 0.642 |
| 1.0 | 0.206 | ssim | 3.008 | 0.629 | 1.481 | 0.718 | 0.631 |

**Phase2（α=0.3）**

| int/tex | score | worst | MI | Qabf | SCD | VIF | SSIM |
|---------|------:|-------|---:|-----:|----:|----:|-----:|
| 0.7/0.3 | 0.397 | ssim | 3.995 | 0.572 | 1.444 | 0.789 | 0.642 |
| 0.6/0.4 | 0.413 | ssim | 3.953 | 0.574 | 1.444 | 0.789 | 0.643 |
| **0.5/0.5** | **0.447** | ssim | 3.885 | 0.577 | 1.449 | 0.789 | 0.645 |
| 0.4/0.6 | 0.417 | ssim | 3.332 | 0.610 | 1.474 | 0.769 | 0.643 |
| 0.3/0.7 | 0.369 | ssim | 2.992 | 0.645 | 1.476 | 0.759 | 0.640 |

**保留结论：** max-min 能压超高 MI 糊模；本轮高分≈SSIM 略好（标尺过敏）；短训 $S$ 低于纯先验；MI/VIF 随细节注入下降。支持「SSIM 不宜死锁、偏好学习必要」，不支持把 A 当最终真理。

---

## 5. 公开方法表上的手工配方

方法均值 → 归一化 → A–G。摘要：

- 跨集名次一致性：**D 平均 $\rho=0.82$（最低 0.70）最优**；G 0.78；A 0.68。
- A 常被 SCD/SSIM 锁短板；D 压力在 Qabf/SF。
- MI 偏科方法在 A 下可进前三、在 D 下常跌出（如 OpIVF SwinFusion 1→9）。
- A–D 名次相关仅 ~0.76 → 不同手工裁判会选不同赢家。

→ 手工 D 是强基线；是否人眼最优，见主文档偏好实验。

---

## 6. 先验网格上的基线打分

### 6.1 `blend`（已完成）

LLVIP val 30；α 网格。原始指标：α↑ 则 MI/VIF↓、Qabf/SF↑。

| α | S_A | S_D | S_F | S_G |
|--:|----:|----:|----:|----:|
| 0.0 | 0.073 | 0.600 | **0.584** | **0.586** |
| 0.3 | **0.247** | 0.568 | 0.550 | 0.553 |
| 1.0 | 0.027 | **0.767** | 0.421 | 0.421 |

| 配方 | 最优 α | α=0 最高？ | 保留结论 |
|------|-------:|:----------:|----------|
| A/B | 0.3 | 否 | 不奖励糊，但锁 SSIM |
| D/E | 1.0 | 否 | 偏好完整 $m_{\mathrm{hf}}$ |
| **F/G** | **0.0** | **是** | **淘汰向**（VIF→max） |

目视与 D 选 α=1 一致；F/G 选最糊 → 含强 VIF 压力的手工式不适合当最终裁判。

![先验条带](runs/score_prior_alpha/qual_strips/00_020435.png)

（更多样例见 `runs/score_prior_alpha/qual_strips/`。）

### 6.2 `inject`（待跑）

```bash
python scripts/score_prior_alpha_grid.py \
  --prior_mode inject --out runs/score_prior_alpha_inject \
  --data /data/ykx/LLVIP/train
```

用于核对「高频注入量」单调性；结果可并入主文档偏好候选与自动单调性测。
