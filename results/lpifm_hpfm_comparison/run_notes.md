# LPIFM 与 HPFM 冻结对比

## 命令

```bash
/home/ykx/micromamba/envs/reconet/bin/python /home/ykx/NLG-fusion/results/lpifm_hpfm_comparison/run_comparison.py
```

实际设备 `cuda:0`，`CUDA_VISIBLE_DEVICES=3`。

## 代码与权重

- NLG-fusion `93b7bf6529ea786f025e3357929e9923203b4569`
- LPIFM `d589acf7c19ca0495a1e19f54660b3855a0ced38`
- HPFM 权重 `/data/ykx/pref_bench/fit/scorer_stage1_best.pt`，epoch 10，arch resnet18_scalar，size 224，tau 1.0。与 `checkpoints/scorer_stage1_best.pt` 及 `hpfm/src/hpfm/weights.pt` 的 `model` 状态字典逐张量相等。推理为 `eval`，输入是 `pref.dataset.load_triple`：灰度、`cv2.INTER_AREA` 到 224、`[0,1]` 的 IR/VI/F，网络内部做 ImageNet 归一化。决策是 `s_A - s_B >= 0` 判 A，否则判 B。没有平局阈值，所以口径 A 与口径 B 的 HPFM 决策相同。分差恰好为 0 时判 A，并单独计数。
- LPIFM 权重 `/home/ykx/LPIFM/checkpoints/lpifm_vifb_baseline_v1.pt`，sha256 `28c98f3fdb1da138c3bc7a0ea952b2707dd25a9ec1531fd6375064ed3349cd3a`。配置来自 checkpoint，推理阈值来自 `/home/ykx/LPIFM/configs/release_inference.yaml`：`t=0.3`，`T_cal=1.0`。checkpoint 内 inference 为 `{"calibration_temperature_Tcal": 1.0, "model_variant": "auto", "tie_threshold_t": 0.3}`。实际载入 `swa`。`load_state_dict(strict=False)` 缺失键 0，意外键 0。图像经官方 `build_default_transform(384)`：非 RGB 先变成三通道，`Resize(441)`、`CenterCrop(384)`、ImageNet 归一化。前向 `d = s(A) - s(B)`，`d_cal = d / T_cal`。`d_cal > 0.3` 为 A，`d_cal < -0.3` 为 B，否则 Tie。口径 B 忽略该阈值，`d_cal > 0` 为 A，`d_cal < 0` 为 B，`d_cal == 0` 为 A。

## 清单

q_A 来自监督文件 `y_A + 0.5 y_T`。y_A 是标签 A 的经验比例，q_A 越大越偏好 A。明确偏好是 `q_A <= 0.4` 或 `q_A >= 0.6`。

- E1：场景 30，对 90，明确偏好 85，q_A=0.5 5，缺文件 0
- E2：场景 30，对 90，明确偏好 85，q_A=0.5 5，缺文件 0
- 域内 Full：场景 29，对 239，明确偏好 209，q_A=0.5 30，缺文件 0
- Natural：137 对，明确偏好 113，明确偏好涉及场景 29
- Public：52 对，明确偏好 44，明确偏好涉及场景 23

Natural 去掉候选名含 `deg_` 的对。Public 只保留两侧都属于 method_a、method_b、method_c、method_d 的对。二者都是 Full 的子集。RoadScene 监督与 pairs 未匹配 0。记录重复键 0。HPFM 域内缓存 239 行，与测试清单同序匹配，A/B 翻转 0，q 不一致 0，缓存缺对 0。域内 HPFM 使用该缓存，不重新推理。RoadScene 没有逐对缓存，使用上述冻结前向。

q_A=0.5 的构成：E1/E2 全部是单次标注且 y_T=1，即观察者当次选择了平局。域内 24 条同样是单次平局，6 条是两次标注 y_A=y_B=0.5、y_T=0，即相反投票聚合为 0.5。这些都只作为描述，不称为已验证的人类平局识别率。

## 交换验证

图对 `e1::RoadScene/34::ddfm__swinfusion`，A=ddfm，B=swinfusion，q_A=1.0（decisive_A）。
`d_cal=3.771775` → A。交换 A/B 后 `d_cal=-3.774027` → B。两次之和 -0.002253。同一顺序重复前向是逐位相同的，所以这个残差来自 triple-perception 对两个候选槽位的交叉计算，不是 A/B 路径接反。符号和官方标签都已对调。该交换记录没有进入测试分母。

## 结果

- e1 / full（明确偏好 85，场景 30）
  - 口径 A：HPFM 69/85 (81.2%)；LPIFM 正确 59、判反 21、判平 5，合计 85/85，准确率 59/85 (69.4%)；明确方向覆盖 80/85，条件准确率 59/80 (73.8%)
  - 口径 B：HPFM 69/85 (81.2%)；LPIFM 63/85 (74.1%)；零分差 HPFM 0、LPIFM 0
  - HPFM−LPIFM：口径 A 11.76 个百分点（95% 场景 bootstrap [2.35, 21.18]），仅 HPFM 对 16，仅 LPIFM 对 6
  - 口径 B 差值 7.06 个百分点（[-1.20, 16.28]），仅 HPFM 对 12，仅 LPIFM 对 6
  - 论文 HPFM 对照值 0.812
- e2 / full（明确偏好 85，场景 30）
  - 口径 A：HPFM 59/85 (69.4%)；LPIFM 正确 60、判反 23、判平 2，合计 85/85，准确率 60/85 (70.6%)；明确方向覆盖 83/85，条件准确率 60/83 (72.3%)
  - 口径 B：HPFM 59/85 (69.4%)；LPIFM 62/85 (72.9%)；零分差 HPFM 0、LPIFM 0
  - HPFM−LPIFM：口径 A -1.18 个百分点（95% 场景 bootstrap [-15.48, 13.10]），仅 HPFM 对 14，仅 LPIFM 对 15
  - 口径 B 差值 -3.53 个百分点（[-17.86, 10.59]），仅 HPFM 对 13，仅 LPIFM 对 16
  - 论文 HPFM 对照值 0.694
- indomain / full（明确偏好 209，场景 29）
  - 口径 A：HPFM 170/209 (81.3%)；LPIFM 正确 113、判反 78、判平 18，合计 209/209，准确率 113/209 (54.1%)；明确方向覆盖 191/209，条件准确率 113/191 (59.2%)
  - 口径 B：HPFM 170/209 (81.3%)；LPIFM 121/209 (57.9%)；零分差 HPFM 0、LPIFM 0
  - HPFM−LPIFM：口径 A 27.27 个百分点（95% 场景 bootstrap [18.35, 36.45]），仅 HPFM 对 70，仅 LPIFM 对 13
  - 口径 B 差值 23.44 个百分点（[14.01, 33.33]），仅 HPFM 对 63，仅 LPIFM 对 14
  - 论文 HPFM 对照值 0.813
- indomain / natural（明确偏好 113，场景 29）
  - 口径 A：HPFM 90/113 (79.6%)；LPIFM 正确 55、判反 44、判平 14，合计 113/113，准确率 55/113 (48.7%)；明确方向覆盖 99/113，条件准确率 55/99 (55.6%)
  - 口径 B：HPFM 90/113 (79.6%)；LPIFM 61/113 (54.0%)；零分差 HPFM 0、LPIFM 0
  - HPFM−LPIFM：口径 A 30.97 个百分点（95% 场景 bootstrap [18.02, 43.24]），仅 HPFM 对 40，仅 LPIFM 对 5
  - 口径 B 差值 25.66 个百分点（[12.62, 38.89]），仅 HPFM 对 35，仅 LPIFM 对 6
  - 论文 HPFM 对照值 0.796
- indomain / public（明确偏好 44，场景 23）
  - 口径 A：HPFM 34/44 (77.3%)；LPIFM 正确 19、判反 19、判平 6，合计 44/44，准确率 19/44 (43.2%)；明确方向覆盖 38/44，条件准确率 19/38 (50.0%)
  - 口径 B：HPFM 34/44 (77.3%)；LPIFM 22/44 (50.0%)；零分差 HPFM 0、LPIFM 0
  - HPFM−LPIFM：口径 A 34.09 个百分点（95% 场景 bootstrap [12.77, 53.85]），仅 HPFM 对 19，仅 LPIFM 对 4
  - 口径 B 差值 27.27 个百分点（[4.55, 50.00]），仅 HPFM 对 17，仅 LPIFM 对 5
  - 论文 HPFM 对照值 0.773

q_A=0.5 上 LPIFM 官方判平数/该子集总数：

- e1: active_tie 0/5（合计 5）
- e2: active_tie 1/5（合计 5）
- indomain: aggregated_half 1/6, active_tie 7/24（合计 30）

见过或未见过的融合方法只相对于 HPFM 训练集：E1 的 TarDAL / CDDFuse / SwinFusion / DDFM 在 HPFM 训练中以 method_a–d 出现，E2 的 EMMA / TextIF / MetaFusion / DCEvo 不在该训练集。这里没有把 E2 说成 LPIFM 也未见过。

Bootstrap 在每个协议内按场景有放回抽取，抽中场景的全部明确偏好对按重复次数计入，种子 443，10000 次，区间是差值的 2.5% 与 97.5% 分位。Public 明确偏好只涉及 23 个场景。
