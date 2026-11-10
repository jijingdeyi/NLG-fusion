# HPFM on LPIFM VIFB splits

A/B 顺序沿用 LPIFM 标签文件，没有按方法名重排。

## all_images_unseen_methods
LPIFM config test_labels_json. All 21 scenes, 6 methods excluded from LPIFM training.
- 全部 315 对，其中标签为 Tie 的 13 对；明确偏好 302 对，场景 21
- 口径 A：HPFM 226/302 (74.8%)；LPIFM 255/302 (84.4%)（正确 255，判反 39，判平 8）
- 口径 B：HPFM 226/302 (74.8%)；LPIFM 260/302 (86.1%)
- HPFM−LPIFM 口径 A -9.60 个百分点 [-15.46, -3.64]；仅 HPFM 25，仅 LPIFM 54
- 口径 B 差值 -11.26 个百分点 [-17.16, -5.55]；仅 HPFM 21，仅 LPIFM 55
- LPIFM 三分类（含人工 Tie，论文式准确率） 256/315 (81.3%)；人工 Tie 上判平 1/13

## unseen_images_unseen_methods
5 validation scenes and the 6 methods excluded from LPIFM training.
- 全部 75 对，其中标签为 Tie 的 2 对；明确偏好 73 对，场景 5
- 口径 A：HPFM 55/73 (75.3%)；LPIFM 59/73 (80.8%)（正确 59，判反 10，判平 4）
- 口径 B：HPFM 55/73 (75.3%)；LPIFM 61/73 (83.6%)
- HPFM−LPIFM 口径 A -5.48 个百分点 [-15.07, 4.11]；仅 HPFM 9，仅 LPIFM 13
- 口径 B 差值 -8.22 个百分点 [-17.81, 1.35]；仅 HPFM 7，仅 LPIFM 13
- LPIFM 三分类（含人工 Tie，论文式准确率） 59/75 (78.7%)；人工 Tie 上判平 0/2

## unseen_images_all_methods
5 scenes held out of LPIFM training. The released checkpoint was selected on this scene split.
- 全部 1500 对，其中标签为 Tie 的 93 对；明确偏好 1407 对，场景 5
- 口径 A：HPFM 909/1407 (64.6%)；LPIFM 1161/1407 (82.5%)（正确 1161，判反 199，判平 47）
- 口径 B：HPFM 909/1407 (64.6%)；LPIFM 1184/1407 (84.2%)
- HPFM−LPIFM 口径 A -17.91 个百分点 [-25.64, -10.30]；仅 HPFM 99，仅 LPIFM 351
- 口径 B 差值 -19.55 个百分点 [-27.65, -11.57]；仅 HPFM 86，仅 LPIFM 361
- LPIFM 三分类（含人工 Tie，论文式准确率） 1179/1500 (78.6%)；人工 Tie 上判平 18/93

## seen_images_seen_methods
16 scenes and 19 methods used to train LPIFM. Not a held-out test.
- 全部 2736 对，其中标签为 Tie 的 168 对；明确偏好 2568 对，场景 16
- 口径 A：HPFM 1619/2568 (63.0%)；LPIFM 2403/2568 (93.6%)（正确 2403，判反 133，判平 32）
- 口径 B：HPFM 1619/2568 (63.0%)；LPIFM 2420/2568 (94.2%)
- HPFM−LPIFM 口径 A -30.53 个百分点 [-36.08, -25.20]；仅 HPFM 72，仅 LPIFM 856
- 口径 B 差值 -31.19 个百分点 [-36.60, -26.04]；仅 HPFM 64，仅 LPIFM 865
- LPIFM 三分类（含人工 Tie，论文式准确率） 2442/2736 (89.3%)；人工 Tie 上判平 39/168

口径与上一轮相同。HPFM 用 s_A-s_B>=0 判 A。LPIFM 用 swa、T_cal=1、阈值 0.3。明确偏好只含标签 0/1。场景 bootstrap 种子 443，10000 次。见过/未见过的方法是相对 LPIFM 训练集；HPFM 没有用 VIFB 训练。
