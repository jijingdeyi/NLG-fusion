# RoadScene v2 frozen HPFM vs LPIFM

Device: cuda. LPIFM state dict: swa.
Pack: /data/ykx/pref_bench_roadscene_v2. HPFM: /data/ykx/pref_bench/fit/scorer_stage1_best.pt. LPIFM: /home/ykx/LPIFM/checkpoints/lpifm_vifb_baseline_v1.pt.
Decisive means q_A <= 0.4 or q_A >= 0.6. q_A=0.5 stays out of the denominator.
Protocol A keeps LPIFM ties as misses. Protocol B forces the sign of d_cal; exact zero predicts A.
original_ids uses pair ids from the previous 30-scene pack and the updated aggregated labels.
paper_hpfm on that subset is the published reference, not a number to force.

e1 v2_full: scenes 40 n 232 HPFM 176/232 LPIFM-A 139/232 (reverse 69 tie 24) LPIFM-B 152/232 diffA 15.95 [8.62, 23.48] diffB 10.34 [3.04, 17.90]
e1 original_ids: scenes 30 n 85 HPFM 69/85 LPIFM-A 59/85 (reverse 21 tie 5) LPIFM-B 63/85 diffA 11.76 [2.35, 21.18] diffB 7.06 [-1.20, 16.28]
e2 v2_full: scenes 40 n 222 HPFM 157/222 LPIFM-A 133/222 (reverse 73 tie 16) LPIFM-B 141/222 diffA 10.81 [1.35, 20.18] diffB 7.21 [-1.84, 16.22]
e2 original_ids: scenes 30 n 80 HPFM 56/80 LPIFM-A 56/80 (reverse 22 tie 2) LPIFM-B 58/80 diffA 0.00 [-14.63, 14.81] diffB -2.50 [-16.89, 12.20]
