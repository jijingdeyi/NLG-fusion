# RoadScene frozen HPFM vs LPIFM

Device: cuda. LPIFM state dict: swa.
Pack: /data/ykx/pref_bench_roadscene. HPFM: /data/ykx/pref_bench/fit/scorer_stage1_best.pt. LPIFM: /home/ykx/LPIFM/checkpoints/lpifm_vifb_baseline_v1.pt.
Decisive means q_A <= 0.4 or q_A >= 0.6. q_A=0.5 stays out of the denominator.
Protocol A keeps LPIFM ties as misses. Protocol B forces the sign of d_cal; exact zero predicts A.

e1 full: scenes 40 n 232 HPFM 176/232 LPIFM-A 139/232 (reverse 69 tie 24) LPIFM-B 152/232 diffA 15.95 [8.62, 23.48] diffB 10.34 [3.04, 17.90]
e2 full: scenes 40 n 222 HPFM 157/222 LPIFM-A 133/222 (reverse 73 tie 16) LPIFM-B 141/222 diffA 10.81 [1.35, 20.18] diffB 7.21 [-1.84, 16.22]
