# Overnight loop training (2026-10-16 08:59)

Goal: enter the hoop, loop round, re-enter, as many times as possible. Score = **laps per episode** (passes in a 90 s episode, same-side laps from all 9 start zones, measured profile, full randomisation). Written by `hardware/onboard/loop_night.py`.

## Race (3 envs each, from C's best model, ring tracker on)

| Run | Extra settings | Steps | Last-10 laps/ep | Best eval | State |
|---|---|---|---|---|---|
| w10_laps_T_kl_tracker (control, 8 envs, earlier) | (control) | 32,630,000 | 0.156 | 0.500 | yardstick |
| w10_steven_OZ_oblique_clearance | 0.3 --bc-coef-min 0.3 --lr 5e-5 --target-kl 0.01 --batch-size 512 --n-epochs 5 --critic-warmup-steps 0 --sway-deg 2 --reward-weights {"r_wrong_way": -20, "drain": false, "r_struck": -15, "w_dwell": 0.05, "w_rim": 1.0, "rim_margin_m": 0.15, "axis_pull": 2.0} --oblique-p 0.4 | 9,448,684 | 1.144 | 1.667 | completed |

## Extension (the 2 best, continued)

| Run | Steps | Last-10 laps/ep | Best eval | State |
|---|---|---|---|---|
| w10_steven_OZ_oblique_clearance_ext | 12,453,400 | 1.231 | 1.472 | completed |

## Ranking (laps task in the sim: same-side laps from all 9 zones, 90 s, measured profile, full randomisation, fixed layouts)

lpe = laps (passes) per 90 s episode. Fly the top row first; the file name is also written into each flight's meta.json (`model`), so a flight log always says which policy flew. Scores from fewer than 36 episodes are listed last. Memory = the policy keeps a history of past frames (onboard/history.py); it flies as it is, `policy.load_policy` keeps the same history on the drone. Oblique entry = share of 30 slanted starts near the ring (35-80 deg off its axis, as from a corner of the real cage) that end in a pass rather than a strike; per-zone scores are in each file's .meta.json (export_scores.zones).

| # | File | From | Steps | Laps/ep | Best in 1 ep | 2+ passes | 1+ pass | Oblique entry | Memory | Episodes |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `LOOP_20261015_w10-n9-O-oblique_best_1.58lpe.zip` | w10_n9_O_oblique (best) | 6456984 | 1.58 | 4 | 53% | 81% | 67% | yes | 36 |
| 2 | `LOOP_20261015_w10-n9-O-oblique_final_1.44lpe.zip` | w10_n9_O_oblique (final) | 6456984 | 1.44 | 3 | 42% | 78% | 63% | yes | 36 |
| 3 | `LOOP_20261016_w10-steven-OZ-oblique-clearance_best_1.33lpe.zip` | w10_steven_OZ_oblique_clearance (best) | 9453400 | 1.33 | 3 | 42% | 81% | 53% | yes | 36 |
| 4 | `LOOP_20261014_n8-dapg-nodrain_best_1.28lpe.zip` | n8_dapg_nodrain (best) | ? | 1.28 | 2 | 44% | 83% | - | yes | 36 |
| 5 | `LOOP_20261016_w10-steven-OZ-oblique-clearance-ext_final_1.22lpe.zip` | w10_steven_OZ_oblique_clearance_ext (final) | 12459864 | 1.22 | 3 | 33% | 81% | 53% | yes | 36 |
| 6 | `LOOP_20261016_w10-steven-OZ-oblique-clearance_final_1.19lpe.zip` | w10_steven_OZ_oblique_clearance (final) | 9453400 | 1.19 | 3 | 36% | 81% | 57% | yes | 36 |
| 7 | `LOOP_20261016_w10-steven-OZ-oblique-clearance-ext_best_1.11lpe.zip` | w10_steven_OZ_oblique_clearance_ext (best) | 12459864 | 1.11 | 2 | 28% | 83% | 57% | yes | 36 |
| 8 | `LOOP_20261014_n8-dapg-nodrain-cont2_final_1.08lpe.zip` | n8_dapg_nodrain_cont2 (final) | 6768048 | 1.08 | 3 | 22% | 83% | 60% | yes | 36 |
| 9 | `LOOP_20261015_w10-n9-OZ-oblique-clearance_final_1.06lpe.zip` | w10_n9_OZ_oblique_clearance (final) | 6436984 | 1.06 | 3 | 25% | 75% | 60% | yes | 36 |
| 10 | `LOOP_20261014_n8-dapg-nodrain-cont2_best_1.03lpe.zip` | n8_dapg_nodrain_cont2 (best) | 6768048 | 1.03 | 3 | 22% | 78% | 70% | yes | 36 |
| 11 | `LOOP_20261014_n8-dapg-nodrain_final_1.03lpe.zip` | n8_dapg_nodrain (final) | ? | 1.03 | 3 | 22% | 72% | - | yes | 36 |
| 12 | `LOOP_20261015_w10-n9-OZ-oblique-clearance_best_0.94lpe.zip` | w10_n9_OZ_oblique_clearance (best) | 6436984 | 0.94 | 3 | 19% | 72% | 57% | yes | 36 |
| 13 | `LOOP_20261014_w10-laps4-M-np80_final_0.78lpe.zip` | w10_laps4_M_np80 (final) | 35955688 | 0.78 | 1 | 0% | 78% | - | no | 36 |
| 14 | `LOOP_20261014_n6-same-from-W_final_0.75lpe.zip` | n6_same_from_W (final) | 35173940 | 0.75 | 2 | 8% | 67% | - | yes | 36 |
| 15 | `LOOP_20261014_w10-laps4-M-np80_best_0.72lpe.zip` | w10_laps4_M_np80 (best) | 35955688 | 0.72 | 1 | 0% | 72% | - | no | 36 |
| 16 | `LOOP_20261014_n7-dapg-from-dagger_best_0.69lpe.zip` | n7_dapg_from_dagger (best) | 2476000 | 0.69 | 2 | 6% | 64% | - | yes | 36 |
| 17 | `LOOP_20261014_w10-laps4-P-pure-more_final_0.69lpe.zip` | w10_laps4_P_pure_more (final) | 35955688 | 0.69 | 1 | 0% | 69% | - | no | 36 |
| 18 | `LOOP_20261014_W-trk-pure_best_0.67lpe.zip` | w10_laps3_W_trk_pure (best) | 31632944 | 0.67 | 1 | 0% | 67% | - | no | 36 |
| 19 | `LOOP_20261014_w10-laps4-P-pure-more_best_0.67lpe.zip` | w10_laps4_P_pure_more (best) | 35955688 | 0.67 | 1 | 0% | 67% | - | no | 36 |
| 20 | `LOOP_20261014_w10-laps5-K-pure-cont_best_0.64lpe.zip` | w10_laps5_K_pure_cont (best) | 36669800 | 0.64 | 2 | 3% | 61% | - | no | 36 |
| 21 | `LOOP_20261014_W-trk-pure-ext_final_0.64lpe.zip` | w10_laps3_W_trk_pure_ext (final) | 34629800 | 0.64 | 1 | 0% | 64% | - | no | 36 |
| 22 | `LOOP_20261014_w10-laps4-N-np50_best_0.64lpe.zip` | w10_laps4_N_np50 (best) | 35955688 | 0.64 | 1 | 0% | 64% | - | no | 36 |
| 23 | `LOOP_20261014_n6-same-from-W_best_0.58lpe.zip` | n6_same_from_W (best) | 35173940 | 0.58 | 1 | 0% | 58% | - | yes | 36 |
| 24 | `LOOP_20261014_W-trk-pure-ext_best_0.56lpe.zip` | w10_laps3_W_trk_pure_ext (best) | 34629800 | 0.56 | 1 | 0% | 56% | - | no | 36 |
| 25 | `LOOP_20261014_w10-laps4-N-np50_final_0.56lpe.zip` | w10_laps4_N_np50 (final) | 35955688 | 0.56 | 1 | 0% | 56% | - | no | 36 |
| 26 | `LOOP_20261014_w10-laps5-R-refind_best_0.56lpe.zip` | w10_laps5_R_refind (best) | 36690800 | 0.56 | 1 | 0% | 56% | - | no | 36 |
| 27 | `LOOP_20261014_w10-laps5-R-refind_final_0.53lpe.zip` | w10_laps5_R_refind (final) | 36690800 | 0.53 | 2 | 6% | 47% | - | no | 36 |
| 28 | `LOOP_20261014_W-trk-pure_final_0.53lpe.zip` | w10_laps3_W_trk_pure (final) | 31632944 | 0.53 | 1 | 0% | 53% | - | no | 36 |
| 29 | `LOOP_20261014_Y-trk-loopdemo_best_0.39lpe.zip` | w10_laps3_Y_trk_loopdemo (best) | 31632944 | 0.39 | 2 | 3% | 36% | - | no | 36 |
| 30 | `LOOP_20261014_w10-laps5-K-pure-cont_final_0.39lpe.zip` | w10_laps5_K_pure_cont (final) | 36669800 | 0.39 | 1 | 0% | 39% | - | no | 36 |
| 31 | `LOOP_20261016_T-kl-tracker_best_0.39lpe.zip` | w10_laps_T_kl_tracker (best) | 32644656 | 0.39 | 1 | 0% | 39% | 20% | no | 36 |
| 32 | `LOOP_20261014_X-trk-combo_best_0.28lpe.zip` | w10_laps3_X_trk_combo (best) | 31632944 | 0.28 | 1 | 0% | 28% | - | no | 36 |
| 33 | `LOOP_20261014_X-trk-combo_final_0.25lpe.zip` | w10_laps3_X_trk_combo (final) | 31632944 | 0.25 | 1 | 0% | 25% | - | no | 36 |
| 34 | `LOOP_20261016_T-kl-tracker_final_0.17lpe.zip` | w10_laps_T_kl_tracker (final) | 32644656 | 0.17 | 1 | 0% | 17% | 30% | no | 36 |
| 35 | `LOOP_20261014_Y-trk-loopdemo_final_0.14lpe.zip` | w10_laps3_Y_trk_loopdemo (final) | 31632944 | 0.14 | 1 | 0% | 14% | - | no | 36 |
| 36 | `LOOP_20261014_n7-dapg-from-dagger_final_0.08lpe.zip` | n7_dapg_from_dagger (final) | 2476000 | 0.08 | 1 | 0% | 8% | - | yes | 36 |
| 37 | `LOOP_20261013_T-kl-tracker_best_0.25lpe.zip` | w10_laps_T_kl_tracker (best) | 32644656 | 0.25 | 1 | 0% | 25% | - | no | 12 |
| 38 | `LOOP_20261013_T-kl-tracker_final_0.08lpe.zip` | w10_laps_T_kl_tracker (final) | 32644656 | 0.08 | 1 | 0% | 8% | - | no | 12 |

Fly the top one:

```bash
python hardware/onboard/fly_hoop.py --mode policy --model hardware/policies/LOOP_20261015_w10-n9-O-oblique_best_1.58lpe.zip
```
