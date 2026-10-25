# Overnight loop training (2026-10-25 18:38)

Goal: enter the hoop, loop round, re-enter, as many times as possible. Score = **laps per episode** (passes in a 90 s episode, same-side laps from all 9 start zones, measured profile, full randomisation). Written by `hardware/onboard/loop_night.py`.

## Race (3 envs each, from C's best model, ring tracker on)

| Run | Extra settings | Steps | Last-10 laps/ep | Best eval | State |
|---|---|---|---|---|---|
| w10_laps_T_kl_tracker (control, 8 envs, earlier) | (control) | 32,630,000 | 0.156 | 0.500 | yardstick |
| w10_mg18_LO_low_oblique08 | 0.3 --bc-coef-min 0.3 --lr 5e-5 --target-kl 0.01 --batch-size 512 --n-epochs 5 --critic-warmup-steps 0 --sway-deg 2 --reward-weights {"r_wrong_way": -20, "drain": false, "r_struck": -10, "w_dwell": 0.05} --lr 2e-5 --oblique-p 0.8 --near-pass-p 0.3 | 25,857,356 | 3.492 | 4.056 | crashed |
| w10_mg18_LX_lower_lr | 0.3 --bc-coef-min 0.3 --lr 5e-5 --target-kl 0.01 --batch-size 512 --n-epochs 5 --critic-warmup-steps 0 --sway-deg 2 --reward-weights {"r_wrong_way": -20, "drain": false, "r_struck": -10, "w_dwell": 0.05} --lr 1e-5 --oblique-p 0.6 --near-pass-p 0.3 | 25,757,356 | 3.783 | 4.556 | crashed |

## Ranking (laps task in the sim: same-side laps from all 9 zones, 90 s, measured profile, full randomisation, fixed layouts)

lpe = laps (passes) per 90 s episode. Fly the top row first; the file name is also written into each flight's meta.json (`model`), so a flight log always says which policy flew. Scores from fewer than 36 episodes are listed last. Memory = the policy keeps a history of past frames (onboard/history.py); it flies as it is, `policy.load_policy` keeps the same history on the drone. Oblique entry = share of 30 slanted starts near the ring (35-80 deg off its axis, as from a corner of the real cage) that end in a pass rather than a strike; per-zone scores are in each file's .meta.json (export_scores.zones).

| # | File | From | Steps | Laps/ep | Best in 1 ep | 2+ passes | 1+ pass | Oblique entry | Memory | Episodes |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `LOOP_20261025_w10-mg18-LX-lower-lr_best_4.28lpe.zip` | w10_mg18_LX_lower_lr (best) | 26897356 | 4.28 | 10 | 64% | 81% | 67% | yes | 36 |
| 2 | `LOOP_20261025_w10-mg17-LO-low-oblique08_final_4.08lpe.zip` | w10_mg17_LO_low_oblique08 (final) | 26984396 | 4.08 | 9 | 64% | 81% | 63% | yes | 36 |
| 3 | `LOOP_20261024_w10-mg16-LO-low-oblique08_best_4.06lpe.zip` | w10_mg16_LO_low_oblique08 (best) | 23837356 | 4.06 | 9 | 69% | 83% | 60% | yes | 36 |
| 4 | `LOOP_20261024_w10-mg16-CL-loop-lowlr_best_3.97lpe.zip` | w10_mg16_CL_loop_lowlr (best) | 23837356 | 3.97 | 8 | 64% | 81% | 63% | yes | 36 |
| 5 | `LOOP_20261025_w10-mg18-LO-low-oblique08_best_3.92lpe.zip` | w10_mg18_LO_low_oblique08 (best) | 26957356 | 3.92 | 9 | 56% | 83% | 63% | yes | 36 |
| 6 | `LOOP_20261025_w10-mg17-LO-low-oblique08_best_3.89lpe.zip` | w10_mg17_LO_low_oblique08 (best) | 26984396 | 3.89 | 9 | 58% | 78% | 60% | yes | 36 |
| 7 | `LOOP_20261025_w10-mg18-LO-low-oblique08_final_3.89lpe.zip` | w10_mg18_LO_low_oblique08 (final) | 26957356 | 3.89 | 9 | 56% | 83% | 63% | yes | 36 |
| 8 | `LOOP_20261024_w10-mg16-LO-low-oblique08_final_3.75lpe.zip` | w10_mg16_LO_low_oblique08 (final) | 23837356 | 3.75 | 9 | 61% | 78% | 63% | yes | 36 |
| 9 | `LOOP_20261024_w10-mg15-CL-loop-lowlr_best_3.61lpe.zip` | w10_mg15_CL_loop_lowlr (best) | 21830316 | 3.61 | 8 | 61% | 78% | 53% | yes | 36 |
| 10 | `LOOP_20261024_w10-mg15-LE-low-stress_best_3.53lpe.zip` | w10_mg15_LE_low_stress (best) | 21830316 | 3.53 | 7 | 64% | 81% | 60% | yes | 36 |
| 11 | `LOOP_20261024_w10-mg15-LE-low-stress_final_3.44lpe.zip` | w10_mg15_LE_low_stress (final) | 21830316 | 3.44 | 8 | 64% | 81% | 60% | yes | 36 |
| 12 | `LOOP_20261025_w10-mg18-LX-lower-lr_final_3.36lpe.zip` | w10_mg18_LX_lower_lr (final) | 26897356 | 3.36 | 9 | 50% | 81% | 60% | yes | 36 |
| 13 | `LOOP_20261024_w10-mg14-CL-loop-lowlr_final_3.31lpe.zip` | w10_mg14_CL_loop_lowlr (final) | 19823276 | 3.31 | 7 | 72% | 81% | 60% | yes | 36 |
| 14 | `LOOP_20261024_w10-mg16-CL-loop-lowlr_final_3.31lpe.zip` | w10_mg16_CL_loop_lowlr (final) | 23837356 | 3.31 | 8 | 56% | 81% | 60% | yes | 36 |
| 15 | `LOOP_20261024_w10-mg14-CL-loop-lowlr_best_3.25lpe.zip` | w10_mg14_CL_loop_lowlr (best) | 19823276 | 3.25 | 7 | 69% | 89% | 50% | yes | 36 |
| 16 | `LOOP_20261025_w10-mg17-LD-low-dagger_final_3.19lpe.zip` | w10_mg17_LD_low_dagger (final) | 26944396 | 3.19 | 7 | 53% | 83% | 67% | yes | 36 |
| 17 | `LOOP_20261024_w10-mg15-CL-loop-lowlr_final_3.19lpe.zip` | w10_mg15_CL_loop_lowlr (final) | 21830316 | 3.19 | 8 | 50% | 75% | 60% | yes | 36 |
| 18 | `LOOP_20261025_w10-mg17-LD-low-dagger_best_3.11lpe.zip` | w10_mg17_LD_low_dagger (best) | 26944396 | 3.11 | 8 | 61% | 83% | 67% | yes | 36 |
| 19 | `LOOP_20261024_w10-mg13-CL-loop-lowlr_final_3.03lpe.zip` | w10_mg13_CL_loop_lowlr (final) | 17816236 | 3.03 | 7 | 64% | 89% | 53% | yes | 36 |
| 20 | `LOOP_20261024_w10-mg14-LZ-low-clearance_final_2.75lpe.zip` | w10_mg14_LZ_low_clearance (final) | 19823276 | 2.75 | 6 | 72% | 86% | 57% | yes | 36 |
| 21 | `LOOP_20261024_w10-mg14-LZ-low-clearance_best_2.75lpe.zip` | w10_mg14_LZ_low_clearance (best) | 19823276 | 2.75 | 6 | 61% | 86% | 53% | yes | 36 |
| 22 | `LOOP_20261024_w10-mg13-CL-loop-lowlr_best_2.56lpe.zip` | w10_mg13_CL_loop_lowlr (best) | 17816236 | 2.56 | 6 | 61% | 75% | 60% | yes | 36 |
| 23 | `LOOP_20261023_w10-mg12-CL-loop-lowlr_final_2.42lpe.zip` | w10_mg12_CL_loop_lowlr (final) | 15809196 | 2.42 | 6 | 56% | 78% | 60% | yes | 36 |
| 24 | `LOOP_20261019_w10-mg6-C9-loop-practice_final_2.33lpe.zip` | w10_mg6_C9_loop_practice (final) | 13671080 | 2.33 | 5 | 64% | 81% | 57% | yes | 36 |
| 25 | `LOOP_20261023_w10-mg9-CZ-loop-clearance_best_2.31lpe.zip` | w10_mg9_CZ_loop_clearance (best) | 15678120 | 2.31 | 5 | 64% | 81% | 60% | yes | 36 |
| 26 | `LOOP_20261018_w10-mg3-C9-loop-practice_best_2.19lpe.zip` | w10_mg3_C9_loop_practice (best) | 10766692 | 2.19 | 4 | 64% | 89% | 60% | yes | 36 |
| 27 | `LOOP_20261023_w10-hrl-SL-skill-loop_best_2.17lpe.zip` | w10_hrl_SL_skill_loop (best) | 17238120 | 2.17 | 5 | 56% | 86% | 63% | yes | 36 |
| 28 | `LOOP_20261024_w10-mg13-CO-loop-oblique08_best_2.14lpe.zip` | w10_mg13_CO_loop_oblique08 (best) | 17816236 | 2.14 | 5 | 67% | 81% | 60% | yes | 36 |
| 29 | `LOOP_20261023_w10-mg11-C9-loop-practice_final_2.14lpe.zip` | w10_mg11_C9_loop_practice (final) | 15678120 | 2.14 | 5 | 64% | 83% | 60% | yes | 36 |
| 30 | `LOOP_20261023_w10-mg11-CD-loop-dagger_best_2.14lpe.zip` | w10_mg11_CD_loop_dagger (best) | 15678120 | 2.14 | 5 | 58% | 86% | 63% | yes | 36 |
| 31 | `LOOP_20261023_w10-mg12-CL-loop-lowlr_best_2.11lpe.zip` | w10_mg12_CL_loop_lowlr (best) | 15809196 | 2.11 | 6 | 44% | 81% | 63% | yes | 36 |
| 32 | `LOOP_20261019_w10-mg6-C9-loop-practice_best_2.06lpe.zip` | w10_mg6_C9_loop_practice (best) | 13671080 | 2.06 | 5 | 64% | 83% | 53% | yes | 36 |
| 33 | `LOOP_20261018_w10-mg2-E9-stress_best_2.00lpe.zip` | w10_mg2_E9_stress (best) | 8759652 | 2.00 | 5 | 64% | 81% | 53% | yes | 36 |
| 34 | `LOOP_20261023_w10-mg11-C9-loop-practice_best_1.97lpe.zip` | w10_mg11_C9_loop_practice (best) | 15678120 | 1.97 | 5 | 58% | 83% | 53% | yes | 36 |
| 35 | `LOOP_20261023_w10-mg12-C9-loop-practice_best_1.97lpe.zip` | w10_mg12_C9_loop_practice (best) | 15824640 | 1.97 | 5 | 56% | 81% | 60% | yes | 36 |
| 36 | `LOOP_20261018_w10-mg3-C9-loop-practice_final_1.92lpe.zip` | w10_mg3_C9_loop_practice (final) | 10766692 | 1.92 | 4 | 56% | 78% | 60% | yes | 36 |
| 37 | `LOOP_20261024_w10-mg13-CO-loop-oblique08_final_1.92lpe.zip` | w10_mg13_CO_loop_oblique08 (final) | 17816236 | 1.92 | 5 | 50% | 86% | 67% | yes | 36 |
| 38 | `LOOP_20261018_w10-mg4-C9-loop-practice_best_1.89lpe.zip` | w10_mg4_C9_loop_practice (best) | 12773732 | 1.89 | 4 | 67% | 83% | 53% | yes | 36 |
| 39 | `LOOP_20261018_w10-mg5-C9-loop-practice_best_1.89lpe.zip` | w10_mg5_C9_loop_practice (best) | 12773732 | 1.89 | 4 | 67% | 83% | 53% | yes | 36 |
| 40 | `LOOP_20261018_w10-mg5-OF-oblique-zonefocus_best_1.89lpe.zip` | w10_mg5_OF_oblique_zonefocus (best) | 12773732 | 1.89 | 5 | 56% | 78% | 63% | yes | 36 |
| 41 | `LOOP_20261023_w10-hrl-SL-skill-loop_final_1.89lpe.zip` | w10_hrl_SL_skill_loop (final) | 17238120 | 1.89 | 5 | 56% | 75% | 60% | yes | 36 |
| 42 | `LOOP_20261018_w10-mg3-E9-stress_final_1.83lpe.zip` | w10_mg3_E9_stress (final) | 10766692 | 1.83 | 4 | 53% | 83% | 57% | yes | 36 |
| 43 | `LOOP_20261023_w10-mg9-C9-loop-practice_best_1.83lpe.zip` | w10_mg9_C9_loop_practice (best) | 15678120 | 1.83 | 5 | 53% | 78% | 53% | yes | 36 |
| 44 | `LOOP_20261018_w10-mg5-OF-oblique-zonefocus_final_1.81lpe.zip` | w10_mg5_OF_oblique_zonefocus (final) | 12773732 | 1.81 | 4 | 61% | 81% | 63% | yes | 36 |
| 45 | `LOOP_20261023_w10-mg10-C9-loop-practice_best_1.81lpe.zip` | w10_mg10_C9_loop_practice (best) | 15678120 | 1.81 | 4 | 56% | 83% | 57% | yes | 36 |
| 46 | `LOOP_20261023_w10-mg10-C9-loop-practice_final_1.78lpe.zip` | w10_mg10_C9_loop_practice (final) | 15678120 | 1.78 | 4 | 47% | 86% | 70% | yes | 36 |
| 47 | `LOOP_20261023_w10-mg10-CE-loop-stress_final_1.75lpe.zip` | w10_mg10_CE_loop_stress (final) | 15678120 | 1.75 | 5 | 50% | 78% | 60% | yes | 36 |
| 48 | `LOOP_20261019_w10-mg6-OS-oblique-stall30_best_1.72lpe.zip` | w10_mg6_OS_oblique_stall30 (best) | 13633732 | 1.72 | 4 | 56% | 78% | 67% | yes | 36 |
| 49 | `LOOP_20261023_w10-mg12-C9-loop-practice_final_1.72lpe.zip` | w10_mg12_C9_loop_practice (final) | 15824640 | 1.72 | 4 | 56% | 75% | 60% | yes | 36 |
| 50 | `LOOP_20261018_w10-mg4-DE9-dagger-stress_final_1.72lpe.zip` | w10_mg4_DE9_dagger_stress (final) | 12773732 | 1.72 | 4 | 53% | 86% | 67% | yes | 36 |
| 51 | `LOOP_20261018_w10-mg3-E9-stress_best_1.69lpe.zip` | w10_mg3_E9_stress (best) | 10766692 | 1.69 | 4 | 53% | 81% | 70% | yes | 36 |
| 52 | `LOOP_20261019_w10-mg6-OS-oblique-stall30_final_1.69lpe.zip` | w10_mg6_OS_oblique_stall30 (final) | 13633732 | 1.69 | 4 | 50% | 81% | 53% | yes | 36 |
| 53 | `LOOP_20261018_w10-mg2-E9-stress_final_1.67lpe.zip` | w10_mg2_E9_stress (final) | 8759652 | 1.67 | 4 | 47% | 86% | 57% | yes | 36 |
| 54 | `LOOP_20261018_w10-mg4-C9-loop-practice_final_1.67lpe.zip` | w10_mg4_C9_loop_practice (final) | 12773732 | 1.67 | 4 | 47% | 75% | 63% | yes | 36 |
| 55 | `LOOP_20261018_w10-mg5-C9-loop-practice_final_1.67lpe.zip` | w10_mg5_C9_loop_practice (final) | 12773732 | 1.67 | 4 | 47% | 75% | 63% | yes | 36 |
| 56 | `LOOP_20261023_w10-mg11-CD-loop-dagger_final_1.67lpe.zip` | w10_mg11_CD_loop_dagger (final) | 15678120 | 1.67 | 4 | 47% | 78% | 57% | yes | 36 |
| 57 | `LOOP_20261018_w10-mg4-DE9-dagger-stress_best_1.64lpe.zip` | w10_mg4_DE9_dagger_stress (best) | 12773732 | 1.64 | 4 | 50% | 78% | 60% | yes | 36 |
| 58 | `LOOP_20261023_w10-mg10-CE-loop-stress_best_1.64lpe.zip` | w10_mg10_CE_loop_stress (best) | 15678120 | 1.64 | 5 | 47% | 81% | 53% | yes | 36 |
| 59 | `LOOP_20261018_w10-mg2-O-oblique_best_1.61lpe.zip` | w10_mg2_O_oblique (best) | 8776024 | 1.61 | 4 | 47% | 83% | 63% | yes | 36 |
| 60 | `LOOP_20261015_w10-n9-O-oblique_best_1.58lpe.zip` | w10_n9_O_oblique (best) | 6456984 | 1.58 | 4 | 53% | 81% | 67% | yes | 36 |
| 61 | `LOOP_20261018_w10-mg2-O-oblique_final_1.58lpe.zip` | w10_mg2_O_oblique (final) | 8776024 | 1.58 | 4 | 53% | 83% | 57% | yes | 36 |
| 62 | `LOOP_20261023_w10-mg9-C9-loop-practice_final_1.56lpe.zip` | w10_mg9_C9_loop_practice (final) | 15678120 | 1.56 | 4 | 44% | 81% | 60% | yes | 36 |
| 63 | `LOOP_20261015_w10-n9-O-oblique_final_1.44lpe.zip` | w10_n9_O_oblique (final) | 6456984 | 1.44 | 3 | 42% | 78% | 63% | yes | 36 |
| 64 | `LOOP_20261023_w10-hrl-SE-skill-enter_final_1.33lpe.zip` | w10_hrl_SE_skill_enter (final) | 15678120 | 1.33 | 3 | 42% | 83% | 60% | yes | 36 |
| 65 | `LOOP_20261014_n8-dapg-nodrain_best_1.28lpe.zip` | n8_dapg_nodrain (best) | ? | 1.28 | 2 | 44% | 83% | - | yes | 36 |
| 66 | `LOOP_20261023_w10-mg9-CZ-loop-clearance_final_1.22lpe.zip` | w10_mg9_CZ_loop_clearance (final) | 15678120 | 1.22 | 4 | 31% | 78% | 63% | yes | 36 |
| 67 | `LOOP_20261023_w10-hrl-SE-skill-enter_best_1.17lpe.zip` | w10_hrl_SE_skill_enter (best) | 15678120 | 1.17 | 3 | 33% | 78% | 57% | yes | 36 |
| 68 | `LOOP_20261014_n8-dapg-nodrain-cont2_final_1.08lpe.zip` | n8_dapg_nodrain_cont2 (final) | 6768048 | 1.08 | 3 | 22% | 83% | 60% | yes | 36 |
| 69 | `LOOP_20261015_w10-n9-OZ-oblique-clearance_final_1.06lpe.zip` | w10_n9_OZ_oblique_clearance (final) | 6436984 | 1.06 | 3 | 25% | 75% | 60% | yes | 36 |
| 70 | `LOOP_20261014_n8-dapg-nodrain-cont2_best_1.03lpe.zip` | n8_dapg_nodrain_cont2 (best) | 6768048 | 1.03 | 3 | 22% | 78% | 70% | yes | 36 |
| 71 | `LOOP_20261014_n8-dapg-nodrain_final_1.03lpe.zip` | n8_dapg_nodrain (final) | ? | 1.03 | 3 | 22% | 72% | - | yes | 36 |
| 72 | `LOOP_20261015_w10-n9-OZ-oblique-clearance_best_0.94lpe.zip` | w10_n9_OZ_oblique_clearance (best) | 6436984 | 0.94 | 3 | 19% | 72% | 57% | yes | 36 |
| 73 | `LOOP_20261014_w10-laps4-M-np80_final_0.78lpe.zip` | w10_laps4_M_np80 (final) | 35955688 | 0.78 | 1 | 0% | 78% | - | no | 36 |
| 74 | `LOOP_20261014_n6-same-from-W_final_0.75lpe.zip` | n6_same_from_W (final) | 35173940 | 0.75 | 2 | 8% | 67% | - | yes | 36 |
| 75 | `LOOP_20261014_w10-laps4-M-np80_best_0.72lpe.zip` | w10_laps4_M_np80 (best) | 35955688 | 0.72 | 1 | 0% | 72% | - | no | 36 |
| 76 | `LOOP_20261014_n7-dapg-from-dagger_best_0.69lpe.zip` | n7_dapg_from_dagger (best) | 2476000 | 0.69 | 2 | 6% | 64% | - | yes | 36 |
| 77 | `LOOP_20261014_w10-laps4-P-pure-more_final_0.69lpe.zip` | w10_laps4_P_pure_more (final) | 35955688 | 0.69 | 1 | 0% | 69% | - | no | 36 |
| 78 | `LOOP_20261014_W-trk-pure_best_0.67lpe.zip` | w10_laps3_W_trk_pure (best) | 31632944 | 0.67 | 1 | 0% | 67% | - | no | 36 |
| 79 | `LOOP_20261014_w10-laps4-P-pure-more_best_0.67lpe.zip` | w10_laps4_P_pure_more (best) | 35955688 | 0.67 | 1 | 0% | 67% | - | no | 36 |
| 80 | `LOOP_20261014_w10-laps5-K-pure-cont_best_0.64lpe.zip` | w10_laps5_K_pure_cont (best) | 36669800 | 0.64 | 2 | 3% | 61% | - | no | 36 |
| 81 | `LOOP_20261014_W-trk-pure-ext_final_0.64lpe.zip` | w10_laps3_W_trk_pure_ext (final) | 34629800 | 0.64 | 1 | 0% | 64% | - | no | 36 |
| 82 | `LOOP_20261014_w10-laps4-N-np50_best_0.64lpe.zip` | w10_laps4_N_np50 (best) | 35955688 | 0.64 | 1 | 0% | 64% | - | no | 36 |
| 83 | `LOOP_20261014_n6-same-from-W_best_0.58lpe.zip` | n6_same_from_W (best) | 35173940 | 0.58 | 1 | 0% | 58% | - | yes | 36 |
| 84 | `LOOP_20261014_W-trk-pure-ext_best_0.56lpe.zip` | w10_laps3_W_trk_pure_ext (best) | 34629800 | 0.56 | 1 | 0% | 56% | - | no | 36 |
| 85 | `LOOP_20261014_w10-laps4-N-np50_final_0.56lpe.zip` | w10_laps4_N_np50 (final) | 35955688 | 0.56 | 1 | 0% | 56% | - | no | 36 |
| 86 | `LOOP_20261014_w10-laps5-R-refind_best_0.56lpe.zip` | w10_laps5_R_refind (best) | 36690800 | 0.56 | 1 | 0% | 56% | - | no | 36 |
| 87 | `LOOP_20261014_w10-laps5-R-refind_final_0.53lpe.zip` | w10_laps5_R_refind (final) | 36690800 | 0.53 | 2 | 6% | 47% | - | no | 36 |
| 88 | `LOOP_20261014_W-trk-pure_final_0.53lpe.zip` | w10_laps3_W_trk_pure (final) | 31632944 | 0.53 | 1 | 0% | 53% | - | no | 36 |
| 89 | `LOOP_20261014_Y-trk-loopdemo_best_0.39lpe.zip` | w10_laps3_Y_trk_loopdemo (best) | 31632944 | 0.39 | 2 | 3% | 36% | - | no | 36 |
| 90 | `LOOP_20261014_w10-laps5-K-pure-cont_final_0.39lpe.zip` | w10_laps5_K_pure_cont (final) | 36669800 | 0.39 | 1 | 0% | 39% | - | no | 36 |
| 91 | `LOOP_20261018_T-kl-tracker_best_0.39lpe.zip` | w10_laps_T_kl_tracker (best) | 32644656 | 0.39 | 1 | 0% | 39% | 20% | no | 36 |
| 92 | `LOOP_20261019_T-kl-tracker_best_0.39lpe.zip` | w10_laps_T_kl_tracker (best) | 32644656 | 0.39 | 1 | 0% | 39% | 20% | no | 36 |
| 93 | `LOOP_20261023_T-kl-tracker_best_0.39lpe.zip` | w10_laps_T_kl_tracker (best) | 32644656 | 0.39 | 1 | 0% | 39% | 20% | no | 36 |
| 94 | `LOOP_20261024_T-kl-tracker_best_0.39lpe.zip` | w10_laps_T_kl_tracker (best) | 32644656 | 0.39 | 1 | 0% | 39% | 20% | no | 36 |
| 95 | `LOOP_20261025_T-kl-tracker_best_0.39lpe.zip` | w10_laps_T_kl_tracker (best) | 32644656 | 0.39 | 1 | 0% | 39% | 20% | no | 36 |
| 96 | `LOOP_20261014_X-trk-combo_best_0.28lpe.zip` | w10_laps3_X_trk_combo (best) | 31632944 | 0.28 | 1 | 0% | 28% | - | no | 36 |
| 97 | `LOOP_20261014_X-trk-combo_final_0.25lpe.zip` | w10_laps3_X_trk_combo (final) | 31632944 | 0.25 | 1 | 0% | 25% | - | no | 36 |
| 98 | `LOOP_20261018_T-kl-tracker_final_0.17lpe.zip` | w10_laps_T_kl_tracker (final) | 32644656 | 0.17 | 1 | 0% | 17% | 30% | no | 36 |
| 99 | `LOOP_20261019_T-kl-tracker_final_0.17lpe.zip` | w10_laps_T_kl_tracker (final) | 32644656 | 0.17 | 1 | 0% | 17% | 30% | no | 36 |
| 100 | `LOOP_20261023_T-kl-tracker_final_0.17lpe.zip` | w10_laps_T_kl_tracker (final) | 32644656 | 0.17 | 1 | 0% | 17% | 30% | no | 36 |
| 101 | `LOOP_20261024_T-kl-tracker_final_0.17lpe.zip` | w10_laps_T_kl_tracker (final) | 32644656 | 0.17 | 1 | 0% | 17% | 30% | no | 36 |
| 102 | `LOOP_20261025_T-kl-tracker_final_0.17lpe.zip` | w10_laps_T_kl_tracker (final) | 32644656 | 0.17 | 1 | 0% | 17% | 30% | no | 36 |
| 103 | `LOOP_20261014_Y-trk-loopdemo_final_0.14lpe.zip` | w10_laps3_Y_trk_loopdemo (final) | 31632944 | 0.14 | 1 | 0% | 14% | - | no | 36 |
| 104 | `LOOP_20261014_n7-dapg-from-dagger_final_0.08lpe.zip` | n7_dapg_from_dagger (final) | 2476000 | 0.08 | 1 | 0% | 8% | - | yes | 36 |
| 105 | `LOOP_20261013_T-kl-tracker_best_0.25lpe.zip` | w10_laps_T_kl_tracker (best) | 32644656 | 0.25 | 1 | 0% | 25% | - | no | 12 |
| 106 | `LOOP_20261013_T-kl-tracker_final_0.08lpe.zip` | w10_laps_T_kl_tracker (final) | 32644656 | 0.08 | 1 | 0% | 8% | - | no | 12 |

Fly the top one:

```bash
python hardware/onboard/fly_hoop.py --mode policy --model hardware/policies/LOOP_20261025_w10-mg18-LX-lower-lr_best_4.28lpe.zip
```
