# Overnight loop training (2026-10-17 08:08)

Goal: enter the hoop, loop round, re-enter, as many times as possible. Score = **laps per episode** (passes in a 90 s episode, same-side laps from all 9 start zones, measured profile, full randomisation). Written by `hardware/onboard/loop_night.py`.

## Race (3 envs each, from C's best model, ring tracker on)

| Run | Extra settings | Steps | Last-10 laps/ep | Best eval | State |
|---|---|---|---|---|---|
| w10_laps_T_kl_tracker (control, 8 envs, earlier) | (control) | 32,630,000 | 0.156 | 0.500 | yardstick |
| w10_mg1_O_oblique | 0.3 --bc-coef-min 0.3 --lr 5e-5 --target-kl 0.01 --batch-size 512 --n-epochs 5 --critic-warmup-steps 0 --sway-deg 2 --reward-weights {"r_wrong_way": -20, "drain": false, "r_struck": -10, "w_dwell": 0.05} --oblique-p 0.4 | 6,748,984 | 1.331 | 1.583 | running |
| w10_mg1_D9_dagger_relabel | 0.3 --bc-coef-min 0.3 --lr 5e-5 --target-kl 0.01 --batch-size 512 --n-epochs 5 --critic-warmup-steps 0 --sway-deg 2 --reward-weights {"r_wrong_way": -20, "drain": false, "r_struck": -10, "w_dwell": 0.05} --oblique-p 0.4 --demos hardware/flights/demos/dagger_n1_r*.npz hardware/flights/demos/dagger_mg_r*.npz | 6,748,984 | 1.386 | 1.500 | running |
