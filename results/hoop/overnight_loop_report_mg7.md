# Overnight loop training (2026-10-19 23:27)

Goal: enter the hoop, loop round, re-enter, as many times as possible. Score = **laps per episode** (passes in a 90 s episode, same-side laps from all 9 start zones, measured profile, full randomisation). Written by `hardware/onboard/loop_night.py`.

## Race (3 envs each, from C's best model, ring tracker on)

| Run | Extra settings | Steps | Last-10 laps/ep | Best eval | State |
|---|---|---|---|---|---|
| w10_laps_T_kl_tracker (control, 8 envs, earlier) | (control) | 32,630,000 | 0.156 | 0.500 | yardstick |
| w10_mg7_C9_loop_practice | 0.3 --bc-coef-min 0.3 --lr 5e-5 --target-kl 0.01 --batch-size 512 --n-epochs 5 --critic-warmup-steps 0 --sway-deg 2 --reward-weights {"r_wrong_way": -20, "drain": false, "r_struck": -10, "w_dwell": 0.05} --oblique-p 0.6 --near-pass-p 0.3 | 15,791,084 | 1.739 | 2.111 | running |
| w10_mg7_ON_oblique_normreward | 0.3 --bc-coef-min 0.3 --lr 5e-5 --target-kl 0.01 --batch-size 512 --n-epochs 5 --critic-warmup-steps 0 --sway-deg 2 --reward-weights {"r_wrong_way": -20, "drain": false, "r_struck": -10, "w_dwell": 0.05} --oblique-p 0.4 --norm-reward | 15,691,084 | 1.872 | 2.194 | running |
