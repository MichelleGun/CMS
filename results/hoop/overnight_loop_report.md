# Overnight loop training (2026-10-14 01:58)

Goal: enter the hoop, loop round, re-enter, as many times as possible. Score = **laps per episode** (passes in a 90 s episode, same-side laps from all 9 start zones, measured profile, full randomisation). Written by `hardware/onboard/loop_night.py`.

## Race (3 envs each, from C's best model, ring tracker on)

| Run | Extra settings | Steps | Last-10 laps/ep | Best eval | State |
|---|---|---|---|---|---|
| w10_laps_T_kl_tracker (control, 8 envs, earlier) | (control) | 32,630,000 | 0.156 | 0.500 | yardstick |
| w10_laps3_X_trk_combo | --gamma 0.995 --gae-lambda 0.97 --streak 1.0 --streak-cap 4 --bc-coef 0 --bc-coef-min 0 | 31,629,800 | 0.283 | 0.417 | completed |
| w10_laps3_W_trk_pure | --bc-coef 0 --bc-coef-min 0 | 31,629,800 | 0.450 | 0.778 | completed |
| w10_laps3_Y_trk_loopdemo | --gamma 0.995 --gae-lambda 0.97 --streak 1.0 --streak-cap 4 --bc-coef 0.1 --bc-coef-min 0.05 + loop demos | 31,629,800 | 0.303 | 0.417 | completed |
| w10_laps3_U_trk_gamma | --gamma 0.995 --gae-lambda 0.97 | 30,849,878 | 0.103 | 0.361 | stopped: 0.10 laps/episode below T after 60 evaluations |
| w10_laps3_V_trk_streak | --streak 1.0 --streak-cap 4 | 31,629,800 | 0.178 | 0.500 | stopped: 0.11 laps/episode below T after 100 evaluations |
