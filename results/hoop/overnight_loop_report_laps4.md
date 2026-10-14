# Overnight loop training (2026-10-14 12:16)

Goal: enter the hoop, loop round, re-enter, as many times as possible. Score = **laps per episode** (passes in a 90 s episode, same-side laps from all 9 start zones, measured profile, full randomisation). Written by `hardware/onboard/loop_night.py`.

## Race (3 envs each, from C's best model, ring tracker on)

| Run | Extra settings | Steps | Last-10 laps/ep | Best eval | State |
|---|---|---|---|---|---|
| w10_laps_T_kl_tracker (control, 8 envs, earlier) | (control) | 32,630,000 | 0.156 | 0.500 | yardstick |
| w10_laps4_N_np50 | --bc-coef 0 --bc-coef-min 0 --near-pass-p 0.5 | 35,729,800 | 0.631 | 0.778 | running |
| w10_laps4_M_np80 | --bc-coef 0 --bc-coef-min 0 --near-pass-p 0.8 | 35,949,800 | 0.622 | 0.806 | completed |
| w10_laps4_P_pure_more | --bc-coef 0 --bc-coef-min 0 | 35,949,800 | 0.667 | 0.861 | completed |
