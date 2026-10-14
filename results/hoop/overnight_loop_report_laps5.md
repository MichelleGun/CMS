# Overnight loop training (2026-10-14 13:30)

Goal: enter the hoop, loop round, re-enter, as many times as possible. Score = **laps per episode** (passes in a 90 s episode, same-side laps from all 9 start zones, measured profile, full randomisation). Written by `hardware/onboard/loop_night.py`.

## Race (3 envs each, from C's best model, ring tracker on)

| Run | Extra settings | Steps | Last-10 laps/ep | Best eval | State |
|---|---|---|---|---|---|
| w10_laps_T_kl_tracker (control, 8 envs, earlier) | (control) | 32,630,000 | 0.156 | 0.500 | yardstick |
| w10_laps5_L_teacher_loop | --demos hardware/flights/demos/teacher_loop_*.npz --bc-coef 0.2 --bc-coef-min 0.02 --bc-decay-steps 2000000 | 36,509,744 | 0.039 | 0.556 | stopped |
| w10_laps5_Q_teacher_loop_refind | --demos hardware/flights/demos/teacher_loop_*.npz --bc-coef 0.2 --bc-coef-min 0.02 --bc-decay-steps 2000000 --reward-weights {"r_found": 3.0} | 36,469,748 | 0.022 | 0.556 | stopped |
| w10_laps5_R_refind | --bc-coef 0 --bc-coef-min 0 --reward-weights {"r_found": 3.0} | 36,589,736 | 0.447 | 0.722 | running |
| w10_laps5_K_pure_cont | --bc-coef 0 --bc-coef-min 0 | 36,589,736 | 0.617 | 0.750 | running |
