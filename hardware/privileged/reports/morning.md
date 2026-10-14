# Morning report (2026-10-13 14:05)

Goal: enter the hoop, loop round the outside, re-enter, as many times as possible.

Everything below ran on the corrected simulator (crossing fix, measured profile, no speed input, clockwise yaw). Written by `hardware/privileged/autopilot.py`; log in `results/hoop/autopilot.log`.

## 1. Week 10 PPO settings (front runway, one pass per episode, 3M steps each)

| Run | Status | Best (5-eval avg) | Last 5 evals |
|---|---|---|---|
| A baseline | completed | 0.727 | 0.627 |
| B gamma | completed | 0.813 | 0.640 |
| C kl | completed | 0.940 | 0.913 **(winner)** |
| D entropy | completed | 0.593 | 0.420 |
| E sde | None | 0.280 | 0.000 |
| F lambda1 | completed | 0.827 | 0.660 |

Passes per front-runway episode (max 1). Curves and diagnostics: `tutorial/week10_compare.ipynb`.

## 2. The goal: the winner on same-side laps (3M steps)

`w10_laps_C_kl`: completed; laps per 90 s episode, last 5 evaluations: **0.02**, best: 0.37. Notebook: set `PATTERN = "w10_laps_*"`.

## 3. DAgger: the camera-only student taught by the privileged teacher

**Teacher** (sees the true state): 4.78 laps / 90 s, clean 100%, best streak 6. **G4** = student ≥ 3.82 laps / 90 s (80 % of the teacher).
| Round | Ticks collected | Teacher flew | Handover at (noodle) | Student laps / 90 s | Clean | Best streak | G4 | Clone |
|---|---|---|---|---|---|---|---|---|
| 0 | 35529 | 100% | 6 cm | **0.00** | 0% | 0 | – | `hardware\flights\bc\dagger_r0` |
| 1 | 16212 | 19% | 5 cm | **0.06** | 0% | 1 | – | `hardware\flights\bc\dagger_r1` |
| 2 | 21071 | 14% | 4 cm | **0.00** | 0% | 0 | – | `hardware\flights\bc\dagger_r2` |
| 3 | 18835 | 11% | 2 cm | **0.00** | 0% | 0 | – | `hardware\flights\bc\dagger_r3` |
| 4 | 18623 | 5% | 1 cm | **0.00** | 0% | 0 | – | `hardware\flights\bc\dagger_r4` |
| 5 | 18753 | 1% | 0 cm | **0.00** | 0% | 0 | – | `hardware\flights\bc\dagger_r5` |
| 6 | 22313 | 2% | 0 cm | **0.00** | 0% | 0 | – | `hardware\flights\bc\dagger_r6` |
| 7 | 26353 | 2% | 0 cm | **0.00** | 0% | 0 | – | `hardware\flights\bc\dagger_r7` |
| 8 | 22359 | 1% | 0 cm | **0.11** | 0% | 1 | – | `hardware\flights\bc\dagger_r8` |

## What to look at first

1. Section 2: laps per 90 s of the RL policy. The teacher manages ~4.8 (it sees the true state).
2. Section 3: does the student's laps column climb round by round? If it stays near 0, the camera inputs (no speed, ring often out of view on the way round) may not carry what the teacher uses; that is the question to look at next (more memory in the observation, or a privileged critic).
3. The best policy is the candidate for the next real flight, after the 1-minute hand check of the yaw sign on the real drone.
