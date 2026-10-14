# Week 10 race (2026-10-13 08:17)

Yardstick: `w10_front_C_kl` (C). A run is stopped when its last-10-evaluation mean falls more than 0.1 below C's at the same point (or more than 0.05 below twice in a row), checked every 400k steps. Written by `hardware/onboard/w10_race.py`.

| Run | Settings | Steps | Last-10 mean | C at same point | Gap | State |
|---|---|---|---|---|---|---|
| w10_front_G_kl_gamma | --target-kl 0.02 --batch-size 512 --n-epochs 5 --gamma 0.995 --gae-lambda 0.97 | 940k | 0.580 | 0.687 | -0.107 | stopped: 0.17 below C at 800k steps |
| w10_front_H_kl_lambda1 | --target-kl 0.02 --batch-size 512 --n-epochs 5 --gae-lambda 1.0 | 420k | 0.270 | 0.613 | -0.343 | stopped: 0.34 below C at 400k steps |
| w10_front_C_kl_s1 | --target-kl 0.02 --batch-size 512 --n-epochs 5 | 3000k | 0.873 | 0.927 | -0.053 | completed |

Winner: **w10_front_C_kl** (last-10 mean 0.927); w10_front_C_kl 0.927, w10_front_C_kl_s1 0.873

Then: `w10_front_C_kl_long` last-10 0.813; laps stage `w10_laps_C_kl_long` last-10 0.133 laps per episode.
