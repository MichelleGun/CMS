# Hierarchical RL vs the flat policy

2026-10-23 21:17. Same fixed layouts: 72 laps episodes + 60 slanted starts, through the flight loader. Score = laps/ep + 1.0 x oblique entry (as loop_forever).

| Policy | Laps / 90 s | 2+ passes | Best in 1 ep | Oblique entry | Score |
|---|---|---|---|---|---|
| flat C9 (final) | 2.17 | 62% | 5 | 58% | 2.75 |
| fixed switch (enter -> loop) | 1.96 | 54% | 5 | 55% | 2.51 |
| learned manager (best) | 2.15 | 57% | 5 | 58% | 2.74 |
| learned manager (final) | 1.72 | 46% | 5 | 55% | 2.27 |

Skill usage of the best manager evaluation (rows: before / after the first pass; columns: enter, loop, C9): [[0.0, 0.084, 0.916], [0.002, 0.107, 0.89]]
Fixed-switch baseline during training: {'mean': 1.7222222222222223, 'max': 5, 'two_plus': 0.5, 'zones': {'1': 1.75, '2': 0.5, '3': 2.0, '4': 2.0, '5': 1.5, '6': 2.25, '7': 3.25, '8': 2.25, '9': 0.0}, 'usage_before_after_first_pass': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]}
