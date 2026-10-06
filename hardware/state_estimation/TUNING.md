# Tuning the state estimator

The filter (`fusion_kf.py`) blends two imperfect sources:

| source | rate | good at | bad at |
|---|---|---|---|
| overhead camera (ArUco) | ~30 Hz, can drop out | absolute position, never drifts | noise, latency, blackouts (glare, motion blur, occlusion, marker too small) |
| Tello telemetry (velocity, height, attitude) | ~10 Hz | smooth, always there, no latency to speak of | velocity is biased and coarse, so integrating it drifts |

Position is tracked as `[x y z vx vy vz]`. Velocity and height measurements keep it moving smoothly between camera
frames and bridge blackouts; camera positions keep it from drifting. **How much each is trusted is set by a few
numbers in `config.yaml` under `estimator:`.** A Kalman filter trusts a source in proportion to how small its noise
setting is relative to the others.

## The knobs

| setting | meaning | raise it when | lower it when |
|---|---|---|---|
| `sigma_cam_xy` (m) | camera position noise at good quality | camera jitters (bad light, blur, small marker, wide-angle edge distortion): the filter will then lean on dead-reckoning | the camera is clean and sharp; the estimate lags the truth during manoeuvres |
| `sigma_vel` (m/s) | Tello velocity noise | velocity readings are biased / noisy (see below): the filter leans on the camera | you trust Tello's optical-flow velocity (good floor texture, steady light) |
| `q_acc` (m/s^2) | how much the drone may accelerate unpredictably (process noise) | the estimate lags when you change direction (model too stiff): the filter reacts faster to both sources | the estimate is jittery at constant velocity |
| `sigma_z` (m) | Tello height noise | height jumps (ToF dropouts, ground reflections) | height is steady |
| `cam_gate_chi2` | outlier gate on camera positions | you see good detections rejected (counter `kf.cam_rejected` climbing) | a stray marker / reflection keeps pulling the estimate |
| `cam_latency_s`, `latency_*` | camera delay baseline and the online tracker's limits | `calibrate_speed.py` reports a different latency | |
| `yaw_offset_gain`, `yaw_offset_max_rate_dps` | how fast camera yaw may correct Tello's drifting yaw | Tello yaw drifts faster than the offset follows | the heading estimate wobbles |

Two things are applied automatically so you do not have to hand-tune for them:

* **Quality scaling**: each camera detection carries a quality in (0, 1] from the marker's apparent size, and
  its noise is divided by it. A marker far from the camera, or seen at a shallow angle, counts for less.
* **Latency jitter**: while the drone moves, the camera noise is inflated by `(speed * sigma_latency)^2`, where
  `sigma_latency` is the spread of the online latency estimate. Fast motion with jittery camera timing therefore
  trusts the camera less, with no change to the settings.

## Trust by conditions

| condition | what happens to the camera | what to change |
|---|---|---|
| bright, even light, matte markers | low noise | defaults are fine; optionally `sigma_cam_xy` 0.02 |
| dim light, long exposure, motion blur | noisy and late | `sigma_cam_xy` 0.05-0.08, `q_acc` stays; use a camera with manual exposure <= 5 ms if possible |
| glare / glossy floor / markers on shiny paper | outliers, flicker | keep `cam_gate_chi2` at 13.8 (gate catches them); use matte print; `sigma_cam_xy` 0.04 |
| drone high and far from the camera | marker small, quality low | automatic via quality; a bigger drone marker (10 cm) helps most |
| floor with no texture (Tello's downward flow sensor blind) | n/a | raise `sigma_vel` to 0.12-0.2 so the camera dominates |
| long camera blackouts expected (people walking through) | gone for seconds | lower `q_acc` slightly (more dead-reckoning memory); check the health flags below |

## Health flags the safety layer uses

* `pos_health`: `OK`, `DEGRADED` (no camera for `cam_degraded_s`, running on dead-reckoning), `LOST` (position
  std above `pos_lost_std`: the estimate cannot be trusted). `LOST` triggers hover then safe landing.
* `yaw_health`: `DEGRADED` when camera and Tello heading disagree by more than `yaw_degraded_deg` for
  `yaw_degraded_s` (the correction freezes), `LOST` beyond `yaw_lost_deg` or on a sudden jump.
* If `pos_std` grows during a blackout, that is the filter telling you how long dead-reckoning can be believed.
  With the defaults a 1-2 s blackout is bridged comfortably; ~5 s at normal speed is not.

## Procedure

1. Fly (or `--dry-run`) with logging on (the default). Logs in `hardware/flights/*.jsonl` store every raw
   telemetry packet and camera detection.
2. Replay with candidate settings:

   ```
   python hardware/state_estimation/tune_replay.py --log hardware/flights/teleop_<time>.jsonl
   python hardware/state_estimation/tune_replay.py --log ... --grid wide
   ```
   Every 4th camera detection is withheld; the score is how far the filter's prediction at that instant is from
   it (RMS, cm). Its floor is about the camera's own noise, so compare candidates against each other and against
   your current config, not against zero.
3. Read the winner: best `sigma_cam_xy` much larger than the config => your camera is noisier than assumed;
   best `q_acc` large => the model lags your manoeuvres; best `sigma_vel` large => Tello velocity is poor.
4. Copy winning values into `config.yaml` (`estimator:`), re-run step 2 on a *different* flight to check it holds.

## Order of magnitude to expect

Camera 2-4 cm RMS at good lighting; fused position 3-6 cm RMS; error rises to 10-20 cm over a 2 s blackout while
moving. Anything much worse points to a calibration problem (camera intrinsics, `camera.height_m`, wrong marker
positions in `aruco.reference`) rather than to Q/R.
