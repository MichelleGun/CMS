"""
How wrong is the simulator? Measured from the real Tello flight logs, nothing assumed.

    python hardware/privileged/sim_fidelity.py            # all flights in hardware/flights/
    python hardware/privileged/sim_fidelity.py --no-sim   # skip the simulated-detector comparison

Writes hardware/privileged/reports/sim_fidelity.md (+ .json, + plots/*.png).

Every real flight logs, per control tick (15 Hz), the rc sticks sent and the detector's reading
(ticks.jsonl, clock = seconds since takeoff_done), and the Tello's own telemetry at ~5 Hz
(telemetry.csv, wall clock). Lining the two up gives, per channel, "what was asked" against
"what the drone did":

    lr  (rc[0]) -> vgy  (body sideways speed, cm/s)
    fb  (rc[1]) -> vgx  (body forward speed, cm/s)
    ud  (rc[2]) -> h    (height, cm; vgz is unreliable, so height itself is fitted)
    yaw (rc[3]) -> yaw  (heading, deg; fitted on the angle, not its noisy derivative)

Each channel is fitted with the SAME model sim.py uses (dead zone -> transport delay -> first-order
lag, slower to stop than to start) plus a gain. The gain is solved exactly for every candidate
(delay, lag up, lag down, dead zone), because the model is linear in it, and the candidate with the
smallest error wins. Spread comes from refitting on bootstrap resamples of whole flights.

The replay test then drives two copies of that model with each real flight's sticks: one with the
simulator's current constants, one with the fitted ones, and reports how far each lands from what
the drone really did. That error is the trust score for anything trained in the simulator.
"""
import argparse
import csv
import glob
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HW = HERE.parent
for p in (HW,):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from onboard import sim as onboard_sim  # noqa: E402
from onboard import spec  # noqa: E402

FLIGHTS = HW / "flights"
OUT = HERE / "reports"
DT = spec.CTRL_DT
RC_MAX = spec.RC_MAX

# channel -> (rc index, telemetry column, kind). "rate": telemetry IS the speed; "angle"/"height":
# telemetry is the integral of the speed, fitted with a per-flight offset.
CHANNELS = {
    "fb": (1, "vgx", "rate"),
    "lr": (0, "vgy", "rate"),
    "ud": (2, "h", "height"),
    "yaw": (3, "yaw", "angle"),
    # forward speed from the mission pads under the cage (x, y in cm relative to the pad in view),
    # because the Tello's own vgx carries no direction here. Speed magnitude vs |forward stick|,
    # only while the same pad stays in view and the sideways stick is near zero.
    "fb_pad": (1, "padspeed", "speed"),
}
UNITS = {"fb": "cm/s per rc", "lr": "cm/s per rc", "ud": "cm/s per rc", "yaw": "deg/s per rc",
         "fb_pad": "cm/s per rc"}

# What the simulator assumes today, per rc unit and as reported in its telemetry (sim.py).
SIM = onboard_sim.PROFILES["typical"]
SIM_GAIN = {"fb": 100.0 / onboard_sim.RC_PER_MPS, "lr": 100.0 / onboard_sim.RC_PER_MPS,
            "ud": 100.0 / onboard_sim.RC_PER_MPS,
            # sim.step: yaw_deg += -cmd * MAX_YAW_DPS -> reported yaw FALLS for a positive (right) stick
            "yaw": -onboard_sim.YAW_DPS_PER_RC, "fb_pad": 100.0 / onboard_sim.RC_PER_MPS}
SIM_PARAMS = dict(delay=SIM.delay_s, tau_up=SIM.tau_accel, tau_dn=SIM.tau_decel,
                  dead=SIM.deadband * RC_MAX)
DR_SPEED, DR_YAW = onboard_sim.SPEED_SCALE_DR, onboard_sim.YAW_SCALE_DR

GRID = dict(delay=np.array([0.0, 0.067, 0.133, 0.2, 0.267, 0.333, 0.4, 0.5, 0.6, 0.8, 1.0]),
            tau_up=np.array([0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0]),
            tau_dn=np.array([0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0]),
            dead=np.array([0.0, 2.0, 4.0, 6.0, 9.0, 12.0, 15.0]))
WINDOW_S = 3.0   # height / heading are fitted over windows this long, each with its own offset


def _utc(s):
    return datetime.fromisoformat(s).timestamp()


def load_flights(root=FLIGHTS, min_ticks=50):
    out = []
    for d in sorted(glob.glob(str(root / "tello_*"))):
        d = Path(d)
        try:
            ticks = [json.loads(line) for line in open(d / "ticks.jsonl")]
            ticks = [k for k in ticks if k.get("type") == "tick" and k.get("rc") is not None]
            if len(ticks) < min_ticks:
                continue
            events = [json.loads(line) for line in open(d / "events.jsonl")]
            t0 = [_utc(e["utc"]) for e in events if e.get("event") == "takeoff_done"]
            if not t0:
                continue
            tel = list(csv.DictReader(open(d / "telemetry.csv")))
            meta = json.loads((d / "meta.json").read_text())
        except (OSError, ValueError, KeyError):
            continue
        if meta.get("dry_run") is not False:
            continue        # dry runs fly the simulator itself: measuring them would measure the sim
        T = np.array([k["t"] for k in ticks], float)
        tt = np.array([_utc(r["utc"]) - t0[0] for r in tel], float)
        keep = (tt >= T[0]) & (tt <= T[-1])
        if keep.sum() < 10:
            continue
        telv = {c: np.array([float(r[c]) for r in tel], float)[keep] for c in ("vgx", "vgy", "h", "yaw")}
        telv["yaw"] = np.degrees(np.unwrap(np.radians(telv["yaw"])))
        telv["padspeed"] = _pad_speed(tel, tt)[keep]
        out.append(dict(name=d.name, mode=meta.get("mode"), T=T, rc=np.array([k["rc"] for k in ticks], float),
                        ticks=ticks, tt=tt[keep], tel=telv))
    return out


def _pad_speed(tel, tt):
    """Horizontal speed (cm/s) from mission-pad x/y by central difference, NaN unless the same
    pad is seen on both neighbours within 0.6 s (a pad switch jumps the frame)."""
    mid = np.array([int(float(r.get("mid", -1) or -1)) for r in tel])
    x = np.array([float(r.get("x", 0) or 0) for r in tel])
    y = np.array([float(r.get("y", 0) or 0) for r in tel])
    out = np.full(len(tel), np.nan)
    for i in range(1, len(tel) - 1):
        a, b = i - 1, i + 1
        if mid[a] > 0 and mid[a] == mid[b] == mid[i] and 0 < tt[b] - tt[a] < 0.6:
            out[i] = math.hypot(x[b] - x[a], y[b] - y[a]) / (tt[b] - tt[a])
    return out


def _grid():
    g = np.array(np.meshgrid(GRID["delay"], GRID["tau_up"], GRID["tau_dn"], GRID["dead"], indexing="ij"))
    return g.reshape(4, -1).T                                  # (n_candidates, 4)


def simulate(cmd_t, cmd, cand, t_end):
    """Unit-gain response of every candidate at once. cmd: rc values held from cmd_t[i].
    cand: (n, 4) = delay, tau_up, tau_dn, dead. Returns (grid_t, v[n_steps, n])."""
    gt = np.arange(cmd_t[0], t_end + DT, DT)
    n = len(cand)
    v = np.zeros(n)
    out = np.empty((len(gt), n))
    a_up = 1.0 - np.exp(-DT / cand[:, 1])
    a_dn = 1.0 - np.exp(-DT / cand[:, 2])
    idx_cache = {}
    for i, t in enumerate(gt):
        # the rc each candidate sees: the one sent `delay` ago (zero-order hold)
        u = np.empty(n)
        for dly in np.unique(cand[:, 0]):
            j = idx_cache.get((i, dly))
            if j is None:
                j = np.searchsorted(cmd_t, t - dly, side="right") - 1
            sel = cand[:, 0] == dly
            u[sel] = cmd[j] if j >= 0 else 0.0
        u = np.where(np.abs(u) < cand[:, 3], 0.0, u)
        a = np.where(np.abs(u) >= np.abs(v), a_up, a_dn)
        v = v + a * (u - v)
        out[i] = v
    return gt, out


def design(flight, ch, cand):
    """Per candidate: the unit-gain model sampled where the telemetry is, ready for the gain fit."""
    rc_i, col, kind = CHANNELS[ch]
    cmd = flight["rc"][:, rc_i]
    gt, v = simulate(flight["T"], np.abs(cmd) if kind == "speed" else cmd, cand, flight["T"][-1])
    if kind == "speed":
        k = np.clip(np.searchsorted(gt, flight["tt"]), 0, len(gt) - 1)
        X, y = v[k], flight["tel"][col].copy()
        j = np.clip(np.searchsorted(flight["T"], flight["tt"], side="right") - 1, 0, len(flight["T"]) - 1)
        ok = np.isfinite(y) & (np.abs(flight["rc"][j, 0]) <= 3)
        X, y = X[ok], y[ok]
        if len(y):
            # jitter makes a speed MAGNITUDE positive even when hovering: fit the gain from how
            # speed varies with the stick (per-flight offset), not from its level
            X = X - X.mean(axis=0)
            y = y - y.mean()
        return X, y
    if kind == "rate":
        x = v
    else:
        x = np.cumsum(v, axis=0) * DT                        # integrate speed -> height / angle
    k = np.clip(np.searchsorted(gt, flight["tt"]) , 0, len(gt) - 1)
    X = x[k]                                                 # (n_tel, n_cand)
    y = flight["tel"][col].copy()
    if kind != "rate":
        # a free offset per WINDOW_S window: slow drift (ToF over a cluttered floor, compass
        # creep) cannot pile up over a whole flight and swamp the stick response
        w = np.floor((flight["tt"] - flight["tt"][0]) / WINDOW_S).astype(int)
        for k_ in np.unique(w):
            m = w == k_
            X[m] -= X[m].mean(axis=0)
            y[m] -= y[m].mean()
    return X, y


def fit_channel(flights, ch, cand):
    """-> best params, gain, per-candidate data for bootstrap, residual RMS."""
    parts = [design(f, ch, cand) for f in flights]
    XX = np.zeros(len(cand))
    Xy = np.zeros(len(cand))
    yy = 0.0
    per = []
    for X, y in parts:
        a, b, c = (X * X).sum(0), (X * y[:, None]).sum(0), float(y @ y)
        XX += a
        Xy += b
        yy += c
        per.append((a, b, c))
    gain = Xy / np.maximum(XX, 1e-12)
    sse = yy - gain * Xy
    best = int(np.argmin(sse))
    n = sum(len(y) for _, y in parts)
    return dict(best=best, gain=float(gain[best]), params=dict(zip(("delay", "tau_up", "tau_dn", "dead"),
                                                                   map(float, cand[best]))),
                rms=float(math.sqrt(max(sse[best], 0.0) / n)), per=per, n=n)


def bootstrap(fit, cand, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    per = fit["per"]
    gains, delays, tups, tdns = [], [], [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(per), len(per))
        XX = sum(per[i][0] for i in pick)
        Xy = sum(per[i][1] for i in pick)
        yy = sum(per[i][2] for i in pick)
        g = Xy / np.maximum(XX, 1e-12)
        b = int(np.argmin(yy - g * Xy))
        gains.append(g[b])
        delays.append(cand[b, 0])
        tups.append(cand[b, 1])
        tdns.append(cand[b, 2])
    q = lambda a: [float(np.percentile(a, 5)), float(np.percentile(a, 95))]
    return dict(gain=q(gains), delay=q(delays), tau_up=q(tups), tau_dn=q(tdns))


def replay(flights, ch, params, gain):
    """Pooled RMS error of the model (params, gain) against the real telemetry over `flights`,
    and the RMS of the telemetry itself (the error of predicting 'no motion at all')."""
    cand = np.array([[params["delay"], params["tau_up"], params["tau_dn"], params["dead"]]])
    se = sy = n = 0.0
    for f in flights:
        X, y = design(f, ch, cand)
        se += float(((X[:, 0] * gain - y) ** 2).sum())
        sy += float((y ** 2).sum())
        n += len(y)
    return math.sqrt(se / n), math.sqrt(sy / n)


def heldout(flights, ch, cand):
    """Two-fold over flights: fit on one half, score the fit AND the sim's constants on the other."""
    fit_se = sim_se = zero_se = n = 0.0
    for k in (0, 1):
        train = [f for i, f in enumerate(flights) if i % 2 == k]
        test = [f for i, f in enumerate(flights) if i % 2 != k]
        fit = fit_channel(train, ch, cand)
        e_fit, e_zero = replay(test, ch, fit["params"], fit["gain"])
        e_sim, _ = replay(test, ch, SIM_PARAMS, SIM_GAIN[ch])
        m = sum(len(design(f, ch, cand[:1])[1]) for f in test)
        fit_se += e_fit ** 2 * m
        sim_se += e_sim ** 2 * m
        zero_se += e_zero ** 2 * m
        n += m
    return math.sqrt(fit_se / n), math.sqrt(sim_se / n), math.sqrt(zero_se / n)


# --- the detector, real vs simulated --------------------------------------------------------------
RANGE_BINS = [0.0, 0.4, 0.6, 0.9, 1.3, 2.0, 4.0]


def detector_stats(rows):
    """rows: dicts with range_m, cx, axis_ratio, clipped, fresh, found (consecutive ticks)."""
    out = []
    for lo, hi in zip(RANGE_BINS[:-1], RANGE_BINS[1:]):
        dcx, dar, clip, lost = [], [], [], []
        for i in range(1, len(rows)):
            a, b = rows[i - 1], rows[i]
            r = a.get("range_m")
            if not (a.get("fresh") and r is not None and lo <= r < hi):
                continue
            clip.append(bool(a.get("clipped")))
            lost.append(not b.get("fresh"))
            if b.get("fresh"):
                dcx.append(b["cx"] - a["cx"])
                dar.append(b["axis_ratio"] - a["axis_ratio"])
        n = len(clip)
        out.append(dict(bin=f"{lo:.1f}-{hi:.1f} m", n=n,
                        cx_jitter=float(np.std(dcx) / math.sqrt(2)) if len(dcx) > 5 else None,
                        ratio_jitter=float(np.std(dar) / math.sqrt(2)) if len(dar) > 5 else None,
                        clipped=float(np.mean(clip)) if n else None,
                        lost_next=float(np.mean(lost)) if n else None))
    return out


def real_detector_rows(flights):
    rows = []
    for f in flights:
        for k in f["ticks"]:
            rows.append(dict(range_m=k.get("range_m"), cx=k.get("cx", 0.0), axis_ratio=k.get("axis_ratio", 0.0),
                             clipped=k.get("clipped", False), fresh=k.get("fresh", False)))
        rows.append(dict(range_m=None, fresh=False))       # never pair across flights
    return rows


def sim_detector_rows(episodes=40, seed=0):
    """The scripted expert flying in the sim (dr=1, so the detector noise is on), every RingObs kept."""
    from onboard.expert import ExpertPolicy
    from onboard.gym_env import OnboardHoopEnv
    env = OnboardHoopEnv(seed=seed, dr=1.0, hard_start_p=0.5)
    pol = ExpertPolicy()
    rows = []
    for _ in range(episodes):
        obs, _ = env.reset()
        pol.reset()
        cap = []
        orig = env.sim.step

        def step(a, _orig=orig):
            ro = _orig(a)
            cap.append(ro)
            return ro
        env.sim.step = step
        done = False
        while not done:
            obs, _, term, trunc, _ = env.step(pol(obs))
            done = term or trunc
        env.sim.step = orig
        for ro in cap:
            rows.append(dict(range_m=None if not np.isfinite(ro.range_m) else float(ro.range_m), cx=ro.cx,
                             axis_ratio=ro.axis_ratio, clipped=ro.clipped, fresh=ro.fresh))
        rows.append(dict(range_m=None, fresh=False))
    return rows


# --- report ---------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flights", type=Path, default=FLIGHTS)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--no-sim", action="store_true", help="skip the simulated-detector comparison")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)

    flights = load_flights(a.flights)
    modes = {}
    for f in flights:
        modes[f["mode"]] = modes.get(f["mode"], 0) + 1
    print(f"[INFO] {len(flights)} flights, {sum(len(f['T']) for f in flights)} ticks, by mode {modes}")
    cand = _grid()
    res = {"flights": len(flights), "modes": modes, "channels": {}}
    for ch in CHANNELS:
        fit = fit_channel(flights, ch, cand)
        ci = bootstrap(fit, cand, a.boot)
        fit_err, sim_err, zero_err = heldout(flights, ch, cand)
        res["channels"][ch] = dict(gain=fit["gain"], gain_ci=ci["gain"], params=fit["params"],
                                   delay_ci=ci["delay"], tau_up_ci=ci["tau_up"], tau_dn_ci=ci["tau_dn"],
                                   rms=fit["rms"], n=fit["n"], sim_gain=SIM_GAIN[ch],
                                   replay_sim=sim_err, replay_fit=fit_err, replay_zero=zero_err)
        print(f"[FIT] {ch}: gain {fit['gain']:+.3f} {UNITS[ch]} (90% {ci['gain'][0]:+.3f}..{ci['gain'][1]:+.3f}), "
              f"sim {SIM_GAIN[ch]:+.3f}; params {fit['params']}; held-out RMS: sim {sim_err:.2f}, "
              f"fitted {fit_err:.2f}, 'no motion' {zero_err:.2f}")
    real_rows = real_detector_rows(flights)
    res["detector_real"] = detector_stats(real_rows)
    if not a.no_sim:
        res["detector_sim"] = detector_stats(sim_detector_rows())
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "sim_fidelity.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (a.out / "sim_fidelity.md").write_text(render_md(res), encoding="utf-8")
    print(f"[DONE] {a.out / 'sim_fidelity.md'}")
    return res


def _verdict(ch, c):
    sim, g = c["sim_gain"], c["gain"]
    if np.sign(sim) != np.sign(g):
        return "**SIGN FLIPPED**: the sim reports this the opposite way round from the real drone"
    ratio = g / sim
    band = DR_YAW if ch == "yaw" else DR_SPEED
    if abs(ratio - 1) <= band:
        return f"inside the sim's randomisation (x{ratio:.2f}, dr covers ±{band:.0%})"
    return f"**outside randomisation**: real is x{ratio:.2f} of the sim (dr covers ±{band:.0%})"


def render_md(res):
    L = ["# Simulator fidelity, measured from the real Tello logs", "",
         f"{res['flights']} real flights, dry runs excluded ({', '.join(f'{v} {k}' for k, v in res['modes'].items())}). "
         "Generated by `hardware/privileged/sim_fidelity.py`.", "",
         "## Stick → motion, per channel", "",
         "| Channel | Real gain (90% range) | Sim gain | Delay s | Lag up / down s | Dead zone rc | Verdict |",
         "|---|---|---|---|---|---|---|"]
    for ch, c in res["channels"].items():
        p = c["params"]
        L.append(f"| {ch} ({UNITS[ch]}) | {c['gain']:+.3f} ({c['gain_ci'][0]:+.3f}..{c['gain_ci'][1]:+.3f}) | "
                 f"{c['sim_gain']:+.3f} | {p['delay']:.2f} (sim {SIM_PARAMS['delay']:.2f}) | "
                 f"{p['tau_up']:.2f} / {p['tau_dn']:.2f} (sim {SIM_PARAMS['tau_up']:.2f} / {SIM_PARAMS['tau_dn']:.2f}) | "
                 f"{p['dead']:.0f} (sim {SIM_PARAMS['dead']:.1f}) | {_verdict(ch, c)} |")
    L += ["", "Gains are per rc unit (the policy sends at most 30). At full stick the real drone does "
          + ", ".join(f"{ch} {abs(c['gain']) * RC_MAX:.1f} {UNITS[ch].split(' per')[0]}" for ch, c in res["channels"].items())
          + "; the sim does "
          + ", ".join(f"{ch} {abs(c['sim_gain']) * RC_MAX:.1f}" for ch, c in res["channels"].items()) + ".", "",
          "## Replay: real sticks through the sim's model vs the fitted model (held-out flights)", "",
          "Two-fold: fitted on half the flights, scored on the other half. 'No motion' is the error of predicting "
          "the drone never moves; a useful model must beat it.", "",
          "| Channel | RMS error, sim constants | RMS error, fitted | RMS, 'no motion' | Unit |", "|---|---|---|---|---|"]
    unit = {"fb": "cm/s", "lr": "cm/s", "ud": "cm of height", "yaw": "deg of heading",
            "fb_pad": "cm/s (mission pads)"}
    for ch, c in res["channels"].items():
        L.append(f"| {ch} | {c['replay_sim']:.2f} | {c['replay_fit']:.2f} | {c['replay_zero']:.2f} | {unit[ch]} |")
    L += ["", "## Detector, real vs simulated (scripted expert, dr = 1)", "",
          "Per range bin, over ticks with a real detection: tick-to-tick jitter of the ring centre (cx, half-frames) and "
          "roundness (axis ratio), how often the ring overflows the frame (clipped), and how often the lock is lost on the next tick.", "",
          "| Range | n real / sim | cx jitter real / sim | ratio jitter real / sim | clipped real / sim | lost next tick real / sim |",
          "|---|---|---|---|---|---|"]
    sim = res.get("detector_sim") or [{} for _ in res["detector_real"]]
    f = lambda v: "–" if v is None else f"{v:.3f}"
    for r, s in zip(res["detector_real"], sim):
        L.append(f"| {r['bin']} | {r['n']} / {s.get('n', '–')} | {f(r['cx_jitter'])} / {f(s.get('cx_jitter'))} | "
                 f"{f(r['ratio_jitter'])} / {f(s.get('ratio_jitter'))} | {f(r['clipped'])} / {f(s.get('clipped'))} | "
                 f"{f(r['lost_next'])} / {f(s.get('lost_next'))} |")
    L += ["", "## Not answerable from logs", "",
          "- Whether the real yaw reading is clockwise-positive *in the cage frame* is what the yaw sign above measures "
          "against the stick; a 1-minute hand check (drone on the floor, turn it clockwise, watch `yaw`) confirms it independently.",
          "- Lateral speed (vgy) and the Tello's velocity estimate itself lag the truth; fitted delays include that lag.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
