// Environment tab: the training cage on its own (net, ring, drone on the floor), with a preview of the
// per-episode ring variation, an example take-off-and-fly-through animation and an optional overlay of
// the phone scan of the real cage. Geometry comes from /api/config (the same HoopSpec the env uses).
const EnvView = (() => {
  let view, spec, room;
  let ringC, ringN, ringR;
  let wind = null;                 // live sway preview: {dir, axis, gust, w, phase, t0}
  let swayDeg = 20;                // largest swing about the rope knot (the "Max swing" sliders)
  let flying = null;               // {t0, pts, seg, total, speed}
  let scanObj = null, scanLoading = false;
  let start = [-0.6, -0.5, 0.03];
  const $ = (id) => document.getElementById(id);

  const add = (a, b, k = 1) => a.map((x, i) => x + k * b[i]);
  const rnd = () => Math.random() * 2 - 1;

  function ringNormal(yawDeg, tiltDeg) {
    const y = yawDeg * Math.PI / 180, t = tiltDeg * Math.PI / 180;
    return [Math.cos(t) * Math.cos(y), Math.cos(t) * Math.sin(y), Math.sin(t)];
  }

  // Rodrigues rotation of v about a unit axis (same as hoop_net_env.rotate)
  function rotate(v, axis, ang) {
    const c = Math.cos(ang), s = Math.sin(ang), d = axis[0] * v[0] + axis[1] * v[1] + axis[2] * v[2];
    const cr = [axis[1] * v[2] - axis[2] * v[1], axis[2] * v[0] - axis[0] * v[2], axis[0] * v[1] - axis[1] * v[0]];
    return v.map((x, i) => x * c + cr[i] * s + axis[i] * d * (1 - c));
  }

  // In-plane unit vector of a ring with normal n closest to world up (hoop_net_env.ring_up).
  function ringUp(n) {
    const up = [-n[2] * n[0], -n[2] * n[1], 1 - n[2] * n[2]];
    const l = Math.hypot(...up);
    return up.map((x) => x / l);
  }

  const rOut = () => ringR + spec.tube_radius;
  // knot drift at full gust: scales with the swing so the top always moves least (HoopNetAviary.sway_top_amp)
  const topAmp = () => (spec.sway_tilt_deg ? spec.sway_top_amp * swayDeg / spec.sway_tilt_deg : 0);

  // A fresh wind for the sway preview, drawn like hoop_net_env._sample_ring (a strong gust, so it shows).
  function newWind() {
    const phi = Math.random() * 2 * Math.PI;
    const [p0, p1] = spec.sway_period_s;
    wind = { dir: [Math.cos(phi), Math.sin(phi), 0], axis: [-Math.sin(phi), Math.cos(phi), 0],
             gust: 0.6 + 0.4 * Math.random(), w: 2 * Math.PI / (p0 + Math.random() * (p1 - p0)),
             phase: Math.random() * 2 * Math.PI, t0: performance.now() };
  }

  // The ring's live pose in the wind (hoop_net_env._update_ring): it hangs from a rope tied to its
  // top, so the knot drifts a little downwind and the ring swings about it, bottom furthest.
  function swayRing() {
    if (!wind) { view.setHoop(ringC, ringN, ringR); return; }
    const t = (performance.now() - wind.t0) / 1000;
    const swing = wind.gust * Math.sin(wind.w * t + wind.phase);
    const top0 = add(ringC, ringUp(ringN), rOut());
    const knot = add(top0, wind.dir, swing * topAmp());
    const ang = -swing * swayDeg * Math.PI / 180;
    view.setHoop(add(knot, rotate(add(ringC, top0, -1), wind.axis, ang)),
                 rotate(ringN, wind.axis, ang), ringR);
  }

  // Both "Max swing" sliders (Environment preview + Training panel) hold the same value.
  function setSwayDeg(deg) {
    swayDeg = deg;
    const bottom = topAmp() + 4 * rOut() * Math.sin(deg * Math.PI / 360);   // chord of the 2*rOut arm
    document.querySelectorAll(".sway-deg-input").forEach((el) => { el.value = deg; });
    document.querySelectorAll(".sway-deg-val").forEach((el) => {
      el.textContent = `${deg}° · bottom ±${(bottom * 100).toFixed(0)} cm · ` +
                       `top ±${(topAmp() * 100).toFixed(0)} cm`;
    });
    if (ringC) setReadout();
  }

  function setRing(jitter) {
    const j = jitter ? 1 : 0;
    ringC = spec.centre.map((c, i) => c + j * rnd() * spec.centre_jitter[i]);
    ringN = ringNormal(spec.normal_yaw_deg + j * rnd() * spec.yaw_jitter_deg,
                       (spec.normal_tilt_deg || 0) + j * rnd() * spec.tilt_jitter_deg);
    ringR = spec.inner_radius * (1 + j * rnd() * spec.radius_jitter_frac);
    if (wind) newWind();
    swayRing();
    setReadout();
    placeDrone();
  }

  function setReadout() {
    const f = (x) => x.toFixed(2);
    const yaw = Math.atan2(ringN[1], ringN[0]) * 180 / Math.PI, tilt = Math.asin(ringN[2]) * 180 / Math.PI;
    $("env-readout").textContent =
      `net            ${room.join(" x ")} m (floor, 4 walls, roof)\n` +
      `ring centre    ${ringC.map(f).join(", ")}\n` +
      `ring facing    ${ringN.map(f).join(", ")} (turned ${yaw.toFixed(0)} deg, leaning ${tilt.toFixed(0)} deg)\n` +
      `ring opening   ${(2 * ringR * 100).toFixed(0)} cm (drone width ${(2 * spec.drone_radius * 100).toFixed(0)} cm)\n` +
      `wind           swings up to ${swayDeg} deg about the rope knot at its top ` +
      `(knot ${(topAmp() * 100).toFixed(0)} cm), shoved by rim hits\n` +
      `start          on the floor, anywhere inside the net\n` +
      `success        through, along the arrow`;
  }

  function startPoint() {
    const half = [room[0] / 2 - 0.25, room[1] / 2 - 0.25];
    for (let i = 0; i < 100; i++) {
      const p = [rnd() * half[0], rnd() * half[1], 0.03];
      if (Math.hypot(p[0] - ringC[0], p[1] - ringC[1]) > 0.35) return p;
    }
    return [-0.6, -0.5, 0.03];
  }

  function placeDrone() {
    flying = null;
    start = startPoint();
    view.setDroneTarget(start, [0, 0, 0], true);
  }

  function play() {
    const approach = add(ringC, ringN, -spec.approach_dist);
    const exit = add(ringC, ringN, 0.6);
    const pts = [start, [start[0], start[1], 0.6], approach, ringC, exit];
    const seg = [];
    let total = 0;
    for (let i = 1; i < pts.length; i++) {
      const d = Math.hypot(...pts[i].map((x, k) => x - pts[i - 1][k]));
      seg.push(d);
      total += d;
    }
    // the real speed limit is ~0.25 m/s; play 4x faster so it is watchable
    flying = { t0: performance.now(), pts, seg, total, speed: 1.0 };
  }

  function step() {
    if (wind) swayRing();
    if (flying) {
      let d = (performance.now() - flying.t0) / 1000 * flying.speed;
      if (d >= flying.total) {
        view.setDroneTarget(flying.pts[flying.pts.length - 1], [0, 0, 0]);
        flying = null;
      } else {
        let i = 0;
        while (i < flying.seg.length - 1 && d > flying.seg[i]) { d -= flying.seg[i]; i++; }
        const k = d / (flying.seg[i] || 1);
        view.setDroneTarget(flying.pts[i].map((x, n) => x + (flying.pts[i + 1][n] - x) * k), [0, 0, 0]);
      }
    }
    requestAnimationFrame(step);
  }

  // ------------------------------------------------------------- phone scan
  // The scan is ~1.25 m across, turned ~30 deg and off-centre. fitScan() finds its footprint
  // (a robust minimum-area rectangle over the 2-98 percentile of the vertices, so stray floor
  // points don't count) and its floor/top, then scales it uniformly so the footprint matches the
  // sim net, squares it to the net's axes and puts its floor at y = 0. The sliders fine-tune on top.
  let scanFit = null;          // {yaw, scale, cx, cz, floor, size:[w, h, d]}
  let fitGroup = null;         // fitGroup (fit + fine-tune) > scanObj (offset to its own centre)

  function pct(sorted, q) { return sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor(q * (sorted.length - 1))))]; }

  function fitScan() {
    const pts = [];
    scanObj.updateMatrixWorld(true);
    const v = new THREE.Vector3();
    scanObj.traverse((o) => {
      if (!o.isMesh) return;
      const pos = o.geometry.attributes.position;
      const step = Math.max(1, Math.floor(pos.count / 15000));
      for (let i = 0; i < pos.count; i += step) {
        v.fromBufferAttribute(pos, i).applyMatrix4(o.matrixWorld);
        pts.push([v.x, v.y, v.z]);
      }
    });
    const ys = pts.map((q) => q[1]).sort((x, y) => x - y);
    const floor = pct(ys, 0.02), top = pct(ys, 0.98);
    let best = null;
    for (let deg = 0; deg < 90; deg += 0.5) {
      const r = deg * Math.PI / 180, c = Math.cos(r), sn = Math.sin(r);
      const us = pts.map((q) => c * q[0] + sn * q[2]).sort((x, y) => x - y);
      const ws = pts.map((q) => -sn * q[0] + c * q[2]).sort((x, y) => x - y);
      const lu = pct(us, 0.02), hu = pct(us, 0.98), lw = pct(ws, 0.02), hw = pct(ws, 0.98);
      const area = (hu - lu) * (hw - lw);
      if (!best || area < best.area) best = { area, r, c, sn, w: hu - lu, d: hw - lw, uc: (lu + hu) / 2, wc: (lw + hw) / 2 };
    }
    // footprint centre back in scan coordinates (inverse of the rotation used above)
    const cx = best.c * best.uc - best.sn * best.wc;
    const cz = best.sn * best.uc + best.c * best.wc;
    const scale = ((room[0] + room[1]) / 2) / ((best.w + best.d) / 2);
    scanFit = { yaw: best.r, scale, cx, cz, floor, size: [best.w, top - floor, best.d] };

    fitGroup = new THREE.Group();
    view.scene.add(fitGroup);
    fitGroup.add(scanObj);
    scanObj.position.set(-cx, -floor, -cz);
    const s = scanFit.size;
    $("env-scan-status").textContent =
      `fitted: scan ${s[0].toFixed(2)} x ${s[2].toFixed(2)} m, ${s[1].toFixed(2)} m tall ` +
      `-> x${scale.toFixed(2)} (= ${(s[0] * scale).toFixed(2)} x ${(s[2] * scale).toFixed(2)} x ` +
      `${(s[1] * scale).toFixed(2)} m), turned ${(best.r * 180 / Math.PI).toFixed(1)} deg`;
  }

  function resetFineTune() {
    $("env-scan-scale").value = 1; $("env-scan-yaw").value = 0;
    $("env-scan-x").value = 0; $("env-scan-y").value = 0;
    setScanTransform();
  }

  function setScanTransform() {
    const k = parseFloat($("env-scan-scale").value), yaw = parseFloat($("env-scan-yaw").value);
    const dx = parseFloat($("env-scan-x").value), dy = parseFloat($("env-scan-y").value);
    $("env-scan-scale-val").textContent = k.toFixed(2);
    $("env-scan-yaw-val").textContent = yaw;
    $("env-scan-x-val").textContent = dx.toFixed(2);
    $("env-scan-y-val").textContent = dy.toFixed(2);
    if (!fitGroup) return;
    fitGroup.scale.setScalar(scanFit.scale * k);
    // three.js rotation.y by +r maps (x, z) -> (x cos r + z sin r, -x sin r + z cos r): the same
    // rotation that made the footprint rectangle axis-aligned in fitScan()
    fitGroup.rotation.y = scanFit.yaw + yaw * Math.PI / 180;
    fitGroup.position.set(dx, 0, dy);         // sim y is three z
  }

  function toggleScan(on) {
    const status = $("env-scan-status");
    if (scanObj) { fitGroup.visible = on; return; }
    if (!on || scanLoading) return;
    scanLoading = true;
    status.textContent = "loading scan (34 MB)...";
    new THREE.GLTFLoader().load("/assets/3D_net%26hoop.glb", (gltf) => {
      scanObj = gltf.scene;
      scanObj.traverse((o) => {
        if (o.isMesh) {
          [].concat(o.material).forEach((m) => {
            m.side = THREE.DoubleSide; m.transparent = true; m.opacity = 0.6; m.depthWrite = false;
          });
        }
      });
      try {
        fitScan();
      } catch (e) {
        status.textContent = "could not fit the scan: " + (e.message || e);
        fitGroup = new THREE.Group();
        view.scene.add(fitGroup);
        fitGroup.add(scanObj);
        scanFit = { yaw: 0, scale: 1 };
      }
      fitGroup.visible = $("env-scan-chk").checked;
      setScanTransform();
      scanLoading = false;
    }, (e) => {
      if (e && e.total) status.textContent = `loading scan... ${Math.round(100 * e.loaded / e.total)}%`;
    }, (e) => {
      status.textContent = "could not load scan: " + (e.message || e);
      scanLoading = false;
    });
  }

  // Same numbers as scripts/hoop_net_env.HoopSpec, used only if /api/config has no "hoop".
  const DEFAULT_SPEC = {
    centre: [0.04, 0.07, 1.29], inner_radius: 0.15, tube_radius: 0.03, normal_yaw_deg: -29,
    normal_tilt_deg: -18, centre_jitter: [0.1, 0.1, 0.08], yaw_jitter_deg: 20, tilt_jitter_deg: 8,
    radius_jitter_frac: 0.1, drone_radius: 0.07, approach_dist: 0.35,
    sway_tilt_deg: 20, sway_top_amp: 0.02, sway_period_s: [1.5, 3.0],
  };

  async function init() {
    try {
      await setup();
    } catch (e) {
      // never a silent blank view: say what broke (most often a stale cached script)
      console.error(e);
      $("env-readout").textContent =
        "Could not draw the environment: " + (e && e.message ? e.message : e) +
        "\nTry a hard refresh (Ctrl+F5); if it persists, restart the dashboard server.";
    }
  }

  async function setup() {
    const cfg = await Api.getConfig();
    room = cfg.room || [2, 2, 2];
    spec = Object.assign({}, DEFAULT_SPEC, cfg.hoop || {});   // an older backend lacks the wind fields
    view = createScene3D();
    view.init($("env-canvas"), room);
    swayDeg = spec.sway_tilt_deg;
    setRing(false);
    setSwayDeg(swayDeg);
    document.querySelectorAll(".sway-deg-input").forEach((el) =>
      el.addEventListener("input", (e) => setSwayDeg(parseFloat(e.target.value))));
    $("env-random-btn").addEventListener("click", () => setRing(true));
    $("env-nominal-btn").addEventListener("click", () => setRing(false));
    $("env-fly-btn").addEventListener("click", () => { placeDrone(); setTimeout(play, 300); });
    $("env-wind-chk").addEventListener("change", (e) => {
      if (e.target.checked) newWind(); else wind = null;
      swayRing();
    });
    $("env-scan-chk").addEventListener("change", (e) => toggleScan(e.target.checked));
    ["env-scan-scale", "env-scan-yaw", "env-scan-x", "env-scan-y"]
      .forEach((id) => $(id).addEventListener("input", setScanTransform));
    $("env-scan-fit-btn").addEventListener("click", resetFineTune);
    Tabs.onChange((n) => { if (n === "env") view.resize(); });
    requestAnimationFrame(step);
  }

  return { init };
})();
