// Three.js scene: the 2x2x2 m net, the ring (hoop) and the drone. The simulator
// is Z-up with the room's XY centred at the origin and the floor at z=0;
// Three.js is Y-up by convention, so every position goes through one
// conversion helper rather than juggling the axis swap in several places.
function simToThree(pos) {
  return new THREE.Vector3(pos[0], pos[2], pos[1]);
}

const PALETTE = {
  bg: 0x171721,
  room: 0x4a4a6a,
  floor: 0x262638,
  net: 0x6a6a8c,
  hoopRim: 0x2f6fff,          // the real ring is a blue pool noodle
  hoopBoard: 0xeceef5,
  hoopPole: 0x34344a,
  droneBody: 0x872b97,
  droneArm: 0x1f1f2e,
  droneRotor: 0xff3c68,
  targetGhost: 0xff3c68,
};

// A factory, so the Simulator tab and the Environment tab each get their own scene.
function createScene3D() {
  let renderer, scene, camera, controls;
  let room = [2, 2, 2];
  let hoopGroup, droneGroup, netGroup;
  let ringRadius = 0.15;       // inner (clear) radius of the opening, m
  const RING_TUBE = 0.03;
  let dronePosCurrent = new THREE.Vector3(0, 0.1, 0);
  let dronePosTarget = new THREE.Vector3(0, 0.1, 0);
  let droneRotCurrent = new THREE.Euler(0, 0, 0);
  let droneRotTarget = new THREE.Euler(0, 0, 0);

  function buildDrone() {
    return DroneModel.build({ accent: PALETTE.droneBody });     // js/dronemodel.js
  }

  function buildHoop() {
    // A ring whose clear opening is `ringRadius`, with a 3 cm tube, like the real one.
    const g = new THREE.Group();
    const rim = new THREE.Mesh(
      new THREE.TorusGeometry(ringRadius + RING_TUBE, RING_TUBE, 12, 48),
      new THREE.MeshStandardMaterial({ color: PALETTE.hoopRim }),
    );
    g.add(rim);          // torus axis = local +Z, which setHoop() turns onto the ring normal
    const arrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, 0),
      0.3, 0xbcd6ff, 0.07, 0.04);       // shows the direction the drone must pass in
    g.add(arrow);
    return g;
  }

  function buildNet() {
    // Four walls + roof drawn as translucent panels with a mesh grid (the physical net).
    const g = new THREE.Group();
    const [rx, ry, rz] = room;
    const mat = new THREE.MeshBasicMaterial({ color: PALETTE.net, transparent: true, opacity: 0.10,
                                              side: THREE.DoubleSide, depthWrite: false });
    const lineMat = new THREE.LineBasicMaterial({ color: PALETTE.net, transparent: true, opacity: 0.35 });
    const panel = (w, h, pos, rot) => {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), mat);
      m.position.set(...pos); m.rotation.set(...rot);
      g.add(m);
      const pts = [];
      const nx = Math.round(w / 0.25), ny = Math.round(h / 0.25);
      for (let i = 0; i <= nx; i++) pts.push(-w / 2 + (i / nx) * w, -h / 2, 0, -w / 2 + (i / nx) * w, h / 2, 0);
      for (let j = 0; j <= ny; j++) pts.push(-w / 2, -h / 2 + (j / ny) * h, 0, w / 2, -h / 2 + (j / ny) * h, 0);
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
      const l = new THREE.LineSegments(geo, lineMat);
      l.position.set(...pos); l.rotation.set(...rot);
      g.add(l);
    };
    panel(rx, rz, [0, rz / 2, -ry / 2], [0, 0, 0]);
    panel(rx, rz, [0, rz / 2, ry / 2], [0, 0, 0]);
    panel(ry, rz, [-rx / 2, rz / 2, 0], [0, Math.PI / 2, 0]);
    panel(ry, rz, [rx / 2, rz / 2, 0], [0, Math.PI / 2, 0]);
    panel(rx, ry, [0, rz, 0], [Math.PI / 2, 0, 0]);
    return g;
  }

  function buildRoom() {
    const g = new THREE.Group();
    const geo = new THREE.BoxGeometry(room[0], room[2], room[1]);
    const edges = new THREE.EdgesGeometry(geo);
    const lines = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: PALETTE.room }));
    lines.position.set(0, room[2] / 2, 0);
    g.add(lines);

    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(room[0], room[1]),
      new THREE.MeshStandardMaterial({ color: PALETTE.floor, side: THREE.DoubleSide }),
    );
    floor.rotation.x = -Math.PI / 2;
    g.add(floor);

    return g;
  }

  function init(canvas, initialRoom) {
    if (initialRoom) room = initialRoom;
    // Some GPUs refuse only the antialias attribute, so try once without it.
    try {
      renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    } catch (e) {
      try {
        renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: "default" });
      } catch (e2) {
        throw new Error("WebGL unavailable: " + e2.message);
      }
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));

    scene = new THREE.Scene();
    scene.background = new THREE.Color(PALETTE.bg);

    camera = new THREE.PerspectiveCamera(55, 1, 0.05, 100);
    camera.position.set(3, 2.2, 3);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.target.set(0, room[2] / 2, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(2, 4, 3);
    scene.add(dirLight);

    scene.add(buildRoom());
    netGroup = buildNet();
    scene.add(netGroup);

    hoopGroup = buildHoop();
    scene.add(hoopGroup);
    setHoop([0, 0, 1.2]);

    droneGroup = buildDrone();
    scene.add(droneGroup);
    setDroneTarget([0, 0, 0.1], [0, 0, 0], true);

    resize();
    window.addEventListener("resize", resize);
    requestAnimationFrame(tick);
  }

  let lastRender = 0;
  let viewShiftPx = 0;          // shift the picture left so overlays on the right don't hide the cage

  function resize() {
    const wrap = renderer.domElement.parentElement;
    const w = wrap.clientWidth, h = wrap.clientHeight;
    if (!w || !h) return;        // tab hidden: keep the last good size
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    if (viewShiftPx) camera.setViewOffset(w, h, viewShiftPx, 0, w, h);
    else camera.clearViewOffset();
    camera.updateProjectionMatrix();
  }

  function setViewShift(px) {
    viewShiftPx = px;
    resize();
  }

  function setTrailVisible(key, visible) {
    const t = trails[key];
    if (!t) return;
    t.line.visible = visible;
    t.ghosts.forEach((g) => { g.visible = visible; });
    t.marks.forEach((m) => { m.visible = visible; });
  }

  // pos: ring centre (sim coords). normal: unit direction the drone must pass in (sim coords,
  // default +x). radius: clear opening radius in m (rebuilds the torus when it changes).
  function setHoop(pos, normal, radius) {
    if (radius && Math.abs(radius - ringRadius) > 1e-4) {
      ringRadius = radius;
      const old = hoopGroup;
      hoopGroup = buildHoop();
      scene.remove(old);
      scene.add(hoopGroup);
      old.traverse((o) => { if (o.geometry) o.geometry.dispose(); });
    }
    hoopGroup.position.copy(simToThree(pos));
    const n = simToThree(normal || [1, 0, 0]).normalize();
    hoopGroup.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), n);
  }

  function setDroneTarget(pos, rpy, snap = false) {
    dronePosTarget = simToThree(pos);
    // simToThree swaps y and z (a mirror), which flips every rotation's sense: sim
    // R = Rz(yaw) Ry(pitch) Rx(roll) becomes three Ry(-yaw) Rz(-pitch) Rx(-roll), Euler order YZX.
    // (The old symmetric drone hid this; the camera nose now shows the heading.)
    droneRotTarget = new THREE.Euler(-rpy[0], -rpy[2], -rpy[1], "YZX");
    if (snap) {
      dronePosCurrent.copy(dronePosTarget);
      droneRotCurrent.copy(droneRotTarget);
    }
  }

  // ---------------------------------------------------------- route trails
  // One live trail per source ("train" = training env 0, "showcase" = current-policy episode);
  // finished episodes stay as faded ghosts (the last GHOSTS of them).
  const TRAIL_MAX = 2000, GHOSTS = 3;
  const trails = {};
  const TRAIL_COLORS = { train: 0x3cc6ff, showcase: 0x5cf08a };

  function getTrail(key) {
    if (trails[key]) return trails[key];
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(TRAIL_MAX * 3), 3));
    geo.setDrawRange(0, 0);
    const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: TRAIL_COLORS[key] || 0xffffff }));
    line.frustumCulled = false;
    scene.add(line);
    trails[key] = { line, n: 0, ghosts: [], marks: [] };
    return trails[key];
  }

  function addTrailPoint(pos, key = "train") {
    const t = getTrail(key);
    if (t.n >= TRAIL_MAX) return;
    const v = simToThree(pos);
    const arr = t.line.geometry.attributes.position.array;
    arr[t.n * 3] = v.x; arr[t.n * 3 + 1] = v.y; arr[t.n * 3 + 2] = v.z;
    t.n += 1;
    t.line.geometry.attributes.position.needsUpdate = true;
    t.line.geometry.setDrawRange(0, t.n);
  }

  // End the current episode's trail: keep it as a faded ghost, start a fresh one.
  function newEpisode(key = "train", keepGhost = true) {
    const t = getTrail(key);
    if (keepGhost && t.n > 1) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(
        t.line.geometry.attributes.position.array.slice(0, t.n * 3), 3));
      const ghost = new THREE.Line(geo, new THREE.LineBasicMaterial({
        color: TRAIL_COLORS[key] || 0xffffff, transparent: true, opacity: 0.35 }));
      scene.add(ghost);
      t.ghosts.push(ghost);
      while (t.ghosts.length > GHOSTS) {
        const old = t.ghosts.shift();
        scene.remove(old); old.geometry.dispose();
      }
      t.ghosts.forEach((g, i) => { g.material.opacity = 0.12 + 0.25 * (i + 1) / t.ghosts.length; });
    }
    t.marks.forEach((m) => scene.remove(m));
    t.marks = [];
    t.n = 0;
    t.line.geometry.setDrawRange(0, 0);
  }

  function clearTrail(key) {
    const keys = key ? [key] : Object.keys(trails);
    keys.forEach((k) => {
      const t = trails[k];
      if (!t) return;
      newEpisode(k, false);
      t.ghosts.forEach((g) => { scene.remove(g); g.geometry.dispose(); });
      t.ghosts = [];
    });
  }

  const MARK_COLORS = { net: 0xff3c68, rim: 0xffa040, pass: 0x5cf08a };
  function markEvent(pos, kind, key = "train") {
    const t = getTrail(key);
    const m = new THREE.Mesh(new THREE.SphereGeometry(kind === "pass" ? 0.05 : 0.025, 10, 8),
      new THREE.MeshBasicMaterial({ color: MARK_COLORS[kind] || 0xffffff }));
    m.position.copy(simToThree(pos));
    scene.add(m);
    t.marks.push(m);
  }

  function setDroneColor(hex) {
    if (droneGroup && droneGroup.userData.bodyMat) droneGroup.userData.bodyMat.color.setHex(hex);
  }

  function tick() {
    // Don't draw a scene nobody can see (its tab is hidden, or the window is in the background):
    // two scenes at 60 fps would otherwise compete with training for the CPU/GPU.
    // 30 fps is plenty for telemetry that arrives at ~15 Hz, and halves the drawing cost.
    const now = performance.now();
    if (document.hidden || renderer.domElement.offsetParent === null || now - lastRender < 33) {
      requestAnimationFrame(tick);
      return;
    }
    lastRender = now;
    dronePosCurrent.lerp(dronePosTarget, 0.25);
    droneRotCurrent.x += (droneRotTarget.x - droneRotCurrent.x) * 0.25;
    const dYaw = Math.atan2(Math.sin(droneRotTarget.y - droneRotCurrent.y), Math.cos(droneRotTarget.y - droneRotCurrent.y));
    droneRotCurrent.y += dYaw * 0.25;      // the short way round, not a full spin at +-180 deg
    droneRotCurrent.z += (droneRotTarget.z - droneRotCurrent.z) * 0.25;
    droneGroup.position.copy(dronePosCurrent);
    droneGroup.rotation.copy(droneRotCurrent);
    DroneModel.spin(droneGroup, 0.033);

    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(tick);
  }

  return {
    init, setHoop, setDroneTarget, resize,
    addTrailPoint, newEpisode, clearTrail, markEvent, setDroneColor, setViewShift, setTrailVisible,
    droneColors: { train: PALETTE.droneBody, showcase: 0x2fbf6a },
    get scene() { return scene; },
    setNetVisible: (v) => { netGroup.visible = v; },
  };
}

// If three.js failed to load, every method is a no-op and init says why, so the rest of
// the dashboard (tabs, run list) still works.
const Scene3D = typeof THREE !== "undefined" ? createScene3D() : new Proxy({ droneColors: {} }, {
  get: (t, k) => (k in t ? t[k]
    : k === "init" ? () => { throw new Error("three.js did not load"); } : () => {}),
});
