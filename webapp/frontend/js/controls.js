// Draft start/target positions, moved by on-screen arrow buttons or the
// keyboard. Purely client-side until "Start training" is pressed: nothing
// here reaches the backend, it only mutates draftStart/draftTarget, clamped
// to the room bounds from GET /api/config. This is pre-run configuration,
// not a mid-flight teleport, matching hoop_env.py only reading a position
// override at __init__/reset time.
const Controls = (() => {
  let bounds = { room: [2, 2, 2], xyMargin: 0.25, zRange: [0.3, 1.6], startZ: 0.1 };
  let selectedEntity = "drone"; // "drone" | "hoop" | "both"
  let draftStart = [0, 0, 0.1];
  let draftTarget = [0.5, 0.5, 0.6];
  const STEP = 0.08;
  let onChangeCb = null;
  let panelActive = true;
  let envMode = "net";        // net: the drone always starts on the floor; the hoop is the ring centre

  function halfXY() {
    return [bounds.room[0] / 2 - bounds.xyMargin, bounds.room[1] / 2 - bounds.xyMargin];
  }

  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  function clampStart(p) {
    const [hx, hy] = halfXY();
    const z = envMode === "net" ? bounds.startZ : clamp(p[2], 0.05, bounds.room[2] - 0.05);
    return [clamp(p[0], -hx, hx), clamp(p[1], -hy, hy), z];
  }

  let legacyZRange = [0.3, 1.6], netZRange = [1.0, 1.2];
  function setEnvMode(mode) {
    envMode = mode;
    bounds.zRange = mode === "net" ? netZRange : legacyZRange;
    resetCenter();
  }

  function clampTarget(p) {
    const [hx, hy] = halfXY();
    return [clamp(p[0], -hx, hx), clamp(p[1], -hy, hy),
            clamp(p[2], bounds.zRange[0], bounds.zRange[1])];
  }

  function move(dx, dy, dz) {
    if (selectedEntity === "drone" || selectedEntity === "both") {
      draftStart = clampStart([draftStart[0] + dx, draftStart[1] + dy, draftStart[2] + dz]);
    }
    if (selectedEntity === "hoop" || selectedEntity === "both") {
      draftTarget = clampTarget([draftTarget[0] + dx, draftTarget[1] + dy, draftTarget[2] + dz]);
    }
    emitChange();
  }

  function resetCenter() {
    if (envMode === "net") {
      draftStart = [-0.5, -0.4, bounds.startZ];              // on the floor, clear of the ring
      draftTarget = bounds.hoopCentre ? bounds.hoopCentre.slice() : [0, 0, 1.2];
    } else {
      draftStart = [0, 0, 0.1];
      draftTarget = [0.5, 0.5, 0.95];
    }
    emitChange();
  }

  function emitChange() {
    if (onChangeCb) onChangeCb({ start: draftStart.slice(), target: draftTarget.slice(), entity: selectedEntity });
  }

  function setBounds(cfg) {
    bounds = { room: cfg.room, xyMargin: cfg.xy_margin, zRange: cfg.z_range, startZ: cfg.start_z,
               hoopCentre: cfg.hoop ? cfg.hoop.centre : null };
    legacyZRange = [0.3, 1.6];
    netZRange = cfg.z_range;
    resetCenter();
  }

  // positionActive: the start-position panel (setup only). trainingActive: the Training panel,
  // usable whenever no run is training, even while a finished run is in view.
  function setPanelActive(positionActive, trainingActive = positionActive) {
    panelActive = positionActive;
    document.querySelectorAll("#panel-position input, #panel-position button")
      .forEach((el) => { el.disabled = !positionActive; });
    document.querySelectorAll("#panel-training input, #panel-training select, #panel-training button")
      .forEach((el) => { el.disabled = !trainingActive; });
    const seed = document.getElementById("seed-input");
    const rnd = document.getElementById("randomize-toggle");
    if (seed && rnd && positionActive) seed.disabled = !rnd.checked;
  }

  function wireDom() {
    document.querySelectorAll('input[name="entity"]').forEach((el) => {
      el.addEventListener("change", (e) => { selectedEntity = e.target.value; });
    });
    document.getElementById("arrow-up").addEventListener("click", () => move(0, STEP, 0));
    document.getElementById("arrow-down").addEventListener("click", () => move(0, -STEP, 0));
    document.getElementById("arrow-left").addEventListener("click", () => move(-STEP, 0, 0));
    document.getElementById("arrow-right").addEventListener("click", () => move(STEP, 0, 0));
    document.getElementById("arrow-z-up").addEventListener("click", () => move(0, 0, STEP));
    document.getElementById("arrow-z-down").addEventListener("click", () => move(0, 0, -STEP));
    document.getElementById("reset-center-btn").addEventListener("click", resetCenter);

    document.addEventListener("keydown", (e) => {
      if (!panelActive) return;
      if (typeof Tabs !== "undefined" && Tabs.current() !== "sim") return;   // arrows fly the drone there
      const tag = (document.activeElement && document.activeElement.tagName) || "";
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
      switch (e.key) {
        case "ArrowUp": move(0, STEP, 0); e.preventDefault(); break;
        case "ArrowDown": move(0, -STEP, 0); e.preventDefault(); break;
        case "ArrowLeft": move(-STEP, 0, 0); e.preventDefault(); break;
        case "ArrowRight": move(STEP, 0, 0); e.preventDefault(); break;
        case "PageUp": move(0, 0, STEP); e.preventDefault(); break;
        case "PageDown": move(0, 0, -STEP); e.preventDefault(); break;
      }
    });
  }

  function init(cfg) {
    setBounds(cfg);
    wireDom();
  }

  function getDraft() {
    return { start: draftStart.slice(), target: draftTarget.slice() };
  }

  function onChange(cb) { onChangeCb = cb; }

  return { init, getDraft, onChange, setPanelActive, setEnvMode };
})();
