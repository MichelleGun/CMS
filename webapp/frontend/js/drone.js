// The "Real drone" panel: battery, height, ring lock and the drone's own camera, plus
// the two commands a browser is allowed to send.
//
// Same revealing-module shape as Controls and RewardChart. It owns its own polling and
// its own socket, so app.js only has to call Drone.init() once.
//
// There is no takeoff button, and that is a deliberate safety decision rather than an
// omission: the backend refuses to start a real flight at all (see drone_manager.py).
// Every button here either reads state or makes the drone safer, so a page left open in
// a forgotten tab cannot spin propellers. Real flights are started from a terminal by
// someone standing next to the cage, and this panel attaches to whichever flight is
// newest.
const Drone = (() => {
  const STATUS_POLL_MS = 1500;
  const CAMERA_POLL_MS = 700;   // matches the training camera's own refresh

  let flightId = null;
  let active = false;
  let ws = null;
  let statusTimer = null;
  let cameraTimer = null;
  let lastStatus = null;
  let el = {};

  function fmt(n, d = 2) {
    return (n === null || n === undefined || n === "") ? "-" : Number(n).toFixed(d);
  }

  // ------------------------------------------------------------------ readout

  function setPill(text, cls) {
    el.pill.textContent = text;
    el.pill.className = "pill " + cls;
  }

  function phaseClass(phase, status) {
    if (status === "killed" || status === "crashed" || phase === "abort") return "pill-crashed";
    if (status === "refused" || status === "interrupted") return "pill-stopped";
    if (active) return "pill-training";
    return "pill-idle";
  }

  function renderReadout(s) {
    if (!s) {
      el.readout.textContent = "no flight recorded yet";
      return;
    }
    const lock = s.ring_found
      ? (s.ring_clipped ? "locked (clipped)" : "locked")
      : "no ring";
    const rng = s.range_m === null || s.range_m === undefined ? "-" : fmt(s.range_m, 2) + " m";
    const travel = s.max_travel_m
      ? `${fmt(s.travel_m, 2)} / ${fmt(s.max_travel_m, 2)} m`
      : `${fmt(s.travel_m, 2)} m`;
    // Pre-formatted block, like #position-readout. .readout already uses tabular
    // numerals, so these columns do not jitter as the values change.
    el.readout.style.whiteSpace = "pre";
    el.readout.textContent =
      `battery ${s.battery === undefined ? "-" : s.battery} %   temp ${s.temp_c === undefined ? "-" : s.temp_c} C\n` +
      `height  ${fmt(s.height_cm, 0)} cm   airborne ${fmt(s.airborne_s, 1)} s\n` +
      `ring    ${lock}   range ${rng}\n` +
      `offset  x ${fmt(s.cx, 2)}  y ${fmt(s.cy, 2)}  round ${fmt(s.axis_ratio, 2)}\n` +
      `travel  ${travel}\n` +
      `rc      ${(s.action || []).map((v) => fmt(v, 2)).join("  ")}`;
  }

  function applyStatus(payload) {
    // /api/drone/status returns the whole flight record; the socket sends just the
    // status object. Accept either.
    const s = payload && payload.status !== undefined && typeof payload.status === "object"
      ? payload.status
      : payload;
    if (payload && payload.flight_id) flightId = payload.flight_id;
    if (payload && payload.active !== undefined) active = !!payload.active;
    lastStatus = s || lastStatus;
    renderReadout(lastStatus);

    const meta = payload && payload.meta ? payload.meta : {};
    const finalStatus = meta.final_status;
    const phase = (lastStatus && lastStatus.phase) || "-";
    const dry = lastStatus && lastStatus.dry_run;
    if (!flightId) {
      setPill("no flight", "pill-idle");
    } else if (active) {
      setPill(`${phase}${dry ? " (dry)" : ""}`, phaseClass(phase, finalStatus));
    } else {
      setPill(finalStatus || "finished", phaseClass(phase, finalStatus));
    }
    el.flightIdEl.textContent = flightId || "";

    // Land and Emergency are only meaningful while something is flying. They stay
    // enabled for a dry run too, so the flag path gets exercised before it matters.
    el.landBtn.disabled = !flightId || !active;
    el.killBtn.disabled = !flightId || !active;
    el.dryRunBtn.disabled = active;
    el.stopBtn.disabled = !active;
    if (payload && payload.camera_enabled !== undefined
        && el.cameraToggle.checked !== payload.camera_enabled) {
      el.cameraToggle.checked = payload.camera_enabled;
      syncCamera();
    }
  }

  // ------------------------------------------------------------------ camera

  function syncCamera() {
    const on = el.cameraToggle.checked && !!flightId;
    el.cameraFrame.hidden = !on;
    el.cameraPlaceholder.hidden = on;
    if (cameraTimer) { clearInterval(cameraTimer); cameraTimer = null; }
    if (!on) return;
    const refresh = () => { el.cameraFrame.src = Api.droneCameraUrl(flightId); };
    refresh();
    cameraTimer = setInterval(refresh, CAMERA_POLL_MS);
  }

  async function onCameraToggle() {
    if (!flightId) { el.cameraToggle.checked = false; return; }
    try {
      await Api.setDroneCamera(flightId, el.cameraToggle.checked);
    } catch (e) {
      el.note.textContent = "camera toggle failed: " + e.message;
    }
    syncCamera();
  }

  // ------------------------------------------------------------------ socket

  function openSocket(id) {
    closeSocket();
    ws = Api.openDroneSocket(id, {
      onStatus: (msg) => applyStatus(msg),
      onEnded: (msg) => {
        active = false;
        setPill(msg.status || "finished", phaseClass("", msg.status));
        refresh().catch(() => {});
      },
      onEvent: (msg) => {
        if (msg.event) el.note.textContent = `${msg.event}${msg.reason ? ": " + msg.reason : ""}`;
      },
    });
  }

  function closeSocket() {
    if (ws) { ws.close(); ws = null; }
  }

  // ------------------------------------------------------------------ polling

  async function refresh() {
    const payload = await Api.getDroneStatus();
    const previousId = flightId;
    applyStatus(payload);
    if (flightId && flightId !== previousId) {
      openSocket(flightId);
      if (el.cameraToggle.checked) syncCamera();
    }
    if (!flightId) closeSocket();
    return payload;
  }

  // ------------------------------------------------------------------ actions

  async function land() {
    if (!flightId) return;
    try {
      await Api.landDrone(flightId);
      el.note.textContent = "landing requested";
    } catch (e) {
      el.note.textContent = "land failed: " + e.message;
    }
  }

  async function kill() {
    if (!flightId) return;
    // Cutting the motors makes the drone fall. Worth one confirmation, but it is still
    // offered because an uncontrolled drone is worse.
    if (!window.confirm("Cut the motors? The drone will drop where it is.")) return;
    try {
      await Api.killDrone(flightId);
      el.note.textContent = "EMERGENCY STOP sent";
    } catch (e) {
      el.note.textContent = "emergency stop failed: " + e.message;
    }
  }

  async function startDryRun() {
    try {
      await Api.startDryRun({
        mode: el.mode.value,
        profile: el.profile.value,
        no_forward: el.noForward.checked,
        dry_run: true,
      });
      el.note.textContent = "dry run starting";
      setTimeout(() => refresh().catch(() => {}), 800);
    } catch (e) {
      el.note.textContent = "could not start dry run: " + e.message;
    }
  }

  async function stopDryRun() {
    try {
      await Api.stopDroneFlight(flightId || "tello_none");
      el.note.textContent = "stop requested";
    } catch (e) {
      el.note.textContent = "stop failed: " + e.message;
    }
  }

  // ------------------------------------------------------------------ init

  function fillSelect(select, items) {
    // replaceChildren + textContent rather than innerHTML: the labels come from the
    // backend, so they are never interpolated into markup.
    select.replaceChildren();
    (items || []).forEach((it) => {
      const opt = document.createElement("option");
      opt.value = it.id;
      opt.textContent = it.label;
      select.appendChild(opt);
    });
  }

  function init() {
    el = {
      pill: document.getElementById("drone-pill"),
      flightIdEl: document.getElementById("drone-flight-id"),
      readout: document.getElementById("drone-readout"),
      cameraToggle: document.getElementById("drone-camera-toggle"),
      cameraFrame: document.getElementById("drone-camera-frame"),
      cameraPlaceholder: document.getElementById("drone-camera-placeholder"),
      landBtn: document.getElementById("drone-land-btn"),
      killBtn: document.getElementById("drone-kill-btn"),
      note: document.getElementById("drone-note"),
      mode: document.getElementById("drone-mode"),
      profile: document.getElementById("drone-profile"),
      noForward: document.getElementById("drone-no-forward"),
      dryRunBtn: document.getElementById("drone-dryrun-btn"),
      stopBtn: document.getElementById("drone-stop-btn"),
    };

    el.cameraToggle.addEventListener("change", onCameraToggle);
    el.landBtn.addEventListener("click", land);
    el.killBtn.addEventListener("click", kill);
    el.dryRunBtn.addEventListener("click", startDryRun);
    el.stopBtn.addEventListener("click", stopDryRun);

    refresh()
      .then((payload) => {
        fillSelect(el.mode, payload.modes);
        fillSelect(el.profile, payload.profiles);
        el.profile.value = "typical";
      })
      .catch(() => { /* no hardware side yet; the poll below will retry */ });

    // A backend mid-restart is a normal transient, as it is for the run list.
    statusTimer = setInterval(() => refresh().catch(() => {}), STATUS_POLL_MS);
  }

  return { init, refresh, get flightId() { return flightId; } };
})();
