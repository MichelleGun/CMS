// The Drone tab: start a flight (dry or real), watch it (camera, telemetry, events),
// fly it from the keyboard, land / cut it, and answer "save this run?" after landing.
//
// Safety lives in the backend, not here: a real takeoff needs confirm:true and is only
// accepted from this computer, one flight at a time; keys stop reaching the drone 0.4 s
// after this page stops sending them. This module asks once before a real takeoff.
const Drone = (() => {
  const STATUS_POLL_MS = 1000;
  const CAMERA_POLL_MS = 400;

  // The README's Stage C steps, in order. Each one only fills the options form.
  const STAGES = [
    { id: "hover", title: "1. Hover test", mode: "hover", max_airborne: 20,
      text: "Take off, sweep to find the ring, hold, land. Never moves forward." },
    { id: "align", title: "2. Alignment only", mode: "expert", no_forward: true,
      text: "Rotate and climb to centre the ring. No forward motion." },
    { id: "approach", title: "3. Bounded approach", mode: "expert", no_commit: true, max_travel: 0.4,
      text: "Short approach, 0.4 m cap, no blind push. Measure how far it really went." },
    { id: "full", title: "4. Full flight", mode: "expert", max_travel: 1.8,
      text: "Align, approach, and the blind push through the ring." },
    { id: "human", title: "5. Fly it yourself", mode: "human", record: true, max_travel: 1.8, max_airborne: 60,
      text: "Keyboard flight, recorded as a demonstration. Save it after landing." },
    { id: "policy", title: "6. Fly a trained policy", mode: "policy", max_travel: 1.8,
      text: "A clone from the Imitation tab flies the same safety layers." },
  ];

  let flightId = null;
  let active = false;
  let meta = {};
  let ws = null;
  let cameraTimer = null;
  let lastStatus = null;
  let stage = STAGES[0];
  let answeredFor = null;     // the flight whose save question was answered here
  let el = {};
  const liveListeners = [];

  const $ = (id) => document.getElementById(id);
  function fmt(n, d = 2) {
    return (n === null || n === undefined || n === "") ? "-" : Number(n).toFixed(d);
  }
  function isReal() { return el.target.querySelector("input:checked").value === "real"; }

  // ------------------------------------------------------------------ start form

  function renderStages() {
    el.stages.replaceChildren();
    STAGES.forEach((s) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "stage-btn" + (s === stage ? " selected" : "");
      const t = document.createElement("strong");
      t.textContent = s.title;
      const d = document.createElement("span");
      d.textContent = s.text;
      b.append(t, d);
      b.addEventListener("click", () => selectStage(s.id));
      el.stages.appendChild(b);
    });
  }

  function selectStage(id) {
    stage = STAGES.find((s) => s.id === id) || STAGES[0];
    el.maxTravel.value = stage.max_travel === undefined ? "" : stage.max_travel;
    el.maxAirborne.value = stage.max_airborne || 40;
    el.record.checked = !!stage.record;
    renderStages();
    syncForm();
  }

  function syncForm() {
    const real = isReal();
    el.realWarning.hidden = !real;
    el.modelRow.hidden = stage.mode !== "policy";
    el.gainRow.hidden = stage.mode !== "human";
    el.profileRow.hidden = real;
    el.startBtn.textContent = real ? "Take off (real drone)" : "Start dry run";
    el.startBtn.classList.toggle("danger-btn", real);
    el.startBtn.classList.toggle("primary-btn", !real);
    el.startBtn.disabled = active;
  }

  async function loadModels(selectPath) {
    try {
      const { models } = await Api.getModels();
      el.model.replaceChildren();
      models.filter((m) => m.flyable).forEach((m) => {
        const opt = document.createElement("option");
        opt.value = m.model;
        const ev = (m.evals && m.evals[0]) || m.train_eval;
        opt.textContent = m.name + (ev ? `  (${ev.passed}/${ev.episodes} in sim)` : "");
        el.model.appendChild(opt);
      });
      if (selectPath) el.model.value = selectPath;
    } catch (e) { /* the Imitation tab shows the error */ }
  }

  async function start() {
    const real = isReal();
    const payload = {
      mode: stage.mode,
      no_forward: !!stage.no_forward,
      no_commit: !!stage.no_commit,
      max_travel: el.maxTravel.value === "" ? null : Number(el.maxTravel.value),
      max_airborne: Number(el.maxAirborne.value) || 40,
      record: el.record.checked,
      model: stage.mode === "policy" ? el.model.value || null : null,
      human_gain: Number(el.gain.value) || 0.6,
      profile: el.profile.value,
      dry_run: !real,
      confirm: false,
    };
    if (stage.mode === "policy" && !payload.model) {
      el.flyNote.textContent = "pick a policy first (train one in the Imitation tab)";
      return;
    }
    if (real) {
      const ok = window.confirm(
        `REAL FLIGHT: ${stage.title}\n\nCage closed, nobody inside, battery above 30 %, ` +
        "laptop on the TELLO Wi-Fi.\nLand: L or the Land button. Motor cut: X.\n\nTake off?");
      if (!ok) return;
      payload.confirm = true;
    }
    el.startBtn.disabled = true;
    try {
      await Api.startFlight(payload);
      el.flyNote.textContent = real ? "taking off in 3 s..." : "dry run starting";
      if (stage.mode === "human") el.flyNote.textContent += " Keys are live once it is up.";
      setTimeout(() => refresh().catch(() => {}), 700);
    } catch (e) {
      el.flyNote.textContent = "could not start: " + e.message;
      el.startBtn.disabled = active;
    }
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
    if (!s) { el.readout.textContent = "no flight recorded yet"; return; }
    const lock = s.ring_found ? (s.ring_clipped ? "locked (clipped)" : "locked") : "no ring";
    const rng = s.range_m === null || s.range_m === undefined ? "-" : fmt(s.range_m, 2) + " m";
    const travel = s.max_travel_m ? `${fmt(s.travel_m, 2)} / ${fmt(s.max_travel_m, 2)} m`
      : `${fmt(s.travel_m, 2)} m`;
    el.readout.textContent =
      `battery ${s.battery ?? "-"} %   temp ${s.temp_c ?? "-"} C\n` +
      `height  ${fmt(s.height_cm, 0)} cm   airborne ${fmt(s.airborne_s, 1)} s\n` +
      `ring    ${lock}   range ${rng}\n` +
      `offset  x ${fmt(s.cx, 2)}  y ${fmt(s.cy, 2)}  round ${fmt(s.axis_ratio, 2)}\n` +
      `travel  ${travel}\n` +
      `rc      ${(s.action || []).map((v) => fmt(v, 2)).join("  ")}`;
  }

  function addEvent(ev) {
    if (!ev || !ev.event) return;
    const li = document.createElement("li");
    const time = (ev.utc || "").slice(11, 19);
    const extra = Object.entries(ev)
      .filter(([k]) => !["utc", "event", "type"].includes(k))
      .map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`).join(" ");
    li.textContent = `${time}  ${ev.event}${extra ? "  " + extra : ""}`;
    el.events.prepend(li);
    while (el.events.children.length > 80) el.events.lastChild.remove();
  }

  function applyStatus(payload) {
    const s = payload && payload.status !== undefined && typeof payload.status === "object"
      ? payload.status : payload;
    if (payload && payload.flight_id) flightId = payload.flight_id;
    if (payload && payload.active !== undefined) active = !!payload.active;
    if (payload && payload.meta) meta = payload.meta || {};
    lastStatus = s || lastStatus;
    renderReadout(lastStatus);

    const finalStatus = meta.final_status;
    const phase = (lastStatus && lastStatus.phase) || "-";
    const dry = meta.dry_run !== false;
    if (!flightId) setPill("no flight", "pill-idle");
    else if (active) setPill(`${phase}${dry ? " (dry)" : " (REAL)"}`, phaseClass(phase, finalStatus));
    else setPill(finalStatus || "finished", phaseClass(phase, finalStatus));
    el.flightIdEl.textContent = flightId ? `${flightId} · ${meta.mode || ""}` : "";

    el.landBtn.disabled = !flightId || !active;
    el.killBtn.disabled = !flightId || !active;
    el.stopBtn.disabled = !active || !dry;
    el.startBtn.disabled = active;

    const awaiting = active && phase === "awaiting_save" && payload.save_answered !== true
      && answeredFor !== flightId;
    el.savePanel.hidden = !awaiting;
    if (awaiting) el.saveSamples.textContent =
      `${lastStatus.samples || "?"} samples recorded. The drone has landed.`;

    const flying = active && meta.mode === "human" && phase !== "awaiting_save";
    if (flying && flightId) Keys.engage(flightId);
    else if (Keys.engaged) Keys.disengage();

    if (payload && payload.camera_enabled !== undefined
        && el.cameraToggle.checked !== payload.camera_enabled) {
      el.cameraToggle.checked = payload.camera_enabled;
      syncCamera();
    }
    liveListeners.forEach((fn) => fn(active, flightId, !dry));
  }

  // ------------------------------------------------------------------ camera

  function syncCamera() {
    const on = el.cameraToggle.checked && !!flightId;
    el.cameraFrame.hidden = !on;
    el.cameraPlaceholder.hidden = on;
    if (cameraTimer) { clearInterval(cameraTimer); cameraTimer = null; }
    if (!on) return;
    const tick = () => {
      if (Tabs.current() === "drone") el.cameraFrame.src = Api.droneCameraUrl(flightId);
    };
    tick();
    cameraTimer = setInterval(tick, CAMERA_POLL_MS);
  }

  async function onCameraToggle() {
    if (!flightId) { el.cameraToggle.checked = false; return; }
    try { await Api.setDroneCamera(flightId, el.cameraToggle.checked); }
    catch (e) { el.note.textContent = "camera toggle failed: " + e.message; }
    syncCamera();
  }

  // ------------------------------------------------------------------ socket + poll

  function openSocket(id) {
    if (ws) ws.close();
    el.events.replaceChildren();
    ws = Api.openDroneSocket(id, {
      onStatus: (msg) => applyStatus(msg),
      onEnded: () => refresh().catch(() => {}),
      onEvent: (msg) => {
        addEvent(msg);
        if (msg.event === "awaiting_save") refresh().catch(() => {});
      },
    });
  }

  async function refresh() {
    const payload = await Api.getDroneStatus();
    const previousId = flightId;
    applyStatus(payload);
    if (flightId && flightId !== previousId) {
      openSocket(flightId);
      syncCamera();
    }
    return payload;
  }

  // ------------------------------------------------------------------ actions

  async function land() {
    if (!flightId) return;
    try { await Api.landDrone(flightId); el.note.textContent = "landing requested"; }
    catch (e) { el.note.textContent = "land failed: " + e.message; }
  }

  async function kill(ask = true) {
    if (!flightId) return;
    if (ask && !window.confirm("Cut the motors? The drone will drop where it is.")) return;
    try { await Api.killDrone(flightId); el.note.textContent = "EMERGENCY MOTOR CUT sent"; }
    catch (e) { el.note.textContent = "motor cut failed: " + e.message; }
  }

  async function stopDryRun() {
    try { await Api.stopDroneFlight(flightId || "tello_none"); el.note.textContent = "stop requested"; }
    catch (e) { el.note.textContent = "stop failed: " + e.message; }
  }

  async function answer(save, passed) {
    try {
      await Api.answerSave(flightId, save, passed);
      answeredFor = flightId;
      el.savePanel.hidden = true;
      el.note.textContent = !save ? "run discarded" : passed ? "saved as a pass (demo.npz)"
        : "saved as a miss (demo_failed.npz)";
    } catch (e) {
      el.note.textContent = "could not answer: " + e.message;
    }
  }

  // ------------------------------------------------------------------ init

  function init() {
    el = {
      target: $("fly-target"), realWarning: $("fly-real-warning"), stages: $("stage-list"),
      modelRow: $("opt-model-row"), model: $("opt-model"), maxTravel: $("opt-max-travel"),
      maxAirborne: $("opt-max-airborne"), gainRow: $("opt-gain-row"), gain: $("opt-gain"),
      profileRow: $("opt-profile-row"), profile: $("opt-profile"), record: $("opt-record"),
      startBtn: $("fly-start-btn"), flyNote: $("fly-note"),
      pill: $("drone-pill"), flightIdEl: $("drone-flight-id"), readout: $("drone-readout"),
      cameraToggle: $("drone-camera-toggle"), cameraFrame: $("drone-camera-frame"),
      cameraPlaceholder: $("drone-camera-placeholder"), events: $("drone-events"),
      landBtn: $("drone-land-btn"), killBtn: $("drone-kill-btn"), stopBtn: $("drone-stop-btn"),
      note: $("drone-note"), savePanel: $("save-panel"), saveSamples: $("save-samples"),
      keysState: $("keys-state"), keysHeld: $("keys-held"),
    };
    el.readout.style.whiteSpace = "pre";

    el.target.addEventListener("change", syncForm);
    el.startBtn.addEventListener("click", start);
    el.cameraToggle.addEventListener("change", onCameraToggle);
    el.landBtn.addEventListener("click", land);
    el.killBtn.addEventListener("click", () => kill(true));
    el.stopBtn.addEventListener("click", stopDryRun);
    $("save-pass-btn").addEventListener("click", () => answer(true, true));
    $("save-miss-btn").addEventListener("click", () => answer(true, false));
    $("save-discard-btn").addEventListener("click", () => answer(false, false));

    Keys.init({
      // The keyboard motor cut is immediate, as it was in the pygame window: the pilot
      // reaching for X has no time for a dialog. The button still asks.
      onPress: (what) => (what === "land" ? land() : kill(false)),
      onChange: (held) => {
        if (held === null) {
          el.keysState.textContent = "keys off";
          el.keysState.className = "pill pill-idle";
          el.keysHeld.textContent = "-";
        } else {
          el.keysState.textContent = "keys live";
          el.keysState.className = "pill pill-training";
          el.keysHeld.textContent = Keys.describe(held);
        }
      },
    });

    selectStage("hover");
    loadModels();
    Tabs.onChange((name) => { if (name === "drone") loadModels(el.model.value); });
    refresh().catch(() => {});
    setInterval(() => refresh().catch(() => {}), STATUS_POLL_MS);
  }

  // Called from the Imitation tab's "Fly this clone".
  async function flyModel(path) {
    Tabs.show("drone");
    selectStage("policy");
    await loadModels(path);
  }

  return {
    init, refresh, flyModel,
    onLive: (fn) => liveListeners.push(fn),
    get flightId() { return flightId; },
  };
})();
