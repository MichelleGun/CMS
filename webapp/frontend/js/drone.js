// The Drone tab: start a flight (dry or real), watch it (camera, telemetry, events),
// fly it from the keyboard, land / cut it, and answer "save this run?" after landing.
//
// Safety lives in the backend, not here: a real takeoff needs confirm:true and is only
// accepted from this computer, one flight at a time; keys stop reaching the drone 0.4 s
// after this page stops sending them. This module asks once before a real takeoff.
const Drone = (() => {
  const STATUS_POLL_MS = 1000;
  const CAMERA_POLL_MS = 200;

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
    { id: "human", title: "5. Fly it yourself", mode: "human", record: true,
      text: "A pop-up window opens on this computer: click it and fly with the keyboard. " +
            "Takes off, hovers, climbs to the start height, then the keys are yours." },
    { id: "policy", title: "6. Fly a trained policy", mode: "policy",
      text: "Pick a policy below (★ = recommended). Takes off, hovers, climbs to the start " +
            "height, then the policy flies inside the same safety layers. No travel cap: " +
            "it laps until the airborne limit, the battery, or you press Land." },
  ];

  let flightId = null;
  let active = false;
  let meta = {};
  let ws = null;
  let cameraTimer = null;
  let cameraRetryDelay = CAMERA_POLL_MS;
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
    el.maxAirborne.value = stage.max_airborne || 150;
    el.record.checked = !!stage.record;
    renderStages();
    syncForm();
  }

  function syncForm() {
    const real = isReal();
    el.realWarning.hidden = !real;
    el.modelRow.hidden = stage.mode !== "policy";
    el.gainRow.hidden = stage.mode !== "human";
    el.keysRow.hidden = stage.mode !== "human";
    showModelInfo();
    el.profileRow.hidden = real;
    el.startBtn.textContent = real ? "Take off (real drone)" : "Start dry run";
    el.startBtn.classList.toggle("danger-btn", real);
    el.startBtn.classList.toggle("primary-btn", !real);
    el.startBtn.disabled = active;
  }

  let models = [];

  // "<title> — <what it does> · <score>", grouped: recommended, RL policies, clones.
  // (Before: the raw folder name and "(undefined/undefined in sim)" for RL runs.)
  function modelLabel(m) {
    const title = m.title || m.name;
    const star = m.recommended ? "★ " : (m.starred ? "☆ " : "");
    return `${star}${title} — ${m.task_label || "hoop"} · ${m.score || "not evaluated yet"}`;
  }

  function showModelInfo() {
    if (!el.modelInfo) return;
    const m = models.find((x) => x.model === el.model.value);
    el.modelInfo.hidden = stage.mode !== "policy" || !m;
    if (!m) return;
    const extras = (m.extras && m.extras.length) ? ` with ${m.extras.join(" + ")}` : "";
    el.modelInfo.textContent =
      `${m.method || m.kind}${extras}. Trained for: ${m.task_label || "?"}. ` +
      `Score: ${m.score || "not evaluated yet"}. Run: ${m.run_id || m.name}`;
  }

  async function loadModels(selectPath) {
    try {
      const res = await Api.getModels();
      models = (res.models || []).filter((m) => m.flyable);
      el.model.replaceChildren();
      const groups = [
        ["Recommended", models.filter((m) => m.recommended)],
        ["RL policies (PPO)", models.filter((m) => !m.recommended && m.kind === "rl")],
        ["Behaviour clones / DAgger students", models.filter((m) => !m.recommended && m.kind !== "rl")],
      ];
      groups.forEach(([name, list]) => {
        if (!list.length) return;
        const g = document.createElement("optgroup");
        g.label = name;
        list.forEach((m) => {
          const opt = document.createElement("option");
          opt.value = m.model;
          opt.textContent = modelLabel(m);
          opt.title = m.model;
          g.appendChild(opt);
        });
        el.model.appendChild(g);
      });
      if (selectPath) el.model.value = selectPath;
      showModelInfo();
    } catch (e) { /* the Imitation tab shows the error */ }
  }

  async function start() {
    const real = isReal();
    const payload = {
      mode: stage.mode,
      no_forward: !!stage.no_forward,
      no_commit: !!stage.no_commit,
      max_travel: el.maxTravel.value === "" ? null : Number(el.maxTravel.value),
      max_airborne: Number(el.maxAirborne.value) || 150,
      max_height: Number(el.maxHeight.value) || 190,
      start_height: el.startHeight.value === "" ? 80 : Number(el.startHeight.value),
      keys: el.keys.value || "popup",
      clearml: el.clearml.checked,
      record: el.record.checked,
      model: stage.mode === "policy" ? el.model.value || null : null,
      // the camera calibration preset (js/calib.js fills this select); "default" = detector.json
      detector_preset: (document.getElementById("opt-detector") || {}).value || null,
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
      if (stage.mode === "human") el.flyNote.textContent += payload.keys === "browser"
        ? " Keys are live once it is up."
        : " A pop-up window opens: CLICK IT, then fly from it (W/S A/D, arrows; L lands, X cuts).";
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
    // the upload runs in its own process now, so it never holds the Start button
    if (!active && payload && payload.clearml_uploading) {
      el.flyNote.textContent = "uploading the last flight to ClearML in the background; you can start the next one";
    }

    const awaiting = active && phase === "awaiting_save" && payload.save_answered !== true
      && answeredFor !== flightId;
    el.savePanel.hidden = !awaiting;
    if (awaiting) el.saveSamples.textContent =
      `${lastStatus.samples || "?"} samples recorded. The drone has landed.`;

    // browser keys only for a flight that listens for them (keys_port); the pop-up window
    // reads the keyboard itself
    const flying = active && meta.mode === "human" && phase !== "awaiting_save" && !!meta.keys_port;
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

  // The frame stays hidden behind the placeholder until one has actually loaded. Assigning
  // the URL straight to the visible <img> blanked the pane on every 404 -- and before
  // takeoff there was never a frame, so a flight that failed to take off showed nothing.
  function syncCamera() {
    const on = el.cameraToggle.checked && !!flightId;
    if (cameraTimer) { clearTimeout(cameraTimer); cameraTimer = null; }
    Frames.reset(el.cameraFrame);
    el.cameraFrame.hidden = true;
    el.cameraPlaceholder.hidden = false;
    el.cameraPlaceholder.textContent = on ? "waiting for the first camera frame" : "camera off";
    if (!on) return;
    const id = flightId;
    cameraRetryDelay = CAMERA_POLL_MS;
    let lastFrameTimestamp = null;
    let lastFrameAt = 0;
    const poll = async () => {
      if (flightId !== id || !el.cameraToggle.checked) return;
      if (Tabs.current() !== "drone") {
        cameraTimer = setTimeout(poll, CAMERA_POLL_MS);
        return;
      }
      await Frames.swap(el.cameraFrame, Api.droneCameraUrl(id), (timestamp) => {
        if (flightId !== id) return;
        const now = performance.now();
        if (timestamp !== lastFrameTimestamp) {
          lastFrameTimestamp = timestamp;
          lastFrameAt = now;
        }
        const age = (now - lastFrameAt) / 1000;
        if (age > 2) {
          el.cameraFrame.hidden = true;
          el.cameraPlaceholder.hidden = false;
          el.cameraPlaceholder.textContent = active
            ? `Last camera frame is ${age.toFixed(1)}s old; reconnecting…`
            : "Flight ended; camera is no longer live.";
          cameraRetryDelay = Math.min(cameraRetryDelay * 2, 1200);
          return;
        }
        el.cameraFrame.hidden = false;
        el.cameraPlaceholder.hidden = true;
        cameraRetryDelay = CAMERA_POLL_MS;
      }, (error) => {
        if (flightId !== id) return;
        el.cameraFrame.hidden = true;
        el.cameraPlaceholder.hidden = false;
        el.cameraPlaceholder.textContent = `Camera reconnecting (${error.message})…`;
        cameraRetryDelay = Math.min(cameraRetryDelay * 2, 1200);
      });
      if (flightId === id && el.cameraToggle.checked) {
        cameraTimer = setTimeout(poll, cameraRetryDelay);
      }
    };
    poll();
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
      modelInfo: $("opt-model-info"), keysRow: $("opt-keys-row"), keys: $("opt-keys"),
      maxHeight: $("opt-max-height"), startHeight: $("opt-start-height"), clearml: $("opt-clearml"),
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
    el.model.addEventListener("change", showModelInfo);
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
