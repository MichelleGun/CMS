// Wires api.js, scene3d.js, controls.js and chart.js together: run history,
// start/stop, live telemetry, the reward chart, the camera toggle and the
// replay panel.
(function () {
  let config = null;
  let runs = [];
  let currentRunId = null;
  let bestRunId = null;
  let viewedRunId = null;
  let ws = null;
  let isViewedRunTraining = false;
  let cameraPollTimer = null;
  let replayPollTimer = null;
  let lastReplayId = null;

  // uiMode: "setup" (configuring, no run in view) | "starting" (run picked,
  // waiting for its first data) | "run" (the darkened overlay is showing).
  let uiMode = "setup";
  let overlayView = "newest"; // "newest" | "best", which card set the overlay shows
  let overlayReplayPollTimer = null;
  let overlayBestReplayRunId = null; // avoids re-rendering the same best replay repeatedly
  // Auto-resume into the most recent run on first load (e.g. after a page
  // refresh mid-run), but never again once the user has deliberately gone
  // back to setup, otherwise the 2.5s run-list poll would pull them right
  // back into the overlay.
  let autoSelectEnabled = true;
  const lastBestSeen = {}; // run_id -> highest best_eval_reward observed, for the toast
  let toastTimer = null;

  const statusPill = document.getElementById("status-pill");
  const runListEl = document.getElementById("run-list");
  const readoutEl = document.getElementById("position-readout");
  const startBtn = document.getElementById("start-btn");
  const stopBtn = document.getElementById("stop-btn");
  const algoSelect = document.getElementById("algo-select");
  const dqnNote = document.getElementById("dqn-note");
  const randomizeToggle = document.getElementById("randomize-toggle");
  const seedInput = document.getElementById("seed-input");
  const cameraToggle = document.getElementById("camera-toggle");
  const cameraFrame = document.getElementById("camera-frame");
  const cameraPlaceholder = document.getElementById("camera-off-placeholder");
  const replayStatusEl = document.getElementById("replay-status");
  const replayVideo = document.getElementById("replay-video");
  const replayViewToggle = document.getElementById("replay-view-toggle");
  let lastReplayUrls = null;

  const sceneCanvas = document.getElementById("scene-canvas");
  const runScrim = document.getElementById("run-scrim");
  const runLoading = document.getElementById("run-loading");
  const runOverlay = document.getElementById("run-overlay");
  const viewNewestBtn = document.getElementById("view-newest-btn");
  const viewBestBtn = document.getElementById("view-best-btn");
  const backToSetupBtn = document.getElementById("back-to-setup-btn");
  const bestToast = document.getElementById("best-toast");
  const overlayTerminal = document.getElementById("overlay-terminal");
  const overlayCameraFrame = document.getElementById("overlay-camera-frame");
  const overlayCameraPlaceholder = document.getElementById("overlay-camera-placeholder");
  const overlayCameraPane = document.getElementById("overlay-camera-pane");
  const overlayReplayPane = document.getElementById("overlay-replay-pane");
  const overlayReplayStatus = document.getElementById("overlay-replay-status");
  const overlayReplayVideo = document.getElementById("overlay-replay-video");
  const MAX_LOG_LINES = 500;

  function fmt(n, d = 2) { return (n === null || n === undefined) ? "-" : Number(n).toFixed(d); }

  function setStatusPill(status) {
    statusPill.textContent = status || "idle";
    statusPill.className = "pill " + (
      status === "training" ? "pill-training" :
      (status === "stopped" || status === "crashed") ? "pill-stopped" : "pill-idle"
    );
  }

  function updateReadout() {
    const { start, target } = Controls.getDraft();
    readoutEl.textContent =
      `drone start  x ${fmt(start[0])}  y ${fmt(start[1])}  z ${fmt(start[2])}\n` +
      `hoop target  x ${fmt(target[0])}  y ${fmt(target[1])}  z ${fmt(target[2])}`;
    readoutEl.style.whiteSpace = "pre";
  }

  function showDraftPreview() {
    const { start, target } = Controls.getDraft();
    Scene3D.setHoop(target);
    Scene3D.setDroneTarget(start, [0, 0, 0], true);
  }

  // ------------------------------------------------------------- ui mode

  function applyPanelState() {
    const activeRun = runs.find((r) => r.status === "training");
    startBtn.disabled = !!activeRun || uiMode !== "setup";
    Controls.setPanelActive(uiMode === "setup" && !activeRun);
  }

  function setUiMode(mode) {
    uiMode = mode;
    runScrim.classList.toggle("visible", mode !== "setup");
    sceneCanvas.classList.toggle("dimmed", mode !== "setup");
    runLoading.hidden = mode !== "starting";
    runOverlay.hidden = mode !== "run";
    if (mode === "run") {
      setOverlayView("newest");
    } else if (overlayReplayPollTimer) {
      clearInterval(overlayReplayPollTimer);
      overlayReplayPollTimer = null;
    }
    applyPanelState();
  }

  function setOverlayView(which) {
    overlayView = which;
    viewNewestBtn.classList.toggle("active", which === "newest");
    viewBestBtn.classList.toggle("active", which === "best");
    overlayCameraPane.hidden = which !== "newest";
    overlayReplayPane.hidden = which !== "best";
    if (which === "best") {
      requestOverlayBestReplay();
    } else if (overlayReplayPollTimer) {
      clearInterval(overlayReplayPollTimer);
      overlayReplayPollTimer = null;
    }
  }

  function backToSetup() {
    autoSelectEnabled = false;
    viewedRunId = null;
    if (ws) { ws.close(); ws = null; }
    stopCameraPolling();
    cameraToggle.checked = false;
    renderRunList();
    showDraftPreview();
    setUiMode("setup");
  }

  function clearTerminal() { overlayTerminal.textContent = ""; }

  function appendLogLine(line) {
    const atBottom = overlayTerminal.scrollTop + overlayTerminal.clientHeight >= overlayTerminal.scrollHeight - 4;
    overlayTerminal.appendChild(document.createTextNode(line + "\n"));
    while (overlayTerminal.childNodes.length > MAX_LOG_LINES) {
      overlayTerminal.removeChild(overlayTerminal.firstChild);
    }
    if (atBottom) overlayTerminal.scrollTop = overlayTerminal.scrollHeight;
  }

  function showBestToast() {
    bestToast.hidden = false;
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { bestToast.hidden = true; }, 4000);
  }

  // -------------------------------------------------------- best replay

  function requestOverlayBestReplay() {
    if (!bestRunId) return;
    if (overlayBestReplayRunId === bestRunId && !overlayReplayVideo.hidden) return;
    overlayBestReplayRunId = bestRunId;
    overlayReplayStatus.textContent = "rendering...";
    overlayReplayVideo.hidden = true;
    Api.startReplay(bestRunId, { variant: "best", pov: false, episodes: 1 }).then((res) => {
      pollOverlayReplay(bestRunId, res.replay_id);
    }).catch((e) => {
      overlayReplayStatus.textContent = "failed: " + e.message;
    });
  }

  function pollOverlayReplay(runId, replayId) {
    if (overlayReplayPollTimer) clearInterval(overlayReplayPollTimer);
    overlayReplayPollTimer = setInterval(async () => {
      try {
        const status = await Api.getReplayStatus(runId, replayId, false);
        if (status.status === "done") {
          clearInterval(overlayReplayPollTimer);
          overlayReplayPollTimer = null;
          overlayReplayStatus.textContent = "";
          overlayReplayVideo.src = status.video_url;
          overlayReplayVideo.hidden = false;
          overlayReplayVideo.play().catch(() => {});
        } else if (status.status === "failed") {
          clearInterval(overlayReplayPollTimer);
          overlayReplayPollTimer = null;
          overlayReplayStatus.textContent = "failed (see server log, ffmpeg may be missing)";
        }
      } catch (e) {
        clearInterval(overlayReplayPollTimer);
        overlayReplayPollTimer = null;
        overlayReplayStatus.textContent = "failed: " + e.message;
      }
    }, 2000);
  }

  // ----------------------------------------------------------------- runs

  async function refreshRuns() {
    const data = await Api.getRuns();
    runs = data.runs;
    currentRunId = data.current_run_id;
    bestRunId = data.best_run_id;
    renderRunList();

    stopBtn.disabled = !(viewedRunId && runs.find((r) => r.run_id === viewedRunId && r.status === "training"));
    viewBestBtn.disabled = !bestRunId;
    applyPanelState();

    if (!viewedRunId && currentRunId && autoSelectEnabled) selectRun(currentRunId);
    if (bestRunId) {
      Api.getRun(bestRunId).then((r) => {
        if (r.curve) RewardChart.setBestCurve(r.curve.timesteps, r.curve.mean_reward);
      }).catch(() => {});
    }
  }

  function renderRunList() {
    runListEl.innerHTML = "";
    runs.slice().reverse().forEach((r) => {
      const row = document.createElement("div");
      row.className = "run-row" +
        (r.run_id === viewedRunId ? " selected" : "") +
        (r.status === "training" ? " active-training" : "");
      const best = r.best_eval_reward !== null && r.best_eval_reward !== undefined
        ? r.best_eval_reward.toFixed(1) : "-";
      row.innerHTML = `<span class="algo-badge">${r.algo}</span><br>` +
        `${r.status} &middot; ${r.current_timestep}/${r.total_timesteps}<br>` +
        `best ${best}`;
      row.addEventListener("click", () => selectRun(r.run_id));
      runListEl.appendChild(row);
    });
  }

  // 20 retries at 500ms = 10s: importing torch/pybullet in the training
  // subprocess before it writes its first meta.json can genuinely take
  // several seconds (observed ~5-8s cold start), so this needs real slack,
  // not just a couple of quick retries.
  const SELECT_RUN_MAX_RETRIES = 20;

  function selectRun(runId, retriesLeft = SELECT_RUN_MAX_RETRIES) {
    if (viewedRunId === runId && retriesLeft === SELECT_RUN_MAX_RETRIES) return;
    if (retriesLeft === SELECT_RUN_MAX_RETRIES) {
      viewedRunId = runId;
      renderRunList();
      if (ws) { ws.close(); ws = null; }
      RewardChart.clearCurrent();
      clearTerminal();
      replayStatusEl.textContent = "";
      replayVideo.hidden = true;
      replayViewToggle.hidden = true;
      setUiMode("starting");
    }

    Api.getRun(runId).then((r) => {
      const status = (r.meta.final_status) || (r.live_status && r.live_status.status) || "unknown";
      isViewedRunTraining = status === "training";
      setStatusPill(status);
      if (r.curve) RewardChart.setCurrentCurve(r.curve.timesteps, r.curve.mean_reward);
      if (r.meta.target_override) Scene3D.setHoop(r.meta.target_override);
      if (isViewedRunTraining) openLiveSocket(runId);
      setUiMode("run");
    }).catch(() => {
      // The training subprocess may not have written meta.json yet (it
      // takes a moment to start): retry briefly before giving up, rather
      // than leaving this run stuck with no live view for its whole life.
      if (retriesLeft > 0 && viewedRunId === runId) {
        setTimeout(() => selectRun(runId, retriesLeft - 1), 500);
      } else if (viewedRunId === runId) {
        setUiMode("setup");
      }
    });
  }

  function openLiveSocket(runId) {
    ws = Api.openRunSocket(runId, {
      onTelemetry: (msg) => {
        Scene3D.setDroneTarget(msg.pos, msg.rpy);
        Scene3D.setHoop(msg.target);
      },
      onStatus: (msg) => {
        setStatusPill("training");
        RewardChart.appendCurrentPoint(msg.current_timestep, msg.latest_eval_reward);
        const prevBest = lastBestSeen[runId];
        if (msg.best_eval_reward !== undefined && msg.best_eval_reward !== null) {
          if (prevBest !== undefined && msg.best_eval_reward > prevBest) showBestToast();
          lastBestSeen[runId] = msg.best_eval_reward;
        }
      },
      onRunEnded: (msg) => {
        setStatusPill(msg.status);
        isViewedRunTraining = false;
        showDraftPreview();
        refreshRuns();
      },
      onLog: (msg) => appendLogLine(msg.line),
    });
  }

  // ------------------------------------------------------------- training

  async function startTraining() {
    const algo = algoSelect.value;
    const payload = {
      algo,
      timesteps: parseInt(document.getElementById("timesteps-input").value, 10) || 1000000,
      n_envs: parseInt(document.getElementById("nenvs-input").value, 10) || 4,
    };
    if (randomizeToggle.checked) {
      payload.seed = parseInt(seedInput.value, 10) || 0;
    } else {
      const { start, target } = Controls.getDraft();
      payload.start_pos = { x: start[0], y: start[1], z: start[2] };
      payload.target_pos = { x: target[0], y: target[1], z: target[2] };
    }
    startBtn.disabled = true;
    try {
      const res = await Api.startRun(payload);
      viewedRunId = null; // force re-select so the socket opens
      selectRun(res.run_id);
      await refreshRuns();
    } catch (e) {
      alert("could not start training: " + e.message);
      startBtn.disabled = false;
    }
  }

  async function stopTraining() {
    if (!viewedRunId) return;
    stopBtn.disabled = true;
    try {
      await Api.stopRun(viewedRunId);
    } catch (e) {
      alert("could not stop training: " + e.message);
    }
  }

  // --------------------------------------------------------------- camera

  function startCameraPolling() {
    stopCameraPolling();
    cameraFrame.hidden = false;
    cameraPlaceholder.hidden = true;
    overlayCameraFrame.hidden = false;
    overlayCameraPlaceholder.hidden = true;
    cameraPollTimer = setInterval(() => {
      if (!viewedRunId) return;
      const url = Api.cameraLatestUrl(viewedRunId);
      cameraFrame.src = url;
      overlayCameraFrame.src = url;
    }, 700);
  }

  function stopCameraPolling() {
    if (cameraPollTimer) clearInterval(cameraPollTimer);
    cameraPollTimer = null;
    cameraFrame.hidden = true;
    cameraPlaceholder.hidden = false;
    overlayCameraFrame.hidden = true;
    overlayCameraPlaceholder.hidden = false;
  }

  async function onCameraToggle() {
    if (!viewedRunId) { cameraToggle.checked = false; return; }
    try {
      await Api.setCamera(viewedRunId, cameraToggle.checked);
    } catch (e) { /* best effort */ }
    if (cameraToggle.checked) startCameraPolling(); else stopCameraPolling();
  }

  // --------------------------------------------------------------- replay

  async function requestReplay(variant) {
    if (!viewedRunId) return;
    const pov = document.getElementById("pov-toggle").checked;
    replayStatusEl.textContent = "rendering...";
    replayVideo.hidden = true;
    replayViewToggle.hidden = true;
    try {
      const res = await Api.startReplay(viewedRunId, { variant, pov, episodes: 1 });
      lastReplayId = res.replay_id;
      pollReplay(viewedRunId, res.replay_id, pov);
    } catch (e) {
      replayStatusEl.textContent = "failed: " + e.message;
    }
  }

  function pollReplay(runId, replayId, pov) {
    if (replayPollTimer) clearInterval(replayPollTimer);
    replayPollTimer = setInterval(async () => {
      try {
        const status = await Api.getReplayStatus(runId, replayId, pov);
        if (status.status === "done") {
          clearInterval(replayPollTimer);
          replayStatusEl.textContent = "ready";
          lastReplayUrls = status;
          replayVideo.src = status.video_url;
          replayVideo.hidden = false;
          replayViewToggle.hidden = !status.pov_video_url;
          replayVideo.play().catch(() => {});
        } else if (status.status === "failed") {
          clearInterval(replayPollTimer);
          replayStatusEl.textContent = "failed (see server log, ffmpeg may be missing)";
        }
      } catch (e) {
        clearInterval(replayPollTimer);
        replayStatusEl.textContent = "failed: " + e.message;
      }
    }, 2000);
  }

  document.querySelectorAll('input[name="replay-view"]').forEach((el) => {
    el.addEventListener("change", (e) => {
      if (!lastReplayUrls) return;
      const wantPov = e.target.value === "pov";
      replayVideo.src = wantPov ? lastReplayUrls.pov_video_url : lastReplayUrls.video_url;
      replayVideo.play().catch(() => {});
    });
  });

  // ------------------------------------------------------------------ boot

  async function boot() {
    config = await Api.getConfig();

    algoSelect.innerHTML = "";
    config.algorithms.forEach((a) => {
      const opt = document.createElement("option");
      opt.value = a.id;
      opt.textContent = a.label;
      algoSelect.appendChild(opt);
    });
    algoSelect.addEventListener("change", () => {
      dqnNote.hidden = algoSelect.value !== "dqn";
    });

    Scene3D.init(document.getElementById("scene-canvas"), config.room);
    Controls.init(config);
    Controls.onChange(() => {
      updateReadout();
      if (!isViewedRunTraining) showDraftPreview();
    });
    updateReadout();
    showDraftPreview();

    RewardChart.init(document.getElementById("reward-chart"));

    startBtn.addEventListener("click", startTraining);
    stopBtn.addEventListener("click", stopTraining);
    cameraToggle.addEventListener("change", onCameraToggle);
    viewNewestBtn.addEventListener("click", () => setOverlayView("newest"));
    viewBestBtn.addEventListener("click", () => setOverlayView("best"));
    backToSetupBtn.addEventListener("click", backToSetup);
    document.getElementById("replay-current-btn").addEventListener("click", () => requestReplay("current"));
    document.getElementById("replay-best-btn").addEventListener("click", () => requestReplay("best"));

    randomizeToggle.addEventListener("change", () => {
      seedInput.disabled = !randomizeToggle.checked;
    });

    await refreshRuns().catch(() => {});
    // .catch here too: if the backend is mid-restart when a poll lands, this
    // is a normal transient state, not worth an unhandled-rejection console
    // error every 2.5s until it comes back.
    setInterval(() => refreshRuns().catch(() => {}), 2500);
  }

  window.addEventListener("DOMContentLoaded", boot);
})();
