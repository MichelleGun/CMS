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

  // ----------------------------------------------------------------- runs

  async function refreshRuns() {
    const data = await Api.getRuns();
    runs = data.runs;
    currentRunId = data.current_run_id;
    bestRunId = data.best_run_id;
    renderRunList();

    const activeRun = runs.find((r) => r.status === "training");
    startBtn.disabled = !!activeRun;
    stopBtn.disabled = !(viewedRunId && runs.find((r) => r.run_id === viewedRunId && r.status === "training"));
    Controls.setPanelActive(!activeRun);

    if (!viewedRunId && currentRunId) selectRun(currentRunId);
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
      row.className = "run-row" + (r.run_id === viewedRunId ? " selected" : "");
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
    viewedRunId = runId;
    renderRunList();
    if (ws) { ws.close(); ws = null; }
    RewardChart.clearCurrent();
    replayStatusEl.textContent = "";
    replayVideo.hidden = true;
    replayViewToggle.hidden = true;

    Api.getRun(runId).then((r) => {
      const status = (r.meta.final_status) || (r.live_status && r.live_status.status) || "unknown";
      isViewedRunTraining = status === "training";
      setStatusPill(status);
      if (r.curve) RewardChart.setCurrentCurve(r.curve.timesteps, r.curve.mean_reward);
      if (r.meta.target_override) Scene3D.setHoop(r.meta.target_override);
      if (isViewedRunTraining) openLiveSocket(runId);
    }).catch(() => {
      // The training subprocess may not have written meta.json yet (it
      // takes a moment to start): retry briefly before giving up, rather
      // than leaving this run stuck with no live view for its whole life.
      if (retriesLeft > 0 && viewedRunId === runId) {
        setTimeout(() => selectRun(runId, retriesLeft - 1), 500);
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
      },
      onRunEnded: (msg) => {
        setStatusPill(msg.status);
        isViewedRunTraining = false;
        showDraftPreview();
        refreshRuns();
      },
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
    cameraPollTimer = setInterval(() => {
      if (!viewedRunId) return;
      cameraFrame.src = Api.cameraLatestUrl(viewedRunId);
    }, 700);
  }

  function stopCameraPolling() {
    if (cameraPollTimer) clearInterval(cameraPollTimer);
    cameraPollTimer = null;
    cameraFrame.hidden = true;
    cameraPlaceholder.hidden = false;
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
