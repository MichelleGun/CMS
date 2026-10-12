// Wires api.js, scene3d.js, controls.js and chart.js together: run history,
// start/stop, live telemetry, the reward chart, the camera toggle and the
// replay panel.
(function () {
  let config = null;
  let runs = [];
  let currentRunId = null;
  let bestRunId = null;
  let bestRunIds = {};          // the best run per environment (net, onboard): most passes per episode
  const isBestRun = (r) => !!r && Object.values(bestRunIds).includes(r.run_id);
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
  const envSelect = document.getElementById("env-select");
  const netOptions = document.getElementById("net-options");
  const hudSource = document.getElementById("hud-source");
  const hudLine = document.getElementById("hud-line");
  const showcaseBtn = document.getElementById("showcase-btn");
  const showcaseStatus = document.getElementById("showcase-status");
  const povSource = document.getElementById("pov-source");
  const realtimeToggle = document.getElementById("realtime-toggle");
  const initSelect = document.getElementById("init-select");
  const onboardOptions = document.getElementById("onboard-options");
  const onboardDemoList = document.getElementById("onboard-demo-list");
  const teacherRunSelect = document.getElementById("teacher-run-select");
  const teacherNote = document.getElementById("teacher-note");
  const capacityNote = document.getElementById("capacity-note");
  let capacity = null;          // {live_runs, max_runs, cpu_used, cpu_budget} from GET /api/runs
  let onboardModels = [];       // clones + onboard RL runs (GET /api/imitation/models)

  // Live far-view state for the run in view
  const live = { lastMsgAt: 0, simT: 0, ep: null, epStart: null, episodes: 0, passes: 0, lastT: 0, evalSuccess: null,
                 showcase: false, showcaseTimer: null, showcaseTarget: null };

  // Load a camera frame off-screen and swap it in only when complete; a frame that fails
  // (mid-write, not there yet) just keeps the previous one on screen.
  const frameLoading = new WeakMap();
  function swapFrame(img, url) {
    if (frameLoading.get(img)) return;            // previous request still in flight
    const pre = new Image();
    frameLoading.set(img, true);
    pre.onload = () => { img.src = pre.src; frameLoading.set(img, false); };
    pre.onerror = () => { frameLoading.set(img, false); };
    pre.src = url;
  }

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

  function untilStopped() {
    const el = document.querySelector('input[name="duration"]:checked');
    return !!el && el.value === "until";
  }

  function applyDuration() {
    const until = untilStopped();
    document.getElementById("until-note").hidden = !until;
    const ts = document.getElementById("timesteps-input");
    if (until) ts.disabled = true;
    else if (!startBtn.disabled) ts.disabled = false;
  }

  function stepsText(r) {
    const n = (r.current_timestep || 0).toLocaleString();
    if (r.until_stopped) {
      const h = r.elapsed_sec ? ` (${(r.elapsed_sec / 3600).toFixed(1)} h)` : "";
      return `${n} / \u221e${h}`;
    }
    return `${n} / ${r.total_timesteps === null || r.total_timesteps === undefined ? "?" : r.total_timesteps.toLocaleString()}`;
  }

  // Up to capacity.max_runs runs train side by side, so the Training panel stays usable while
  // one runs; the backend also refuses a run that would need more cores than are free.
  function atCapacity() { return !!capacity && capacity.live_runs >= capacity.max_runs; }

  function applyPanelState() {
    Controls.setPanelActive(uiMode === "setup", !atCapacity() && uiMode !== "starting");
    startBtn.disabled = atCapacity() || uiMode === "starting";
    applyDuration();
    renderCapacity();
  }

  function renderCapacity() {
    if (uiMode === "starting") {
      capacityNote.className = "note";
      capacityNote.textContent = "loading the run\u2026 (the Training panel unlocks once it is shown)";
      return;
    }
    if (!capacity) { capacityNote.textContent = ""; return; }
    capacityNote.className = "note" + (atCapacity() ? " bad" : "");
    capacityNote.textContent =
      `${capacity.live_runs} of ${capacity.max_runs} runs training · ` +
      `${capacity.cpu_used} of ${capacity.cpu_budget} cores in use` +
      (atCapacity() ? " — stop one to start another" : "");
  }

  // ------------------------------------------------- start-from / onboard options

  function fillSelect(sel, items, keep) {
    // the run list is re-polled every 2.5 s: leave an unchanged dropdown alone, since rebuilding
    // it closes the list if it happens to be open
    const same = sel.options.length === items.length &&
      items.every(([v, l], i) => sel.options[i].value === v && sel.options[i].textContent === l);
    if (same) return;
    sel.replaceChildren();
    items.forEach(([value, label]) => {
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = label;
      sel.appendChild(opt);
    });
    if (keep !== undefined && items.some(([v]) => v === keep)) sel.value = keep;
  }

  function runLabel(r) {
    const best = r.best_eval_reward !== null && r.best_eval_reward !== undefined
      ? `best ${r.best_eval_reward.toFixed(1)}` : "no eval yet";
    const succ = RunNames.success(r);
    const when = r.label ? ` \u00b7 ${RunNames.when(r.created_at)}` : "";
    const tag = isBestRun(r) ? "BEST \u00b7 " : "";
    return `${tag}${RunNames.title(r)}${when} (${(r.current_timestep || 0).toLocaleString()} steps, ` +
      `${best}${succ ? ", " + succ : ""})`;
  }

  // newest first, but the best run and the starred ones on top, so the keepers are easy to pick
  function keepersFirst(list) {
    const rank = (r) => (isBestRun(r) ? 0 : r.starred ? 1 : 2);
    return list.slice().reverse().map((r, i) => [r, i])
      .sort((a, b) => rank(a[0]) - rank(b[0]) || a[1] - b[1]).map(([r]) => r);
  }

  async function toggleStar(r, ev) {
    if (ev) ev.stopPropagation();          // the star sits on a clickable run row
    try { await Api.setRunStarred(r.run_id, !r.starred); }
    catch (e) { alert("could not star: " + e.message); return; }
    await refreshRuns().catch(() => {});
  }

  function renderInitOptions() {
    const env = envSelect.value;
    const items = [["", "a new policy"]];
    if (env === "onboard") {
      onboardModels.filter((m) => m.kind === "clone" && m.flyable).forEach((m) => {
        const ev = m.train_eval;
        items.push([`clone:${m.name}`, `clone ${m.name}` +
          (ev && ev.pass_rate !== undefined && ev.pass_rate !== null ? ` (${Math.round(ev.pass_rate * 100)}% in sim)` : "")]);
      });
    }
    keepersFirst(runs)
      .filter((r) => r.has_model && (r.env || "legacy") === env &&
                     (env === "onboard" || r.algo === algoSelect.value))
      .forEach((r) => items.push([`run:${r.run_id}`, runLabel(r)]));
    fillSelect(initSelect, items, initSelect.value);
    // teacher candidates: PyBullet net runs with continuous actions
    const teachers = keepersFirst(runs)
      .filter((r) => r.has_model && r.env === "net" && r.algo !== "dqn")
      .map((r) => [r.run_id, runLabel(r)]);
    fillSelect(teacherRunSelect, teachers.length ? teachers : [["", "no PyBullet net run with a model"]],
               teacherRunSelect.value);
  }

  async function loadOnboardLists() {
    try {
      const [{ models }, { demos }] = await Promise.all([Api.getModels(), Api.getDemos()]);
      onboardModels = models;
      const usable = demos.filter((d) => d.task === "onboard_hoop" && !d.error);
      onboardDemoList.replaceChildren();
      if (!usable.length) onboardDemoList.textContent = "no onboard-task demos yet";
      usable.forEach((d) => {
        const row = document.createElement("label");
        row.className = "checkbox-row";
        const box = document.createElement("input");
        box.type = "checkbox";
        box.value = d.id;
        // your passed flights and teacher demos are ticked by default; misses never are
        box.checked = (d.source === "human" && d.label === "pass") || d.source === "sim_teacher";
        row.append(box, ` ${d.id} (${d.source || "?"}, ${d.samples} samples)`);
        onboardDemoList.appendChild(row);
      });
    } catch (e) {
      onboardDemoList.textContent = "could not load demos: " + e.message;
    }
    renderInitOptions();
  }

  function applyEnvChoice() {
    const env = envSelect.value;
    netOptions.hidden = env !== "net";
    onboardOptions.hidden = env !== "onboard";
    document.getElementById("algo-block").hidden = env === "onboard";
    if (env !== "onboard") Controls.setEnvMode(env);
    if (env === "onboard") loadOnboardLists(); else renderInitOptions();
  }

  async function makeTeacherDemos() {
    const runId = teacherRunSelect.value;
    if (!runId) return;
    const name = ("teacher_" + runId).replace(/[^A-Za-z0-9_-]/g, "_").slice(0, 40);
    teacherNote.className = "note";
    teacherNote.textContent = "starting…";
    try {
      const { job_id } = await Api.teacherDemos({ run_id: runId, name, episodes: 150 });
      teacherNote.textContent = `flying ${runId} as the teacher (150 episodes)…`;
      const timer = setInterval(async () => {
        const job = await Api.getJob(job_id).catch(() => null);
        if (!job || job.status === "running") return;
        clearInterval(timer);
        teacherNote.className = "note" + (job.status === "done" ? "" : " bad");
        teacherNote.textContent = job.status === "done"
          ? `saved demos/${name}.npz (see the job log in the Imitation tab for its pass rate)`
          : "failed: no episode passed, or see the job log in the Imitation tab";
        loadOnboardLists();
      }, 3000);
    } catch (e) {
      teacherNote.className = "note bad";
      teacherNote.textContent = "could not start: " + e.message;
    }
  }

  function setUiMode(mode) {
    uiMode = mode;
    runScrim.classList.toggle("visible", mode === "starting");
    sceneCanvas.classList.toggle("dimmed", mode === "starting");
    runLoading.hidden = mode !== "starting";
    runOverlay.hidden = mode !== "run";
    // the run overlay's right column covers ~340 px of the viewport: keep the cage in the clear part
    Scene3D.setViewShift(mode === "run" ? 170 : 0);
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
    closeLiveSocket();
    stopCameraPolling();
    cameraToggle.checked = false;
    endShowcaseView();
    Scene3D.clearTrail();
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

  // The viewed run's own best policy, flown on a fresh random ring/start/wind each time.
  function requestOverlayBestReplay() {
    const runId = viewedRunId;
    if (!runId) return;
    if (overlayBestReplayRunId === runId && (!overlayReplayVideo.hidden || overlayReplayPollTimer)) return;
    overlayBestReplayRunId = runId;
    overlayReplayStatus.textContent = "rendering...";
    overlayReplayVideo.hidden = true;
    Api.startReplay(runId, { variant: "best", pov: false, episodes: 1 }).then((res) => {
      pollOverlayReplay(runId, res.replay_id);
    }).catch((e) => {
      overlayBestReplayRunId = null;             // let a later click try again
      overlayReplayStatus.textContent = "failed: " + e.message;
    });
  }

  function pollOverlayReplay(runId, replayId) {
    if (overlayReplayPollTimer) clearInterval(overlayReplayPollTimer);
    overlayReplayPollTimer = setInterval(async () => {
      if (runId !== viewedRunId) {               // another run is in view now
        clearInterval(overlayReplayPollTimer);
        overlayReplayPollTimer = null;
        return;
      }
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

  // ------------------------------------------------------- run name + rename

  let renaming = false;

  function renderOverlayTitle(r) {
    if (renaming) return;
    const listed = runs.find((x) => x.run_id === r.run_id);
    if (listed) r = Object.assign({}, r, { starred: listed.starred, label: listed.label });
    const star = document.getElementById("star-btn");
    star.textContent = r.starred ? "\u2605" : "\u2606";
    star.classList.toggle("on", !!r.starred);
    star.title = r.starred ? "Unstar this run" : "Star this run (a keeper)";
    document.getElementById("overlay-best-tag").hidden = !isBestRun(r);
    document.getElementById("overlay-run-name").textContent = RunNames.title(Object.assign({}, r, { starred: false }));
    document.getElementById("overlay-run-when").textContent =
      (r.label ? RunNames.when(r.created_at) + " \u00b7 " : "") + (r.run_id || "");
  }

  function startRename() {
    if (!viewedRunId || renaming) return;
    const runId = viewedRunId;
    const nameEl = document.getElementById("overlay-run-name");
    const run = runs.find((r) => r.run_id === runId) || {};
    const input = document.createElement("input");
    input.type = "text";
    input.maxLength = 60;
    input.value = run.label || "";
    input.placeholder = "name this run (empty = date & time)";
    renaming = true;
    nameEl.replaceChildren(input);
    input.focus();
    let done = false;
    const finish = async (save) => {
      if (done) return;
      done = true;
      renaming = false;
      if (save) {
        try { await Api.setRunLabel(runId, input.value.trim()); }
        catch (e) { alert("could not rename: " + e.message); }
      }
      nameEl.textContent = "";
      await refreshRuns().catch(() => {});
      const r = runs.find((x) => x.run_id === runId);
      if (r) renderOverlayTitle(r);
    };
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") finish(true);
      else if (e.key === "Escape") finish(false);
    });
    input.addEventListener("blur", () => finish(true));
  }

  // Start from an earlier run: show (and keep) that run's reward, so continuing it doesn't
  // silently change the MDP under the critic.
  async function onInitChoice() {
    const v = initSelect.value;
    if (!v.startsWith("run:")) { Reward.setParent(null); return; }
    try {
      const r = await Api.getRun(v.slice(4));
      Reward.setParent(r.meta);
    } catch (e) { Reward.setParent(null); }
  }

  // ----------------------------------------------------------------- runs

  async function refreshRuns() {
    const data = await Api.getRuns();
    runs = data.runs;
    currentRunId = data.current_run_id;
    bestRunId = data.best_run_id;
    bestRunIds = data.best_run_ids || { net: data.best_run_id };
    capacity = data.capacity || null;
    renderRunList();
    if (document.activeElement !== initSelect && document.activeElement !== teacherRunSelect) {
      renderInitOptions();
    }

    stopBtn.disabled = !(viewedRunId && runs.find((r) => r.run_id === viewedRunId && r.status === "training"));
    const viewed = runs.find((r) => r.run_id === viewedRunId);
    viewBestBtn.disabled = !(viewed && viewed.has_model) || (viewed && viewed.env === "onboard");
    viewBestBtn.title = viewed && viewed.env === "onboard"
      ? "Replays are PyBullet-only: an onboard-camera run has no replay video"
      : "A replay of this run's best saved policy (a new random ring and start each time)";
    if (viewed && !renaming) renderOverlayTitle(viewed);
    applyPanelState();

    // On load, jump into a run only if one is actually training; otherwise stay in setup so
    // the Training panel is ready to use (finished runs are one click away in the list).
    const trainingRun = runs.find((r) => r.status === "training");
    if (!viewedRunId && trainingRun && autoSelectEnabled) selectRun(trainingRun.run_id);
    const bestForView = (viewed && bestRunIds[viewed.env]) || bestRunId;
    if (bestForView) {
      Api.getRun(bestForView).then((r) => {
        if (r.curve) RewardChart.setBestCurve(r.curve.timesteps, r.curve.mean_reward);
      }).catch(() => {});
    }
  }

  function renderRunList() {
    runListEl.innerHTML = "";
    runs.slice().reverse().forEach((r) => {
      const row = document.createElement("div");
      const isBest = isBestRun(r);
      row.className = "run-row" +
        (r.run_id === viewedRunId ? " selected" : "") +
        (r.status === "training" ? " active-training" : "") +
        (isBest ? " best-run" : "") + (r.starred ? " starred" : "");
      const star = document.createElement("button");
      star.type = "button";
      star.className = "icon-btn star-btn" + (r.starred ? " on" : "");
      star.textContent = r.starred ? "\u2605" : "\u2606";
      star.title = r.starred ? "Unstar" : "Star this run (a keeper: listed first in Start from)";
      star.addEventListener("click", (e) => toggleStar(r, e));
      row.append(star);
      if (isBest) {
        const tag = document.createElement("span");
        tag.className = "tag-best";
        tag.textContent = "BEST";
        tag.title = "Most passes per evaluation episode for this environment" +
          (r.best_mean_passes !== null && r.best_mean_passes !== undefined
            ? ` (${r.best_mean_passes.toFixed(2)})` : "") + (RunNames.success(r) ? `, ${RunNames.success(r)}` : "");
        row.append(tag);
      }
      const best = r.best_eval_reward !== null && r.best_eval_reward !== undefined
        ? r.best_eval_reward.toFixed(1) : "-";
      const title = document.createElement("strong");
      title.textContent = r.label || RunNames.title(Object.assign({}, r, { starred: false }));
      row.append(title, document.createElement("br"));
      if (r.label) {
        const when = document.createElement("span");
        when.className = "run-when";
        when.textContent = RunNames.when(r.created_at);
        row.append(when, document.createElement("br"));
      }
      row.insertAdjacentHTML("beforeend", `<span class="algo-badge">${r.algo}</span> <span class="note">${r.env || ""}</span><br>` +
        `${r.status} &middot; ${stepsText(r)}<br>` +
        `best ${best}` + (RunNames.success(r) ? ` &middot; ${RunNames.success(r)}` : ""));
      row.title = r.run_id;
      row.addEventListener("click", () => selectRun(r.run_id));
      runListEl.appendChild(row);
    });
  }

  // 80 retries at 500ms = 40s: importing torch/pybullet in the training
  // subprocess before it writes its first meta.json can genuinely take
  // several seconds (observed ~5-8s cold start), so this needs real slack,
  // not just a couple of quick retries.
  const SELECT_RUN_MAX_RETRIES = 80;

  function selectRun(runId, retriesLeft = SELECT_RUN_MAX_RETRIES) {
    if (viewedRunId === runId && retriesLeft === SELECT_RUN_MAX_RETRIES) return;
    if (retriesLeft === SELECT_RUN_MAX_RETRIES) {
      viewedRunId = runId;
      renderRunList();
      closeLiveSocket();
      RewardChart.clearCurrent();
      clearTerminal();
      resetLiveView();
      replayStatusEl.textContent = "";
      replayVideo.hidden = true;
      replayViewToggle.hidden = true;
      overlayBestReplayRunId = null;
      overlayReplayVideo.hidden = true;
      overlayReplayStatus.textContent = "";
      setUiMode("starting");
    }

    Api.getRun(runId).then((r) => {
      const status = r.status || (r.meta.final_status) || (r.live_status && r.live_status.status) || "unknown";
      isViewedRunTraining = status === "training";
      setStatusPill(status);
      if (r.curve) RewardChart.setCurrentCurve(r.curve.timesteps, r.curve.mean_reward);
      RewardChart.setPasses(r.passes);
      const rw = r.meta.reward_weights || {};
      live.streakGain = rw.budget_streak !== undefined ? rw.budget_streak
        : (r.meta.streak !== undefined ? r.meta.streak : 0);
      if (r.meta.target_override) Scene3D.setHoop(r.meta.target_override);
      renderOverlayTitle(Object.assign({}, r.meta, { label: r.label, starred: r.starred }));
      Reward.renderCard(r.meta);
      live.swayMax = r.meta.sway_deg === null ? undefined : r.meta.sway_deg;
      openLiveSocket(runId);              // also carries showcase episodes for finished runs
      realtimeToggle.checked = false;
      if (isViewedRunTraining) Api.setRealtime(runId, false).catch(() => {});
      live.evalEvery = r.meta.eval_every_timesteps || null;
      const onboard = r.meta.env === "onboard";
      if (isViewedRunTraining && (r.meta.env === "net" || onboard)) {
        // the drone POV is what you come here to see: turn it on for live net / onboard runs
        cameraToggle.checked = true;
        onCameraToggle();
      }
      showcaseStatus.className = "note";
      showcaseBtn.disabled = onboard;
      showcaseStatus.textContent = onboard
        ? "Onboard-camera runs train in their own simulator (not PyBullet), so there is no policy episode " +
          "or replay video here: the far view and drone POV above are the training env live. " +
          "Fly the saved policy from the Drone tab (dry run first)."
        : isViewedRunTraining
        ? "Plays one real-time episode with what the policy has learned so far (no exploration). Available after the first evaluation."
        : "Plays one real-time episode of this run's latest saved policy.";
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

  // The far view depends on this one socket for hours, so it must survive a dropped connection
  // or a backend restart: reconnect while the same run is in view, and reopen it if it goes
  // quiet for longer than an evaluation pause. The drone POV is polled separately over HTTP.
  const sock = { wanted: null, lastMsgAt: 0, retryMs: 1000, timer: null };

  function closeLiveSocket() {
    sock.wanted = null;
    if (sock.timer) { clearTimeout(sock.timer); sock.timer = null; }
    if (ws) { ws.onclose = null; ws.close(); ws = null; }
  }

  function openLiveSocket(runId) {
    if (ws) { ws.onclose = null; ws.close(); ws = null; }
    sock.wanted = runId;
    sock.lastMsgAt = Date.now();
    ws = Api.openRunSocket(runId, {
      onAny: () => { sock.lastMsgAt = Date.now(); sock.retryMs = 1000; },
      onClose: () => {
        ws = null;
        if (sock.wanted !== runId) return;              // closed on purpose
        if (uiMode === "run" && !live.showcase) hudLine.textContent = "reconnecting to the live view\u2026";
        sock.timer = setTimeout(() => { if (sock.wanted === runId) openLiveSocket(runId); }, sock.retryMs);
        sock.retryMs = Math.min(5000, sock.retryMs * 2);
      },
      onTelemetry: onTrainTelemetry,
      onShowcase: onShowcaseMsg,
      onStatus: (msg) => {
        setStatusPill("training");
        if (msg.latest_success_rate !== undefined && msg.latest_success_rate !== null) {
          live.evalSuccess = msg.latest_success_rate;
          renderHud();
        }
        RewardChart.appendCurrentPoint(msg.current_timestep, msg.latest_eval_reward);
        RewardChart.appendPassesPoint(msg.current_timestep, msg.latest_mean_passes);
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

  // ------------------------------------------------- live far view + HUD

  function resetLiveView() {
    live.ep = null; live.epStart = null; live.episodes = 0; live.passes = 0;
    live.evalSuccess = null; live.lastT = 0; live.swing = null;
    endShowcaseView();
    Scene3D.clearTrail();
    hudLine.textContent = "waiting for the first episode\u2026";
  }

  function renderHud(extra) {
    if (live.showcase) return;
    const t = live.epStart !== null ? live.simT.toFixed(1) : "-";      // simulated seconds
    const sr = live.evalSuccess !== null ? `${Math.round(live.evalSuccess * 100)}%`
      : live.evalEvery ? `first at ${live.evalEvery.toLocaleString()} steps` : "-";
    hudLine.textContent =
      `episode ${live.ep === null ? "-" : live.ep + 1}  \u00b7  ${t} s  \u00b7  ` +
      `${live.reversed ? "back\u2192front  \u00b7  " : ""}` +
      `passes this episode ${live.epPasses || 0}${streakText(live.epPasses)}  \u00b7  ` +
      (live.zone ? `start: ${live.zone}  \u00b7  ` : "") +
      `passes seen ${live.passes}/${live.episodes}  \u00b7  eval success ${sr}  \u00b7  step ${live.lastT}` +
      (live.swing ? `\n${live.swing}` : "") +
      (extra ? `  \u00b7  ${extra}` : "");
  }

  const ZONE_NAMES = ["top-left", "top-middle", "top-right", "middle-right", "bottom-right",
                      "bottom-middle", "bottom-left", "middle-left", "under the hoop"];

  // the multiplier the NEXT pass will be paid with (streak bonus, capped at 3x)
  function streakText(n) {
    const g = live.streakGain || 0;
    if (!g || !n) return "";
    return ` (next \u00d7${Math.min(1 + g * n, 3).toFixed(1)})`;
  }

  function eventMarks(msg, key) {
    if (msg.net_contact) Scene3D.markEvent(msg.pos, "net", key);
    if (msg.rim_contact) Scene3D.markEvent(msg.pos, "rim", key);
  }

  function onTrainTelemetry(msg) {
    // a record older than one already shown (a replay after a reconnect) would jump the drone back
    if (msg.t !== undefined && msg.t < live.lastT) return;
    live.lastMsgAt = Date.now();
    live.lastT = msg.t || live.lastT;
    if (msg.step !== undefined) live.simT = msg.step / (msg.ctrl_hz || 30);
    if (msg.reversed !== undefined) live.reversed = msg.reversed;
    if (msg.passes !== undefined) live.epPasses = msg.passes;
    if (msg.start_zone) live.zone = ZONE_NAMES[msg.start_zone - 1] || `zone ${msg.start_zone}`;
    if (msg.swing_deg !== undefined) {
      // why the ring may look calm: wind ramp x this episode's gust x the run's "Max swing"
      live.swing = `ring swing this episode ${msg.swing_deg.toFixed(0)}\u00b0` +
        (live.swayMax !== undefined ? ` of ${live.swayMax}\u00b0 max` : "") +
        `  \u00b7  wind ramp ${Math.round((msg.wind_scale || 0) * 100)}%` +
        `  \u00b7  period ${(msg.swing_period_s || 0).toFixed(1)} s` +
        (realtimeToggle.checked ? "" : "  (view runs faster than real time)");
    }
    if (msg.ep !== undefined && msg.ep !== live.ep) {
      if (live.ep !== null) Scene3D.newEpisode("train");
      live.ep = msg.ep;
      live.epStart = Date.now();
    }
    if (live.showcase) return;          // the far view is showing the showcase episode
    Scene3D.setDroneTarget(msg.pos, msg.rpy);
    Scene3D.setHoop(msg.target, msg.hoop_normal, msg.hoop_radius);
    Scene3D.addTrailPoint(msg.pos, "train");
    eventMarks(msg, "train");
    if (msg.done) {
      live.episodes += 1;
      if (msg.success) { live.passes += 1; Scene3D.markEvent(msg.target, "pass", "train"); }
      renderHud(msg.success ? "PASSED THROUGH" : "episode over");
    } else {
      renderHud();
    }
  }

  // ----------------------------------------------------------- showcase

  function beginShowcaseView() {
    live.showcase = true;
    Scene3D.clearTrail("showcase");
    Scene3D.setTrailVisible("train", false);
    Scene3D.setTrailVisible("showcase", true);
    showcaseStatus.className = "note";
    showcaseStatus.textContent = realtimeToggle.checked
      ? "playing one episode in real time\u2026" : "playing one episode at full speed\u2026";
    Scene3D.setDroneColor(Scene3D.droneColors.showcase);
    hudSource.innerHTML = 'Far view &middot; <span class="hud-show">current policy (' +
      (realtimeToggle.checked ? "real time" : "full speed") + ')</span>';
    povSource.textContent = "current policy";
    if (live.showcaseTimer) clearInterval(live.showcaseTimer);
    live.showcaseTimer = setInterval(() => {
      if (!viewedRunId) return;
      overlayCameraFrame.hidden = false;
      overlayCameraPlaceholder.hidden = true;
      swapFrame(overlayCameraFrame, Api.showcaseCamUrl(viewedRunId));
    }, 200);
  }

  function endShowcaseView() {
    if (live.showcaseTimer) clearInterval(live.showcaseTimer);
    live.showcaseTimer = null;
    if (!live.showcase) return;
    live.showcase = false;
    Scene3D.setDroneColor(Scene3D.droneColors.train);
    Scene3D.setTrailVisible("showcase", false);
    Scene3D.setTrailVisible("train", true);
    hudSource.innerHTML = 'Far view &middot; <span class="hud-train">training env (exploring)</span>';
    povSource.textContent = "training env";
    if (!cameraPollTimer) { overlayCameraFrame.hidden = true; overlayCameraPlaceholder.hidden = false; }
    showcaseBtn.disabled = false;
  }

  function onShowcaseMsg(msg) {
    if (msg.summary) {
      const passed = msg.success;
      showcaseStatus.className = passed ? "note ok" : "note bad";
      showcaseStatus.textContent = passed
        ? `Passed through the ring${msg.reversed ? " back to front" : ""} in ${msg.duration_s} s (net bumps: ${msg.net_hits}, rim hits: ${msg.rim_hits}).`
        : `Did not pass (${msg.ended} after ${msg.duration_s} s; net bumps: ${msg.net_hits}, rim hits: ${msg.rim_hits}).`;
      if (passed && live.showcaseTarget) Scene3D.markEvent(live.showcaseTarget, "pass", "showcase");
      hudLine.textContent = showcaseStatus.textContent;
      // keep the result on screen for a moment, then go back to the training view
      setTimeout(endShowcaseView, 4000);
      return;
    }
    if (!live.showcase) beginShowcaseView();
    live.showcaseLastAt = Date.now();
    live.showcaseTarget = msg.target;
    Scene3D.setDroneTarget(msg.pos, msg.rpy);
    Scene3D.setHoop(msg.target, msg.hoop_normal, msg.hoop_radius);
    Scene3D.addTrailPoint(msg.pos, "showcase");
    eventMarks(msg, "showcase");
    hudLine.textContent = `current policy  \u00b7  ${(msg.step / 30).toFixed(1)} s  \u00b7  passes ${msg.passes || 0}`;
  }

  async function startShowcase() {
    if (!viewedRunId) return;
    showcaseBtn.disabled = true;
    showcaseStatus.className = "note";
    showcaseStatus.textContent = "loading the policy (10-30 s while training is running)\u2026";
    try {
      await Api.startShowcase(viewedRunId, { variant: "latest", realtime: realtimeToggle.checked });
      // re-enable if nothing arrives (e.g. the process failed to start)
      setTimeout(() => { if (!live.showcase) showcaseBtn.disabled = false; }, 60000);
    } catch (e) {
      showcaseStatus.className = "note bad";
      showcaseStatus.textContent = "could not start: " + e.message;
      showcaseBtn.disabled = false;
    }
  }

  // ------------------------------------------------------------- training

  async function startTraining() {
    const onboard = envSelect.value === "onboard";
    const algo = onboard ? "ppo" : algoSelect.value;
    const init = initSelect.value;
    const payload = {
      algo,
      init_run: init.startsWith("run:") ? init.slice(4) : null,
      init_clone: init.startsWith("clone:") ? init.slice(6) : null,
      timesteps: parseInt(document.getElementById("timesteps-input").value, 10) || 1000000,
      until_stopped: untilStopped(),
      n_envs: parseInt(document.getElementById("nenvs-input").value, 10) || 4,
      env: envSelect.value,
      ring_jitter: document.getElementById("ring-jitter-toggle").checked ? 1.0 : 0.0,
      radius_start: document.getElementById("radius-curriculum-toggle").checked ? 1.5 : 1.0,
      sway_deg: parseFloat(document.getElementById("train-sway-deg").value),
      wash_deg: parseFloat(document.getElementById("train-wash-deg").value) || 0,
      start_z: document.getElementById("hover-start-toggle").checked ? 0.8 : 0.0,
      same_side: document.getElementById("same-side-toggle").checked,
      start_zones: document.getElementById("start-zones-toggle").checked,
      streak: Math.max(0, parseFloat(document.getElementById("streak-input").value) || 0),
      carry_start: document.getElementById("carry-start-toggle").checked ? 0.7 : 0.0,
      reverse_prob: document.getElementById("reverse-toggle").checked ? 0.5 : 0.0,
      lean_flip_prob: document.getElementById("lean-flip-toggle").checked ? 0.3 : 0.0,
      wind_steps: Math.max(0, parseInt(document.getElementById("wind-steps-input").value, 10) || 0),
    };
    if (envSelect.value === "net") Object.assign(payload, Reward.payload());
    if (onboard) {
      payload.demo_ids = Array.from(onboardDemoList.querySelectorAll("input:checked")).map((b) => b.value);
      payload.human_weight = parseFloat(document.getElementById("human-weight").value) || 0;
      payload.bc_coef = parseFloat(document.getElementById("bc-coef").value) || 0;
      payload.profile = document.getElementById("onboard-profile").value;
      payload.onboard_dr = document.getElementById("onboard-dr-toggle").checked ? 1.0 : 0.0;
      payload.onboard_sway_deg = parseFloat(document.getElementById("onboard-sway").value) || 0;
      payload.onboard_wash_deg = parseFloat(document.getElementById("onboard-wash").value) || 0;
      payload.hard_start_p = Math.min(100, Math.max(0, parseFloat(document.getElementById("hard-start-input").value) || 0)) / 100;
      payload.dr_steps = Math.max(0, parseInt(document.getElementById("dr-steps-input").value, 10) || 0);
      payload.bc_augment = document.getElementById("bc-augment-toggle").checked;
      payload.seed = parseInt(seedInput.value, 10) || 0;       // the onboard sim draws its own layouts
    } else if (randomizeToggle.checked) {
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
    const run = runs.find((r) => r.run_id === viewedRunId);
    if (run && run.until_stopped &&
        !confirm("Stop this until-stopped training run? The policy learned so far is saved.")) return;
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
      swapFrame(cameraFrame, url);
      if (!live.showcase) swapFrame(overlayCameraFrame, url);
    }, 200);
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

  // One failing part (a library that did not load, a server hiccup) must not stop the
  // rest: run each part on its own, and say on screen which one failed.
  const bootFailures = [];

  function safe(name, fn) {
    try { fn(); return true; } catch (e) {
      console.error(`[boot] ${name} failed:`, e);
      bootFailures.push({ name, message: (e && e.message) || String(e) });
      showBootError();
      return false;
    }
  }

  function showBootError() {
    const el = document.getElementById("boot-error");
    if (!el) return;
    const libs = (typeof THREE === "undefined" ? ["three.js"] : [])
      .concat(typeof Chart === "undefined" ? ["Chart.js"] : []);
    const webgl = bootFailures.some((f) => /webgl/i.test(f.message));
    el.textContent = "Part of the dashboard did not start: " +
      bootFailures.map((f) => `${f.name} (${f.message})`).join("; ") + "." +
      (libs.length ? ` ${libs.join(" and ")} did not load.` : "") +
      (webgl
        ? " This browser has WebGL (3D graphics) turned off. Turn on Settings > System > " +
          "\"Use graphics acceleration when available\" (edge://settings/system or chrome://settings/system), " +
          "restart the browser, and check edge://gpu (or chrome://gpu) shows WebGL as hardware accelerated. " +
          "Open the page in Edge or Chrome directly, not in VS Code's preview."
        : " Hard-refresh with Ctrl+F5; the browser console (F12) has the details.");
    el.hidden = false;
  }

  // Replace an object's methods with no-ops, so later calls from the run view don't throw.
  function stubOut(obj) {
    if (!obj) return;
    Object.keys(obj).forEach((k) => { if (typeof obj[k] === "function") obj[k] = () => {}; });
  }

  async function loadConfig() {
    for (;;) {
      try { return await Api.getConfig(); } catch (e) {
        console.warn("[boot] waiting for the server:", e.message);
        await new Promise((r) => setTimeout(r, 2000));
      }
    }
  }

  async function boot() {
    // Tabs first: navigation must work whatever else fails.
    safe("tabs", () => Tabs.init());
    config = await loadConfig();

    algoSelect.innerHTML = "";
    config.algorithms.forEach((a) => {
      const opt = document.createElement("option");
      opt.value = a.id;
      opt.textContent = a.label;
      algoSelect.appendChild(opt);
    });
    algoSelect.addEventListener("change", () => {
      dqnNote.hidden = algoSelect.value !== "dqn";
      renderInitOptions();
    });
    document.getElementById("teacher-btn").addEventListener("click", makeTeacherDemos);

    if (!safe("3D view", () => Scene3D.init(document.getElementById("scene-canvas"), config.room))) {
      stubOut(Scene3D);
      document.getElementById("viewport-hint").textContent =
        "3D view unavailable (see the message at the top).";
    }
    safe("position controls", () => {
      Controls.init(config);
      Controls.onChange(() => {
        updateReadout();
        if (!isViewedRunTraining) showDraftPreview();
      });
      updateReadout();
      showDraftPreview();
    });

    if (!safe("reward chart", () => RewardChart.init(document.getElementById("reward-chart")))) {
      stubOut(RewardChart);
    }
    // The other tabs own their own polling and sockets, so they only need starting.
    safe("Drone tab", () => Drone.init());
    safe("Environment tab", () => EnvView.init());
    safe("Imitation tab", () => Imitation.init());
    safe("History tab", () => History.init());
    // A live flight is visible from every tab, so it is never out of sight.
    const livePill = document.getElementById("live-flight-pill");
    livePill.addEventListener("click", () => Tabs.show("drone"));
    Drone.onLive((isLive, id, real) => {
      livePill.hidden = !isLive;
      livePill.textContent = real ? "REAL flight live" : "dry run live";
      livePill.className = "pill " + (real ? "pill-crashed" : "pill-training");
    });

    startBtn.addEventListener("click", startTraining);
    showcaseBtn.addEventListener("click", startShowcase);
    // Training pauses while the policy is evaluated (10 episodes on a separate env, every 20k
    // steps): say so, rather than letting the far view look frozen.
    setInterval(() => {
      // socket went quiet for longer than any evaluation pause: reopen it
      if (ws && sock.wanted && isViewedRunTraining && Date.now() - sock.lastMsgAt > 45000) {
        sock.lastMsgAt = Date.now();
        ws.close();                                     // onClose reconnects
      }
      // a showcase that stopped sending (its process died) must not hold the far view forever
      if (live.showcase && live.showcaseLastAt && Date.now() - live.showcaseLastAt > 15000) {
        endShowcaseView();
      }
    }, 5000);
    setInterval(() => {
      // (no socket = reconnecting: its own message is on the HUD)
      if (uiMode !== "run" || !isViewedRunTraining || live.showcase || !live.lastMsgAt || !ws) return;
      if (Date.now() - live.lastMsgAt > 2500) {
        hudLine.textContent = `evaluating the current policy (training paused for a few seconds)\u2026  \u00b7  step ${live.lastT}`;
      }
    }, 1000);
    realtimeToggle.addEventListener("change", () => {
      // paces a live training run right away; a showcase picks the setting up when started
      if (viewedRunId && isViewedRunTraining) {
        Api.setRealtime(viewedRunId, realtimeToggle.checked).catch(() => {});
      }
      if (!live.showcase) {
        hudSource.innerHTML = 'Far view &middot; <span class="hud-train">training env (' +
          (realtimeToggle.checked ? "real time" : "full speed") + ', exploring)</span>';
      }
    });
    document.getElementById("overlay-terminal-toggle").addEventListener("click", () => {
      document.getElementById("overlay-terminal-card").classList.toggle("collapsed");
    });
    envSelect.addEventListener("change", applyEnvChoice);
    stopBtn.addEventListener("click", stopTraining);
    document.querySelectorAll('input[name="duration"]').forEach((el) => el.addEventListener("change", applyDuration));
    cameraToggle.addEventListener("change", onCameraToggle);
    viewNewestBtn.addEventListener("click", () => setOverlayView("newest"));
    viewBestBtn.addEventListener("click", () => setOverlayView("best"));
    backToSetupBtn.addEventListener("click", backToSetup);
    document.getElementById("replay-current-btn").addEventListener("click", () => requestReplay("current"));
    document.getElementById("replay-best-btn").addEventListener("click", () => requestReplay("best"));

    safe("reward editor", () => Reward.init(config.reward));
    initSelect.addEventListener("change", onInitChoice);
    document.getElementById("rename-btn").addEventListener("click", startRename);
    const onboardSway = document.getElementById("onboard-sway");
    onboardSway.addEventListener("input", () => {
      document.getElementById("onboard-sway-val").textContent = `${onboardSway.value}°`;
    });
    for (const [inp, val] of [["onboard-wash", "onboard-wash-val"], ["train-wash-deg", "train-wash-val"]]) {
      const el = document.getElementById(inp);
      el.addEventListener("input", () => { document.getElementById(val).textContent = `${el.value}°`; });
    }
    document.getElementById("star-btn").addEventListener("click", () => {
      const r = runs.find((x) => x.run_id === viewedRunId);
      if (r) toggleStar(r);
    });

    randomizeToggle.addEventListener("change", () => {
      seedInput.disabled = !randomizeToggle.checked;
    });

    await refreshRuns().catch(() => {});
    // .catch here too: if the backend is mid-restart when a poll lands, this
    // is a normal transient state, not worth an unhandled-rejection console
    // error every 2.5s until it comes back.
    setInterval(() => refreshRuns().catch(() => {}), 2500);
  }

  // the History tab opens a run here
  window.SimApp = { selectRun: (runId) => { autoSelectEnabled = false; viewedRunId = null; selectRun(runId); } };

  window.addEventListener("DOMContentLoaded", boot);
})();
