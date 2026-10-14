// Thin fetch/WebSocket wrappers. Relative URLs everywhere, no hardcoded host,
// so moving the backend elsewhere later only changes what serves this page.
const API_BASE = "";
const WS_BASE = (location.protocol === "https:" ? "wss://" : "ws://") + location.host;

async function apiGet(path) {
  const res = await fetch(API_BASE + path);
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`);
  return res.json();
}

async function apiPost(path, body) {
  const res = await fetch(API_BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let detail = `${res.status}`;
    try { detail = (await res.json()).detail || detail; } catch (e) { /* ignore */ }
    throw new Error(detail);
  }
  return res.json();
}

async function apiPut(path, body) {
  const res = await fetch(API_BASE + path, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = `${res.status}`;
    try { detail = (await res.json()).detail || detail; } catch (e) { /* ignore */ }
    throw new Error(detail);
  }
  return res.json();
}

// How a training run is named everywhere (run list, Start from, History, overlay): its own
// name if it was given one, else its algorithm and start time. `when` is always the date & time.
const RunNames = {
  when(iso) {
    if (!iso) return "";
    const d = new Date(iso);
    if (isNaN(d)) return iso.replace("T", " ").slice(0, 16);
    return d.toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
  },
  title(r) {
    const star = r.starred ? "\u2605 " : "";
    if (r.label) return star + r.label;
    const algo = (r.algo || "run").toUpperCase();
    const w = RunNames.when(r.created_at);
    return star + (w ? `${algo} \u00b7 ${w}` : r.run_id);
  },
  // "best success 40%" for tooltips / dropdowns (null when the run has no success data)
  success(r) {
    return r.best_success_rate === null || r.best_success_rate === undefined ? null
      : `${Math.round(r.best_success_rate * 100)}% success`;
  },
};

const Api = {
  getConfig: () => apiGet("/api/config"),
  getRuns: () => apiGet("/api/runs"),
  getRun: (runId) => apiGet(`/api/runs/${encodeURIComponent(runId)}`),
  startRun: (payload) => apiPost("/api/runs", payload),
  getRunConfig: (runId) => apiGet(`/api/runs/${encodeURIComponent(runId)}/config`),
  previewRunCommand: (payload) => apiPost("/api/runs/command", Object.assign({ env: "onboard", algo: "ppo" }, payload)),
  setRunStarred: (runId, starred) => apiPut(`/api/runs/${encodeURIComponent(runId)}/star`, { starred }),
  setRunLabel: (runId, label) => apiPut(`/api/runs/${encodeURIComponent(runId)}/label`, { label }),
  stopRun: (runId) => apiPost(`/api/runs/${encodeURIComponent(runId)}/stop`),
  startReplay: (runId, payload) => apiPost(`/api/runs/${encodeURIComponent(runId)}/replay`, payload),
  getReplayStatus: (runId, replayId, pov) =>
    apiGet(`/api/runs/${encodeURIComponent(runId)}/replay/${encodeURIComponent(replayId)}?pov=${pov ? "true" : "false"}`),
  setCamera: (runId, enabled) => apiPost(`/api/runs/${encodeURIComponent(runId)}/camera`, { enabled }),
  cameraLatestUrl: (runId) => `/api/runs/${encodeURIComponent(runId)}/camera/latest.jpg?t=${Date.now()}`,
  setRealtime: (runId, enabled) => apiPost(`/api/runs/${encodeURIComponent(runId)}/realtime`, { enabled }),
  startShowcase: (runId, payload) => apiPost(`/api/runs/${encodeURIComponent(runId)}/showcase`, payload || {}),
  showcaseCamUrl: (runId) => `/api/runs/${encodeURIComponent(runId)}/showcase/cam.jpg?t=${Date.now()}`,

  openRunSocket(runId, { onTelemetry, onStatus, onRunEnded, onLog, onShowcase, onError, onAny, onClose } = {}) {
    const ws = new WebSocket(`${WS_BASE}/ws/runs/${encodeURIComponent(runId)}`);
    ws.onmessage = (evt) => {
      let msg;
      try { msg = JSON.parse(evt.data); } catch (e) { return; }
      if (onAny) onAny(msg);
      if (msg.type === "telemetry" && onTelemetry) onTelemetry(msg);
      else if (msg.type === "status" && onStatus) onStatus(msg);
      else if (msg.type === "run_ended" && onRunEnded) onRunEnded(msg);
      else if (msg.type === "log" && onLog) onLog(msg);
      else if (msg.type === "showcase" && onShowcase) onShowcase(msg);
    };
    if (onError) ws.onerror = onError;
    if (onClose) ws.onclose = onClose;
    return ws;
  },

  // --- real drone. A real takeoff needs confirm:true and is accepted only from
  // this computer (backend main.py); land and kill are always accepted.
  getDroneStatus: () => apiGet("/api/drone/status"),
  getDroneFlights: (limit = 25) => apiGet("/api/drone/flights?limit=" + limit),
  startFlight: (payload) => apiPost("/api/drone/flights", payload),
  getDroneTicks: (id) => apiGet(`/api/drone/flights/${encodeURIComponent(id)}/ticks`),
  getDroneEvents: (id) => apiGet(`/api/drone/flights/${encodeURIComponent(id)}/events`),
  answerSave: (id, save, passed) =>
    apiPost(`/api/drone/flights/${encodeURIComponent(id)}/save`, { save, passed }),
  openKeysSocket: (id) =>
    new WebSocket(`${WS_BASE}/ws/drone/${encodeURIComponent(id)}/keys`),

  // --- imitation learning and jobs
  getDemos: () => apiGet("/api/imitation/demos"),
  generateDemos: (payload) => apiPost("/api/imitation/demos/generate", payload),
  teacherDemos: (payload) => apiPost("/api/imitation/demos/teacher", payload),
  labelDemo: (demo_id, passed) => apiPost("/api/imitation/demos/label", { demo_id, passed }),
  getModels: () => apiGet("/api/imitation/models"),
  trainClone: (payload) => apiPost("/api/imitation/train", payload),
  evaluateClone: (name, payload) =>
    apiPost(`/api/imitation/models/${encodeURIComponent(name)}/evaluate`, payload),
  getJobs: (limit = 50) => apiGet("/api/jobs?limit=" + limit),
  getJob: (id) => apiGet(`/api/jobs/${encodeURIComponent(id)}`),
  getJobLog: (id, offset) => apiGet(`/api/jobs/${encodeURIComponent(id)}/log?offset=${offset || 0}`),
  stopJob: (id) => apiPost(`/api/jobs/${encodeURIComponent(id)}/stop`),

  getDroneFlight: (id) => apiGet(`/api/drone/flights/${encodeURIComponent(id)}`),
  landDrone: (id) => apiPost(`/api/drone/flights/${encodeURIComponent(id)}/land`),
  killDrone: (id) => apiPost(`/api/drone/flights/${encodeURIComponent(id)}/kill`),
  stopDroneFlight: (id) => apiPost(`/api/drone/flights/${encodeURIComponent(id)}/stop`),
  setDroneCamera: (id, enabled) =>
    apiPost(`/api/drone/flights/${encodeURIComponent(id)}/camera`, { enabled }),
  droneCameraUrl: (id) =>
    `/api/drone/flights/${encodeURIComponent(id)}/camera/latest.jpg?t=${Date.now()}`,

  openDroneSocket(flightId, { onTick, onStatus, onEvent, onEnded, onError } = {}) {
    const ws = new WebSocket(`${WS_BASE}/ws/drone/${encodeURIComponent(flightId)}`);
    ws.onmessage = (evt) => {
      let msg;
      try { msg = JSON.parse(evt.data); } catch (e) { return; }
      if (msg.type === "drone_tick" && onTick) onTick(msg);
      else if (msg.type === "drone_status" && onStatus) onStatus(msg);
      else if (msg.type === "drone_event" && onEvent) onEvent(msg);
      else if (msg.type === "drone_ended" && onEnded) onEnded(msg);
    };
    if (onError) ws.onerror = onError;
    return ws;
  },
};
