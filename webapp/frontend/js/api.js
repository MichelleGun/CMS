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

const Api = {
  getConfig: () => apiGet("/api/config"),
  getRuns: () => apiGet("/api/runs"),
  getRun: (runId) => apiGet(`/api/runs/${encodeURIComponent(runId)}`),
  startRun: (payload) => apiPost("/api/runs", payload),
  stopRun: (runId) => apiPost(`/api/runs/${encodeURIComponent(runId)}/stop`),
  startReplay: (runId, payload) => apiPost(`/api/runs/${encodeURIComponent(runId)}/replay`, payload),
  getReplayStatus: (runId, replayId, pov) =>
    apiGet(`/api/runs/${encodeURIComponent(runId)}/replay/${encodeURIComponent(replayId)}?pov=${pov ? "true" : "false"}`),
  setCamera: (runId, enabled) => apiPost(`/api/runs/${encodeURIComponent(runId)}/camera`, { enabled }),
  cameraLatestUrl: (runId) => `/api/runs/${encodeURIComponent(runId)}/camera/latest.jpg?t=${Date.now()}`,

  openRunSocket(runId, { onTelemetry, onStatus, onRunEnded, onLog, onError } = {}) {
    const ws = new WebSocket(`${WS_BASE}/ws/runs/${encodeURIComponent(runId)}`);
    ws.onmessage = (evt) => {
      let msg;
      try { msg = JSON.parse(evt.data); } catch (e) { return; }
      if (msg.type === "telemetry" && onTelemetry) onTelemetry(msg);
      else if (msg.type === "status" && onStatus) onStatus(msg);
      else if (msg.type === "run_ended" && onRunEnded) onRunEnded(msg);
      else if (msg.type === "log" && onLog) onLog(msg);
    };
    if (onError) ws.onerror = onError;
    return ws;
  },

  // --- real drone. There is deliberately no "take off" call: the backend only
  // accepts dry runs, and real flights are started from a terminal. Everything
  // callable here either reads state or makes the drone safer.
  getDroneStatus: () => apiGet("/api/drone/status"),
  getDroneFlights: () => apiGet("/api/drone/flights"),
  getDroneFlight: (id) => apiGet(`/api/drone/flights/${encodeURIComponent(id)}`),
  startDryRun: (payload) => apiPost("/api/drone/flights", payload),
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
