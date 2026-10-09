// The History tab: every flight (real or dry), every simulator training run and every
// imitation job, with a detail view. Read-only; everything comes from the folders the
// scripts already write, so nothing here needs to be "saved".
const History = (() => {
  let charts = [];
  let selected = null;
  let el = {};

  const $ = (id) => document.getElementById(id);
  function node(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = text;
    return n;
  }
  function filterValue() { return el.filter.querySelector("input:checked").value; }
  function when(iso) { return iso ? iso.replace("T", " ").slice(0, 16) : ""; }

  function clearCharts() { charts.forEach((c) => c.destroy()); charts = []; }

  function lineChart(canvas, series, xLabel) {
    const colours = ["#ff7130", "#872b97", "#3cc6ff", "#ff3c68", "#8be28b", "#e2d48b"];
    charts.push(new Chart(canvas, {
      type: "line",
      data: {
        datasets: series.map((s, i) => ({
          label: s.label, data: s.data, borderColor: colours[i % colours.length],
          backgroundColor: colours[i % colours.length], pointRadius: 0, tension: 0.1,
          yAxisID: s.axis || "y",
        })),
      },
      options: {
        animation: false, parsing: false, responsive: true, maintainAspectRatio: false,
        scales: {
          x: { type: "linear", title: { display: true, text: xLabel, color: "#9494ab" },
               ticks: { color: "#9494ab" }, grid: { color: "#2a2a3c" } },
          y: { ticks: { color: "#9494ab" }, grid: { color: "#2a2a3c" } },
          y2: { position: "right", display: series.some((s) => s.axis === "y2"),
                ticks: { color: "#9494ab" }, grid: { display: false } },
        },
        plugins: { legend: { labels: { color: "#eceef5" } } },
      },
    }));
  }

  function chartBox(title) {
    const wrap = node("div", "chart-box");
    wrap.append(node("h3", "", title));
    const holder = node("div", "chart-holder");
    const canvas = document.createElement("canvas");
    holder.append(canvas);
    wrap.append(holder);
    return { wrap, canvas };
  }

  function kv(pairs) {
    const t = node("div", "kv");
    pairs.filter(([, v]) => v !== undefined && v !== null && v !== "").forEach(([k, v]) => {
      t.append(node("span", "k", k), node("span", "v", String(v)));
    });
    return t;
  }

  // ------------------------------------------------------------------ list

  function row(title, sub, tags, onClick, key) {
    const r = node("div", "list-row clickable" + (key === selected ? " selected" : ""));
    const body = node("div", "row-body");
    const top = node("div", "row-top");
    top.append(node("strong", "", title));
    (tags || []).filter(Boolean).forEach(([text, cls]) => top.append(node("span", "tag " + (cls || ""), text)));
    body.append(top, node("div", "note", sub));
    r.append(body);
    r.addEventListener("click", () => { selected = key; onClick(); renderList(); });
    return r;
  }

  let cache = { flights: [], runs: [], jobs: [] };

  function outcomeTag(f) {
    const ended = ["crashed", "refused", "killed", "interrupted"];
    const o = ended.includes(f.status) ? f.status
      : [f.sim_outcome, f.outcome, f.status].find((v) => v && v !== "unknown") || "unknown";
    const good = ["passed", "human_landed", "ok", "landed"].includes(o);
    const bad = ["struck", "crashed", "killed", "refused", "missed"].includes(o);
    return [o, good ? "tag-ok" : bad ? "tag-bad" : ""];
  }

  function renderList() {
    const f = filterValue();
    el.list.replaceChildren();
    let items = [];
    if (f === "real" || f === "dry") {
      items = cache.flights.filter((x) => (f === "dry") === (x.dry_run !== false)).map((x) =>
        row(when(x.created_at) || x.flight_id, `${x.mode || "?"} · travel ${x.travel_m ?? "-"} m`,
            [outcomeTag(x), x.demo && ["demo: " + x.demo, x.demo === "pass" ? "tag-ok" : "tag-bad"]],
            () => showFlight(x.flight_id), "f:" + x.flight_id));
    } else if (f === "runs") {
      items = [...cache.runs].reverse().map((r) =>
        row(`${(r.algo || "").toUpperCase()} ${when(r.created_at)}`,
            `${r.current_timestep ?? 0} / ${r.until_stopped ? "\u221e" : (r.total_timesteps ?? "?")} steps · best ${
              r.best_eval_reward === null || r.best_eval_reward === undefined ? "-" : r.best_eval_reward.toFixed(1)}`,
            [[r.status, r.status === "completed" ? "tag-ok" : ""]],
            () => showRun(r.run_id), "r:" + r.run_id));
    } else {
      items = cache.jobs.map((j) =>
        row(j.label || j.kind, when(j.started_at),
            [[j.status, j.status === "done" ? "tag-ok" : j.status === "failed" ? "tag-bad" : ""]],
            () => showJob(j.job_id), "j:" + j.job_id));
    }
    if (!items.length) el.list.append(node("div", "note", "nothing here yet"));
    items.forEach((i) => el.list.append(i));
  }

  async function load() {
    const [flights, runs, jobs] = await Promise.all([
      Api.getDroneFlights(0).catch(() => ({ flights: [] })),
      Api.getRuns().catch(() => ({ runs: [] })),
      Api.getJobs(100).catch(() => ({ jobs: [] })),
    ]);
    cache = { flights: flights.flights, runs: runs.runs, jobs: jobs.jobs };
    renderList();
  }

  // ------------------------------------------------------------------ details

  function detail(title) {
    clearCharts();
    el.detail.replaceChildren(node("h2", "", title));
    return el.detail;
  }

  async function showFlight(id) {
    const d = detail(id);
    const [f, ticks, events] = await Promise.all([
      Api.getDroneFlight(id), Api.getDroneTicks(id).catch(() => ({ ticks: [] })),
      Api.getDroneEvents(id).catch(() => ({ events: [] })),
    ]);
    const m = f.meta || {}, s = f.summary || {};
    d.append(kv([
      ["mode", m.mode], ["real drone", m.dry_run === false ? "yes" : "no (dry run)"],
      ["started", when(m.created_at)], ["status", m.final_status], ["outcome", s.outcome],
      ["simulated result", s.sim_outcome], ["forward travel", s.travel_m !== undefined ? s.travel_m + " m" : null],
      ["travel budget", m.max_travel_m ? m.max_travel_m + " m" : null], ["policy", m.model],
      ["demonstration", f.demo], ["ticks", ticks.ticks.length],
    ]));
    const t = ticks.ticks;
    if (t.length) {
      const a = chartBox("Ring range, height and forward travel");
      const b = chartBox("Commands sent (−1 … 1)");
      d.append(a.wrap, b.wrap);
      const pts = (fn) => t.filter((r) => fn(r) !== null && fn(r) !== undefined)
        .map((r) => ({ x: r.t, y: fn(r) }));
      lineChart(a.canvas, [
        { label: "range to ring (m)", data: pts((r) => r.range_m) },
        { label: "forward travel (m)", data: pts((r) => r.travel_m) },
        { label: "height (cm)", data: pts((r) => r.h), axis: "y2" },
      ], "seconds airborne");
      const names = ["strafe", "forward", "up", "rotate"];
      lineChart(b.canvas, names.map((n, i) => ({
        label: n, data: pts((r) => (r.action ? r.action[i] : null)) })), "seconds airborne");
    } else {
      d.append(node("div", "note", "no control ticks were recorded (it never took off)"));
    }
    if (f.has_camera_frame) {
      d.append(node("h3", "", "Last camera frame"));
      const img = node("img", "history-frame");
      img.src = Api.droneCameraUrl(id);
      img.alt = "last camera frame";
      d.append(img);
    }
    d.append(node("h3", "", "Events"));
    const ol = node("ol", "event-list");
    events.events.forEach((e) => {
      const extra = Object.entries(e).filter(([k]) => !["utc", "event"].includes(k))
        .map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`).join(" ");
      ol.append(node("li", "", `${(e.utc || "").slice(11, 19)}  ${e.event}  ${extra}`));
    });
    d.append(ol);
  }

  async function showRun(id) {
    const d = detail(id);
    const r = await Api.getRun(id);
    const m = r.meta || {}, ls = r.live_status || {};
    d.append(kv([
      ["algorithm", (m.algo || "").toUpperCase()], ["status", m.final_status || ls.status],
      ["started", when(m.created_at)], ["steps", m.final_timesteps || ls.current_timestep],
      ["requested", m.until_stopped ? "until stopped" : m.timesteps_requested], ["best eval reward", ls.best_eval_reward],
    ]));
    if (r.curve && r.curve.timesteps.length) {
      const c = chartBox("Evaluation reward");
      d.append(c.wrap);
      lineChart(c.canvas, [{ label: "mean reward",
        data: r.curve.timesteps.map((x, i) => ({ x, y: r.curve.mean_reward[i] })) }], "timesteps");
    }
    const open = node("button", "secondary-btn", "Open in the Simulator tab (replays)");
    open.type = "button";
    open.addEventListener("click", () => Tabs.show("sim"));
    d.append(open);
  }

  async function showJob(id) {
    const d = detail(id);
    const j = await Api.getJob(id);
    d.append(kv([["what", j.label], ["status", j.status], ["started", when(j.started_at)],
                 ["finished", when(j.finished_at)], ["exit code", j.returncode]]));
    const log = await Api.getJobLog(id, 0);
    d.append(node("pre", "log", log.text || "(empty log)"));
  }

  function init() {
    el = { filter: $("hist-filter"), list: $("hist-list"), detail: $("hist-detail") };
    el.filter.addEventListener("change", renderList);
    Tabs.onChange((name) => { if (name === "history") load().catch(() => {}); });
    if (Tabs.current() === "history") load().catch(() => {});
  }

  return { init, load };
})();
