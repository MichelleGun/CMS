// The Imitation tab: demonstrations (recorded and simulated), training a clone on the
// ticked ones, evaluating clones, and the running job's log. Jobs run one at a time in
// the backend (jobs.py); this page polls the active one.
const Imitation = (() => {
  const POLL_MS = 1500;
  let demos = [];
  let ticked = new Set();
  let tickedInit = false;
  let jobId = null;
  let logOffset = 0;
  let el = {};

  const $ = (id) => document.getElementById(id);

  function filterValue() { return el.filter.querySelector("input:checked").value; }

  function node(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }

  // ------------------------------------------------------------------ demos

  function renderDemos() {
    const f = filterValue();
    const shown = demos.filter((d) => f === "all" || (f === "real") === !!d.real);
    el.demoList.replaceChildren();
    if (!shown.length) el.demoList.appendChild(node("div", "note", "no demonstrations yet"));
    shown.forEach((d) => {
      const row = node("label", "list-row");
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = ticked.has(d.id);
      cb.disabled = !!d.error || d.task !== "onboard_hoop";
      cb.addEventListener("change", () => {
        if (cb.checked) ticked.add(d.id); else ticked.delete(d.id);
        summarise();
      });
      const body = node("div", "row-body");
      const top = node("div", "row-top");
      const name = d.real ? d.id.split("/")[0].replace("tello_", "") : d.file;
      top.append(node("strong", "", name));
      if (d.label) top.append(node("span", "tag " + (d.label === "pass" ? "tag-ok" : "tag-bad"), d.label));
      top.append(node("span", "tag", d.real ? (d.dry_run ? "dry run" : "real") : "sim"));
      const sub = d.error ? `unreadable: ${d.error}`
        : `${d.samples} samples · ${d.episodes} episode(s) · ${d.seconds} s · ${d.source || ""}`;
      body.append(top, node("div", "note", sub));
      row.append(cb, body);
      if (d.real && d.label) {
        const flip = node("button", "link-btn", d.label === "pass" ? "mark miss" : "mark pass");
        flip.type = "button";
        flip.addEventListener("click", async (e) => {
          e.preventDefault();
          try {
            const r = await Api.labelDemo(d.id, d.label !== "pass");
            if (ticked.delete(d.id)) ticked.add(r.demo_id);
            await loadDemos();
          } catch (err) { el.trainNote.textContent = "relabel failed: " + err.message; }
        });
        row.append(flip);
      }
      el.demoList.appendChild(row);
    });
    summarise();
  }

  function summarise() {
    const sel = demos.filter((d) => ticked.has(d.id));
    const n = sel.reduce((a, d) => a + (d.samples || 0), 0);
    el.demoSummary.textContent = sel.length
      ? `${sel.length} file(s) ticked, ${n} samples (${(n / 15).toFixed(0)} s of flying)`
      : "tick the demonstrations to train on";
  }

  async function loadDemos() {
    const r = await Api.getDemos();
    demos = r.demos;
    const ids = new Set(demos.map((d) => d.id));
    ticked = new Set([...ticked].filter((id) => ids.has(id)));
    if (!tickedInit) {     // default: every real flight that went through
      demos.filter((d) => d.real && d.label === "pass" && !d.dry_run).forEach((d) => ticked.add(d.id));
      tickedInit = true;
    }
    renderDemos();
  }

  // ------------------------------------------------------------------ clones

  function evalText(ev) {
    if (!ev) return "";
    return `${ev.passed}/${ev.episodes} passed${ev.render ? " (real detector)" : ""}`;
  }

  async function loadModels() {
    const { models } = await Api.getModels();
    el.modelList.replaceChildren();
    if (!models.length) el.modelList.appendChild(node("div", "note", "no clones yet"));
    models.forEach((m) => {
      const row = node("div", "list-row col");
      const top = node("div", "row-top");
      top.append(node("strong", "", m.name));
      if (!m.flyable) top.append(node("span", "tag tag-bad", "simulator task"));
      row.append(top);
      const lines = [`${m.samples ?? "?"} samples · ${(m.demos || []).join(", ")}`];
      if (m.train_eval) lines.push("after training: " + evalText(m.train_eval));
      (m.evals || []).slice(0, 3).forEach((ev) => lines.push(`${ev.saved}: ${evalText(ev)}`));
      lines.forEach((t) => row.append(node("div", "note", t)));
      const btns = node("div", "button-row");
      const evalBtn = node("button", "secondary-btn", "Evaluate");
      evalBtn.type = "button";
      evalBtn.addEventListener("click", () => startJob(() => Api.evaluateClone(m.name, {
        episodes: Number(el.evalEpisodes.value) || 12, render: el.evalRender.checked })));
      btns.append(evalBtn);
      if (m.flyable) {
        const fly = node("button", "primary-btn", "Fly this clone");
        fly.type = "button";
        fly.addEventListener("click", () => Drone.flyModel(m.model));
        btns.append(fly);
      }
      row.append(btns);
      el.modelList.appendChild(row);
    });
  }

  // ------------------------------------------------------------------ jobs

  async function startJob(fn) {
    try {
      const r = await fn();
      jobId = r.job_id;
      logOffset = 0;
      el.log.textContent = "";
      el.trainNote.textContent = "";
      await pollJob();
    } catch (e) {
      el.trainNote.textContent = e.message;
    }
  }

  async function pollJob() {
    if (!jobId) {
      const r = await Api.getJobs(1);
      const j = r.active || r.jobs[0];
      if (!j) return;
      jobId = j.job_id;
      logOffset = 0;
      el.log.textContent = "";
    }
    const job = await Api.getJob(jobId);
    const chunk = await Api.getJobLog(jobId, logOffset);
    if (chunk.text) {
      el.log.textContent += chunk.text;
      if (el.log.textContent.length > 60000) el.log.textContent = el.log.textContent.slice(-50000);
      el.log.scrollTop = el.log.scrollHeight;
    }
    logOffset = chunk.offset;
    el.jobLabel.textContent = job.label || job.kind;
    el.jobPill.textContent = job.status;
    el.jobPill.className = "pill " + (job.status === "running" ? "pill-training"
      : job.status === "failed" ? "pill-crashed" : "pill-idle");
    el.jobStop.disabled = job.status !== "running";
    el.trainBtn.disabled = el.genBtn.disabled = job.status === "running";
    if (job.status !== "running" && el.jobPill.dataset.was === "running") {
      loadDemos().catch(() => {});
      loadModels().catch(() => {});
    }
    el.jobPill.dataset.was = job.status;
  }

  // ------------------------------------------------------------------ init

  function init() {
    el = {
      filter: $("demo-filter"), demoList: $("demo-list"), demoSummary: $("demo-summary"),
      genName: $("gen-name"), genEpisodes: $("gen-episodes"), genProfile: $("gen-profile"),
      genRender: $("gen-render"), genBtn: $("gen-btn"),
      trainName: $("train-name"), trainEpochs: $("train-epochs"), trainLr: $("train-lr"),
      trainUse: $("train-use"), trainEval: $("train-eval"), trainBtn: $("train-btn"),
      trainNote: $("train-note"), jobPill: $("job-pill"), jobLabel: $("job-label"),
      jobStop: $("job-stop-btn"), log: $("job-log"),
      evalEpisodes: $("eval-episodes"), evalRender: $("eval-render"), modelList: $("model-list"),
    };
    el.filter.addEventListener("change", renderDemos);
    el.genBtn.addEventListener("click", () => startJob(() => Api.generateDemos({
      name: el.genName.value.trim(), episodes: Number(el.genEpisodes.value) || 60,
      profile: el.genProfile.value, render: el.genRender.checked,
      ...($("gen-robust") && $("gen-robust").checked ? { dr: 1.0, sway_deg: 25, hard_start_p: 0.5 } : {}) })));
    el.trainBtn.addEventListener("click", () => startJob(() => Api.trainClone({
      name: el.trainName.value.trim(), demo_ids: [...ticked],
      epochs: Number(el.trainEpochs.value) || 200, lr: Number(el.trainLr.value) || 0.003,
      use: el.trainUse.value, eval_episodes: Number(el.trainEval.value) || 0 })));
    el.jobStop.addEventListener("click", async () => {
      if (jobId) { try { await Api.stopJob(jobId); } catch (e) { /* poll shows it */ } }
    });

    const loadAll = () => {
      loadDemos().catch((e) => { el.demoSummary.textContent = "could not list demos: " + e.message; });
      loadModels().catch(() => {});
    };
    loadAll();
    Tabs.onChange((name) => { if (name === "imitation") loadAll(); });
    setInterval(() => {
      if (Tabs.current() === "imitation") pollJob().catch(() => {});
    }, POLL_MS);
  }

  return { init };
})();
