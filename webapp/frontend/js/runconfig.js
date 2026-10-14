// The Config card: what a run was started with (settings, PPO hyperparameters, reward weights,
// the network's architecture), and an editable copy of those values for the NEXT run, with the
// equivalent command line updating as you type. Everything comes from GET /api/runs/{id}/config;
// the form is built from the field description the backend sends (webapp/backend/run_config.py),
// so a new setting there appears here without touching this file.
const RunConfig = (() => {
  const $ = (id) => document.getElementById(id);
  function node(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = text;
    return n;
  }

  function fmt(v) {
    if (v === null || v === undefined || v === "") return "—";            // em dash: not recorded / off
    if (typeof v === "boolean") return v ? "on" : "off";
    if (typeof v === "number") {
      if (Number.isInteger(v)) return v.toLocaleString("en-US").replace(/,/g, " ");
      if (v !== 0 && Math.abs(v) < 0.001) return v.toExponential(1);
      return String(Math.round(v * 1e6) / 1e6);
    }
    return String(v);
  }

  // an <input> for one field, pre-filled; onChange(value) gets the parsed value (null = empty)
  function control(f, value, onChange) {
    let inp;
    if (f.type === "bool") {
      inp = document.createElement("input");
      inp.type = "checkbox";
      inp.checked = !!value;
      inp.addEventListener("change", () => onChange(inp.checked));
    } else if (f.type === "choice") {
      inp = document.createElement("select");
      f.choices.forEach((c) => inp.append(Object.assign(document.createElement("option"), { value: c, textContent: c })));
      inp.value = value;
      inp.addEventListener("change", () => onChange(inp.value));
    } else {
      inp = document.createElement("input");
      inp.type = "number";
      inp.step = f.sci ? "any" : (f.step || (f.type === "int" ? 1 : "any"));
      if (f.min !== undefined && !f.sci) inp.min = f.min;
      if (f.max !== undefined && !f.sci) inp.max = f.max;
      inp.value = value === null || value === undefined ? "" : value;
      if (f.null_ok) inp.placeholder = "default";
      inp.addEventListener("input", () => {
        if (inp.value === "") { onChange(null); return; }
        const n = f.type === "int" ? parseInt(inp.value, 10) : parseFloat(inp.value);
        onChange(Number.isFinite(n) ? n : null);
      });
    }
    inp.className = "cfg-input";
    return inp;
  }

  const same = (a, b) => (a === b) || ((a === null || a === undefined) && (b === null || b === undefined)) ||
    (typeof a === "number" && typeof b === "number" && Math.abs(a - b) < 1e-12);

  async function mount(container, runId, opts = {}) {
    container.replaceChildren(node("div", "note", "loading the configuration…"));
    let cfg;
    try {
      cfg = await Api.getRunConfig(runId);
    } catch (e) {
      container.replaceChildren(node("div", "note", "could not load the configuration: " + e.message));
      return;
    }
    render(container, runId, cfg, opts);
  }

  function render(container, runId, cfg, opts) {
    const root = node("div", "cfg");
    const draft = {};                       // field key -> edited value
    const draftReward = {};                 // reward name -> edited value
    cfg.fields.forEach((f) => { draft[f.key] = f.value; });
    cfg.reward.forEach((r) => { draftReward[r.key] = r.value; });
    const rows = [];                        // {row, changed()} so highlights refresh together

    // ---- header
    const head = node("div", "cfg-head");
    head.append(node("strong", "", runId), node("span", "pill pill-idle", cfg.status));
    if (!cfg.editable) head.append(node("span", "note", " Settings of this kind of run are shown read-only (editing is for onboard-camera runs)."));
    root.append(head);

    // ---- read-only info
    const info = node("div", "kv cfg-info");
    cfg.info.forEach((r) => info.append(node("span", "k", r.k), node("span", "v", fmt(r.v))));
    root.append(info);

    if (cfg.editable) {
      const legend = node("div", "note cfg-legend",
        "Left: what this run used. Right: the values for the next run (start as a copy of this run's; " +
        "changed values are highlighted). Nothing here changes the run you are looking at.");
      root.append(legend);

      // ---- grouped field tables
      const groups = [];
      cfg.fields.forEach((f) => { if (!groups.includes(f.group)) groups.push(f.group); });
      groups.forEach((g) => {
        const sec = node("div", "cfg-group");
        sec.append(node("h4", "", g));
        const t = node("table", "cfg-table");
        t.append(Object.assign(document.createElement("thead"), { innerHTML: "<tr><th>Setting</th><th>This run</th><th>Next run</th></tr>" }));
        const tb = document.createElement("tbody");
        cfg.fields.filter((f) => f.group === g).forEach((f) => {
          const tr = document.createElement("tr");
          const lab = node("td", "cfg-label", f.label);
          if (f.help) lab.title = f.help;
          const old = node("td", "cfg-old", fmt(f.value));
          const cell = node("td", "cfg-new");
          const inp = control(f, f.value, (v) => { draft[f.key] = v; refresh(); });
          cell.append(inp);
          tr.append(lab, old, cell);
          if (f.help) tr.title = f.help;
          tb.append(tr);
          rows.push({ tr, changed: () => !same(draft[f.key], f.value), reset: () => {
            draft[f.key] = f.value;
            if (f.type === "bool") inp.checked = !!f.value; else inp.value = f.value === null || f.value === undefined ? "" : f.value;
          } });
        });
        t.append(tb);
        sec.append(t);
        root.append(sec);
      });

      // ---- reward weights
      const rsec = node("div", "cfg-group");
      rsec.append(node("h4", "", "Reward weights"));
      const rt = node("table", "cfg-table");
      rt.append(Object.assign(document.createElement("thead"), { innerHTML: "<tr><th>Weight</th><th>Default</th><th>This run</th><th>Next run</th></tr>" }));
      const rb = document.createElement("tbody");
      cfg.reward.forEach((r) => {
        const tr = document.createElement("tr");
        const f = { key: r.key, type: r.type === "bool" ? "bool" : "float", step: "any" };
        const inp = control(f, r.value, (v) => { draftReward[r.key] = v === null ? r.default : v; refresh(); });
        const lab = node("td", "cfg-label", r.key);
        if (r.help) { lab.title = r.help; tr.title = r.help; }
        tr.append(lab, node("td", "cfg-old", fmt(r.default)), node("td", "cfg-old" + (!same(r.value, r.default) ? " cfg-nondefault" : ""), fmt(r.value)),
                  Object.assign(node("td", "cfg-new"), {}));
        tr.lastChild.append(inp);
        rb.append(tr);
        rows.push({ tr, changed: () => !same(draftReward[r.key], r.value), reset: () => {
          draftReward[r.key] = r.value;
          if (f.type === "bool") inp.checked = !!r.value; else inp.value = r.value;
        } });
      });
      rt.append(rb);
      rsec.append(rt);
      rsec.append(node("div", "note", "Hover a row for what it does. Orange 'this run' = that run already changed it from the default."));
      root.append(rsec);

      // ---- start from + demos
      const ssec = node("div", "cfg-group");
      ssec.append(node("h4", "", "Start the next run"));
      const from = document.createElement("select");
      from.className = "cfg-input cfg-wide";
      from.append(Object.assign(document.createElement("option"), { value: "run", textContent: "Continue from this run's latest model (step count carries on)" }));
      from.append(Object.assign(document.createElement("option"), { value: "fresh", textContent: "Fresh network (no parent)" }));
      if (cfg.info.some((i) => i.k === "init_parent")) {
        const parent = cfg.info.find((i) => i.k === "init_parent").v;
        from.append(Object.assign(document.createElement("option"), { value: "parent", textContent: "Same parent as this run (" + parent + ")" }));
      }
      from.addEventListener("change", refresh);
      ssec.append(from);

      const demoBox = node("div", "cfg-demos note", "loading demos…");
      ssec.append(node("div", "note", "Demos for the behaviour-cloning term (none ticked = no demo term):"), demoBox);
      Api.getDemos().then(({ demos }) => {
        demoBox.replaceChildren();
        const usable = demos.filter((d) => d.task === "onboard_hoop" && !d.error);
        if (!usable.length) demoBox.textContent = "no onboard-task demos yet";
        const used = new Set(cfg.demo_paths || []);
        const hadDemos = !!(cfg.demo_names && cfg.demo_names.length);
        usable.forEach((d) => {
          const lab = node("label", "checkbox-row");
          const cb = document.createElement("input");
          cb.type = "checkbox";
          cb.value = d.id;
          // a run that recorded its demo paths: exactly those. An older one only has file names, so
          // tick the same defaults as the Training panel (passed human flights + teacher demos)
          cb.checked = cfg.demo_paths ? used.has(d.id)
            : hadDemos && ((d.source === "human" && d.label === "pass") || d.source === "sim_teacher");
          cb.addEventListener("change", refresh);
          lab.append(cb, document.createTextNode(` ${d.id} (${d.source || "?"}, ${d.samples} samples)`));
          demoBox.append(lab);
        });
        if (!cfg.demo_paths && hadDemos)
          demoBox.prepend(node("div", "note", "This run's record only has its demo file names, so the usual defaults are ticked; adjust as you like."));
        refresh();
      }).catch((e) => { demoBox.textContent = "could not load demos: " + e.message; });
      root.append(ssec);

      // ---- command preview + buttons
      const csec = node("div", "cfg-group");
      csec.append(node("h4", "", "Command for the next run"));
      const cmd = node("pre", "cfg-cmd", cfg.command || "");
      const err = node("div", "cfg-error");
      err.hidden = true;
      const bar = node("div", "button-row cfg-buttons");
      const copy = node("button", "secondary-btn", "Copy command");
      const reset = node("button", "secondary-btn", "Reset to this run");
      const go = node("button", "primary-btn", "Start next run");
      [copy, reset, go].forEach((b) => { b.type = "button"; });
      bar.append(copy, reset, go);
      csec.append(cmd, err, bar);
      root.append(csec);

      function payload() {
        const p = { algo: "ppo", env: "onboard" };
        cfg.fields.forEach((f) => { if (draft[f.key] !== null && draft[f.key] !== undefined) p[f.key] = draft[f.key]; });
        const over = {};
        cfg.reward.forEach((r) => { if (!same(draftReward[r.key], r.default)) over[r.key] = typeof draftReward[r.key] === "boolean" ? (draftReward[r.key] ? 1 : 0) : draftReward[r.key]; });
        if (Object.keys(over).length) p.reward_weights = over;
        // an emptied "default" field (target_kl, log-std reset) is sent as null = the trainer's default
        cfg.fields.forEach((f) => { if (f.null_ok && draft[f.key] === null) p[f.key] = null; });
        if (from.value === "run") p.init_run = runId;
        if (from.value === "parent") {
          const parent = cfg.info.find((i) => i.k === "init_parent").v;
          p.init_run = parent;
        }
        p.demo_ids = Array.from(demoBox.querySelectorAll("input:checked")).map((b) => b.value);
        return p;
      }

      let timer = null, seq = 0;
      function refresh() {
        rows.forEach((r) => r.tr.classList.toggle("changed", r.changed()));
        clearTimeout(timer);
        timer = setTimeout(async () => {
          const mine = ++seq;
          try {
            const res = await Api.previewRunCommand(payload());
            if (mine !== seq) return;
            cmd.textContent = res.command;
            err.hidden = true;
            go.disabled = false;
          } catch (e) {
            if (mine !== seq) return;
            err.textContent = "⚠ " + e.message;
            err.hidden = false;
            go.disabled = true;
          }
        }, 250);
      }

      copy.addEventListener("click", async () => {
        try { await navigator.clipboard.writeText(cmd.textContent); copy.textContent = "Copied"; }
        catch (e) { copy.textContent = "Select and copy by hand"; }
        setTimeout(() => { copy.textContent = "Copy command"; }, 1500);
      });
      reset.addEventListener("click", () => { rows.forEach((r) => r.reset()); refresh(); });
      go.addEventListener("click", async () => {
        const changed = rows.filter((r) => r.changed()).length;
        if (!confirm(`Start a new training run${changed ? ` with ${changed} changed setting${changed > 1 ? "s" : ""}` : " with this run's settings"}?`)) return;
        go.disabled = true;
        try {
          const res = await Api.startRun(payload());
          if (opts.onStarted) opts.onStarted(res.run_id);
        } catch (e) {
          err.textContent = "could not start: " + e.message;
          err.hidden = false;
          go.disabled = false;
        }
      });
      refresh();
    } else {
      root.append(node("div", "note", "No editable settings for this kind of run. The raw record is in its meta.json."));
    }

    // ---- architecture (read-only)
    if (cfg.architecture) {
      const a = cfg.architecture;
      const sec = node("div", "cfg-group");
      sec.append(node("h4", "", "Network architecture" + (a.model_file ? ` (from ${a.model_file})` : "")));
      const kv = node("div", "kv cfg-info");
      const add = (k, v) => { if (v !== null && v !== undefined && v !== "") kv.append(node("span", "k", k), node("span", "v", typeof v === "object" ? JSON.stringify(v) : String(v))); };
      add("input / output", `${(a.obs_dim || []).join("x")} → ${(a.act_dim || []).join("x")}`);
      add("activation", a.activation);
      add("policy_kwargs", a.policy_kwargs);
      Object.entries(a.layers || {}).forEach(([k, v]) => add(k, v));
      add("log_std values", a.log_std);
      Object.entries(a.saved || {}).forEach(([k, v]) => add("saved ppo." + k, v));
      add("note", a.error);
      sec.append(kv);
      root.append(sec);
    }
    container.replaceChildren(root);
  }

  // the Config button on the run overlay opens the card as a dialog over the dashboard
  function openModal(runId) {
    const modal = $("config-modal");
    const body = $("config-modal-body");
    $("config-modal-title").textContent = "Configuration · " + runId;
    modal.hidden = false;
    mount(body, runId, {
      onStarted: (newId) => {
        modal.hidden = true;
        if (window.SimApp) window.SimApp.selectRun(newId);
      },
    });
  }
  function closeModal() { $("config-modal").hidden = true; }

  window.addEventListener("DOMContentLoaded", () => {
    const close = $("config-modal-close");
    if (close) close.addEventListener("click", closeModal);
    const modal = $("config-modal");
    if (modal) modal.addEventListener("click", (e) => { if (e.target === modal) closeModal(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && modal && !modal.hidden) closeModal(); });
  });

  return { mount, openModal, closeModal };
})();
