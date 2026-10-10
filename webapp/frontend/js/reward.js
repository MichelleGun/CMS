// The net env's reward (MDP): the editable "Reward (MDP)" block in the Training panel and the
// read-only card in the run overlay. The defaults and term descriptions come from GET /api/config
// (hoop_net_env.DEFAULT_REWARD_WEIGHTS), so nothing is duplicated here.
const Reward = (() => {
  let info = null;            // {modes, defaults, terms: {shaped: [[name, text]], budget: [...]}}
  let parentMeta = null;      // meta.json of the run picked in "Start from", if any
  const $ = (id) => document.getElementById(id);

  function fmt(v) { return Number(v).toString(); }

  function mode() { return $("reward-mode").value; }

  function weightInputs() { return Array.from($("reward-weights").querySelectorAll("input")); }

  // The weights a run trained with (older runs did not save them: the defaults then).
  function metaWeights(meta) { return Object.assign({}, info.defaults, (meta && meta.reward_weights) || {}); }

  function renderInputs(values) {
    const box = $("reward-weights");
    box.replaceChildren();
    info.terms[mode()].forEach(([name, text]) => {
      const row = document.createElement("div");
      row.className = "field-row";
      row.title = text;
      const label = document.createElement("label");
      label.textContent = name.replace(/_/g, " ");
      const input = document.createElement("input");
      input.type = "number";
      input.step = "any";
      input.min = "0";
      input.dataset.name = name;
      input.value = fmt(values[name]);
      input.addEventListener("input", renderSummary);
      row.append(label, input);
      box.append(row);
    });
    $("laps-row").hidden = mode() !== "shaped";
    $("budget-row").hidden = mode() !== "budget";
    renderSummary();
  }

  function current() {
    const w = {};
    weightInputs().forEach((i) => {
      const v = parseFloat(i.value);
      w[i.dataset.name] = Number.isFinite(v) && v >= 0 ? v : info.defaults[i.dataset.name];
    });
    return w;
  }

  // one line that reads like the MDP, e.g. "+20 per pass (+10 centred) · −3 rim · ..."
  function summaryText(m, w, laps, budget) {
    if (m === "budget") {
      return `+${fmt(w.budget_pass)} per pass · −${fmt(w.budget_rim)} rim · ` +
        `−${fmt(w.budget_time)}/step · net ends it · episode ends at −${fmt(budget)}`;
    }
    return `+${fmt(w.pass_bonus)} per pass (+${fmt(w.centre_bonus)} centred) · ` +
      `−${fmt(w.rim_penalty)} rim · −${fmt(w.net_penalty)} net · ` +
      `−${fmt(w.wrong_way_penalty)} wrong way · −${fmt(w.time_penalty)}/step · ` +
      `shaping ×${fmt(w.shaping_gain)}` + (laps === 1 ? "" : ` · ${laps === 0 ? "endless" : laps} passes`);
  }

  function changedFrom(base, w) {
    return Object.keys(w).filter((k) => Math.abs(w[k] - base[k]) > 1e-12);
  }

  function renderSummary() {
    const w = Object.assign({}, info.defaults, current());
    $("reward-summary").textContent = summaryText(mode(), w,
      parseInt($("laps-input").value, 10) || 0, parseFloat($("budget-input").value) || 3);
    const note = $("reward-note");
    const base = parentMeta ? metaWeights(parentMeta) : info.defaults;
    const changed = changedFrom(base, current());
    const parentMode = parentMeta && (parentMeta.reward_mode || "shaped");
    if (parentMeta && (changed.length || parentMode !== mode())) {
      note.className = "note bad";
      note.textContent = `Differs from the run you start from (${changed.join(", ") || "mode"}): ` +
        "its value estimates re-fit to the new reward over the first updates.";
    } else {
      note.className = "note";
      note.textContent = parentMeta ? "Same reward as the run you start from."
        : (changed.length ? `Changed from the defaults: ${changed.join(", ")}.` : "Default reward.");
    }
  }

  function setForm(meta) {
    $("reward-mode").value = (meta && meta.reward_mode) || "shaped";
    $("laps-input").value = meta && meta.laps !== undefined && meta.laps !== null ? meta.laps : 1;
    $("budget-input").value = (meta && meta.budget) || 3;
    renderInputs(metaWeights(meta));
  }

  function init(rewardInfo) {
    info = rewardInfo;
    if (!info) { $("reward-block").hidden = true; return; }
    $("reward-mode").addEventListener("change", () => renderInputs(Object.assign(metaWeights(parentMeta), current())));
    $("laps-input").addEventListener("input", renderSummary);
    $("budget-input").addEventListener("input", renderSummary);
    $("reward-reset-btn").addEventListener("click", () => setForm(null));
    setForm(null);
  }

  // "Start from" changed: show that run's reward (null = a new policy: keep what is there)
  function setParent(meta) {
    if (!info) return;
    parentMeta = meta && meta.env === "net" ? meta : null;
    if (parentMeta) setForm(parentMeta); else renderSummary();
  }

  // fields for POST /api/runs; only the weights shown for this mode, and only when changed
  function payload() {
    if (!info) return {};
    const out = { reward_mode: mode() };
    if (mode() === "budget") out.budget = parseFloat($("budget-input").value) || 3;
    else out.laps = Math.max(0, parseInt($("laps-input").value, 10) || 0);
    const w = current();
    const changed = changedFrom(info.defaults, w);
    if (changed.length) out.reward_weights = Object.fromEntries(changed.map((k) => [k, w[k]]));
    return out;
  }

  // read-only card in the run overlay
  function renderCard(meta) {
    const box = $("overlay-reward");
    box.replaceChildren();
    if (meta && meta.env === "onboard") { renderOnboardCard(box, meta); return; }
    if (!info || !meta || meta.env !== "net") {
      box.textContent = meta ? "legacy point-reaching task (reward: minus the distance)" : "–";
      return;
    }
    const m = meta.reward_mode || "shaped";
    const w = metaWeights(meta);
    const head = document.createElement("div");
    head.style.gridColumn = "1 / -1";
    head.textContent = `${m}` + (meta.reward_weights ? "" : " (defaults at the time; this run did not save its weights)");
    box.append(head);
    info.terms[m].forEach(([name, text]) => {
      const k = document.createElement("span");
      k.textContent = name.replace(/_/g, " ");
      k.title = text;
      const v = document.createElement("span");
      v.className = "rw-val" + (Math.abs(w[name] - info.defaults[name]) > 1e-12 ? " rw-changed" : "");
      v.textContent = fmt(w[name]);
      v.title = text;
      box.append(k, v);
    });
    const extra = document.createElement("div");
    extra.style.gridColumn = "1 / -1";
    extra.textContent = m === "budget" ? `episode ends at −${fmt(meta.budget || 3)}`
      : `${meta.laps === 0 ? "endless" : (meta.laps || 1)} pass(es) per episode` +
        (meta.sway_deg !== undefined && meta.sway_deg !== null ? ` · swing up to ${meta.sway_deg}°` : "") +
        (meta.wind_steps ? ` · ramp ${(meta.wind_steps / 1e6).toFixed(1)}M steps` : "");
    box.append(extra);
  }

  // onboard-camera runs: PPO on hardware/onboard/gym_env.py's MDP, plus the imitation term
  // that pulls the policy towards the demos (strong at first, fading)
  const ONBOARD_TERMS = [
    ["r_passed", "+ through the opening (ends the episode)"],
    ["r_struck", "touched the noodle anywhere (ends it)"],
    ["w_rim", "- per step at most, as the drone's edge nears the noodle"],
    ["rim_margin_m", "clearance (m) at which the rim penalty starts"],
    ["r_missed", "past the ring (ends it)"],
    ["r_out_of_cage", "left the cage (ends it)"],
    ["w_shape", "x metres progress: to the point in front, then through to the one behind"],
    ["axis_pull", "x metres off the axis near the ring (inside the shaping)"],
    ["r_wrong_way", "back through the opening the wrong way (no pass)"],
    ["arm_dist_m", "laps: fly this far clear before the next pass counts"],
    ["w_time", "- per control step (15 per second)"],
  ];

  function row(box, k, v, tip) {
    const a = document.createElement("span");
    a.textContent = k;
    a.title = tip || "";
    const b = document.createElement("span");
    b.className = "rw-val";
    b.textContent = v;
    b.title = tip || "";
    box.append(a, b);
  }

  function line(box, text) {
    const d = document.createElement("div");
    d.style.gridColumn = "1 / -1";
    d.textContent = text;
    box.append(d);
  }

  function renderOnboardCard(box, meta) {
    line(box, "onboard MDP" + (meta.reward ? "" : " (weights not saved by this run: the defaults)") +
      (meta.bc ? " + imitation (demo) term" : ", plain PPO (no demos)"));
    const w = meta.reward || {};
    ONBOARD_TERMS.forEach(([k, tip]) => {
      if (w[k] !== undefined) row(box, k.replace(/_/g, " "), fmt(w[k]), tip);
    });
    const rb = meta.robust;
    if (rb) {
      line(box, rb.dr > 0 || rb.hard_start_p > 0
        ? `robust: randomisation ${fmt(rb.dr)}, swing ≤${fmt(rb.sway_deg)}°, ` +
          `hard starts ${Math.round(rb.hard_start_p * 100)}%, ramp ${(rb.dr_steps / 1e6).toFixed(1)}M steps` +
          (rb.bc_augment ? ", demos mirrored + noisy" : "")
        : "no randomisation (the measured front approach only)");
    }
    const bc = meta.bc;
    if (bc) {
      const samples = Object.entries(bc.samples || {}).map(([src, n]) => `${n} ${src}`).join(", ");
      row(box, "demo pull", `${fmt(bc.coef)} → ${fmt(bc.coef_min)}`,
          `strength of the demo term, fading over ${(bc.decay_steps || 0).toLocaleString()} steps`);
      const hw = bc.source_weights && bc.source_weights.human;
      if (hw !== undefined) row(box, "your demos count", `×${fmt(hw)}`, "weight of one human sample vs a simulated one");
      if (samples) line(box, `demos: ${samples} samples`);
    }
  }

  return { init, payload, setParent, renderCard };
})();
