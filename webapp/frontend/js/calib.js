// Camera calibration panel (Drone tab): live view of what the ring detector sees, click
// auto-fit, sliders, and named presets (day / night / other) that a flight then uses.
// Backend: webapp/backend/calib_manager.py; the camera runs in hardware/onboard/calib_session.py,
// ground only (it never sends takeoff or stick commands).
// Why: daytime flights locked onto the window instead of the hoop with the night calibration.
const Calib = (() => {
  const $ = (id) => document.getElementById(id);
  // [label, param key, index within the key (or null), min, max, step, group]
  const SLIDERS = [
    ["Hue low", "hsv_lo", 0, 0, 179, 1, "Colour"],
    ["Hue high", "hsv_hi", 0, 0, 179, 1, "Colour"],
    ["Saturation low", "hsv_lo", 1, 0, 255, 1, "Colour"],
    ["Saturation high", "hsv_hi", 1, 0, 255, 1, "Colour"],
    ["Brightness low", "hsv_lo", 2, 0, 255, 1, "Colour"],
    ["Brightness high", "hsv_hi", 2, 0, 255, 1, "Colour"],
    ["Min ring size (px)", "min_major_px", null, 0, 400, 1, "Shape"],
    ["Min arc when clipped (deg)", "min_arc_deg", null, 0, 360, 1, "Shape"],
    ["Fill ratio low", "fill_ratio", 0, 0, 2.5, 0.05, "Shape"],
    ["Fill ratio high", "fill_ratio", 1, 0, 2.5, 0.05, "Shape"],
    ["Opening ratio low", "opening_ratio", 0, 0, 1, 0.01, "Shape"],
    ["Max ellipse residual", "residual_max", null, 0, 1, 0.01, "Shape"],
  ];
  // ring_detect.DetectorParams() shape defaults (the "Reset shape gates" button)
  const SHAPE_DEFAULTS = {
    min_area_px: 400, min_side_px: 20, opening_ratio: [0.15, 0.97], fill_ratio: [0.55, 1.45],
    residual_max: 0.3, min_axis_ratio: 0.15, allow_edge: true, close_kernel: 5, border_tol: 2,
    min_arc_deg: 110, min_major_px: 24,
  };

  let params = null;          // what the session uses (from status)
  let active = false;
  let pollTimer = null;
  let lastEdit = 0;
  let pushTimer = null;
  let clicks = { ring: [], reject: [] };
  let loading = { overlay: false, mask: false };
  let presets = [];

  function inputs() { return SLIDERS.map((s, i) => $(`calib-sl-${i}`)); }

  function buildSliders() {
    const box = $("calib-sliders");
    let group = null;
    box.innerHTML = "";
    SLIDERS.forEach(([label, , , min, max, step, g], i) => {
      if (g !== group) {
        group = g;
        const h = document.createElement("div");
        h.className = "calib-group";
        h.textContent = g;
        box.appendChild(h);
      }
      const row = document.createElement("label");
      row.className = "calib-slider";
      row.innerHTML = `<span>${label}</span><input id="calib-sl-${i}" type="range" min="${min}" max="${max}" step="${step}" disabled><output id="calib-out-${i}"></output>`;
      box.appendChild(row);
      row.querySelector("input").addEventListener("input", (ev) => {
        $(`calib-out-${i}`).textContent = ev.target.value;
        lastEdit = Date.now();
        clearTimeout(pushTimer);
        pushTimer = setTimeout(pushSliders, 150);
      });
    });
  }

  function slidersToParams() {
    const out = {};
    SLIDERS.forEach(([, key, idx], i) => {
      const v = Number($(`calib-sl-${i}`).value);
      if (idx === null) out[key] = v;
      else {
        out[key] = out[key] || [...(params && params[key] ? params[key] : [0, 0, 0])];
        out[key][idx] = v;
      }
    });
    for (let k = 0; k < 3; k++) {          // keep each low <= its high
      if (out.hsv_lo[k] > out.hsv_hi[k]) out.hsv_hi[k] = out.hsv_lo[k];
    }
    if (out.fill_ratio[0] > out.fill_ratio[1]) out.fill_ratio[1] = out.fill_ratio[0];
    return out;
  }

  function paramsToSliders(p) {
    if (!p) return;
    SLIDERS.forEach(([, key, idx], i) => {
      const v = idx === null ? p[key] : (p[key] || [])[idx];
      if (v === undefined) return;
      $(`calib-sl-${i}`).value = v;
      $(`calib-out-${i}`).textContent = Number(v).toString();
    });
  }

  async function pushSliders() {
    if (!active) return;
    try {
      params = (await Api.calibParams(slidersToParams())).params;
    } catch (e) {
      $("calib-state").textContent = `slider change refused: ${e.message}`;
    }
  }

  function setActive(on) {
    active = on;
    $("calib-start").disabled = on;
    $("calib-stop").disabled = !on;
    ["calib-source", "calib-webcam", "calib-start-preset"].forEach((id) => { $(id).disabled = on; });
    ["calib-autofit", "calib-forget", "calib-reset-shape", "calib-snapshot"].forEach((id) => { $(id).disabled = !on; });
    inputs().forEach((el) => { el.disabled = !on; });
  }

  function refreshImage(kind) {
    if (loading[kind]) return;
    const img = $(`calib-${kind}`);
    loading[kind] = true;
    const done = () => { loading[kind] = false; };
    img.onload = done;
    img.onerror = done;
    img.src = `/api/calib/frame/${kind}.jpg?t=${Date.now()}`;
  }

  function describe(st) {
    if (!st) return "starting the camera...";
    const parts = [`ring found: <b>${st.found ? "yes" : "no"}</b>`, `candidates: ${st.candidates}`];
    if (st.range_m !== null && st.range_m !== undefined) parts.push(`range ${st.range_m.toFixed(2)} m`);
    if (st.clipped) parts.push("clipped at the frame edge");
    parts.push(`mask covers ${(100 * st.mask_fraction).toFixed(1)}% of the frame`);
    if (st.bat !== undefined && st.bat !== null) parts.push(`battery ${st.bat}%`);
    if (st.temph !== undefined && st.temph !== null) parts.push(`board ${st.temph} °C`);
    parts.push(`${st.fps} fps`);
    if (st.samples && (st.samples.ring_px || st.samples.reject_px)) {
      parts.push(`samples so far: ${st.samples.ring_px} hoop px, ${st.samples.reject_px} ignore px`);
    }
    if (st.image) parts.push(`frame ${st.image}`);
    if (st.candidates > 1) parts.push("<span class='warn-text'>more than one candidate: something else looks like the hoop</span>");
    return parts.join(" · ");
  }

  async function poll() {
    let s;
    try { s = await Api.calibStatus(); } catch (e) { return; }
    setActive(!!s.active);
    if (s.active) {
      if (s.source && $("calib-source").value !== s.source) { $("calib-source").value = s.source; sourceChanged(); }
      $("calib-state").textContent = `session ${s.session} (${s.source})`;
      $("calib-status").innerHTML = describe(s.status);
      if (s.status) {
        refreshImage("overlay");
        refreshImage("mask");
        if (s.status.params && Date.now() - lastEdit > 1500) {
          params = s.status.params;
          paramsToSliders(params);
        }
      }
    } else if (s.session) {
      const tail = s.log_tail ? ` Last output: ${s.log_tail.split("\n").filter(Boolean).slice(-2).join(" / ")}` : "";
      $("calib-state").textContent = `camera stopped.${tail}`;
      $("calib-status").textContent = "camera off";
    }
  }

  function drawDots() {
    const box = $("calib-dots");
    box.innerHTML = "";
    for (const kind of ["ring", "reject"]) {
      for (const [x, y] of clicks[kind]) {
        const d = document.createElement("span");
        d.className = `calib-dot calib-dot-${kind}`;
        d.style.left = `${100 * x}%`;
        d.style.top = `${100 * y}%`;
        box.appendChild(d);
      }
    }
    $("calib-autofit").title = `${clicks.ring.length} hoop and ${clicks.reject.length} ignore clicks`;
  }

  function onImageClick(ev) {
    if (!active) return;
    const r = $("calib-overlay").getBoundingClientRect();
    const x = (ev.clientX - r.left) / r.width;
    const y = (ev.clientY - r.top) / r.height;
    if (x < 0 || x > 1 || y < 0 || y > 1) return;
    const kind = document.querySelector("input[name=calib-click]:checked").value;
    clicks[kind].push([Number(x.toFixed(4)), Number(y.toFixed(4))]);
    drawDots();
  }

  async function autofit() {
    const out = $("calib-autofit-result");
    if (!clicks.ring.length && !clicks.reject.length) { out.textContent = "Click the hoop (and anything to ignore) first."; return; }
    out.textContent = "fitting...";
    try {
      const r = await Api.calibAutofit(clicks.ring, clicks.reject);
      if (!r.ok) { out.textContent = `Auto-fit failed: ${r.error}`; return; }
      clicks = { ring: [], reject: [] };       // sampled now: the session keeps them for the next fit
      drawDots();
      params = r.params;
      paramsToSliders(params);
      lastEdit = Date.now();
      const pct = (v) => `${Math.round(100 * v)}%`;
      out.innerHTML = `Hoop pixels kept: <b>${pct(r.after.ring_kept)}</b> (was ${pct(r.before.ring_kept)}).`
        + (r.reject_px ? ` Ignored pixels still counted as hoop: <b>${pct(r.after.rejected_in)}</b> (was ${pct(r.before.rejected_in)}).` : "")
        + (r.after.rejected_in > 0.1 ? " <span class='warn-text'>Still a lot: they share the hoop's colour. Try the brightness/saturation sliders, or close the blinds.</span>" : "");
    } catch (e) {
      out.textContent = `Auto-fit failed: ${e.message}`;
    }
  }

  async function start() {
    const source = $("calib-source").value;
    const body = { source, preset: $("calib-start-preset").value || null };
    if (source === "image") body.image = $("calib-image").value || null;
    if (source === "webcam") body.webcam_index = Number($("calib-webcam").value) || 0;
    clicks = { ring: [], reject: [] };
    drawDots();
    $("calib-state").textContent = "starting...";
    try {
      await Api.calibStart(body);
      setActive(true);
    } catch (e) {
      $("calib-state").textContent = `could not start: ${e.message}`;
    }
  }

  async function stop() {
    try { await Api.calibStop(); } catch (e) { /* the status poll shows the state */ }
    setTimeout(poll, 300);
  }

  function presetOption(p, suggested) {
    const o = document.createElement("option");
    o.value = p.slug;
    const tag = p.is_default ? "" : ` (${p.condition})`;
    o.textContent = p.name + tag + (p.slug === suggested ? "  ← suggested" : "");
    return o;
  }

  async function loadPresets() {
    let r;
    try { r = await Api.calibPresets(); } catch (e) { return; }
    presets = r.presets;
    // the flight form's picker keeps its choice
    for (const id of ["opt-detector", "calib-start-preset"]) {
      const sel = $(id);
      if (!sel) continue;
      const keep = sel.value;
      sel.innerHTML = "";
      presets.forEach((p) => sel.appendChild(presetOption(p, r.suggested)));
      if (keep && presets.some((p) => p.slug === keep)) sel.value = keep;
    }
    const hint = $("opt-detector-hint");
    if (hint) {
      const s = presets.find((p) => p.slug === r.suggested);
      const now = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      const period = new Date().getHours() >= 7 && new Date().getHours() < 18 ? "day" : "night";
      hint.textContent = s ? `Suggested now (${now}): ${s.name}`
        : `It's ${period} (${now}) and there is no ${period} preset yet.`;
    }
    const box = $("calib-presets");
    box.innerHTML = "";
    presets.forEach((p) => {
      const card = document.createElement("div");
      card.className = "calib-preset";
      const thumb = p.thumb ? `<img src="/api/calib/presets/${encodeURIComponent(p.slug)}/thumb.jpg?t=${Date.now()}" alt="">` : "<div class='calib-nothumb'>no picture</div>";
      const lo = (p.params.hsv_lo || []).join("/"), hi = (p.params.hsv_hi || []).join("/");
      card.innerHTML = `${thumb}<div><b>${p.name}</b> <span class="pill pill-idle">${p.is_default ? "default" : p.condition}</span>
        <div class="note">${p.created_at ? p.created_at.replace("T", " ") : "detector.json"}${p.notes ? " · " + p.notes : ""}</div>
        <div class="note mono">HSV ${lo} … ${hi} · min ring ${p.params.min_major_px ?? "?"} px</div>
        <div class="button-row">
          <button class="secondary-btn" data-act="load" type="button">Load into sliders</button>
          ${p.is_default ? "" : '<button class="secondary-btn" data-act="default" type="button">Set as default</button><button class="secondary-btn" data-act="delete" type="button">Delete</button>'}
        </div></div>`;
      card.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => presetAction(b.dataset.act, p)));
      box.appendChild(card);
    });
  }

  async function presetAction(act, p) {
    const note = $("calib-save-note");
    try {
      if (act === "load") {
        if (!active) { note.textContent = "Start the camera first, then load a preset to see it live."; return; }
        params = (await Api.calibLoadPreset(p.slug)).params;
        paramsToSliders(params);
        lastEdit = Date.now();
        note.textContent = `Loaded "${p.name}".`;
      } else if (act === "default") {
        await Api.calibDefaultPreset(p.slug);
        note.textContent = `"${p.name}" is now the default (detector.json): flights without a preset use it.`;
      } else if (act === "delete") {
        if (!confirm(`Delete the preset "${p.name}"?`)) return;
        await Api.calibDeletePreset(p.slug);
        note.textContent = `Deleted "${p.name}".`;
      }
    } catch (e) {
      note.textContent = e.message;
    }
    loadPresets();
  }

  async function save() {
    const name = $("calib-name").value.trim();
    const note = $("calib-save-note");
    if (!name) { note.textContent = "Give the preset a name (e.g. Day)."; return; }
    try {
      const p = await Api.calibSavePreset({ name, condition: $("calib-condition").value, notes: $("calib-notes").value });
      note.textContent = `Saved "${p.name}". Pick it under Camera preset when you fly.`;
      await loadPresets();
      const sel = $("opt-detector");
      if (sel) sel.value = p.slug;
    } catch (e) {
      note.textContent = `Not saved: ${e.message}`;
    }
  }

  async function loadImages() {
    const sel = $("calib-image");
    sel.innerHTML = "";
    try {
      const r = await Api.calibImages();
      r.images.forEach((im) => {
        const o = document.createElement("option");
        o.value = im.path;
        o.textContent = `${im.taken.replace("T", " ")} · ${im.path.split("/").slice(-2, -1)[0]}`;
        sel.appendChild(o);
      });
      if (!r.images.length) sel.innerHTML = "<option value=''>no saved frames yet</option>";
    } catch (e) { /* ignore */ }
  }

  function sourceChanged() {
    const s = $("calib-source").value;
    $("calib-image").hidden = s !== "image";
    $("calib-webcam").hidden = s !== "webcam";
    if (s === "image") loadImages();
  }

  function open() {
    $("calib-modal").hidden = false;
    loadPresets();
    poll();
    clearInterval(pollTimer);
    pollTimer = setInterval(poll, 400);
  }

  function close() {
    $("calib-modal").hidden = true;
    clearInterval(pollTimer);
    pollTimer = null;
  }

  function init() {
    if (!$("calib-modal")) return;
    buildSliders();
    setActive(false);
    $("calib-open-btn").addEventListener("click", open);
    $("calib-close").addEventListener("click", close);
    $("calib-modal").addEventListener("click", (ev) => { if (ev.target.id === "calib-modal") close(); });
    $("calib-source").addEventListener("change", sourceChanged);
    $("calib-start").addEventListener("click", start);
    $("calib-stop").addEventListener("click", stop);
    $("calib-overlay").addEventListener("click", onImageClick);
    $("calib-autofit").addEventListener("click", autofit);
    $("calib-clear-clicks").addEventListener("click", () => { clicks = { ring: [], reject: [] }; drawDots(); });
    $("calib-forget").addEventListener("click", async () => {
      clicks = { ring: [], reject: [] };
      drawDots();
      try { await Api.calibAutofit([], [], true); $("calib-autofit-result").textContent = "Samples forgotten: start again by clicking the hoop."; }
      catch (e) { $("calib-autofit-result").textContent = e.message; }
    });
    $("calib-image").addEventListener("change", async () => {
      if (!active || $("calib-source").value !== "image") return;
      try { await Api.calibImage($("calib-image").value); clicks = { ring: [], reject: [] }; drawDots(); }
      catch (e) { $("calib-state").textContent = e.message; }
    });
    $("calib-reset-shape").addEventListener("click", async () => {
      try { params = (await Api.calibParams(SHAPE_DEFAULTS)).params; paramsToSliders(params); lastEdit = Date.now(); }
      catch (e) { $("calib-state").textContent = e.message; }
    });
    $("calib-snapshot").addEventListener("click", async () => {
      try { await Api.calibSnapshot(); $("calib-state").textContent = "snapshot saved (pick it later under Saved frame)"; }
      catch (e) { $("calib-state").textContent = e.message; }
    });
    $("calib-save").addEventListener("click", save);
    document.addEventListener("keydown", (ev) => { if (ev.key === "Escape" && !$("calib-modal").hidden) close(); });
    sourceChanged();
    loadPresets();
  }

  document.addEventListener("DOMContentLoaded", init);
  return { open, loadPresets };
})();
