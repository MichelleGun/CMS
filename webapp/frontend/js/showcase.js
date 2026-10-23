// The Overview, Policies and Training tabs: results written by scripts/build_site.py into
// data/*.json (leaderboard, milestones, curves, optimiser rounds, HRL comparison, timeline)
// and recorded replays in data/replays/. Plain files, so these tabs read the same way from
// the local server and from the online copy (GitHub Pages). Each tab renders on first view.
const Show = (() => {
  async function data(name) {
    const r = await fetch(`data/${name}`);
    if (!r.ok) throw new Error(`data/${name}: ${r.status}`);
    return r.json();
  }
  const pct = (v) => (v === null || v === undefined ? "–" : `${Math.round(v * 100)}%`);
  const num = (v, d = 2) => (v === null || v === undefined ? "–" : Number(v).toFixed(d));
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  // markdown-ish cell text from HANDOVER: **bold**, `code`
  const md = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/`(.+?)`/g, "<code>$1</code>");
  const bar = (v) => v === null || v === undefined ? "–"
    : `<div class="bar" title="${pct(v)}"><i style="width:${Math.round(v * 100)}%"></i></div> ${pct(v)}`;
  return { data, pct, num, esc, md, bar };
})();

const Showcase = (() => {
  const done = {};
  const $ = (id) => document.getElementById(id);

  function fail(el, e) {
    el.innerHTML = `<p class="note">Results not built yet (${Show.esc(e.message)}). Run
      <code>python scripts/build_site.py --record</code>.</p>`;
  }

  async function overview() {
    const [meta, board, ms, tl] = await Promise.all([Show.data("meta.json"), Show.data("leaderboard.json"),
      Show.data("milestones.json"), Show.data("timeline.json")]);
    const best = board[0];
    const stat = (v, k) => `<div class="stat"><div class="v">${v}</div><div class="k">${k}</div></div>`;
    $("ov-stats").innerHTML =
      stat(Show.num(best.lpe), "passes per 90 s episode, best policy (sim)") +
      stat(Show.pct(best.two_plus), "of episodes with 2+ passes (it loops)") +
      stat(best.best, "most passes in one episode") +
      stat(board.length, "trained policies scored");
    const hero = ms.find((m) => m.id === "C9") || ms[ms.length - 1];
    new Replay($("ov-hero"), { id: hero.id, title: hero.label, tall: true });
    $("ov-timeline").innerHTML = tl.map((t) =>
      `<div class="item"><div class="date">${Show.esc(t.date)}</div><div>${Show.md(t.change)}</div>
       <div class="imp"><b>Why:</b> ${Show.md(t.why)}<br><b>Impact:</b> ${Show.md(t.impact)}</div></div>`).join("");
    $("ov-built").textContent = `Results built ${meta.built.replace("T", " ")}.`;
  }

  async function policies() {
    const [ms, board] = await Promise.all([Show.data("milestones.json"), Show.data("leaderboard.json")]);
    const box = $("pol-replays");
    ms.forEach((m) => {
      const card = document.createElement("div");
      card.innerHTML = `<div class="replay"></div><p class="note">${Show.esc(m.what)}</p>`;
      box.appendChild(card);
      new Replay(card.querySelector(".replay"), { id: m.id, title: m.label, autoplay: false });
    });
    $("pol-board").innerHTML = board.slice(0, 40).map((r, i) =>
      `<tr${i === 0 ? ' class="hi"' : ""}><td class="n">${i + 1}</td><td><code>${Show.esc(r.file)}</code></td>
       <td class="n">${Show.num(r.lpe)}</td><td>${Show.bar(r.two_plus)}</td><td class="n">${r.best}</td>
       <td>${Show.bar(r.oblique)}</td><td>${r.memory ? '<span class="chip yes">yes</span>' : '<span class="chip">no</span>'}</td></tr>`).join("");
  }

  async function training() {
    const [curves, ms, opt, hrl] = await Promise.all([Show.data("curves.json"), Show.data("milestones.json"),
      Show.data("optimiser.json"), Show.data("hrl.json")]);
    const label = (id) => (ms.find((m) => m.id === id) || { label: id }).label;
    const palette = ["#9aa0b4", "#5a8dee", "#c45ad6", "#2fb38a", "#e0a32c", "#ff7130", "#ff3c68"];
    const dim = "#9494ab", grid = "#2c2c40";
    new Chart($("tr-curves"), {
      type: "line",
      data: { datasets: curves.map((c, i) => ({ label: label(c.id), data: c.points.map(([x, y]) => ({ x, y })),
        borderColor: palette[i % palette.length], backgroundColor: palette[i % palette.length],
        borderWidth: 2, pointRadius: 0, tension: 0.25 })) },
      options: { maintainAspectRatio: false, animation: false, interaction: { mode: "nearest", intersect: false },
        scales: { x: { type: "linear", title: { display: true, text: "million steps added by this run", color: dim }, grid: { color: grid }, ticks: { color: dim } },
                  y: { min: 0, title: { display: true, text: "passes per 90 s", color: dim }, grid: { color: grid }, ticks: { color: dim } } },
        plugins: { legend: { labels: { color: dim, boxWidth: 12 } } } },
    });

    $("tr-opt-head").innerHTML = `Current best: <code>${Show.esc(opt.parent)}</code>`
      + (opt.parent_score ? ` (score ${Show.num(opt.parent_score)})` : "");
    $("tr-rounds").innerHTML = opt.rounds.map((r) =>
      `<tr><td class="n">${r.round}</td><td>${r.keys.map(Show.esc).join(" vs ")}</td><td>${r.results.map((x) =>
        `<b>${Show.esc(x.key)}</b>: ${x.score === null || x.score === undefined ? "not scored" :
          `${Show.num(x.score)} · ${Show.num(x.lpe)} · ${Show.pct(x.oblique)}`}`).join("<br>")}</td>
       <td><code>${Show.esc(r.parent_after)}</code></td></tr>`).join("");

    $("tr-hrl").innerHTML = hrl.length ? `<div class="scroll"><table>
      <thead><tr><th>Policy</th><th>Passes / 90 s</th><th>2+ passes</th><th>Corner entry</th><th>Score</th></tr></thead>
      <tbody>${hrl.map((r) => `<tr><td>${Show.esc(r.name)}</td><td class="n">${Show.num(r.lpe)}</td><td class="n">${Show.pct(r.two_plus)}</td>
        <td class="n">${Show.pct(r.oblique_entry)}</td><td class="n">${Show.num(r.score)}</td></tr>`).join("")}</tbody></table></div>`
      : `<p class="card">Training now: the comparison appears here when it is done. Meanwhile the
         Policies tab has replays of the fixed switch and the manager so far.</p>`;
  }

  const RENDER = { overview: [overview, "ov-hero"], policies: [policies, "pol-replays"], training: [training, "tr-hrl"] };

  // On the online copy, say plainly which tabs only look and which need the code.
  const STATIC_NOTES = {
    sim: "This is the training screen. Pick a run on the left to see its curves; starting a training needs the code on your own computer.",
    env: "The simulated cage, hoop and drone. Editing it, and training in it, runs on your own computer.",
    drone: "Flying the Tello and calibrating its camera happen on a laptop next to the drone.",
    imitation: "Recording and training from demonstrations runs on your own computer.",
    history: "A snapshot of selected training runs. Flights and jobs stay on the computer that ran them.",
  };

  function onTab(name) {
    if (RENDER[name] && !done[name]) {
      done[name] = true;
      const [fn, el] = RENDER[name];
      fn().catch((e) => { console.error(e); fail($(el), e); });
    }
    const banner = $("static-banner");
    if (!window.DQ_STATIC || !STATIC_NOTES[name]) { banner.hidden = true; return; }
    banner.innerHTML = `<b>Online copy.</b> ${STATIC_NOTES[name]}
      <a href="${window.DQ_REPO}" target="_blank" rel="noopener">Get the code on GitHub</a>
      &middot; <a href="#overview" data-go="overview">How to run it</a>`;
    banner.hidden = false;
  }

  function init() {
    document.querySelectorAll(".repo-link").forEach((a) => { a.href = window.DQ_REPO; });
    document.addEventListener("click", (e) => {
      const a = e.target.closest("[data-go]");
      if (!a) return;
      e.preventDefault();
      Tabs.show(a.dataset.go);
      if (a.dataset.go === "overview") $("ov-run").scrollIntoView({ behavior: "smooth" });
    });
    Tabs.onChange(onTab);
    onTab(Tabs.current());
  }

  return { init };
})();
