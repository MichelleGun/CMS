// Hash-based tabs in two levels. The header has five groups; a group with more than one page
// shows a sub-tab row under the header:
//   Overview | Results (Policies, Training progress, Run history) | Simulator (Train, Environment)
//   | Drone (Fly, Demos) | Try it yourself
// The hash and the listeners use the PAGE name (#sim, #env, #history...), so old links and every
// module's Tabs.onChange(name) keep working. No router, no build step. The online copy opens on
// the Overview, the local dashboard on Simulator > Train.
// Pages are <main> elements; the simulator page is #app-main, the others #tab-<name>.
const Tabs = (() => {
  const GROUPS = {
    overview: ["overview"],
    results: ["policies", "training", "history"],
    simulator: ["sim", "env"],
    drone: ["drone", "imitation"],
    try: ["try"],
  };
  const NAMES = Object.values(GROUPS).flat();
  const groupOf = (name) => Object.keys(GROUPS).find((g) => GROUPS[g].includes(name));
  const HOME = window.DQ_STATIC ? "overview" : "sim";
  const lastSub = {};                     // group -> the sub-page shown last, reopened by its tab
  const listeners = [];
  let current = HOME;

  function page(name) {
    return document.getElementById(name === "sim" ? "app-main" : "tab-" + name);
  }

  // A page name always means that page ("drone" is the Fly page, though the group shares its name).
  function show(name) {
    if (!NAMES.includes(name)) name = HOME;
    const group = groupOf(name);
    current = name;
    lastSub[group] = name;
    NAMES.forEach((n) => { page(n).hidden = n !== name; });
    document.querySelectorAll(".tab-btn[data-group]").forEach((b) => {
      b.classList.toggle("active", b.dataset.group === group);
    });
    // one row of sub-tabs per group; single-page groups have none
    let anySub = false;
    document.querySelectorAll(".subtab-row").forEach((row) => {
      row.hidden = row.dataset.group !== group;
      anySub = anySub || !row.hidden;
    });
    document.body.classList.toggle("has-subtabs", anySub);
    document.querySelectorAll(".subtab-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.tab === name);
    });
    // The header pill is the simulator's training status; elsewhere it misleads.
    document.getElementById("status-pill").hidden = name !== "sim";
    if (location.hash !== "#" + name) history.replaceState(null, "", "#" + name);
    // The 3D canvas sizes itself on resize; it measured zero while hidden.
    window.dispatchEvent(new Event("resize"));
    listeners.forEach((fn) => { try { fn(name); } catch (e) { console.error(e); } });
  }

  function init() {
    document.querySelectorAll(".tab-btn[data-group]").forEach((b) => {
      // a header tab reopens the page last seen in its group
      b.addEventListener("click", () => show(lastSub[b.dataset.group] || GROUPS[b.dataset.group][0]));
    });
    document.querySelectorAll(".subtab-btn").forEach((b) => {
      b.addEventListener("click", () => show(b.dataset.tab));
    });
    window.addEventListener("hashchange", () => show(location.hash.slice(1)));
    show(location.hash.slice(1) || HOME);
  }

  return {
    init,
    show,
    current: () => current,
    onChange: (fn) => listeners.push(fn),
  };
})();
