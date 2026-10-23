// Hash-based tabs: #overview, #policies, #training (results, js/showcase.js) and the dashboard's
// #sim, #env, #drone, #imitation, #history. No router, no build step. The online copy opens on
// the Overview, the local dashboard on the Simulator.
// Pages are <main> elements; the simulator page is #app-main, the others #tab-<name>.
const Tabs = (() => {
  const NAMES = ["overview", "policies", "training", "sim", "env", "drone", "imitation", "history"];
  const HOME = window.DQ_STATIC ? "overview" : "sim";
  const listeners = [];
  let current = HOME;

  function page(name) {
    return document.getElementById(name === "sim" ? "app-main" : "tab-" + name);
  }

  function show(name) {
    if (!NAMES.includes(name)) name = HOME;
    current = name;
    NAMES.forEach((n) => { page(n).hidden = n !== name; });
    document.querySelectorAll(".tab-btn[data-tab]").forEach((b) => {
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
    document.querySelectorAll(".tab-btn[data-tab]").forEach((b) => {
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
