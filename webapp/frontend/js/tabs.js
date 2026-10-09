// Hash-based tabs: #sim, #drone, #imitation, #history. No router, no build step.
// Pages are <main> elements; the simulator page is #app-main, the others #tab-<name>.
const Tabs = (() => {
  const NAMES = ["sim", "drone", "imitation", "history"];
  const listeners = [];
  let current = "sim";

  function page(name) {
    return document.getElementById(name === "sim" ? "app-main" : "tab-" + name);
  }

  function show(name) {
    if (!NAMES.includes(name)) name = "sim";
    current = name;
    NAMES.forEach((n) => { page(n).hidden = n !== name; });
    document.querySelectorAll(".tab-btn").forEach((b) => {
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
    document.querySelectorAll(".tab-btn").forEach((b) => {
      b.addEventListener("click", () => show(b.dataset.tab));
    });
    window.addEventListener("hashchange", () => show(location.hash.slice(1)));
    show(location.hash.slice(1) || "sim");
  }

  return {
    init,
    show,
    current: () => current,
    onChange: (fn) => listeners.push(fn),
  };
})();
