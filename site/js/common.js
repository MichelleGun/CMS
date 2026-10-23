// Shared bits: theme toggle, nav highlight, data loading, number formatting.
const Site = (() => {
  const KEY = "hoop-site-theme";
  function applyTheme(t) {
    if (t) document.documentElement.setAttribute("data-theme", t);
    else document.documentElement.removeAttribute("data-theme");
  }
  try { applyTheme(localStorage.getItem(KEY)); } catch (e) { /* storage blocked: follow the system */ }

  function isDark() {
    const t = document.documentElement.getAttribute("data-theme");
    if (t) return t === "dark";
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }

  function initNav() {
    const here = location.pathname.split("/").pop() || "index.html";
    document.querySelectorAll("a.link").forEach((a) => {
      if (a.getAttribute("href") === here) a.setAttribute("aria-current", "page");
    });
    const b = document.getElementById("theme-btn");
    if (b) {
      const label = () => { b.textContent = isDark() ? "Light" : "Dark"; };
      label();
      b.addEventListener("click", () => {
        const next = isDark() ? "light" : "dark";
        applyTheme(next);
        try { localStorage.setItem(KEY, next); } catch (e) { /* ignore */ }
        label();
        document.dispatchEvent(new CustomEvent("themechange"));
      });
    }
  }

  async function data(name) {
    const r = await fetch(`data/${name}`);
    if (!r.ok) throw new Error(`data/${name}: ${r.status}`);
    return r.json();
  }

  const pct = (v) => (v === null || v === undefined ? "–" : `${Math.round(v * 100)}%`);
  const num = (v, d = 2) => (v === null || v === undefined ? "–" : Number(v).toFixed(d));
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  // markdown-ish cell text from HANDOVER: **bold**, `code`
  const md = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/`(.+?)`/g, "<code>$1</code>");

  document.addEventListener("DOMContentLoaded", initNav);
  return { data, pct, num, css, esc, md, isDark };
})();
