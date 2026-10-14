// Keyboard flight from the browser. Tracks which keys are HELD (keydown/keyup) and
// streams the set to the backend over /ws/drone/<id>/keys, which forwards it to the
// flight as UDP. The flight hovers if it hears nothing for 0.4 s, so this sends a
// heartbeat every 100 ms while engaged, and an empty set whenever the page loses focus.
//
// L / Esc and X / Backspace are not streamed: they call the land / kill endpoints, the
// same flag files the buttons use.
const Keys = (() => {
  const HEARTBEAT_MS = 100;
  const HELD = {
    KeyW: "fb+", KeyS: "fb-", KeyD: "lr+", KeyA: "lr-",
    ArrowUp: "ud+", ArrowDown: "ud-", ArrowRight: "yaw+", ArrowLeft: "yaw-",
  };
  const ONCE = { KeyL: "land", Escape: "land", KeyX: "kill", Backspace: "kill" };
  const LABEL = {
    "fb+": "forward", "fb-": "back", "lr+": "strafe right", "lr-": "strafe left",
    "ud+": "up", "ud-": "down", "yaw+": "rotate right", "yaw-": "rotate left",
  };

  let flightId = null;
  let ws = null;
  let timer = null;
  let onPress = () => {};
  let onChange = () => {};
  const held = new Set();

  function engaged() { return flightId !== null; }

  function send() {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ held: [...held] }));
    }
  }

  function release() {
    held.clear();
    send();
    onChange([...held]);
  }

  // Inputs that take no typing. Any INPUT used to count as typing, so after clicking the
  // Camera checkbox mid-flight every flight key -- land and kill included -- was ignored
  // until something else was clicked.
  const NOT_TYPING = new Set(["checkbox", "radio", "button", "submit", "reset", "range",
                              "color", "file", "image"]);

  function typing(e) {
    const t = e.target;
    const tag = (t && t.tagName) || "";
    if (tag === "INPUT") return !NOT_TYPING.has((t.type || "text").toLowerCase());
    return tag === "SELECT" || tag === "TEXTAREA" || Boolean(t && t.isContentEditable);
  }

  function onKeyDown(e) {
    if (!engaged() || Tabs.current() !== "drone" || typing(e)) return;
    if (ONCE[e.code]) {
      e.preventDefault();
      if (!e.repeat) onPress(ONCE[e.code]);
      return;
    }
    const name = HELD[e.code];
    if (!name) return;
    e.preventDefault();                       // arrows would scroll the page
    if (held.has(name)) return;               // auto-repeat
    held.add(name);
    send();
    onChange([...held]);
  }

  function onKeyUp(e) {
    const name = HELD[e.code];
    if (!name || !held.has(name)) return;
    held.delete(name);
    send();
    onChange([...held]);
  }

  function engage(id) {
    if (flightId === id && ws) return;
    disengage();
    flightId = id;
    if (document.activeElement && document.activeElement.blur) document.activeElement.blur();
    ws = Api.openKeysSocket(id);
    ws.onclose = () => { ws = null; };
    timer = setInterval(send, HEARTBEAT_MS);
    onChange([]);
  }

  function disengage() {
    if (timer) { clearInterval(timer); timer = null; }
    held.clear();
    if (ws) {
      try { send(); ws.close(); } catch (e) { /* already closed */ }
      ws = null;
    }
    flightId = null;
    onChange(null);
  }

  function init(handlers) {
    onPress = handlers.onPress || onPress;
    onChange = handlers.onChange || onChange;
    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", release);
    document.addEventListener("visibilitychange", () => { if (document.hidden) release(); });
    Tabs.onChange((name) => { if (name !== "drone") release(); });
  }

  return {
    init, engage, disengage,
    get engaged() { return engaged(); },
    describe: (names) => (names || []).map((n) => LABEL[n] || n).join(" + ") || "hovering",
  };
})();
