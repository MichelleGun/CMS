// Load each camera frame once, then share the decoded image between its views.
const Frames = (() => {
  const states = new WeakMap();
  const REQUEST_TIMEOUT_MS = 2500;

  function stateFor(img) {
    if (!states.has(img)) {
      states.set(img, { generation: 0, controller: null, objectUrl: null, timestamp: null });
    }
    return states.get(img);
  }

  function reset(images) {
    for (const img of Array.isArray(images) ? images : [images]) {
      const state = stateFor(img);
      state.generation += 1;
      if (state.controller) state.controller.abort();
      state.controller = null;
      if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
      state.objectUrl = null;
      state.timestamp = null;
      img.removeAttribute("src");
    }
  }

  async function swap(images, url, onShown, onError) {
    const targets = [...new Set(Array.isArray(images) ? images : [images])];
    const targetStates = targets.map(stateFor);
    if (targetStates.some((state) => state.controller)) return null;

    const controller = new AbortController();
    const generations = targetStates.map((state) => state.generation);
    const current = () => targetStates.every((state, i) =>
      state.controller === controller && state.generation === generations[i]);
    targetStates.forEach((state) => { state.controller = controller; });
    let objectUrl = null;
    let preloader = null;
    let timedOut = false;
    const timeout = setTimeout(() => {
      timedOut = true;
      controller.abort();
      if (preloader) preloader.src = "";
    }, REQUEST_TIMEOUT_MS);

    try {
      const response = await fetch(url, { cache: "no-store", signal: controller.signal });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const timestampValue = Number(response.headers.get("X-Frame-Timestamp"));
      const timestamp = Number.isFinite(timestampValue) && timestampValue > 0
        ? timestampValue : null;
      if (timestamp !== null && targetStates.every((state) => state.timestamp === timestamp)) {
        if (onShown) onShown(timestamp);
        return timestamp;
      }

      const blob = await response.blob();
      if (!current()) return null;
      objectUrl = URL.createObjectURL(blob);
      preloader = new Image();
      await new Promise((resolve, reject) => {
        preloader.onload = resolve;
        preloader.onerror = () => reject(new Error("image decode failed"));
        preloader.src = objectUrl;
      });
      if (!current()) return null;

      const oldUrls = new Set(targetStates.map((state) => state.objectUrl).filter(Boolean));
      targets.forEach((img, i) => {
        img.src = objectUrl;
        targetStates[i].objectUrl = objectUrl;
        targetStates[i].timestamp = timestamp;
      });
      objectUrl = null;
      oldUrls.forEach((oldUrl) => URL.revokeObjectURL(oldUrl));
      if (onShown) onShown(timestamp);
      return timestamp;
    } catch (error) {
      if (current() && onError) {
        onError(new Error(timedOut ? "request timed out" : error.message));
      }
      return null;
    } finally {
      clearTimeout(timeout);
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      targetStates.forEach((state) => {
        if (state.controller === controller) state.controller = null;
      });
    }
  }

  return { reset, swap };
})();
