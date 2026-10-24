// A recorded simulator episode played back in 3D (data/replays/<id>.json, written by
// scripts/build_site.py --record). Sim frame: x, y on the floor, z up -> three.js: (x, z, -y).
// Used by the Overview and Policies tabs, locally and on the online copy alike.
class Replay {
  constructor(el, { id, title, tall = false, autoplay = true } = {}) {
    this.el = el;
    this.id = id;
    this.speed = 1;
    this.playing = autoplay;
    this.i = 0;
    this.ep = 0;
    el.classList.add("viewer");
    if (tall) el.classList.add("tall");
    el.innerHTML = `<div class="hud">loading…</div><canvas aria-label="3D replay of ${Show.esc(title || id)}"></canvas>
      <div class="ctl"><span class="title"></span>
      <button type="button" data-a="play">Pause</button>
      <select data-a="ep" aria-label="start layout"></select>
      <select data-a="speed" aria-label="speed"><option value="1">1×</option><option value="2">2×</option><option value="4">4×</option></select>
      <input type="range" min="0" max="1" value="0" data-a="seek" aria-label="time"></div>`;
    this.hud = el.querySelector(".hud");
    this.canvas = el.querySelector("canvas");
    this.ctl = (a) => el.querySelector(`[data-a="${a}"]`);
    this.ctl("play").addEventListener("click", () => { this.playing = !this.playing; this.ctl("play").textContent = this.playing ? "Pause" : "Play"; });
    this.ctl("speed").addEventListener("change", (e) => { this.speed = Number(e.target.value); });
    this.ctl("ep").addEventListener("change", (e) => this.load(Number(e.target.value)));
    this.ctl("seek").addEventListener("input", (e) => { this.i = Number(e.target.value); this.draw(true); });
    if (!autoplay) this.ctl("play").textContent = "Play";
    this.init3d();
    this.show(id, title);
    new IntersectionObserver((es) => { this.visible = es[0].isIntersecting; }).observe(el);
  }

  // Swap in another policy's recording; the same WebGL renderer is kept (browsers allow only a few
  // of them per page, and losing one turns a canvas white).
  show(id, title) {
    this.id = id;
    this.ticks = null;
    this.el.querySelector(".title").textContent = title || id;
    this.canvas.setAttribute("aria-label", `3D replay of ${title || id}`);
    this.hud.textContent = "loading…";
    const want = id;
    fetch(`data/replays/${id}.json`).then((r) => r.json()).then((d) => {
      if (this.id !== want) return;               // picked something else meanwhile
      this.data = d;
      this.ctl("ep").innerHTML = d.episodes.map((e, k) =>
        `<option value="${k}">start ${k + 1}: ${Show.esc(e.start.replace("zone_", "zone "))} · ${e.passes} pass${e.passes === 1 ? "" : "es"}</option>`).join("");
      this.load(Math.min(this.ep, d.episodes.length - 1));
    }).catch(() => { this.hud.textContent = "replay not available"; });
  }

  v(x, y, z) { return new THREE.Vector3(x, z, -y); }

  init3d() {
    const T = THREE;
    this.renderer = new T.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.canvas.addEventListener("webglcontextlost", (e) => {
      e.preventDefault();
      this.lost = true;
      this.hud.textContent = "3D view paused: the browser ran out of graphics memory. Reload the page.";
    });
    this.scene = new T.Scene();
    this.camera = new T.PerspectiveCamera(45, 1, 0.05, 50);
    this.camera.position.set(1.55, 1.75, 1.85);
    this.controls = new T.OrbitControls(this.camera, this.canvas);
    this.controls.target.set(0, 0.9, 0);
    this.controls.enableDamping = true;
    this.scene.add(new T.HemisphereLight(0xffffff, 0x666677, 1.0));
    const sun = new T.DirectionalLight(0xffffff, 0.6);
    sun.position.set(2, 4, 1);
    this.scene.add(sun);
    // cage: 2 x 2 x 2 m net, floor grid
    this.cageMat = new T.LineBasicMaterial({ transparent: true, opacity: 0.55 });
    this.cage = new T.LineSegments(new T.EdgesGeometry(new T.BoxGeometry(2, 2, 2)), this.cageMat);
    this.cage.position.set(0, 1, 0);
    this.scene.add(this.cage);
    this.grid = new T.GridHelper(2, 10);
    this.scene.add(this.grid);
    // hoop (torus) and drone
    this.ringMat = new T.MeshStandardMaterial({ color: 0x2f6fe0, roughness: 0.6 });
    this.ring = new T.Mesh(new T.TorusGeometry(0.21, 0.06, 16, 48), this.ringMat);
    this.scene.add(this.ring);
    this.drone = DroneModel.build();       // js/dronemodel.js, the same drone as the live view
    this.scene.add(this.drone);
    this.trailGeom = new T.BufferGeometry();
    this.trailMat = new T.LineBasicMaterial({ color: 0x3cc6ff, transparent: true, opacity: 0.8 });
    this.trail = new T.Line(this.trailGeom, this.trailMat);
    this.scene.add(this.trail);
    this.theme();
    const resize = () => {
      const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
      this.renderer.setSize(w, h, false);
      this.camera.aspect = w / Math.max(h, 1);
      this.camera.updateProjectionMatrix();
    };
    new ResizeObserver(resize).observe(this.canvas);
    resize();
    let last = performance.now();
    const loop = (now) => {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      if (this.ticks && this.playing && this.visible !== false) {
        this.acc = (this.acc || 0) + dt * this.data.hz * this.speed;
        const n = Math.floor(this.acc);
        if (n > 0) {
          this.acc -= n;
          this.i += n;
          if (this.i >= this.ticks.length) { this.i = 0; }      // loop the episode
          this.draw();
        }
      }
      if (!this.lost && (this.visible !== false || !this.rendered)) {      // off-screen viewers sleep
        if (this.playing) DroneModel.spin(this.drone, dt * this.speed);
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
        this.rendered = true;
      }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  theme() {
    // the dashboard's colours (css/style.css: --bg-panel, the room and net lines)
    this.scene.background = new THREE.Color(0x1f1f2e);
    this.cageMat.color = new THREE.Color(0x8a8aa6);
    this.scene.remove(this.grid);
    this.grid = new THREE.GridHelper(2, 10, 0x4a4a6a, 0x30304a);
    this.scene.add(this.grid);
  }

  load(k) {
    this.ep = k;
    const e = this.data.episodes[k];
    this.ticks = e.ticks;
    this.events = e.events;
    this.i = 0;
    this.acc = 0;
    const r = e.ring_radius + e.ring_noodle / 2;
    this.ring.geometry.dispose();
    this.ring.geometry = new THREE.TorusGeometry(r, e.ring_noodle / 2, 16, 48);
    const seek = this.ctl("seek");
    seek.max = String(this.ticks.length - 1);
    this.draw(true);
  }

  draw(rebuildTrail) {
    if (!this.ticks || !this.ticks.length) return;            // not loaded yet
    this.i = Math.max(0, Math.min(this.ticks.length - 1, this.i | 0));
    const t = this.ticks[this.i];
    const [x, y, z, yaw, rx, ry, rz, nx, ny, nz, passes, skill] = t;
    this.drone.position.copy(this.v(x, y, z));
    this.drone.rotation.y = yaw;
    this.ring.position.copy(this.v(rx, ry, rz));
    this.ring.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), this.v(nx, ny, nz).normalize());
    // trail: the last 6 s
    const from = Math.max(0, this.i - this.data.hz * 6);
    const pts = [];
    for (let j = from; j <= this.i; j++) pts.push(this.v(this.ticks[j][0], this.ticks[j][1], this.ticks[j][2]));
    this.trailGeom.setFromPoints(pts);
    const flash = (this.events || []).find((ev) => ev[1] === "passed" && this.i >= ev[0] && this.i - ev[0] < this.data.hz * 0.6);
    this.ringMat.color.set(flash ? 0x2fd17a : 0x2f6fe0);
    const ended = this.i === this.ticks.length - 1 ? ` · ${this.data.episodes[this.ep].end}` : "";
    const sk = this.data.skills && skill >= 0 ? ` · skill: ${this.data.skills[skill] || skill}` : "";
    this.hud.textContent = `${(this.i / this.data.hz).toFixed(1)} s · passes ${passes}${sk}${ended}`;
    this.ctl("seek").value = String(this.i);
  }
}
