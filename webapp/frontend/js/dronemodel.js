// The drone as a little Tello: body, camera nose, four arms, motors, spinning two-blade
// propellers and round prop guards. Shared by the live 3D view (scene3d.js) and the recorded
// replays (replay.js). Local frame: +x is the nose (where the camera looks), +y is up.
// A real Tello is about 10 cm across; SCALE draws it a little larger so it reads in a 2 m cage.
const DroneModel = (() => {
  const SCALE = 1.6;

  function build({ body = 0xe9eaf0, accent = 0x872b97, scale = SCALE } = {}) {
    const T = THREE;
    const g = new T.Group();
    const parts = new T.Group();
    parts.scale.setScalar(scale);
    g.add(parts);

    const bodyMat = new T.MeshStandardMaterial({ color: body, roughness: 0.55 });
    const accentMat = new T.MeshStandardMaterial({ color: accent, roughness: 0.5 });
    const darkMat = new T.MeshStandardMaterial({ color: 0x1f1f2e, roughness: 0.7 });
    const propMat = new T.MeshStandardMaterial({ color: 0x2a2a3a, roughness: 0.6,
                                                 transparent: true, opacity: 0.85 });
    const guardMat = new T.MeshStandardMaterial({ color: 0xcfd2dc, roughness: 0.6 });

    // body: a flat box with a coloured top plate and a battery bump at the back
    const hull = new T.Mesh(new T.BoxGeometry(0.056, 0.022, 0.044), bodyMat);
    parts.add(hull);
    const top = new T.Mesh(new T.BoxGeometry(0.046, 0.006, 0.036), accentMat);
    top.position.y = 0.014;
    parts.add(top);
    const battery = new T.Mesh(new T.BoxGeometry(0.02, 0.016, 0.03), darkMat);
    battery.position.set(-0.03, 0.002, 0);
    parts.add(battery);

    // the camera on the nose (black lens in a grey bezel), which is how you tell front from back
    const bezel = new T.Mesh(new T.CylinderGeometry(0.008, 0.008, 0.006, 16), bodyMat);
    bezel.rotation.z = Math.PI / 2;
    bezel.position.set(0.03, 0.0, 0);
    parts.add(bezel);
    const lens = new T.Mesh(new T.CylinderGeometry(0.005, 0.005, 0.004, 16), darkMat);
    lens.rotation.z = Math.PI / 2;
    lens.position.set(0.0335, 0.0, 0);
    parts.add(lens);

    // arms to four motors on the diagonals, each with a propeller and a guard ring
    const reach = 0.046;
    const props = [];
    [[1, 1], [1, -1], [-1, 1], [-1, -1]].forEach(([sx, sz], i) => {
      const len = Math.hypot(reach, reach);
      const arm = new T.Mesh(new T.BoxGeometry(len, 0.006, 0.008), bodyMat);
      arm.position.set(sx * reach / 2, 0, sz * reach / 2);
      arm.rotation.y = -Math.atan2(sz, sx);
      parts.add(arm);

      const motor = new T.Mesh(new T.CylinderGeometry(0.007, 0.007, 0.012, 12), darkMat);
      motor.position.set(sx * reach, 0.004, sz * reach);
      parts.add(motor);

      const prop = new T.Group();
      for (const a of [0, Math.PI]) {
        const blade = new T.Mesh(new T.BoxGeometry(0.03, 0.0015, 0.007), propMat);
        blade.position.x = Math.cos(a) * 0.015;
        blade.rotation.x = 0.25;            // a little pitch, so the blades catch the light
        const holder = new T.Group();
        holder.rotation.y = a;
        holder.add(blade);
        prop.add(holder);
      }
      prop.position.set(sx * reach, 0.011, sz * reach);
      prop.userData.dir = i === 0 || i === 3 ? 1 : -1;   // diagonal pairs spin opposite ways
      parts.add(prop);
      props.push(prop);

      const guard = new T.Mesh(new T.TorusGeometry(0.034, 0.0025, 6, 28), guardMat);
      guard.rotation.x = Math.PI / 2;
      guard.position.set(sx * reach, 0.011, sz * reach);
      parts.add(guard);
    });

    g.userData.props = props;
    g.userData.bodyMat = accentMat;   // setDroneColor() tints the top plate
    return g;
  }

  // Turn the propellers; dt in seconds. Shown slowed down (a real one is a blur).
  function spin(g, dt) {
    (g.userData.props || []).forEach((p) => { p.rotation.y += p.userData.dir * dt * 30; });
  }

  return { build, spin, SCALE };
})();
