// Three.js scene: the 2x2x2 m cage, one hoop, and the drone. The simulator
// is Z-up with the room's XY centred at the origin and the floor at z=0;
// Three.js is Y-up by convention, so every position goes through one
// conversion helper rather than juggling the axis swap in several places.
function simToThree(pos) {
  return new THREE.Vector3(pos[0], pos[2], pos[1]);
}

const PALETTE = {
  bg: 0x171721,
  room: 0x4a4a6a,
  floor: 0x262638,
  hoopRim: 0xff7130,
  hoopBoard: 0xeceef5,
  hoopPole: 0x34344a,
  droneBody: 0x872b97,
  droneArm: 0x1f1f2e,
  droneRotor: 0xff3c68,
  targetGhost: 0xff3c68,
};

const Scene3D = (() => {
  let renderer, scene, camera, controls;
  let room = [2, 2, 2];
  let hoopGroup, droneGroup;
  let dronePosCurrent = new THREE.Vector3(0, 0.1, 0);
  let dronePosTarget = new THREE.Vector3(0, 0.1, 0);
  let droneRotCurrent = new THREE.Euler(0, 0, 0);
  let droneRotTarget = new THREE.Euler(0, 0, 0);

  function buildDrone() {
    const g = new THREE.Group();
    const bodyMat = new THREE.MeshStandardMaterial({ color: PALETTE.droneBody });
    const armMat = new THREE.MeshStandardMaterial({ color: PALETTE.droneArm });
    const rotorMat = new THREE.MeshStandardMaterial({ color: PALETTE.droneRotor });

    const body = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.03, 0.06), bodyMat);
    g.add(body);

    const armLen = 0.09;
    const armPositions = [
      [1, 1], [1, -1], [-1, 1], [-1, -1],
    ];
    armPositions.forEach(([sx, sz]) => {
      const arm = new THREE.Mesh(
        new THREE.CylinderGeometry(0.004, 0.004, armLen, 6),
        armMat,
      );
      arm.rotation.z = Math.PI / 2;
      arm.position.set(sx * armLen / 2 * 0.8, 0, sz * armLen / 2 * 0.8);
      arm.rotation.y = Math.atan2(sz, sx);
      arm.rotation.z = Math.PI / 2;
      g.add(arm);

      const rotor = new THREE.Mesh(
        new THREE.CircleGeometry(0.035, 12),
        rotorMat,
      );
      rotor.rotation.x = -Math.PI / 2;
      rotor.position.set(sx * armLen * 0.8, 0.01, sz * armLen * 0.8);
      g.add(rotor);
    });

    return g;
  }

  function buildHoop() {
    const g = new THREE.Group();
    const rimR = 0.12;
    const rim = new THREE.Mesh(
      new THREE.TorusGeometry(rimR, 0.01, 8, 24),
      new THREE.MeshStandardMaterial({ color: PALETTE.hoopRim }),
    );
    rim.rotation.x = Math.PI / 2;
    g.add(rim);

    const board = new THREE.Mesh(
      new THREE.PlaneGeometry(0.4, 0.25),
      new THREE.MeshStandardMaterial({ color: PALETTE.hoopBoard, side: THREE.DoubleSide, transparent: true, opacity: 0.85 }),
    );
    board.position.set(0, 0.15, -rimR - 0.03);
    g.add(board);

    return g;
  }

  function buildRoom() {
    const g = new THREE.Group();
    const geo = new THREE.BoxGeometry(room[0], room[2], room[1]);
    const edges = new THREE.EdgesGeometry(geo);
    const lines = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: PALETTE.room }));
    lines.position.set(0, room[2] / 2, 0);
    g.add(lines);

    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(room[0], room[1]),
      new THREE.MeshStandardMaterial({ color: PALETTE.floor, side: THREE.DoubleSide }),
    );
    floor.rotation.x = -Math.PI / 2;
    g.add(floor);

    return g;
  }

  function init(canvas, initialRoom) {
    if (initialRoom) room = initialRoom;
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio || 1);

    scene = new THREE.Scene();
    scene.background = new THREE.Color(PALETTE.bg);

    camera = new THREE.PerspectiveCamera(55, 1, 0.05, 100);
    camera.position.set(3, 2.2, 3);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.target.set(0, room[2] / 2, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(2, 4, 3);
    scene.add(dirLight);

    scene.add(buildRoom());

    hoopGroup = buildHoop();
    scene.add(hoopGroup);
    setHoop([0.5, 0.5, 0.6]);

    droneGroup = buildDrone();
    scene.add(droneGroup);
    setDroneTarget([0, 0, 0.1], [0, 0, 0], true);

    resize();
    window.addEventListener("resize", resize);
    requestAnimationFrame(tick);
  }

  function resize() {
    const wrap = renderer.domElement.parentElement;
    const w = wrap.clientWidth, h = wrap.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  function setHoop(pos) {
    const p = simToThree(pos);
    hoopGroup.position.copy(p);
    // face the open side toward the room centre
    const angle = Math.atan2(-pos[1], -pos[0]);
    hoopGroup.rotation.y = angle;
  }

  function setDroneTarget(pos, rpy, snap = false) {
    dronePosTarget = simToThree(pos);
    // sim roll/pitch/yaw (about x, y, z) -> three (x, -z, y) axis remap for
    // a body rotated with the same convention as simToThree's position swap.
    droneRotTarget = new THREE.Euler(rpy[0], rpy[2], -rpy[1], "XYZ");
    if (snap) {
      dronePosCurrent.copy(dronePosTarget);
      droneRotCurrent.copy(droneRotTarget);
    }
  }

  function tick() {
    dronePosCurrent.lerp(dronePosTarget, 0.25);
    droneRotCurrent.x += (droneRotTarget.x - droneRotCurrent.x) * 0.25;
    droneRotCurrent.y += (droneRotTarget.y - droneRotCurrent.y) * 0.25;
    droneRotCurrent.z += (droneRotTarget.z - droneRotCurrent.z) * 0.25;
    droneGroup.position.copy(dronePosCurrent);
    droneGroup.rotation.copy(droneRotCurrent);

    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(tick);
  }

  return { init, setHoop, setDroneTarget, resize };
})();
