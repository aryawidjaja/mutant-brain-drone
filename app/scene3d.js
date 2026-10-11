// Mutant brain drone: the 3D view (three.js). Classic script; defines window.makeScene3D(THREE, OrbitControls, host).
// Frames: the sim is ENU (x east, y north, z up); three.js is Y-up, so ENU (x, y, z) -> three (x, z, -y), a proper
// rotation (-90 deg about x). A body->world quaternion (w, x, y, z) in ENU maps to three as (x, z, -y, w).
// The drone mesh is drawn at roughly true Crazyflie size and is visual only (the physics never reads it).
'use strict';
window.makeScene3D = function (THREE, OrbitControls, host) {
  const V = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
  const E = (x, y, z = 0) => V(x, z, -y);
  let renderer, NO_GL = false;
  try {
    const probe = document.createElement('canvas'), gl = probe.getContext('webgl2') || probe.getContext('webgl');
    if (!gl) throw new Error('no WebGL');
    const lose = gl.getExtension('WEBGL_lose_context'); if (lose) lose.loseContext();
    renderer = new THREE.WebGLRenderer({ antialias: true });
  } catch (e) {
    NO_GL = true;
    const d = document.createElement('div');
    d.textContent = '3D view unavailable: this browser could not start WebGL. The map, panels and timeline still work.';
    d.style.cssText = 'position:absolute;inset:0;display:grid;place-items:center;padding:24px;text-align:center;opacity:.7';
    renderer = new Proxy({ domElement: d }, { get: (t, k) => (k in t ? t[k] : () => {}) });
  }
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  host.prepend(renderer.domElement); renderer.domElement.setAttribute('aria-hidden', 'true');
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(55, 1, 0.02, 3000);
  const hemi = new THREE.HemisphereLight(0xffffff, 0x556655, 1.4); scene.add(hemi);
  const sun = new THREE.DirectionalLight(0xffffff, 1.5); scene.add(sun);
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(4000, 4000), new THREE.MeshLambertMaterial({ color: 0x6b8a4c }));
  ground.rotation.x = -Math.PI / 2; scene.add(ground);
  const grid = new THREE.GridHelper(400, 40, 0x000000, 0x000000); grid.material.transparent = true; grid.material.opacity = 0.08; grid.position.y = 0.005; scene.add(grid);
  const worldG = new THREE.Group(); scene.add(worldG);

  // ---------- drone (X configuration, visual only): body x forward, y left, z up
  const role = new THREE.MeshStandardMaterial({ color: 0xf08a3c, roughness: 0.5 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x2a2d31, roughness: 0.7 }), pcb = new THREE.MeshStandardMaterial({ color: 0x3c4148, roughness: 0.6 });
  const drone = new THREE.Group(), body = new THREE.Group(); drone.add(body);
  const B = (x, y, z) => V(x, z, -y); // body FLU -> three local
  const box = (w, d, h, m, p) => { const o = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), m); o.position.copy(p); body.add(o); return o; };
  box(0.034, 0.03, 0.006, pcb, B(0, 0, 0));
  box(0.012, 0.01, 0.006, role, B(0.02, 0, 0.002)); // nose marker (+x body = forward)
  const props = [];
  [[1, -1], [-1, -1], [-1, 1], [1, 1]].forEach(([sx, sy], i) => { // motor order follows the C4 mixer (1 front-right, then clockwise)
    const p = B(0.0325 * sx, 0.0325 * sy, 0);
    const arm = box(0.046, 0.004, 0.003, dark, p.clone().multiplyScalar(0.5)); arm.rotation.y = Math.atan2(-p.z, p.x);
    const m = new THREE.Mesh(new THREE.CylinderGeometry(0.004, 0.004, 0.012, 8), dark); m.position.copy(p).add(V(0, 0.004, 0)); body.add(m);
    const prop = new THREE.Group(); prop.position.copy(p).add(V(0, 0.011, 0)); body.add(prop);
    const blade = new THREE.Mesh(new THREE.BoxGeometry(0.046, 0.0008, 0.005), new THREE.MeshBasicMaterial({ color: 0x9aa0a8, transparent: true, opacity: 0.85 })); prop.add(blade);
    const disc = new THREE.Mesh(new THREE.CircleGeometry(0.023, 20), new THREE.MeshBasicMaterial({ color: 0x9aa0a8, transparent: true, opacity: 0.0, side: THREE.DoubleSide, depthWrite: false }));
    disc.rotation.x = -Math.PI / 2; prop.add(disc);
    props.push({ prop, blade, disc, dir: i % 2 ? 1 : -1 });
  });
  scene.add(drone);
  // screen-space dot: the drone is ~10 cm across, so overviews mark it with a constant-size dot
  const dotTex = (() => { const c = document.createElement('canvas'); c.width = c.height = 64; const x = c.getContext('2d'); x.fillStyle = '#fff'; x.beginPath(); x.arc(32, 32, 26, 0, 7); x.fill(); x.lineWidth = 6; x.strokeStyle = '#111'; x.stroke(); return new THREE.CanvasTexture(c); })();
  const dot = new THREE.Sprite(new THREE.SpriteMaterial({ map: dotTex, sizeAttenuation: false, depthTest: false, color: 0xf08a3c }));
  dot.scale.set(0.022, 0.022, 1); dot.renderOrder = 3; scene.add(dot);

  // trail
  const TRAIL_N = 30000, trailPos = new Float32Array(TRAIL_N * 3);
  const trailGeo = new THREE.BufferGeometry(); trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPos, 3)); trailGeo.setDrawRange(0, 0);
  const trailLine = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color: 0xf08a3c })); trailLine.frustumCulled = false; scene.add(trailLine);
  // arrows: home estimate (solid), home truth (faint), wind (gust yellow)
  const mkArrow = (col, op = 1) => { const a = new THREE.ArrowHelper(V(1, 0, 0), V(), 1, col, 0.6, 0.3); a.line.material.transparent = a.cone.material.transparent = op < 1; a.line.material.opacity = a.cone.material.opacity = op; a.renderOrder = 2; scene.add(a); return a; };
  const estA = mkArrow(0xffffff), truA = mkArrow(0xffffff, 0.35), windA = mkArrow(0xffd23f);
  estA.visible = truA.visible = windA.visible = false; drone.visible = dot.visible = false; // until the first frame

  // ---------- world
  const disposeAll = (g) => g.traverse((o) => { if (o.geometry) o.geometry.dispose(); if (o.material) [].concat(o.material).forEach((m) => { if (m.map) m.map.dispose(); m.dispose(); }); });
  const col3 = (a) => new THREE.Color().setRGB(a[0], a[1], a[2], THREE.SRGBColorSpace);
  function patternTex(ap, kind) {
    const c = document.createElement('canvas'); c.width = c.height = 128; const x = c.getContext('2d');
    const css = (a) => `rgb(${a.map((v) => Math.round(v * 255)).join(',')})`;
    x.fillStyle = css(ap.base); x.fillRect(0, 0, 128, 128); x.fillStyle = x.strokeStyle = css(ap.accent); x.lineWidth = 9;
    if (ap.pattern === 'stripes') for (let k = -128; k < 256; k += 32) { x.beginPath(); x.moveTo(k, 0); x.lineTo(k + 64, 128); x.stroke(); }
    else if (ap.pattern === 'chevron') for (let k = 8; k < 128; k += 36) { x.beginPath(); x.moveTo(14, k + 20); x.lineTo(64, k - 4); x.lineTo(114, k + 20); x.stroke(); }
    else for (let i = 16; i < 128; i += 32) for (let j = 16; j < 128; j += 32) { x.beginPath(); x.arc(i, j, 7, 0, 7); x.fill(); }
    if (kind === 'pad') { x.strokeStyle = css(ap.accent); x.lineWidth = 6; x.strokeRect(5, 5, 118, 118); }
    const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
  }
  function objectMesh(o) {
    const m = new THREE.MeshStandardMaterial({ map: patternTex(o.appearance, o.kind), roughness: 0.75, flatShading: true });
    let g;
    const [w, d, h] = o.dims || { tent: [2.2, 1.8, 1.3], pad: [2, 2, 0.12], backpack: [0.6, 0.4, 0.7] }[o.kind]; // dims = [x, y, height] m
    if (o.kind === 'tent') { g = new THREE.ConeGeometry(Math.SQRT1_2, 1, 4, 1); g.rotateY(Math.PI / 4); g.scale(w, h, d); g.translate(0, h / 2, 0); }
    else if (o.kind === 'pad') { g = new THREE.BoxGeometry(w, Math.max(0.03, h), d); g.translate(0, Math.max(0.03, h) / 2, 0); }
    else { g = new THREE.BoxGeometry(w, h, d); g.translate(0, h / 2, 0); }
    const mesh = new THREE.Mesh(g, m); mesh.position.copy(E(o.x, o.y, o.z || 0)); mesh.rotation.y = ((o.yaw_deg || 0) * Math.PI) / 180;
    return mesh;
  }
  const objMeshes = [];
  let bbox = { x0: -40, x1: 40, y0: -40, y1: 40 };
  function setWorld(world) {
    disposeAll(worldG); worldG.clear(); objMeshes.length = 0;
    if (!world) return;
    ground.material.color.copy(col3(world.ground ? world.ground.rgb : [0.4, 0.55, 0.3]));
    const ob = world.obstacles || [], by = (t) => ob.filter((o) => o.type === t);
    const inst = (list, geo, mat, place) => {
      if (!list.length) return;
      const im = new THREE.InstancedMesh(geo, mat, list.length), M = new THREE.Matrix4(), q = new THREE.Quaternion(), s = V(), p = V();
      list.forEach((o, i) => { const r = place(o, p, s, q); M.compose(p, q, s); im.setMatrixAt(i, M); im.setColorAt(i, r); });
      worldG.add(im);
    };
    const up = V(0, 1, 0);
    const trees = by('tree');
    inst(trees, new THREE.CylinderGeometry(1, 1, 1, 6).translate(0, 0.5, 0), new THREE.MeshLambertMaterial(), (o, p, s, q) => { p.copy(E(o.x, o.y)); s.set(0.12 + 0.12 * o.r, o.h * 0.45, 0.12 + 0.12 * o.r); q.identity(); return new THREE.Color(0x5b4632); }); // r = obstacle (canopy) radius
    inst(trees, new THREE.ConeGeometry(1, 1, 7).translate(0, 0.5, 0), new THREE.MeshLambertMaterial({ flatShading: true }), (o, p, s, q) => { p.copy(E(o.x, o.y, o.h * 0.3)); s.set(o.r, o.h * 0.7, o.r); q.setFromAxisAngle(up, (o.x * 7.1 + o.y * 3.3) % 6.28); return col3(o.rgb); });
    inst(by('building'), new THREE.BoxGeometry(1, 1, 1).translate(0, 0.5, 0), new THREE.MeshLambertMaterial(), (o, p, s, q) => { p.copy(E(o.x, o.y)); s.set(o.w, o.h, o.d); q.setFromAxisAngle(up, ((o.yaw_deg || 0) * Math.PI) / 180); return col3(o.rgb); });
    inst(by('rock'), new THREE.DodecahedronGeometry(1, 0), new THREE.MeshLambertMaterial({ flatShading: true }), (o, p, s, q) => { p.copy(E(o.x, o.y, o.h * 0.35)); s.set(o.r, o.h * 0.7, o.r * 0.9); q.setFromAxisAngle(up, o.x + o.y); return col3(o.rgb); });
    for (const o of world.objects || []) { const m = objectMesh(o); worldG.add(m); objMeshes.push([o, m]); }
    const home = new THREE.Mesh(new THREE.CylinderGeometry(0.6, 0.6, 0.02, 24), new THREE.MeshLambertMaterial({ color: 0xbfc4ca }));
    home.position.copy(E(world.home[0], world.home[1], 0.01)); worldG.add(home);
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.45, 0.55, 24), new THREE.MeshBasicMaterial({ color: 0x222222 })); ring.rotation.x = -Math.PI / 2; ring.position.copy(E(world.home[0], world.home[1], 0.025)); worldG.add(ring);
    const xs = [world.home[0], ...(world.objects || []).map((o) => o.x)], ys = [world.home[1], ...(world.objects || []).map((o) => o.y)];
    bbox = { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) };
    if (world.sun) { const az = (world.sun.azimuth_deg * Math.PI) / 180, el = (world.sun.elevation_deg * Math.PI) / 180; sun.position.copy(E(Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el)).multiplyScalar(200)); }
    env.light = world.light ?? 1; env.fog = world.fog_m ?? null; applyEnv();
    trailGeo.setDrawRange(0, 0);
  }
  // ---------- theme + light + fog
  const env = { theme: 'dark', light: 1, fog: null, role: '#f08a3c' };
  const SKY = { dark: 0x2a3340, light: 0xc9d8e8 };
  function applyEnv() {
    const L = env.light, sky = new THREE.Color(SKY[env.theme]).multiplyScalar(0.25 + 0.75 * L);
    scene.background = sky;
    scene.fog = env.fog ? new THREE.Fog(sky, Math.max(0.5, env.fog * 0.15), env.fog) : null;
    hemi.intensity = 0.25 + 1.15 * L; sun.intensity = 1.6 * L;
    grid.material.color.set(env.theme === 'light' ? 0x000000 : 0xffffff);
  }
  function setTheme(theme, css, arbCol) {
    env.theme = theme; env.role = arbCol; applyEnv();
    role.color.set(arbCol); dot.material.color.set(arbCol); trailLine.material.color.set(arbCol);
    estA.setColor(new THREE.Color(css.ink)); truA.setColor(new THREE.Color(css.ink)); windA.setColor(new THREE.Color(css.gust));
  }

  // ---------- camera
  const controls = OrbitControls && !NO_GL ? new OrbitControls(camera, renderer.domElement) : null;
  if (controls) { controls.enabled = false; controls.enableDamping = true; controls.dampingFactor = 0.12; controls.minDistance = 0.2; controls.maxDistance = 800; }
  const cam = { mode: 'chase', d: 0.7, yaw: 0, focus: V(), prevFocus: V() };
  renderer.domElement.addEventListener('wheel', (e) => { if (cam.mode !== 'chase') return; e.preventDefault(); cam.d = Math.min(40, Math.max(0.25, cam.d * Math.exp(e.deltaY * 0.0015))); }, { passive: false });
  function setCam(mode) {
    if (mode === 'free' && !controls) return;
    cam.mode = mode;
    if (controls) { controls.enabled = mode === 'free'; if (mode === 'free') { controls.target.copy(cam.focus); controls.update(); } }
    renderer.domElement.style.touchAction = mode === 'free' ? 'none' : 'auto';
  }
  function placeCamera(dt) {
    if (cam.mode === 'free') { const d = cam.focus.clone().sub(cam.prevFocus); camera.position.add(d); controls.target.add(d); controls.update(); }
    else if (cam.mode === 'chase') {
      const k = 1 - Math.exp(-dt * 4); let dy = cam.targetYaw - cam.yaw; dy = Math.atan2(Math.sin(dy), Math.cos(dy)); cam.yaw += dy * k;
      const f = E(Math.cos(cam.yaw), Math.sin(cam.yaw), 0);
      camera.up.set(0, 1, 0);
      camera.position.copy(cam.focus).addScaledVector(f, -cam.d).add(V(0, cam.d * 0.35 + 0.05, 0));
      if (camera.position.y < 0.08) camera.position.y = 0.08;
      camera.lookAt(cam.focus.clone().addScaledVector(f, cam.d * 0.6));
    } else { // top: frame home, objects and the drone, north up
      const x0 = Math.min(bbox.x0, cam.focusE[0]) - 8, x1 = Math.max(bbox.x1, cam.focusE[0]) + 8, y0 = Math.min(bbox.y0, cam.focusE[1]) - 8, y1 = Math.max(bbox.y1, cam.focusE[1]) + 8;
      const tv = Math.tan((camera.fov * Math.PI) / 360), h = Math.max((y1 - y0) / 2 / tv, (x1 - x0) / 2 / (tv * camera.aspect)) * 1.05;
      const c = E((x0 + x1) / 2, (y0 + y1) / 2, 0);
      camera.position.copy(c).add(V(0, h, 0.001)); camera.up.set(0, 0, -1); camera.lookAt(c);
    }
    cam.prevFocus.copy(cam.focus);
  }
  cam.targetYaw = 0; cam.focusE = [0, 0];

  // ---------- per-frame update
  const q3 = new THREE.Quaternion();
  function update(f, trail, opt) {
    if (!f) return;
    const p = f.quad.pos, q = f.quad.quat;
    drone.visible = true;
    drone.position.copy(E(p[0], p[1], p[2])); q3.set(q[1], q[3], -q[2], q[0]); drone.quaternion.copy(q3);
    dot.position.copy(drone.position); dot.visible = cam.mode !== 'chase' || cam.d > 4;
    cam.focus.copy(drone.position); cam.focusE = [p[0], p[1]]; cam.targetYaw = (f.quad.euler_deg[2] * Math.PI) / 180;
    const ms = f.quad.motors || [0, 0, 0, 0];
    props.forEach((pr, i) => { const w = ms[i] || 0; pr.prop.rotation.y += pr.dir * Math.min(1.1, w * 2.2); pr.disc.material.opacity = Math.min(0.35, w * 0.5); pr.blade.visible = w < 0.08; });
    const n = Math.min(trail.n, TRAIL_N);
    if (n !== trailGeo.drawRange.count) {
      for (let i = trailGeo.drawRange.count; i < n; i++) { trailPos[3 * i] = trail.xyz[3 * i]; trailPos[3 * i + 1] = trail.xyz[3 * i + 2]; trailPos[3 * i + 2] = -trail.xyz[3 * i + 1]; }
      if (n < trailGeo.drawRange.count) for (let i = 0; i < n; i++) { trailPos[3 * i] = trail.xyz[3 * i]; trailPos[3 * i + 1] = trail.xyz[3 * i + 2]; trailPos[3 * i + 2] = -trail.xyz[3 * i + 1]; }
      trailGeo.attributes.position.needsUpdate = true; trailGeo.setDrawRange(0, n);
    }
    const o = drone.position.clone().add(V(0, -0.03, 0));
    const setA = (a, v, show) => { const L = Math.hypot(v[0], v[1]); a.visible = show && L > 0.05; if (!a.visible) return; a.position.copy(o); a.setDirection(E(v[0] / L, v[1] / L, 0)); const hd = Math.min(L * 0.3, 0.12 + L * 0.02); a.setLength(L, hd, hd * 0.5); };
    setA(estA, f.pi ? f.pi.home_est : [0, 0], !!f.pi && opt.arrows);
    setA(truA, f.pi ? f.pi.home_true : [0, 0], !!f.pi && opt.arrows);
    const w = f.wind || [0, 0, 0], ws = Math.hypot(w[0], w[1], w[2]);
    windA.visible = ws > 0.05; if (windA.visible) { windA.position.copy(drone.position).add(V(0, 0.12, 0)); windA.setDirection(E(w[0] / ws, w[1] / ws, w[2] / ws)); const L = 0.05 * ws * (cam.mode === 'chase' ? Math.max(1, cam.d / 0.7) : 1); windA.setLength(L, Math.min(L * 0.3, 0.04), Math.min(L * 0.08, 0.012)); }
    for (const [ob, m] of objMeshes) m.visible = !opt.present || opt.present(ob);
  }
  function resize() { const w = host.clientWidth, h = host.clientHeight; renderer.setSize(w, h, false); camera.aspect = w / Math.max(1, h); camera.updateProjectionMatrix(); }
  let lastT = performance.now();
  function render() { const now = performance.now(), dt = Math.min(0.1, (now - lastT) / 1000); lastT = now; placeCamera(dt); renderer.render(scene, camera); }
  const tmp = V();
  function project(x, y, z) { tmp.copy(E(x, y, z)).project(camera); if (tmp.z > 1) return null; return [(tmp.x * 0.5 + 0.5) * host.clientWidth, (-tmp.y * 0.5 + 0.5) * host.clientHeight]; }
  function clearTrail() { trailGeo.setDrawRange(0, 0); }
  return { setWorld, update, render, resize, setCam, setTheme, project, clearTrail, NO_GL, get cam() { return cam.mode; } };
};
