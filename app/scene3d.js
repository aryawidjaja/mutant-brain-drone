// Mutant brain drone: the 3D view (three.js). Classic script; defines window.makeScene3D(THREE, OrbitControls, host, X).
// X (optional addons): { Sky, OBJLoader, mergeGeometries, mergeVertices }. Without them the view falls back to simpler parts.
// Frames: the sim is ENU (x east, y north, z up); three.js is Y-up, so ENU (x, y, z) -> three (x, z, -y), a proper
// rotation (-90 deg about x). A body->world quaternion (w, x, y, z) in ENU maps to three as (x, z, -y, w).
// Render only. Obstacle footprints and heights are drawn exactly as world.js gives them (trees and rocks: cylinders of
// radius r and height h; buildings: boxes w x d x h); grass, textures, foliage shape, terrain relief and clouds are decorative
// and never reach the simulation (the drone's eye and collisions use the simplified geometry in app/sim).
'use strict';
window.makeScene3D = function (THREE, OrbitControls, host, X = {}) {
  const V = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
  const E = (x, y, z = 0) => V(x, z, -y);
  const HTTP = location.protocol !== 'file:';
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
  let renderer, NO_GL = false;
  try {
    const probe = document.createElement('canvas'), gl = probe.getContext('webgl2') || probe.getContext('webgl');
    if (!gl) throw new Error('no WebGL');
    const lose = gl.getExtension('WEBGL_lose_context'); if (lose) lose.loseContext();
    renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  } catch (e) {
    NO_GL = true;
    const d = document.createElement('div');
    d.textContent = '3D view unavailable: this browser could not start WebGL. The map, panels and timeline still work.';
    d.style.cssText = 'position:absolute;inset:0;display:grid;place-items:center;padding:24px;text-align:center;opacity:.7';
    renderer = new Proxy({ domElement: d, shadowMap: {} }, { get: (t, k) => (k in t ? t[k] : () => {}) });
  }
  renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  host.prepend(renderer.domElement); renderer.domElement.setAttribute('aria-hidden', 'true');
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(55, 1, 0.02, 4000);
  const hemi = new THREE.HemisphereLight(0xcfe0ff, 0x4a5a38, 0.6); scene.add(hemi);
  const sun = new THREE.DirectionalLight(0xfff2e0, 3); scene.add(sun, sun.target);
  sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.03;
  const sunDir = V(0.5, 0.7, 0.3).normalize();

  // ---------- quality levels (render only)
  const QUAL = {
    high: { dpr: 2, shadow: 2048, grass: 220000, patch: 40, lod: 70, clouds: 1, seg: 1 },
    balanced: { dpr: 1.5, shadow: 1024, grass: 90000, patch: 28, lod: 40, clouds: 1, seg: 1 },
    low: { dpr: 1, shadow: 0, grass: 20000, patch: 16, lod: 20, clouds: 0, seg: 0 },
  };
  let Q = QUAL.high, qName = 'high';

  // ---------- noise (JS, deterministic)
  const hash2 = (x, y) => { const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453; return s - Math.floor(s); };
  const vnoise = (x, y, P = 0) => {
    const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi, u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
    const w = (i) => (P ? ((i % P) + P) % P : i);
    const a = hash2(w(xi), w(yi)), b = hash2(w(xi + 1), w(yi)), c = hash2(w(xi), w(yi + 1)), d = hash2(w(xi + 1), w(yi + 1));
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
  };
  const fbm = (x, y, o = 4, P = 0) => { let s = 0, a = 0.5, f = 1, n = 0; for (let i = 0; i < o; i++) { s += a * vnoise(x * f, y * f, P ? P * f : 0); n += a; f *= 2; a *= 0.5; } return s / n; };
  const n3 = (x, y, z) => (fbm(x, y, 3) + fbm(y + 7.3, z, 3) + fbm(z - 3.1, x, 3)) / 3;

  // ---------- textures (CC0, ambientCG; see assets/textures/CREDITS.txt). Over file:// browsers refuse them in WebGL.
  const texL = new THREE.TextureLoader(), TEX = {};
  function tex(name, srgb, rep = 1) {
    if (!HTTP || NO_GL) return null;
    const key = name + rep; if (TEX[key]) return TEX[key];
    const t = texL.load('assets/textures/' + name + '.jpg', undefined, undefined, () => console.warn('texture missing: ' + name));
    t.wrapS = t.wrapT = THREE.RepeatWrapping; t.anisotropy = 8; t.repeat.set(rep, rep); if (srgb) t.colorSpace = THREE.SRGBColorSpace;
    return (TEX[key] = t);
  }
  const px = (rgba) => { const t = new THREE.DataTexture(new Uint8Array(rgba), 1, 1); t.needsUpdate = true; return t; };
  const WHITE = px([255, 255, 255, 255]);
  const noiseTex = (() => { // tileable value-noise, 256^2, for macro variation on the ground
    const N = 256, d = new Uint8Array(N * N * 4);
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) { const v = fbm(i / 32, j / 32, 5, 8); const k = 4 * (j * N + i); d[k] = d[k + 1] = d[k + 2] = clamp(v * 255, 0, 255); d[k + 3] = 255; }
    const t = new THREE.DataTexture(d, N, N); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.magFilter = THREE.LinearFilter; t.minFilter = THREE.LinearMipmapLinearFilter; t.generateMipmaps = true; t.needsUpdate = true; return t;
  })();
  const canvasTex = (w, h, draw) => { const c = document.createElement('canvas'); c.width = w; c.height = h; draw(c.getContext('2d'), w, h); const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t; };
  // alpha masks (alphaMap reads the green channel): white = opaque shadow
  const blobTex = canvasTex(64, 64, (x) => { x.fillStyle = '#000'; x.fillRect(0, 0, 64, 64); const g = x.createRadialGradient(32, 32, 0, 32, 32, 31); g.addColorStop(0, '#fff'); g.addColorStop(0.5, '#999'); g.addColorStop(1, '#000'); x.fillStyle = g; x.fillRect(0, 0, 64, 64); });
  const rectTex = canvasTex(64, 64, (x) => { x.fillStyle = '#000'; x.fillRect(0, 0, 64, 64); x.filter = 'blur(6px)'; x.fillStyle = '#fff'; x.fillRect(12, 12, 40, 40); });

  // ---------- sky (three.js Sky addon, Preetham model) + procedural clouds + fog, patched into its fragment shader
  let sky = null;
  if (X.Sky && !NO_GL) {
    sky = new X.Sky(); sky.scale.setScalar(1000); sky.frustumCulled = false;
    const m = sky.material, u = m.uniforms;
    Object.assign(u, { uCloud: { value: 1 }, uGain: { value: 0.6 }, uCloudOff: { value: new THREE.Vector2() }, uCloudCol: { value: new THREE.Color(1, 1, 1) }, uFog: { value: 0 }, uFogCol: { value: new THREE.Color() } });
    u.turbidity.value = 4.5; u.rayleigh.value = 1.4; u.mieCoefficient.value = 0.004; u.mieDirectionalG.value = 0.82;
    m.fragmentShader = m.fragmentShader.replace('void main() {', `
      uniform float uCloud, uGain; uniform vec2 uCloudOff; uniform vec3 uCloudCol; uniform float uFog; uniform vec3 uFogCol;
      float h21(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
      float vn(vec2 p) { vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
        return mix(mix(h21(i), h21(i + vec2(1, 0)), f.x), mix(h21(i + vec2(0, 1)), h21(i + vec2(1, 1)), f.x), f.y); }
      float cfbm(vec2 p) { float s = 0.0, a = 0.5; for (int i = 0; i < 6; i++) { s += a * vn(p); p = p * 2.03 + vec2(1.7, 9.2); a *= 0.5; } return s; }
      void main() {`).replace('gl_FragColor = vec4( retColor, 1.0 );', `
      retColor *= uGain;
      if (uCloud > 0.0 && direction.y > 0.0) {
        vec2 cp = direction.xz / (direction.y + 0.06) * 1.3 + uCloudOff;
        float n = cfbm(cp), n2 = cfbm(cp * 2.1 + 3.7);
        float cov = smoothstep(0.50, 0.74, n) * uCloud;
        float hz = smoothstep(0.0, 0.22, direction.y);
        float sunSide = pow(max(dot(direction, vSunDirection), 0.0), 8.0);
        float shade = 1.0 - 0.38 * smoothstep(0.55, 0.9, n2 * 0.5 + n * 0.5);
        vec3 cc = uCloudCol * shade * (1.0 + 0.8 * sunSide);
        retColor = mix(retColor, cc, cov * hz * 0.92);
      }
      gl_FragColor = vec4( retColor, 1.0 );`).replace('#include <colorspace_fragment>', '#include <colorspace_fragment>\n gl_FragColor.rgb = mix(gl_FragColor.rgb, uFogCol, uFog); // three applies fog after tone mapping, so match it here');
    scene.add(sky);
  }
  const pmrem = NO_GL ? null : new THREE.PMREMGenerator(renderer);
  let envRT = null;
  function bakeEnv() {
    if (!sky || !pmrem) return;
    const s = new THREE.Scene(), sk = new THREE.Mesh(sky.geometry, sky.material); sk.scale.setScalar(1000); s.add(sk);
    const rt = pmrem.fromScene(s, 0, 0.1, 2000);
    if (envRT) envRT.dispose(); envRT = rt; scene.environment = rt.texture;
  }

  const worldG = new THREE.Group(); scene.add(worldG);

  // ---------- drone: Crazyflie 2 meshes from MuJoCo Menagerie (MIT), body x forward, y left, z up
  const role = new THREE.MeshStandardMaterial({ color: 0xf08a3c, roughness: 0.4, emissive: 0xf08a3c, emissiveIntensity: 0.6 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x2a2d31, roughness: 0.7 }), pcb = new THREE.MeshStandardMaterial({ color: 0x3c4148, roughness: 0.6 });
  const drone = new THREE.Group(), body = new THREE.Group(), fallback = new THREE.Group(); drone.add(body); body.add(fallback);
  const B = (x, y, z) => V(x, z, -y); // body FLU -> three local
  const box = (w, d, h, m, p, parent = fallback) => { const o = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), m); o.position.copy(p); parent.add(o); return o; };
  box(0.034, 0.03, 0.006, pcb, B(0, 0, 0));
  const nose = box(0.006, 0.004, 0.003, role, B(0.0145, 0, 0.0165), body); // role-coloured marker on the nose (+x body = forward)
  const blurTex = canvasTex(128, 128, (x) => { // motion-blurred rotor disc
    const g = x.createRadialGradient(64, 64, 6, 64, 64, 63); g.addColorStop(0, 'rgba(200,206,225,.05)'); g.addColorStop(0.75, 'rgba(200,206,225,.32)'); g.addColorStop(0.92, 'rgba(225,230,245,.55)'); g.addColorStop(1, 'rgba(225,230,245,0)');
    x.fillStyle = g; x.beginPath(); x.arc(64, 64, 64, 0, 7); x.fill();
  });
  let props = [];
  const MOTORS = [[1, -1], [-1, -1], [-1, 1], [1, 1]]; // motor order follows the quad.js mixer (1 front-right, then clockwise)
  const pivot = (sx, sy) => B(sx > 0 ? 0.032 : -0.0333, sy > 0 ? 0.032 : -0.0334, 0); // motor axes measured from cf2_4.obj
  const mkDisc = (parent, y) => {
    const disc = new THREE.Mesh(new THREE.CircleGeometry(0.0245, 32), new THREE.MeshBasicMaterial({ map: blurTex, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false }));
    disc.rotation.x = -Math.PI / 2; disc.position.y = y; parent.add(disc); return disc;
  };
  MOTORS.forEach(([sx, sy], i) => {
    const p = B(0.0325 * sx, 0.0325 * sy, 0);
    const arm = box(0.046, 0.004, 0.003, dark, p.clone().multiplyScalar(0.5)); arm.rotation.y = Math.atan2(-p.z, p.x);
    const m = new THREE.Mesh(new THREE.CylinderGeometry(0.004, 0.004, 0.012, 8), dark); m.position.copy(p).add(V(0, 0.004, 0)); fallback.add(m);
    const prop = new THREE.Group(); prop.position.copy(p).add(V(0, 0.011, 0)); fallback.add(prop);
    const blade = new THREE.Mesh(new THREE.BoxGeometry(0.046, 0.0008, 0.005), new THREE.MeshBasicMaterial({ color: 0x9aa0a8 })); prop.add(blade);
    props.push({ prop, blade, disc: mkDisc(prop, 0), dir: i % 2 ? 1 : -1 });
  });
  if (X.OBJLoader && HTTP && !NO_GL) {
    const C = (r, g, b) => new THREE.Color().setRGB(r, g, b, THREE.SRGBColorSpace);
    const MATS = [ // cf2.xml materials, in mesh order cf2_0 ... cf2_6
      { color: C(0.792, 0.820, 0.933), roughness: 0.35, metalness: 0 }, { color: C(0.109, 0.184, 0.0), roughness: 0.45, metalness: 0 },
      { color: C(0.969, 0.878, 0.6), roughness: 0.3, metalness: 1 }, { color: C(0.631, 0.659, 0.678), roughness: 0.25, metalness: 0 },
      { color: C(0.898, 0.898, 0.898), roughness: 0.3, metalness: 1 }, { color: C(0.102, 0.102, 0.102), roughness: 0.55, metalness: 0 },
      { color: C(1, 1, 1), roughness: 0.5, metalness: 0 }];
    const ld = new X.OBJLoader();
    Promise.all(MATS.map((_, i) => ld.loadAsync(`assets/models/cf2/cf2_${i}.obj`))).then((objs) => {
      const geos = objs.map((g) => { let geo = null; g.traverse((o) => { if (o.isMesh && !geo) geo = o.geometry; }); geo.rotateX(-Math.PI / 2); return geo; });
      const cf = new THREE.Group();
      geos.forEach((geo, i) => { if (i) { const m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial(MATS[i])); m.castShadow = true; cf.add(m); } });
      // cf2_0 holds all four propellers: split it by quadrant so each spins about its own motor axis
      const pos = geos[0].attributes.position, nor = geos[0].attributes.normal, nP = [];
      MOTORS.forEach(([sx, sy], i) => {
        const P = [], N = [], c = pivot(sx, sy);
        for (let t = 0; t < pos.count; t += 3) {
          const cx = (pos.getX(t) + pos.getX(t + 1) + pos.getX(t + 2)) / 3, cy = -(pos.getZ(t) + pos.getZ(t + 1) + pos.getZ(t + 2)) / 3; // body x, y
          if (Math.sign(cx) !== sx || Math.sign(cy) !== sy) continue;
          for (let k = t; k < t + 3; k++) { P.push(pos.getX(k) - c.x, pos.getY(k), pos.getZ(k) - c.z); N.push(nor.getX(k), nor.getY(k), nor.getZ(k)); }
        }
        const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(P, 3)); g.setAttribute('normal', new THREE.Float32BufferAttribute(N, 3));
        const prop = new THREE.Group(); prop.position.set(c.x, 0, c.z); cf.add(prop);
        const blade = new THREE.Mesh(g, new THREE.MeshStandardMaterial(MATS[0])); prop.add(blade);
        nP.push({ prop, blade, disc: mkDisc(prop, 0.013), dir: i % 2 ? 1 : -1 });
      });
      body.add(cf); fallback.visible = false; props = nP;
    }).catch((e) => console.warn('Crazyflie meshes not loaded; using the simple drone', e));
  }
  drone.traverse((o) => { if (o.isMesh) o.castShadow = true; });
  scene.add(drone);
  // screen-space dot: the drone is ~9 cm across, so overviews mark it with a constant-size dot
  const dotTex = (() => { const c = document.createElement('canvas'); c.width = c.height = 64; const x = c.getContext('2d'); x.fillStyle = '#fff'; x.beginPath(); x.arc(32, 32, 26, 0, 7); x.fill(); x.lineWidth = 6; x.strokeStyle = '#111'; x.stroke(); return new THREE.CanvasTexture(c); })();
  const dot = new THREE.Sprite(new THREE.SpriteMaterial({ map: dotTex, sizeAttenuation: false, depthTest: false, color: 0xf08a3c, toneMapped: false, fog: false }));
  dot.scale.set(0.022, 0.022, 1); dot.renderOrder = 3; scene.add(dot);
  const droneBlob = new THREE.Mesh(new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ alphaMap: blobTex, color: 0x000000, transparent: true, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -8 }));
  scene.add(droneBlob);

  // trail: fades out toward its old end
  const TRAIL_N = 30000, trailPos = new Float32Array(TRAIL_N * 3);
  const trailGeo = new THREE.BufferGeometry(); trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPos, 3)); trailGeo.setDrawRange(0, 0);
  trailGeo.setAttribute('aIdx', new THREE.BufferAttribute(Float32Array.from({ length: TRAIL_N }, (_, i) => i), 1));
  const trailMat = new THREE.LineBasicMaterial({ color: 0xf08a3c, transparent: true, depthWrite: false, toneMapped: false });
  const trailU = { uN: { value: 0 } };
  trailMat.onBeforeCompile = (s) => {
    s.uniforms.uN = trailU.uN;
    s.vertexShader = 'attribute float aIdx; uniform float uN; varying float vA;\n' + s.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\nvA = clamp(0.2 + (aIdx - uN + 1800.0) / 1800.0, 0.2, 0.95);');
    s.fragmentShader = 'varying float vA;\n' + s.fragmentShader.replace('#include <opaque_fragment>', '#include <opaque_fragment>\ngl_FragColor.a *= vA;');
  };
  const trailLine = new THREE.Line(trailGeo, trailMat); trailLine.frustumCulled = false; scene.add(trailLine);
  // arrows: home estimate (solid), home truth (faint), wind (gust yellow)
  const mkArrow = (col, op = 1) => { const a = new THREE.ArrowHelper(V(1, 0, 0), V(), 1, col, 0.6, 0.3); [a.line.material, a.cone.material].forEach((m) => { m.transparent = op < 1; m.opacity = op; m.toneMapped = false; m.fog = false; }); a.renderOrder = 2; scene.add(a); return a; };
  const estA = mkArrow(0xffffff), truA = mkArrow(0xffffff, 0.35), windA = mkArrow(0xffd23f);
  estA.visible = truA.visible = windA.visible = false; drone.visible = dot.visible = droneBlob.visible = false; // until the first frame

  // ---------- terrain: flat at z = 0 near home and objects, faint relief elsewhere in the flying area, hills beyond it
  let land = 'meadow', rect = { x0: -60, x1: 60, y0: -60, y1: 60 }, keyPts = [[0, 0]];
  function hgt(x, y) {
    let dmin = 1e9; for (const [px, py] of keyPts) dmin = Math.min(dmin, Math.hypot(x - px, y - py));
    let h = 0.2 * (fbm(x / 22 + 3.1, y / 22 - 7.7, 3) - 0.5) * 2 * smooth(4, 14, dmin);
    const dx = Math.max(rect.x0 - x, 0, x - rect.x1), dy = Math.max(rect.y0 - y, 0, y - rect.y1), ramp = smooth(15, 320, Math.hypot(dx, dy));
    if (ramp > 0) h += ramp * (4 + 70 * Math.pow(fbm(x / 210 + 11, y / 210 - 5, 5), 1.6));
    return h;
  }
  const MASK_N = 256;
  const maskData = new Uint8Array(MASK_N * MASK_N * 4);
  const maskTex = new THREE.DataTexture(maskData, MASK_N, MASK_N); maskTex.magFilter = maskTex.minFilter = THREE.LinearFilter;
  const rectU = new THREE.Vector4(); // x0, y0 (ENU), 1/width, 1/height
  function grassDensity(x, y, world) {
    let d = land === 'meadow' ? 0.5 + 0.5 * smooth(0.25, 0.6, fbm(x / 9, y / 9, 3))
      : land === 'forest' ? 0.6 * smooth(0.42, 0.72, fbm(x / 12, y / 12, 3))
        : 0.95 * smooth(0.56, 0.66, fbm(x / 16 + 5, y / 16, 3));
    for (const o of world.obstacles || []) {
      if (o.type === 'building') { if (Math.abs(x - o.x) < o.w / 2 + 1.6 && Math.abs(y - o.y) < o.d / 2 + 1.6) return 0; continue; }
      const r = Math.hypot(x - o.x, y - o.y);
      if (o.type === 'tree') { if (r < 0.15 * o.r + 0.25) return 0; if (r < o.r) d *= 0.6 + 0.4 * r / o.r; }
      else if (r < o.r * 1.05) return 0;
    }
    for (const o of world.objects || []) { const s = Math.max(o.dims ? Math.max(o.dims[0], o.dims[1]) / 2 : 1, 0.4); if (Math.hypot(x - o.x, y - o.y) < s + 0.1) return 0; }
    if (Math.hypot(x - world.home[0], y - world.home[1]) < 0.9) return 0;
    return d;
  }
  function buildMask(world) {
    const W = rect.x1 - rect.x0, H = rect.y1 - rect.y0;
    for (let j = 0; j < MASK_N; j++) for (let i = 0; i < MASK_N; i++) {
      const x = rect.x0 + ((i + 0.5) / MASK_N) * W, y = rect.y0 + ((j + 0.5) / MASK_N) * H, k = 4 * (j * MASK_N + i);
      maskData[k] = clamp(grassDensity(x, y, world) * 255, 0, 255);
      maskData[k + 1] = clamp(((hgt(x, y) + 0.3) / 0.6) * 255, 0, 255); maskData[k + 2] = 0; maskData[k + 3] = 255;
    }
    maskTex.needsUpdate = true; rectU.set(rect.x0, rect.y0, 1 / W, 1 / H);
  }
  const LANDTEX = { // [A (where grass grows), B (bare), scale A m, scale B m, tint]
    meadow: ['grass', 'dirt', 3, 4, [1, 1, 1]], forest: ['dirt', 'soil', 4, 3, [0.7, 0.75, 0.62]], urban: ['grass', 'asphalt', 3, 5, [1, 1, 1]] };
  const groundU = { tA: { value: WHITE }, tB: { value: WHITE }, tN: { value: noiseTex }, uMask: { value: maskTex }, uRect: { value: rectU }, uSA: { value: 3 }, uSB: { value: 4 }, uTint: { value: new THREE.Color(1, 1, 1) }, uFlat: { value: new THREE.Color(0.36, 0.52, 0.24) }, uHasTex: { value: 0 }, uFarG: { value: 1 } };
  const groundMat = new THREE.MeshStandardMaterial({ roughness: 0.95, metalness: 0 });
  groundMat.onBeforeCompile = (s) => {
    Object.assign(s.uniforms, groundU);
    s.vertexShader = 'varying vec3 vGW;\n' + s.vertexShader.replace('#include <project_vertex>', '#include <project_vertex>\nvGW = (modelMatrix * vec4(transformed, 1.0)).xyz;');
    s.fragmentShader = `varying vec3 vGW; uniform sampler2D tA, tB, tN, uMask; uniform vec4 uRect; uniform float uSA, uSB, uHasTex, uFarG; uniform vec3 uTint, uFlat;\n` + s.fragmentShader.replace('#include <map_fragment>', `
      vec2 gp = vec2(vGW.x, -vGW.z);
      vec2 mu = (gp - uRect.xy) * uRect.zw;
      float ins = step(0.0, mu.x) * step(mu.x, 1.0) * step(0.0, mu.y) * step(mu.y, 1.0);
      float nz = texture2D(tN, gp / 90.0).r, nz2 = texture2D(tN, gp / 23.0 + 0.4).r;
      float g = mix(nz * uFarG, texture2D(uMask, mu).r * 0.85 + (nz - 0.5) * 0.3, ins);
      float m = smoothstep(0.2, 0.55, g);
      vec3 cA = mix(texture2D(tA, gp / uSA).rgb, texture2D(tA, gp / (uSA * 4.37) + 0.31).rgb, 0.45);
      vec3 cB = mix(texture2D(tB, gp / uSB).rgb, texture2D(tB, gp / (uSB * 3.71) + 0.17).rgb, 0.35);
      vec3 col = mix(cB * 0.78, cA, m) * uTint * (0.82 + 0.36 * nz2);
      diffuseColor.rgb *= mix(uFlat * (0.85 + 0.3 * nz2), col, uHasTex);`);
  };
  let ground = null;
  function buildGround() {
    const N = Q.seg ? 320 : 200, cx = (rect.x0 + rect.x1) / 2, cy = (rect.y0 + rect.y1) / 2, A = Math.max(rect.x1 - rect.x0, rect.y1 - rect.y0) * 0.6, Bc = 2600 - A;
    const map = (u) => A * u + Bc * Math.pow(u, 5);
    const P = new Float32Array((N + 1) * (N + 1) * 3), UV = new Float32Array((N + 1) * (N + 1) * 2), I = [];
    for (let j = 0; j <= N; j++) for (let i = 0; i <= N; i++) {
      const x = cx + map((2 * i) / N - 1), y = cy + map((2 * j) / N - 1), k = j * (N + 1) + i;
      P.set([x, hgt(x, y), -y], 3 * k); UV.set([x / 3, y / 3], 2 * k);
      if (i < N && j < N) I.push(k, k + 1, k + N + 1, k + 1, k + N + 2, k + N + 1);
    }
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(P, 3)); g.setAttribute('uv', new THREE.BufferAttribute(UV, 2)); g.setIndex(I); g.computeVertexNormals();
    if (ground) { ground.geometry.dispose(); ground.geometry = g; } else { ground = new THREE.Mesh(g, groundMat); ground.receiveShadow = true; scene.add(ground); }
  }

  // ---------- grass: instanced blades placed on a world-fixed lattice that wraps around the camera; wind sway from the frame
  const grassU = { uC: { value: new THREE.Vector2() }, uT: { value: 40 }, uMask: { value: maskTex }, uRect: { value: rectU }, uWind: { value: V() }, uTime: { value: 0 }, uOn: { value: 1 }, uGA: { value: new THREE.Color(0.16, 0.3, 0.07) }, uGB: { value: new THREE.Color(0.3, 0.42, 0.1) }, uHS: { value: 1 } };
  const grassMat = new THREE.MeshLambertMaterial({ side: THREE.DoubleSide, vertexColors: true });
  grassMat.onBeforeCompile = (s) => {
    Object.assign(s.uniforms, grassU);
    s.vertexShader = `attribute vec4 aOff; uniform vec2 uC; uniform float uT; uniform sampler2D uMask; uniform vec4 uRect; uniform vec3 uWind; uniform float uTime, uOn, uHS; uniform vec3 uGA, uGB;
      vec3 gPos; vec3 gNrm; vec3 gCol;
      void grassCalc() {
        vec2 o = uC - 0.5 * uT;
        vec2 b = o + mod(aOff.xy * uT - o, uT);
        vec2 mu = (vec2(b.x, -b.y) - uRect.xy) * uRect.zw;
        vec4 m = textureLod(uMask, mu, 0.0);
        float ins = step(0.0, mu.x) * step(mu.x, 1.0) * step(0.0, mu.y) * step(mu.y, 1.0);
        float keep = step(aOff.w, m.r) * ins * (1.0 - smoothstep(0.3 * uT, 0.5 * uT, length(b - uC))) * uOn;
        float r2 = fract(aOff.w * 17.13), r3 = fract(aOff.w * 31.7);
        float H = (0.06 + 0.13 * r2) * uHS * keep * (0.55 + 0.45 * m.r);
        float W = 0.014 + 0.012 * r3;
        float t = position.y, ca = cos(aOff.z), sa = sin(aOff.z);
        vec3 p = vec3(position.x * W * ca, t * H, position.x * W * sa);
        vec2 w = uWind.xz; float ws = length(w);
        float ph = uTime * (1.5 + 0.3 * ws) + dot(b, vec2(0.23, 0.17));
        float sw = sin(ph) * 0.6 + sin(ph * 2.3 + 1.7) * 0.25;
        vec2 dir = ws > 0.01 ? w / ws : vec2(0.7071);
        vec2 lean = dir * (min(ws * 0.1, 0.65) + sw * (0.05 + 0.04 * ws)) + vec2(sin(aOff.z * 3.0), cos(aOff.z * 3.0)) * 0.18;
        p.xz += lean * t * t * H * 1.3;
        p.y *= 1.0 - 0.3 * dot(lean, lean) * t;
        gPos = p + vec3(b.x, (m.g - 0.5) * 0.6 - 0.01, b.y);
        gNrm = normalize(vec3(-sa * 0.3, 1.0, ca * 0.3));
        gCol = mix(uGA, uGB, r3) * mix(0.35, 1.0, t);
      }\n` + s.vertexShader
      .replace('#include <color_vertex>', 'grassCalc(); vColor = gCol;')
      .replace('#include <beginnormal_vertex>', 'vec3 objectNormal = gNrm;')
      .replace('#include <begin_vertex>', 'vec3 transformed = gPos;');
    s.fragmentShader = s.fragmentShader.replace('#include <normal_fragment_begin>', 'float faceDirection = 1.0; vec3 normal = normalize(vNormal); vec3 nonPerturbedNormal = normal;');
  };
  const GRASS_MAX = QUAL.high.grass;
  const grassGeo = (() => {
    const g = new THREE.InstancedBufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute([-0.5, 0, 0, 0.5, 0, 0, -0.42, 0.38, 0, 0.42, 0.38, 0, -0.26, 0.72, 0, 0.26, 0.72, 0, 0, 1, 0], 3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute(new Array(21).fill(0).map((_, i) => (i % 3 === 1 ? 1 : 0)), 3));
    g.setIndex([0, 1, 3, 0, 3, 2, 2, 3, 5, 2, 5, 4, 4, 5, 6]);
    const off = new Float32Array(GRASS_MAX * 4); let s = 1;
    const rnd = () => { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646; };
    for (let i = 0; i < GRASS_MAX; i++) off.set([rnd(), rnd(), rnd() * 6.2832, rnd()], 4 * i);
    g.setAttribute('aOff', new THREE.InstancedBufferAttribute(off, 4)); g.instanceCount = GRASS_MAX; return g;
  })();
  const grass = new THREE.Mesh(grassGeo, grassMat); grass.frustumCulled = false; grass.receiveShadow = true; scene.add(grass);

  // ---------- shared materials for obstacles and objects
  const barkMat = new THREE.MeshStandardMaterial({ color: 0x8a7a68, map: tex('bark_color', true), normalMap: tex('bark_normal'), roughness: 0.95 });
  if (barkMat.map) { barkMat.map.repeat.set(2, 3); barkMat.normalMap.repeat.set(2, 3); }
  const leafMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.8 });
  leafMat.onBeforeCompile = (s) => { // leafy breakup: world-space noise darkens gaps and tilts normals (no textures needed)
    s.vertexShader = 'varying vec3 vLW;\n' + s.vertexShader.replace('#include <project_vertex>', `#include <project_vertex>
      vec4 lw = vec4(transformed, 1.0);
      #ifdef USE_INSTANCING
      lw = instanceMatrix * lw;
      #endif
      vLW = (modelMatrix * lw).xyz;`);
    s.fragmentShader = `varying vec3 vLW;
      float h31(vec3 p) { p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
      float vn3(vec3 x) { vec3 i = floor(x), f = fract(x); f = f * f * (3.0 - 2.0 * f);
        return mix(mix(mix(h31(i), h31(i + vec3(1, 0, 0)), f.x), mix(h31(i + vec3(0, 1, 0)), h31(i + vec3(1, 1, 0)), f.x), f.y),
                   mix(mix(h31(i + vec3(0, 0, 1)), h31(i + vec3(1, 0, 1)), f.x), mix(h31(i + vec3(0, 1, 1)), h31(i + vec3(1, 1, 1)), f.x), f.y), f.z); }\n` + s.fragmentShader
      .replace('#include <color_fragment>', `#include <color_fragment>
        float lf = vn3(vLW * 7.0), lf2 = vn3(vLW * 19.0 + 3.1);
        diffuseColor.rgb *= 0.5 + 0.75 * smoothstep(0.15, 0.85, lf * 0.6 + lf2 * 0.4) + 0.12 * lf2;`)
      .replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>
        { vec3 q = vLW * 13.0; float n0 = vn3(q);
          vec3 gr = vec3(vn3(q + vec3(0.15, 0, 0)), vn3(q + vec3(0, 0.15, 0)), vn3(q + vec3(0, 0, 0.15))) - n0;
          normal = normalize(normal + (viewMatrix * vec4(gr * 5.0, 0.0)).xyz); }`);
  };
  const rockMat = new THREE.MeshStandardMaterial({ map: tex('rock_color', true), normalMap: tex('rock_normal'), roughness: 0.9 });
  const wallMat = new THREE.MeshStandardMaterial({ map: tex('concrete_color', true), normalMap: tex('concrete_normal'), normalScale: new THREE.Vector2(0.6, 0.6), vertexColors: true, roughness: 0.85 });
  if (wallMat.map) { wallMat.map.repeat.set(0.25, 0.25); wallMat.normalMap.repeat.set(0.25, 0.25); }
  const roofMat = new THREE.MeshStandardMaterial({ map: tex('asphalt_color', true, 0.4), color: 0xd8d2c8, roughness: 0.95 });
  wallMat.onBeforeCompile = (s) => {
    s.vertexShader = 'attribute vec3 aInfo; varying vec3 vInfo; varying vec2 vM;\n' + s.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\nvInfo = aInfo; vM = uv;');
    s.fragmentShader = `varying vec3 vInfo; varying vec2 vM;
      float bnd(float lo, float hi, float x, float e) { return smoothstep(lo - e, lo + e, x) * (1.0 - smoothstep(hi - e, hi + e, x)); }\n` + s.fragmentShader.replace('#include <roughnessmap_fragment>', `#include <roughnessmap_fragment>
      {
        float isW = step(0.5, vInfo.z), L = vInfo.x, H = vInfo.y, fl = 3.2, bay = 2.8;
        float nb = max(floor((L - 1.2) / bay), 1.0), off = (L - nb * bay) * 0.5;
        float u = vM.x - off, v = vM.y;
        float inU = step(0.0, u) * step(u, nb * bay) * step(0.0, v) * step(v, H - 1.1);
        float cu = fract(u / bay), cv = fract(v / fl), gf = step(v, fl);
        float eu = max(fwidth(u / bay), 0.004), ev = max(fwidth(v / fl), 0.004);
        float wl = mix(0.3, 0.06, gf), wh = mix(0.84, 0.8, gf);
        float win = isW * inU * bnd(0.2, 0.8, cu, eu) * bnd(wl, wh, cv, ev);
        float frm = isW * inU * bnd(0.16, 0.84, cu, eu) * bnd(wl - 0.04, wh + 0.04, cv, ev) * (1.0 - win);
        float sill = isW * step(0.0, v) * (1.0 - smoothstep(0.0, 0.05 + ev, abs(cv - 0.0))) * step(fl * 0.9, v);
        float lit = step(0.86, fract(sin(dot(floor(vec2(u / bay, v / fl)), vec2(12.9898, 78.233))) * 43758.5453));
        vec3 glass = mix(vec3(0.03, 0.045, 0.06), vec3(0.22, 0.2, 0.15), lit * 0.6);
        diffuseColor.rgb *= 1.0 - 0.35 * frm - 0.25 * sill;
        diffuseColor.rgb = mix(diffuseColor.rgb, glass, win);
        roughnessFactor = mix(roughnessFactor, 0.06, win);
      }`);
  };
  const blobMat = new THREE.MeshBasicMaterial({ color: 0x000000, alphaMap: blobTex, transparent: true, opacity: 0.55, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -4 });
  const rectMat = blobMat.clone(); rectMat.alphaMap = rectTex; rectMat.opacity = 0.5;

  // object fabric: the same base/accent pattern the drone's eye uses (app/sim/brain/eye.js hitObj), in object-local metres
  const fabricN = tex('fabric_normal', false, 1);
  const PATS = { stripes: 0, chevron: 1, dots: 2, none: 3 };
  const col3 = (a) => new THREE.Color().setRGB(a[0], a[1], a[2], THREE.SRGBColorSpace);
  function fabricMat(ap, opt = {}) {
    const m = new THREE.MeshStandardMaterial({ roughness: opt.rough ?? 0.8, normalMap: fabricN, normalScale: new THREE.Vector2(0.5, 0.5), side: opt.side ?? THREE.FrontSide });
    const U = { uBase: { value: col3(ap.base) }, uAcc: { value: col3(ap.accent) }, uPer: { value: ap.period_m || 0.5 }, uPat: { value: PATS[ap.pattern] ?? 3 }, uDoor: { value: col3([0.12, 0.10, 0.08]) }, uDoorOn: { value: opt.door ? 1 : 0 }, uHalfD: { value: opt.halfD || 1 }, uH: { value: opt.h || 1 }, uMarkH: { value: opt.markH ? 1 : 0 } };
    m.onBeforeCompile = (s) => {
      Object.assign(s.uniforms, U);
      s.vertexShader = 'varying vec3 vLP; varying vec3 vLN;\n' + s.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\nvLP = position; vLN = normal;');
      s.fragmentShader = `varying vec3 vLP; varying vec3 vLN; uniform vec3 uBase, uAcc, uDoor; uniform float uPer, uDoorOn, uHalfD, uH, uMarkH; uniform int uPat;\n` + s.fragmentShader.replace('#include <map_fragment>', `
        vec3 P = vec3(vLP.x, -vLP.z, vLP.y), N = abs(vec3(vLN.x, -vLN.z, vLN.y));
        int ax = (N.x >= N.y && N.x >= N.z) ? 0 : (N.y >= N.z ? 1 : 2);
        float u = ax == 0 ? P.y : P.x, v = ax == 2 ? P.y : P.z;
        float fu = fract(u / uPer), fv = fract(v / uPer), fc = fract((abs(u) + v) / uPer);
        float e = clamp(fwidth(u / uPer) + fwidth(v / uPer), 0.002, 0.25);
        float on = uPat == 0 ? 1.0 - smoothstep(0.5 - e, 0.5 + e, fu)
                 : uPat == 1 ? 1.0 - smoothstep(0.5 - e, 0.5 + e, fc)
                 : uPat == 2 ? 1.0 - smoothstep(0.09 - e, 0.09 + e, (fu - 0.5) * (fu - 0.5) + (fv - 0.5) * (fv - 0.5)) : 0.0;
        vec3 col = mix(uBase, uAcc, clamp(on, 0.0, 1.0));
        if (uMarkH > 0.5 && ax == 2) { // landing "H" and ring on the top face
          vec2 q = abs(vec2(P.x, P.y)) / uHalfD; float r = length(vec2(P.x, P.y)) / uHalfD;
          float hm = (step(q.y, 0.42) * step(0.22, q.x) * step(q.x, 0.34)) + step(q.x, 0.34) * step(q.y, 0.06);
          float ring = step(0.62, r) * step(r, 0.72);
          col = mix(col, uBase, step(r, 0.62)); // plain centre disc so the H reads on any pattern
          col = mix(col, uAcc, clamp(hm + ring, 0.0, 1.0));
          col = mix(col, uBase * 0.35, step(0.94, max(q.x, q.y)));
        }
        if (uDoorOn > 0.5 && ax == 0 && vLN.x > 0.0 && abs(P.y) < 0.4 * uHalfD && P.z < 0.75 * uH) col = abs(P.y) < 0.008 ? uAcc * 0.6 : uDoor;
        diffuseColor.rgb *= col;`);
    };
    return m;
  }
  const poleMat = new THREE.MeshStandardMaterial({ color: 0x9aa2ab, metalness: 0.9, roughness: 0.35 });
  const lineMat = new THREE.LineBasicMaterial({ color: 0xe8e4d8, transparent: true, opacity: 0.7 });
  const pegMat = new THREE.MeshStandardMaterial({ color: 0xc8ccd0, metalness: 0.6, roughness: 0.4 });

  // ---------- geometry builders (all unit or metric; three local coords: x = ENU x, y = up, z = -ENU y)
  const merge = (gs) => (X.mergeGeometries ? X.mergeGeometries(gs, false) : gs[0]);
  const setCol = (g, f) => { const p = g.attributes.position, c = new Float32Array(p.count * 3); for (let i = 0; i < p.count; i++) c.set(f(p.getX(i), p.getY(i), p.getZ(i)), 3 * i); g.setAttribute('color', new THREE.BufferAttribute(c, 3)); return g; };
  function coniferGeo(hi) {
    const tiers = [[0.2, 0.38, 1.0], [0.37, 0.35, 0.8], [0.53, 0.3, 0.6], [0.68, 0.32, 0.4]], gs = [];
    tiers.forEach(([y0, h, r], k) => {
      const g = new THREE.ConeGeometry(r, h, hi ? 18 : 8, hi ? 4 : 1, false); g.translate(0, y0 + h / 2, 0);
      const p = g.attributes.position;
      for (let i = 0; i < p.count; i++) { // ragged, drooping rim
        const x = p.getX(i), y = p.getY(i), z = p.getZ(i), a = Math.atan2(z, x), rr = Math.hypot(x, z), f = 1 + 0.16 * (vnoise(a * 2.2 + k * 5, y * 6) - 0.5) * 2;
        p.setXYZ(i, x * f, y - (rr / r) * 0.05 * (1 + Math.sin(a * 9 + k)) * 0.5, z * f);
      }
      g.computeVertexNormals();
      setCol(g, (x, y, z) => { const t = clamp((y - y0) / h, 0, 1), rr = Math.hypot(x, z) / r, v = (0.45 + 0.55 * t) * (0.75 + 0.25 * rr) * (0.85 + 0.3 * vnoise(x * 9, z * 9 + y * 5)); return [v, v * 1.02, v * 0.95]; });
      gs.push(g);
    });
    return merge(gs);
  }
  function broadleafGeo(hi) {
    const C = [[0, 0.62, 0, 0.46], [0.46, 0.52, 0.12, 0.4], [-0.44, 0.54, -0.14, 0.4], [0.12, 0.5, -0.47, 0.38], [-0.12, 0.56, 0.46, 0.38], [0.05, 0.8, 0.02, 0.32], [0.3, 0.72, 0.3, 0.3], [-0.3, 0.74, -0.25, 0.28]], gs = [];
    C.forEach(([cx, cy, cz, r], k) => {
      const g = new THREE.IcosahedronGeometry(r, hi ? 2 : 1), p = g.attributes.position, N = new Float32Array(p.count * 3);
      for (let i = 0; i < p.count; i++) {
        const x = p.getX(i), y = p.getY(i), z = p.getZ(i), L = Math.hypot(x, y, z) || 1, f = 1 + 0.32 * (n3(x * 4 + k, y * 4, z * 4) - 0.5) * 2;
        p.setXYZ(i, cx + x * f, cy + y * f * 0.85, cz + z * f); N.set([x / L, y / L, z / L], 3 * i);
      }
      g.setAttribute('normal', new THREE.BufferAttribute(N, 3));
      setCol(g, (x, y, z) => { const v = (0.5 + 0.5 * clamp((y - 0.25) / 0.75, 0, 1)) * (0.8 + 0.35 * n3(x * 7, y * 7, z * 7)); return [v * 1.04, v, v * 0.85]; });
      gs.push(g);
    });
    const g = merge(gs), p = g.attributes.position; // fit inside radius 1, top at 1
    let rmax = 0, ymax = 0; for (let i = 0; i < p.count; i++) { rmax = Math.max(rmax, Math.hypot(p.getX(i), p.getZ(i))); ymax = Math.max(ymax, p.getY(i)); }
    for (let i = 0; i < p.count; i++) p.setXYZ(i, p.getX(i) / rmax, 0.2 + ((p.getY(i) - 0.15) / (ymax - 0.15)) * 0.8, p.getZ(i) / rmax);
    return g;
  }
  const trunkGeo = (hi, top) => { const g = new THREE.CylinderGeometry(0.07, 0.12, top + 0.06, hi ? 10 : 6, 1, true); g.translate(0, (top + 0.06) / 2 - 0.06, 0); return g; };
  function rockGeo(seed, hi) {
    let g = new THREE.IcosahedronGeometry(1, hi ? 4 : 2); g.deleteAttribute('normal'); g.deleteAttribute('uv');
    if (X.mergeVertices) g = X.mergeVertices(g);
    const p = g.attributes.position;
    for (let i = 0; i < p.count; i++) {
      const x = p.getX(i), y = p.getY(i), z = p.getZ(i), f = 1 + 0.42 * (n3(x * 1.3 + seed, y * 1.3, z * 1.3 - seed) - 0.5) * 2 + 0.08 * (n3(x * 5, y * 5 + seed, z * 5) - 0.5);
      p.setXYZ(i, x * f, Math.max(y * f, -0.25), z * f);
    }
    let rmax = 0, y0 = 1e9, y1 = -1e9; for (let i = 0; i < p.count; i++) { rmax = Math.max(rmax, Math.hypot(p.getX(i), p.getZ(i))); y0 = Math.min(y0, p.getY(i)); y1 = Math.max(y1, p.getY(i)); }
    const UV = new Float32Array(p.count * 2);
    for (let i = 0; i < p.count; i++) { const x = p.getX(i) / rmax, y = -0.12 + ((p.getY(i) - y0) / (y1 - y0)) * 1.12, z = p.getZ(i) / rmax; p.setXYZ(i, x, y, z); UV.set([(x + z) * 0.7, y * 0.9 + x * 0.3], 2 * i); }
    g.setAttribute('uv', new THREE.BufferAttribute(UV, 2)); g.computeVertexNormals(); return g;
  }
  function buildingGeo(list) {
    const P = [], N = [], UV = [], C = [], INF = [], groups = [[], []];
    const quad = (pts, n, uvs, col, inf, grp) => { // pts: 4 three-space points (CCW seen from n)
      const b = P.length / 3;
      pts.forEach((p, k) => { P.push(p.x, p.y, p.z); N.push(n.x, n.y, n.z); UV.push(uvs[k][0], uvs[k][1]); C.push(col.r, col.g, col.b); INF.push(inf[0], inf[1], inf[2]); });
      const e1 = pts[1].clone().sub(pts[0]), e2 = pts[2].clone().sub(pts[0]), ok = e1.cross(e2).dot(n) > 0;
      groups[grp].push(...(ok ? [b, b + 1, b + 2, b, b + 2, b + 3] : [b, b + 2, b + 1, b, b + 3, b + 2]));
    };
    for (const o of list) {
      const col = col3(o.rgb).multiplyScalar(1.15), x0 = o.x - o.w / 2, x1 = o.x + o.w / 2, y0 = o.y - o.d / 2, y1 = o.y + o.d / 2, h = o.h, zb = -0.3, ins = 0.3, zr = h - 0.6;
      const walls = [[[x0, y0], [x1, y0], V(0, 0, 1)], [[x1, y0], [x1, y1], V(1, 0, 0)], [[x1, y1], [x0, y1], V(0, 0, -1)], [[x0, y1], [x0, y0], V(-1, 0, 0)]];
      for (const [[ax, ay], [bx, by], n] of walls) {
        const L = Math.hypot(bx - ax, by - ay);
        quad([E(ax, ay, zb), E(bx, by, zb), E(bx, by, h), E(ax, ay, h)], n, [[0, zb], [L, zb], [L, h], [0, h]], col, [L, h, 1], 0);
        // parapet inner face (no windows)
        const ix = (t) => [ax + (bx - ax) * t + (n.x ? -n.x * ins : 0) + (n.x ? 0 : 0), ay + (by - ay) * t + (n.z ? n.z * ins : 0)];
        const [px0, py0] = ix(0), [px1, py1] = ix(1);
        quad([E(px1, py1, zr), E(px0, py0, zr), E(px0, py0, h), E(px1, py1, h)], n.clone().negate(), [[L, zr], [0, zr], [0, h], [L, h]], col.clone().multiplyScalar(0.8), [L, h, 0], 0);
      }
      // parapet top rim (4 strips) and roof deck
      const rim = [[x0, y0, x1, y0 + ins], [x0, y1 - ins, x1, y1], [x0, y0 + ins, x0 + ins, y1 - ins], [x1 - ins, y0 + ins, x1, y1 - ins]];
      for (const [a, b, c, d] of rim) quad([E(a, b, h), E(c, b, h), E(c, d, h), E(a, d, h)], V(0, 1, 0), [[a, b], [c, b], [c, d], [a, d]], col.clone().multiplyScalar(0.9), [1, h, 0], 0);
      quad([E(x0 + ins, y0 + ins, zr), E(x1 - ins, y0 + ins, zr), E(x1 - ins, y1 - ins, zr), E(x0 + ins, y1 - ins, zr)], V(0, 1, 0), [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], new THREE.Color(1, 1, 1), [1, h, 0], 1);
      if (o.w > 5 && o.d > 5) { // rooftop unit, stays below the box top h
        const ux = o.x + (hash2(o.x, o.y) - 0.5) * (o.w - 3), uy = o.y + (hash2(o.y, o.x) - 0.5) * (o.d - 3), a = 0.7, b = 0.5, top = h - 0.15, gcol = new THREE.Color(0.55, 0.56, 0.57);
        quad([E(ux - a, uy - b, top), E(ux + a, uy - b, top), E(ux + a, uy + b, top), E(ux - a, uy + b, top)], V(0, 1, 0), [[0, 0], [1, 0], [1, 1], [0, 1]], gcol, [1, h, 0], 0);
        [[[ux - a, uy - b], [ux + a, uy - b], V(0, 0, 1)], [[ux + a, uy - b], [ux + a, uy + b], V(1, 0, 0)], [[ux + a, uy + b], [ux - a, uy + b], V(0, 0, -1)], [[ux - a, uy + b], [ux - a, uy - b], V(-1, 0, 0)]]
          .forEach(([[p, q], [r, s], n]) => quad([E(p, q, zr), E(r, s, zr), E(r, s, top), E(p, q, top)], n, [[0, 0], [1, 0], [1, 0.4], [0, 0.4]], gcol.clone().multiplyScalar(0.8), [1, h, 0], 0));
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(P, 3)); g.setAttribute('normal', new THREE.Float32BufferAttribute(N, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(UV, 2)); g.setAttribute('color', new THREE.Float32BufferAttribute(C, 3)); g.setAttribute('aInfo', new THREE.Float32BufferAttribute(INF, 3));
    g.setIndex([...groups[0], ...groups[1]]); g.addGroup(0, groups[0].length, 0); g.addGroup(groups[0].length, groups[1].length, 1);
    return g;
  }
  // box-projected UVs in metres (for the fabric normal map)
  const boxUV = (g, s = 0.3) => { const p = g.attributes.position, n = g.attributes.normal, uv = new Float32Array(p.count * 2); for (let i = 0; i < p.count; i++) { const ax = Math.abs(n.getX(i)), ay = Math.abs(n.getY(i)), az = Math.abs(n.getZ(i)); const [a, b] = ax > ay && ax > az ? [p.getZ(i), p.getY(i)] : ay > az ? [p.getX(i), p.getZ(i)] : [p.getX(i), p.getY(i)]; uv.set([a / s, b / s], 2 * i); } g.setAttribute('uv', new THREE.BufferAttribute(uv, 2)); return g; };
  function roundedBox(w, h, d, r, seg = 4) {
    const g = new THREE.BoxGeometry(w, h, d, seg, seg, seg), p = g.attributes.position, hx = w / 2 - r, hy = h / 2 - r, hz = d / 2 - r, v = V(), c = V();
    for (let i = 0; i < p.count; i++) { v.fromBufferAttribute(p, i); c.set(clamp(v.x, -hx, hx), clamp(v.y, -hy, hy), clamp(v.z, -hz, hz)); v.sub(c).normalize().multiplyScalar(r).add(c); p.setXYZ(i, v.x, v.y, v.z); }
    g.computeVertexNormals(); return boxUV(g);
  }
  const tube = (pts, r) => new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 24, r, 6, false);
  function guyLines(grp, pairs) {
    const P = []; pairs.forEach(([a, b]) => { P.push(a.x, a.y, a.z, b.x, b.y, b.z); const peg = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.004, 0.12, 5), pegMat); peg.position.copy(b).add(V(0, 0.03, 0)); grp.add(peg); });
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(P, 3)); grp.add(new THREE.LineSegments(g, lineMat));
  }
  function tentMesh(o, dims) {
    const [w, d, h] = dims, ap = o.appearance, grp = new THREE.Group(), dome = ap.pattern !== 'stripes';
    const mat = fabricMat(ap, { door: true, halfD: d / 2, h, side: THREE.DoubleSide });
    if (!dome) { // ridge tent: ridge along x, door on the +x gable
      const NX = 16, NS = 8, P = [], I = [];
      for (const side of [-1, 1]) {
        const b = P.length / 3;
        for (let j = 0; j <= NS; j++) for (let i = 0; i <= NX; i++) {
          const s = j / NS, x = -w / 2 + (i / NX) * w, sag = 0.07 * Math.sin(Math.PI * s) * Math.pow(Math.sin(Math.PI * i / NX), 0.6);
          const zz = side * (d / 2) * (1 - s), yy = h * s; // outward normal of this panel ~ (0, d/2, side*h) normalised
          const nn = Math.hypot(d / 2, h); P.push(x, yy - sag * (d / 2) / nn, zz - side * sag * h / nn);
          if (i < NX && j < NS) { const k = b + j * (NX + 1) + i; I.push(...(side > 0 ? [k, k + 1, k + NX + 2, k, k + NX + 2, k + NX + 1] : [k, k + NX + 2, k + 1, k, k + NX + 1, k + NX + 2])); }
        }
      }
      for (const sx of [-1, 1]) { // gables, slightly inset
        const b = P.length / 3, xi = sx * (w / 2 - 0.04); P.push(xi, 0, -d / 2, xi, 0, d / 2, xi, h, 0); I.push(...(sx > 0 ? [b + 1, b, b + 2] : [b, b + 1, b + 2]));
      }
      const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(P, 3)); g.setIndex(I); g.computeVertexNormals();
      const m = new THREE.Mesh(boxUV(g), mat); m.castShadow = m.receiveShadow = true; grp.add(m);
      for (const sx of [-1, 1]) { const x = sx * (w / 2 + 0.01); [-1, 1].forEach((sz) => { const pm = new THREE.Mesh(tube([V(x, 0, sz * d / 2 * 0.98), V(x, h * 0.5, sz * d / 4 * 0.98), V(x, h + 0.02, 0)], 0.011), poleMat); pm.castShadow = true; grp.add(pm); }); }
      grp.add(new THREE.Mesh(tube([V(-w / 2, h + 0.02, 0), V(0, h + 0.015, 0), V(w / 2, h + 0.02, 0)], 0.008), poleMat));
      guyLines(grp, [[V(w / 2, h, 0), V(w / 2 + 0.95, 0, 0)], [V(-w / 2, h, 0), V(-w / 2 - 0.95, 0, 0)], [V(w / 2, 0.02, d / 2), V(w / 2 + 0.25, 0, d / 2 + 0.25)], [V(-w / 2, 0.02, d / 2), V(-w / 2 - 0.25, 0, d / 2 + 0.25)], [V(w / 2, 0.02, -d / 2), V(w / 2 + 0.25, 0, -d / 2 - 0.25)], [V(-w / 2, 0.02, -d / 2), V(-w / 2 - 0.25, 0, -d / 2 - 0.25)]]);
    } else { // dome tent: elliptic dome with two crossing poles and a fly-sheet sag between them
      const g = new THREE.SphereGeometry(1, 48, 16, 0, Math.PI * 2, 0, Math.PI / 2), p = g.attributes.position;
      for (let i = 0; i < p.count; i++) {
        const x = p.getX(i), y = p.getY(i), z = p.getZ(i), a = Math.atan2(z, x), sag = 1 - 0.05 * Math.pow(Math.abs(Math.sin(2 * a)), 0.7) * Math.sin(Math.PI * Math.min(1, y * 1.2));
        p.setXYZ(i, x * (w / 2) * sag, y * h * (1 - 0.03 * (1 - Math.abs(Math.cos(2 * a)))), z * (d / 2) * sag);
      }
      g.computeVertexNormals();
      const m = new THREE.Mesh(boxUV(g), mat); m.castShadow = m.receiveShadow = true; grp.add(m);
      for (const s of [-1, 1]) {
        const al = s * Math.atan2(d, w), R = 1 / Math.sqrt(Math.cos(al) ** 2 / (w / 2) ** 2 + Math.sin(al) ** 2 / (d / 2) ** 2), pts = [];
        for (let k = 0; k <= 16; k++) { const t = (k / 16) * Math.PI; pts.push(V(Math.cos(al) * R * Math.cos(t) * 1.01, h * Math.sin(t) * 1.005, -Math.sin(al) * R * Math.cos(t) * 1.01)); }
        const pm = new THREE.Mesh(tube(pts, 0.011), poleMat); pm.castShadow = true; grp.add(pm);
      }
      guyLines(grp, [[V(w / 2 * 0.62, h * 0.72, d / 2 * 0.3), V(w / 2 + 0.7, 0, d / 2 + 0.4)], [V(-w / 2 * 0.62, h * 0.72, d / 2 * 0.3), V(-w / 2 - 0.7, 0, d / 2 + 0.4)], [V(w / 2 * 0.62, h * 0.72, -d / 2 * 0.3), V(w / 2 + 0.7, 0, -d / 2 - 0.4)], [V(-w / 2 * 0.62, h * 0.72, -d / 2 * 0.3), V(-w / 2 - 0.7, 0, -d / 2 - 0.4)]]);
    }
    return grp;
  }
  function padMesh(ap, dims, door = true) {
    const [w, d, h0] = dims, h = Math.max(0.03, h0), grp = new THREE.Group();
    const g = roundedBox(w, h, d, Math.min(0.02, h / 2.5), 2); g.translate(0, h / 2, 0);
    const m = new THREE.Mesh(g, fabricMat(ap, { door, halfD: Math.min(w, d) / 2, h, markH: true, rough: 0.6 })); m.castShadow = m.receiveShadow = true; grp.add(m);
    return grp;
  }
  function backpackMesh(ap, dims) {
    const [w, d, h] = dims, grp = new THREE.Group(), mat = fabricMat(ap, { door: true, halfD: d / 2, h, rough: 0.7 }), strap = new THREE.MeshStandardMaterial({ color: 0x1d1f22, roughness: 0.8 });
    const add = (g, m, x, y, z) => { const o = new THREE.Mesh(g, m); o.position.set(x, y, z); o.castShadow = o.receiveShadow = true; grp.add(o); return o; };
    add(roundedBox(w * 0.9, h * 0.8, d * 0.78, 0.08), mat, -w * 0.02, h * 0.4, 0); // main bag
    add(roundedBox(w * 0.8, h * 0.18, d * 0.72, 0.06), mat, -w * 0.04, h * 0.84, 0); // lid
    add(roundedBox(w * 0.2, h * 0.42, d * 0.55, 0.05), mat, w * 0.38, h * 0.27, 0); // front pocket (+x)
    [-1, 1].forEach((s) => add(roundedBox(w * 0.1, h * 0.3, d * 0.16, 0.03), mat, 0, h * 0.25, s * d * 0.42)); // side pockets
    [-1, 1].forEach((s) => grp.add(new THREE.Mesh(tube([V(-w * 0.45, h * 0.78, s * d * 0.18), V(-w * 0.5, h * 0.5, s * d * 0.2), V(-w * 0.45, h * 0.14, s * d * 0.2)], 0.018), strap)));
    add(new THREE.TorusGeometry(0.035, 0.008, 6, 12), strap, -w * 0.06, h * 0.95, 0).rotation.y = Math.PI / 2; // grab loop
    return grp;
  }
  function objectMesh(o) {
    const dims = o.dims || { tent: [2.2, 1.8, 1.3], pad: [2, 2, 0.12], backpack: [0.6, 0.4, 0.7] }[o.kind] || [1, 1, 0.1]; // dims = [x, y, height] m
    const m = o.kind === 'tent' ? tentMesh(o, dims) : o.kind === 'backpack' ? backpackMesh(o.appearance, dims) : padMesh(o.appearance, dims);
    m.position.copy(E(o.x, o.y, o.z || 0)); m.rotation.y = ((o.yaw_deg || 0) * Math.PI) / 180;
    return m;
  }

  // ---------- world
  const disposeAll = (g) => g.traverse((o) => { if (o.geometry) o.geometry.dispose(); if (o.material) [].concat(o.material).forEach((m) => { if (![barkMat, leafMat, rockMat, wallMat, roofMat, blobMat, rectMat, poleMat, lineMat, pegMat].includes(m)) m.dispose(); }); });
  const objMeshes = [], treeLOD = [];
  let bbox = { x0: -40, x1: 40, y0: -40, y1: 40 }, lodAt = -1e9, lodPos = V(1e9, 0, 0);
  function setWorld(world) {
    disposeAll(worldG); worldG.clear(); objMeshes.length = 0; treeLOD.length = 0;
    if (!world) return;
    const g0 = world.ground && world.ground.rgb ? world.ground.rgb : [0.36, 0.52, 0.24]; // world.js GROUND colours identify the landscape
    land = g0[1] > 0.45 ? 'meadow' : g0[2] > 0.4 ? 'urban' : 'forest';
    const xs = [world.home[0], ...(world.objects || []).map((o) => o.x)], ys = [world.home[1], ...(world.objects || []).map((o) => o.y)];
    bbox = { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) };
    const [SW, SH] = world.size || [80, 80], M = 25;
    rect = { x0: Math.min(-SW / 2, bbox.x0) - M, x1: Math.max(SW / 2, bbox.x1) + M, y0: Math.min(-SH / 2, bbox.y0) - M, y1: Math.max(SH / 2, bbox.y1) + M };
    keyPts = xs.map((x, i) => [x, ys[i]]);
    buildMask(world); buildGround();
    const LT = LANDTEX[land], tA = tex(LT[0] + '_color', true), tB = tex(LT[1] + '_color', true);
    groundU.tA.value = tA || WHITE; groundU.tB.value = tB || WHITE; groundU.uSA.value = LT[2]; groundU.uSB.value = LT[3]; groundU.uTint.value.setRGB(...LT[4]);
    groundU.uHasTex.value = tA && tB ? 1 : 0; groundU.uFarG.value = land === 'forest' ? 0.8 : 1;
    groundU.uFlat.value.copy(col3(world.ground ? world.ground.rgb : [0.36, 0.52, 0.24]));
    groundMat.normalMap = tex((land === 'urban' ? LT[1] : LT[0]) + '_normal', false); groundMat.normalScale.set(0.7, 0.7); groundMat.needsUpdate = true;
    if (land === 'forest') { grassU.uGA.value.setRGB(0.1, 0.2, 0.05); grassU.uGB.value.setRGB(0.22, 0.3, 0.08); grassU.uHS.value = 0.9; }
    else if (land === 'urban') { grassU.uGA.value.setRGB(0.14, 0.26, 0.07); grassU.uGB.value.setRGB(0.28, 0.38, 0.1); grassU.uHS.value = 0.6; }
    else { grassU.uGA.value.setRGB(0.15, 0.29, 0.06); grassU.uGB.value.setRGB(0.36, 0.44, 0.12); grassU.uHS.value = 1; }

    const ob = world.obstacles || [], by = (t) => ob.filter((o) => o.type === t), up = V(0, 1, 0);
    const inst = (geo, mat, n, shadow = true) => { const im = new THREE.InstancedMesh(geo, mat, Math.max(1, n)); im.count = 0; im.castShadow = shadow; im.receiveShadow = true; worldG.add(im); return im; };
    const Mx = new THREE.Matrix4(), q = new THREE.Quaternion(), s = V(), p = V();
    // trees: r = canopy (obstacle) radius, h = height; two shapes, two detail levels, re-sorted by camera distance
    const trees = by('tree');
    const kinds = [[], []]; trees.forEach((o) => kinds[(hash2(o.x * 0.37, o.y * 0.53) < (land === 'forest' ? 0.7 : 0.3)) ? 0 : 1].push(o));
    kinds.forEach((list, v) => {
      if (!list.length) return;
      const top = v === 0 ? 0.42 : 0.55;
      treeLOD.push({ list, hiF: inst(v === 0 ? coniferGeo(true) : broadleafGeo(true), leafMat, list.length), loF: inst(v === 0 ? coniferGeo(false) : broadleafGeo(false), leafMat, list.length),
        hiT: inst(trunkGeo(true, top), barkMat, list.length), loT: inst(trunkGeo(false, top), barkMat, list.length) });
    });
    lodAt = -1e9;
    // rocks: three noise-displaced shapes; footprint radius r, top at h
    const rocks = by('rock'), rv = [[], [], []]; rocks.forEach((o) => rv[Math.floor(hash2(o.x, o.y) * 3)].push(o));
    rv.forEach((list, k) => {
      if (!list.length) return;
      const im = inst(rockGeo(k * 3.7 + 1, Q.seg), rockMat, list.length); im.count = list.length;
      list.forEach((o, i) => { p.copy(E(o.x, o.y, 0)); s.set(o.r, o.h, o.r); q.setFromAxisAngle(up, (o.x + o.y) * 1.7); Mx.compose(p, q, s); im.setMatrixAt(i, Mx); im.setColorAt(i, col3(o.rgb).multiplyScalar(1.7)); });
    });
    // buildings: exact boxes w x d x h with facade detail drawn in the shader
    const bl = by('building');
    if (bl.length) { const m = new THREE.Mesh(buildingGeo(bl), [wallMat, roofMat]); m.castShadow = m.receiveShadow = true; worldG.add(m); }
    // contact shadows (soft blobs) under everything that stands on the ground
    const flat = new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2);
    const round = [...trees.map((o) => [o.x, o.y, o.r * 1.25]), ...rocks.map((o) => [o.x, o.y, o.r * 1.5]), ...(world.objects || []).map((o) => [o.x, o.y, (o.dims ? Math.max(o.dims[0], o.dims[1]) : 1) * 0.75])];
    if (round.length) { const im = new THREE.InstancedMesh(flat, blobMat, round.length); round.forEach(([x, y, r], i) => { Mx.compose(E(x, y, hgt(x, y) + 0.04), q.identity(), s.set(2 * r, 1, 2 * r)); im.setMatrixAt(i, Mx); }); im.renderOrder = 1; worldG.add(im); }
    if (bl.length) { const im = new THREE.InstancedMesh(flat, rectMat, bl.length); bl.forEach((o, i) => { Mx.compose(E(o.x, o.y, 0.03), q.identity(), s.set(o.w + 2.4, 1, o.d + 2.4)); im.setMatrixAt(i, Mx); }); im.renderOrder = 1; worldG.add(im); }
    for (const o of world.objects || []) { const m = objectMesh(o); worldG.add(m); objMeshes.push([o, m]); }
    // home landing pad (v2 worlds carry its appearance; otherwise a plain concrete pad with an H)
    const hp = world.home_pad || { x: world.home[0], y: world.home[1], z: 0, yaw_deg: 0, dims: [1.2, 1.2, 0.03], appearance: { base: [0.6, 0.61, 0.63], accent: [0.96, 0.96, 0.96], pattern: 'none', period_m: 1 } };
    const home = padMesh(hp.appearance, hp.dims, false); home.position.copy(E(hp.x, hp.y, hp.z || 0)); home.rotation.y = ((hp.yaw_deg || 0) * Math.PI) / 180; worldG.add(home);
    if (world.sun) { const az = (world.sun.azimuth_deg * Math.PI) / 180, el = (world.sun.elevation_deg * Math.PI) / 180; sunDir.copy(E(Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el))).normalize(); env.el = world.sun.elevation_deg; }
    env.light = world.light ?? 1; env.fog = world.fog_m ?? null; applyEnv();
    trailGeo.setDrawRange(0, 0);
  }
  function updateLOD() {
    if (!treeLOD.length) return;
    const now = performance.now(); if (now - lodAt < 300 && camera.position.distanceToSquared(lodPos) < 4) return;
    lodAt = now; lodPos.copy(camera.position);
    const Mx = new THREE.Matrix4(), q = new THREE.Quaternion(), s = V(), p = V(), up = V(0, 1, 0), c = new THREE.Color(), L2 = Q.lod * Q.lod;
    for (const t of treeLOD) {
      let nh = 0, nl = 0;
      for (const o of t.list) {
        const dx = o.x - camera.position.x, dz = -o.y - camera.position.z, near = dx * dx + dz * dz < L2;
        p.copy(E(o.x, o.y, 0)); s.set(o.r, o.h, o.r); q.setFromAxisAngle(up, (o.x * 7.1 + o.y * 3.3) % 6.28); Mx.compose(p, q, s); c.copy(col3(o.rgb)).multiplyScalar(1.05);
        const [F, T, i] = near ? [t.hiF, t.hiT, nh++] : [t.loF, t.loT, nl++];
        F.setMatrixAt(i, Mx); F.setColorAt(i, c); T.setMatrixAt(i, Mx);
      }
      t.hiF.count = t.hiT.count = nh; t.loF.count = t.loT.count = nl;
      [t.hiF, t.hiT, t.loF, t.loT].forEach((m) => { m.instanceMatrix.needsUpdate = true; if (m.instanceColor) m.instanceColor.needsUpdate = true; m.computeBoundingSphere(); });
    }
  }

  // ---------- light, sky, fog
  const env = { theme: 'dark', light: 1, fog: null, el: 45, role: '#f08a3c' };
  const hazeCol = new THREE.Color(), fogCol = new THREE.Color();
  function applyEnv() {
    const L = env.light, low = smooth(40, 18, env.el), fogOn = !!env.fog;
    sun.position.copy(sunDir).multiplyScalar(150);
    sun.color.setRGB(1, 0.93 - 0.12 * low, 0.84 - 0.25 * low);
    sun.intensity = (fogOn ? 1.0 : 3.2) * L * (0.55 + 0.45 * smooth(5, 35, env.el));
    hemi.intensity = (fogOn ? 1.1 : 0.8) * L; scene.environmentIntensity = (fogOn ? 0.6 : 0.95) * L;
    renderer.toneMappingExposure = 0.62 * (0.35 + 0.65 * L);
    // fog colours are display (sRGB) values: three mixes fog in after tone mapping and output conversion
    const kL = 0.5 + 0.5 * L;
    hazeCol.setRGB((0.70 + 0.12 * low) * kL, (0.77 + 0.02 * low) * kL, (0.86 - 0.08 * low) * kL, THREE.SRGBColorSpace);
    fogCol.setRGB(0.80 * kL, 0.82 * kL, 0.85 * kL, THREE.SRGBColorSpace); // eye.js FOG_RGB, dimmed with the light level
    if (sky) {
      const u = sky.material.uniforms; u.sunPosition.value.copy(sunDir); u.uFog.value = fogOn ? 1 : 0; fogCol.getRGB(u.uFogCol.value, THREE.SRGBColorSpace);
      u.uCloud.value = Q.clouds; u.uCloudCol.value.setRGB(1.05, 1.0 - 0.1 * low, 0.95 - 0.2 * low).multiplyScalar(0.9 + 0.4 * (1 - low));
      bakeEnv(); scene.background = null;
    } else scene.background = fogOn ? fogCol.clone() : new THREE.Color(0x9fb6cf).multiplyScalar(0.4 + 0.6 * L);
    scene.fog = new THREE.FogExp2(fogOn ? fogCol.clone() : hazeCol.clone(), fogOn ? 1.98 / env.fog : 1.98 / 2200);
  }
  function setTheme(theme, css, arbCol) {
    env.theme = theme; env.role = arbCol;
    role.color.set(arbCol); role.emissive.set(arbCol); dot.material.color.set(arbCol); trailMat.color.set(arbCol);
    estA.setColor(new THREE.Color(css.ink)); truA.setColor(new THREE.Color(css.ink)); windA.setColor(new THREE.Color(css.gust));
  }
  function setQuality(name) {
    if (!QUAL[name]) return; qName = name; Q = QUAL[name];
    renderer.setPixelRatio(Math.min(devicePixelRatio, Q.dpr));
    const sh = Q.shadow > 0;
    if (renderer.shadowMap.enabled !== sh) { renderer.shadowMap.enabled = sh; scene.traverse((o) => { if (o.material) [].concat(o.material).forEach((m) => { m.needsUpdate = true; }); }); }
    sun.castShadow = sh;
    if (sh && sun.shadow.mapSize.x !== Q.shadow) { sun.shadow.mapSize.set(Q.shadow, Q.shadow); if (sun.shadow.map) { sun.shadow.map.dispose(); sun.shadow.map = null; } }
    grassGeo.instanceCount = Q.grass; grassU.uT.value = Q.patch;
    if (sky) sky.material.uniforms.uCloud.value = Q.clouds;
    lodAt = -1e9; resize();
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
  let topHalf = 40;
  function placeCamera(dt) {
    if (cam.mode === 'free') { const d = cam.focus.clone().sub(cam.prevFocus); camera.position.add(d); controls.target.add(d); controls.update(); }
    else if (cam.mode === 'chase') {
      const k = 1 - Math.exp(-dt * 4); let dy = cam.targetYaw - cam.yaw; dy = Math.atan2(Math.sin(dy), Math.cos(dy)); cam.yaw += dy * k;
      const f = E(Math.cos(cam.yaw), Math.sin(cam.yaw), 0);
      camera.up.set(0, 1, 0);
      camera.position.copy(cam.focus).addScaledVector(f, -cam.d).add(V(0, cam.d * 0.35 + 0.05, 0));
      const gy = hgt(camera.position.x, -camera.position.z) + 0.08; if (camera.position.y < gy) camera.position.y = gy;
      camera.lookAt(cam.focus.clone().addScaledVector(f, cam.d * 0.6));
    } else { // top: frame home, objects and the drone, north up
      const x0 = Math.min(bbox.x0, cam.focusE[0]) - 8, x1 = Math.max(bbox.x1, cam.focusE[0]) + 8, y0 = Math.min(bbox.y0, cam.focusE[1]) - 8, y1 = Math.max(bbox.y1, cam.focusE[1]) + 8;
      const tv = Math.tan((camera.fov * Math.PI) / 360), h = Math.max((y1 - y0) / 2 / tv, (x1 - x0) / 2 / (tv * camera.aspect)) * 1.05;
      const c = E((x0 + x1) / 2, (y0 + y1) / 2, 0);
      camera.position.copy(c).add(V(0, h, 0.001)); camera.up.set(0, 0, -1); camera.lookAt(c);
      topHalf = Math.max(x1 - x0, y1 - y0) * 0.6;
    }
    cam.prevFocus.copy(cam.focus);
  }
  cam.targetYaw = 0; cam.focusE = [0, 0];
  const fwd = V();
  function fitShadow() {
    if (!sun.castShadow) return;
    const top = cam.mode === 'top', ext = top ? clamp(topHalf, 20, 140) : 28;
    camera.getWorldDirection(fwd); fwd.y = 0; if (fwd.lengthSq() > 1e-6) fwd.normalize();
    const c = top ? V(camera.position.x, 0, camera.position.z) : cam.focus.clone().addScaledVector(fwd, ext * 0.45), tx = (2 * ext) / Q.shadow;
    c.set(Math.round(c.x / tx) * tx, 0, Math.round(c.z / tx) * tx);
    sun.target.position.copy(c); sun.position.copy(c).addScaledVector(sunDir, 200); sun.target.updateMatrixWorld();
    const sc = sun.shadow.camera; if (sc.right !== ext) { sc.left = sc.bottom = -ext; sc.right = sc.top = ext; sc.near = 1; sc.far = 450; sc.updateProjectionMatrix(); }
  }

  // ---------- per-frame update
  const q3 = new THREE.Quaternion();
  function update(f, trail, opt) {
    if (!f) return;
    const p = f.quad.pos, q = f.quad.quat;
    drone.visible = true;
    drone.position.copy(E(p[0], p[1], p[2])); q3.set(q[1], q[3], -q[2], q[0]); drone.quaternion.copy(q3);
    dot.position.copy(drone.position); dot.visible = camera.position.distanceTo(drone.position) > 4;
    const gz = hgt(p[0], p[1]), alt = Math.max(0, p[2] - gz);
    droneBlob.visible = alt < 12; droneBlob.position.copy(E(p[0], p[1], gz + 0.02)); droneBlob.scale.setScalar(0.11 + alt * 0.04); droneBlob.material.opacity = 0.5 * Math.exp(-alt / 3);
    cam.focus.copy(drone.position); cam.focusE = [p[0], p[1]]; cam.targetYaw = (f.quad.euler_deg[2] * Math.PI) / 180;
    const ms = f.quad.motors || [0, 0, 0, 0];
    props.forEach((pr, i) => { const w = ms[i] || 0; pr.prop.rotation.y += pr.dir * Math.min(0.9, w * 2.2); pr.disc.material.opacity = Math.min(0.9, w * 1.6); pr.blade.visible = w < 0.25; });
    const n = Math.min(trail.n, TRAIL_N);
    if (n !== trailGeo.drawRange.count) {
      const from = n < trailGeo.drawRange.count ? 0 : trailGeo.drawRange.count;
      for (let i = from; i < n; i++) { trailPos[3 * i] = trail.xyz[3 * i]; trailPos[3 * i + 1] = trail.xyz[3 * i + 2]; trailPos[3 * i + 2] = -trail.xyz[3 * i + 1]; }
      trailGeo.attributes.position.needsUpdate = true; trailGeo.setDrawRange(0, n); trailU.uN.value = n;
    }
    const o = drone.position.clone().add(V(0, -0.03, 0));
    const setA = (a, v, show) => { const L = Math.hypot(v[0], v[1]); a.visible = show && L > 0.05; if (!a.visible) return; a.position.copy(o); a.setDirection(E(v[0] / L, v[1] / L, 0)); const hd = Math.min(L * 0.3, 0.12 + L * 0.02); a.setLength(L, hd, hd * 0.5); };
    setA(estA, f.pi ? f.pi.home_est : [0, 0], !!f.pi && opt.arrows);
    setA(truA, f.pi ? f.pi.home_true : [0, 0], !!f.pi && opt.arrows);
    const w = f.wind || [0, 0, 0], ws = Math.hypot(w[0], w[1], w[2]);
    grassU.uWind.value.copy(E(w[0], w[1], w[2]));
    windA.visible = ws > 0.05; if (windA.visible) { windA.position.copy(drone.position).add(V(0, 0.12, 0)); windA.setDirection(E(w[0] / ws, w[1] / ws, w[2] / ws)); const L = 0.05 * ws * (cam.mode === 'chase' ? Math.max(1, cam.d / 0.7) : 1); windA.setLength(L, Math.min(L * 0.3, 0.04), Math.min(L * 0.08, 0.012)); }
    for (const [ob, m] of objMeshes) m.visible = !opt.present || opt.present(ob);
  }
  function resize() { const w = host.clientWidth, h = host.clientHeight; renderer.setSize(w, h, false); camera.aspect = w / Math.max(1, h); camera.updateProjectionMatrix(); }
  let lastT = performance.now(), clock = 0;
  function render() {
    const now = performance.now(), dt = Math.min(0.1, (now - lastT) / 1000); lastT = now; clock += dt;
    placeCamera(dt); fitShadow(); updateLOD();
    if (sky) { sky.position.copy(camera.position); const u = sky.material.uniforms; u.uCloudOff.value.x += dt * 0.004 * (1 + grassU.uWind.value.x * 0.3); u.uCloudOff.value.y += dt * 0.002 * (1 + grassU.uWind.value.z * 0.3); }
    // grass: lattice centred ahead of the camera, hidden from high overviews where blades would be sub-pixel
    camera.getWorldDirection(fwd); const hc = camera.position.y - hgt(camera.position.x, -camera.position.z);
    grass.visible = hc < 25 && Q.grass > 0; grassU.uC.value.set(camera.position.x + fwd.x * Q.patch * 0.3, camera.position.z + fwd.z * Q.patch * 0.3); grassU.uTime.value = clock;
    // the top view looks down through fog from far above: use fog over horizontal distance only, so the map stays legible
    if (scene.fog && env.fog) scene.fog.density = 1.98 / (env.fog + (cam.mode === 'top' ? 2 * hc : 0));
    renderer.render(scene, camera);
  }
  const tmp = V();
  function project(x, y, z) { tmp.copy(E(x, y, z)).project(camera); if (tmp.z > 1) return null; return [(tmp.x * 0.5 + 0.5) * host.clientWidth, (-tmp.y * 0.5 + 0.5) * host.clientHeight]; }
  function clearTrail() { trailGeo.setDrawRange(0, 0); }
  setQuality(qName); applyEnv();
  return { setWorld, update, render, resize, setCam, setTheme, setQuality, project, clearTrail, NO_GL, get quality() { return qName; }, get cam() { return cam.mode; } };
};
